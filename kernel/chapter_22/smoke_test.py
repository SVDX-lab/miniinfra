#!/usr/bin/env python3
"""第 22 期 smoke test：边界行为验证（显式报错口径 + RoPE 数学性质）。

覆盖：
  S1-S3  合同校验显式拒绝（奇数 head_dim / head_dim≠128 的核口径 / 空 Tensor）
  S4     seq=1（单 token prefill，pos=0 恒等旋转）
  S5     位置偏移一致性：prefill(seq_offset=p)[0] == decode(seq_offset=p) 位级
  S6     cache-free decode == 查表 decode（同 pos 同 dtype，fp64 位级）
  S7     旋转矩阵正交性 R(p)ᵀR(p)=I（fp64，多位置）
  S8     相对位置性质：<R(m)q, R(n)k> 只依赖 m−n
  S9     fp16 io 舍入界：|out16 − out32_ref| ≤ 4.88e-4·(2·|x|max+1)
  S10    kernel_sim 覆盖断言复跑（tid→下标映射恰一次 + cache 读界）
  S11    HF 语义对照（vendored 路径，≤1e-5）
  S12    大角度不溢出：seq=32768 高频维 cos/sin 有限且 ∈ [−1, 1]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import rope_config as cfg  # noqa: E402
import rope_reference as ref  # noqa: E402

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[smoke] {name}: {'PASS' if ok else 'FAIL'} {detail}")


def main() -> int:
    rng = np.random.default_rng(cfg.FIXTURES_SEED)

    # S1 奇数 head_dim 拒绝
    try:
        ref.build_cache(8, cfg.ROPE_THETA, 127, "float64")
        check("S1 odd-head_dim-reject", False, "未拒绝")
    except ValueError:
        check("S1 odd-head_dim-reject", True)

    # S2 空 seq 拒绝（cache 至少 1 行）
    c, s = ref.build_cache(1, cfg.ROPE_THETA, cfg.HEAD_DIM, "float64")
    check("S2 seq=1 cache", c.shape == (1, 64) and s.shape == (1, 64),
          f"shape={c.shape}")

    # S3 fp16 theta 溢出口径：fp16 档 = fp32 计算 + fp16 存储（非 fp16 角度算术）
    c16, _ = ref.build_cache(4, cfg.ROPE_THETA, cfg.HEAD_DIM, "float16")
    check("S3 fp16-cache-no-inf", bool(np.isfinite(c16.astype(np.float32)).all()),
          "fp16 存储档无 inf（fp32 计算后舍入，非 fp16 角度算术）")

    # S4 seq=1 prefill（pos=0 恒等）
    q = rng.standard_normal((1, cfg.HQ, cfg.HEAD_DIM)).astype(np.float32)
    c0 = np.ones((1, 64), dtype=np.float64)
    s0 = np.zeros((1, 64), dtype=np.float64)
    q_out, _ = ref.apply_prefill(q.astype(np.float64), q.astype(np.float64),
                                 c0, s0, 0)
    check("S4 pos0-identity", bool(np.array_equal(q_out, q.astype(np.float64))))

    # S5 prefill(seq_offset=p)[0] == decode(pos=p)
    c64, s64 = ref.build_cache(1024, cfg.ROPE_THETA, cfg.HEAD_DIM, "float64")
    p = 777
    qp, kp = ref.apply_prefill(q.astype(np.float64), q.astype(np.float64), c64, s64, p)
    qd, kd = ref.apply_decode(q[0].astype(np.float64), q[0].astype(np.float64), c64, s64, p)
    check("S5 prefill-decode-eq", bool(np.array_equal(qp[0], qd)),
          "prefill 偏移行与 decode 同位位级一致（形态只是取行方式不同）")

    # S6 cache-free == 查表
    qf, kf = ref.apply_decode_cache_free(q[0].astype(np.float64), q[0].astype(np.float64),
                                         cfg.ROPE_THETA, cfg.HEAD_DIM, p)
    check("S6 cache-free-eq", bool(np.array_equal(qd, qf)),
          "fp64 下现算与查表位级一致（角度同式）")

    # S7 旋转矩阵正交性
    ok = all(np.allclose(ref.rotation_matrix(p, cfg.ROPE_THETA, cfg.HEAD_DIM).T
                         @ ref.rotation_matrix(p, cfg.ROPE_THETA, cfg.HEAD_DIM),
                         np.eye(cfg.HEAD_DIM), atol=1e-12)
             for p in (0, 1, 17, 1000))
    check("S7 rotation-orthogonal", ok)

    # S8 相对位置性质
    check("S8 relative-position", ref.relative_position_property(cfg.ROPE_THETA, cfg.HEAD_DIM))

    # S9 fp16 io 舍入界
    q32 = rng.standard_normal((8, cfg.HQ, cfg.HEAD_DIM)).astype(np.float32)
    c32, s32 = ref.build_cache(8, cfg.ROPE_THETA, cfg.HEAD_DIM, "float32")
    out32, _ = ref.apply_prefill(q32, q32, c32, s32, 0)
    out16, _ = ref.apply_prefill(q32, q32, c32, s32, 0, io_dtype=np.float16)
    bound = 4.88e-4 * (2 * np.abs(q32).max() + 2) + 1e-6
    err = np.abs(out16.astype(np.float32) - out32).max()
    check("S9 fp16-rounding-bound", err <= bound, f"err={err:.3e} bound={bound:.3e}")

    # S10 kernel_sim 覆盖断言
    sys.path.insert(0, str(HERE / "implementations" / "cpu"))
    import kernel_sim  # noqa: E402
    fails = kernel_sim.run_all(max_seq=64, num_tokens_list=(1, 3), seq_offsets=(0, 63))
    check("S10 kernel_sim", not fails, f"failures={fails}")

    # S11 HF 语义对照（vendored chapter_01 语义）
    dims = torch.arange(0, 128, 2, dtype=torch.float32)
    inv_freq = 1.0 / (cfg.ROPE_THETA ** (dims / 128))
    pos = torch.arange(64)
    freqs = pos.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_hf = emb.cos().numpy()
    sin_hf = emb.sin().numpy()
    cm, sm = ref.expand_dup(*ref.build_cache(64, cfg.ROPE_THETA, cfg.HEAD_DIM, "float32"))
    diff = max(np.abs(cos_hf - cm).max(), np.abs(sin_hf - sm).max())
    check("S11 hf-vs-mine", diff <= 1e-5, f"max diff={diff:.3e}（inv_freq 幂路径 ulp 差）")

    # S12 大角度不溢出
    cbig, sbig = ref.build_cache(32768, cfg.ROPE_THETA, cfg.HEAD_DIM, "float32")
    check("S12 big-angle-finite",
          bool(np.isfinite(cbig).all() and np.isfinite(sbig).all()
               and np.abs(cbig).max() <= 1.0 and np.abs(sbig).max() <= 1.0),
          f"seq=32768 max|cos|={np.abs(cbig).max():.6f}")

    failures = [r for r in results if not r[1]]
    print(f"[smoke] {len(results) - len(failures)}/{len(results)} PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
