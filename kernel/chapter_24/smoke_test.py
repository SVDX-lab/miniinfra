#!/usr/bin/env python3
"""第 24 期边界行为验证（smoke test）。

CPU 部分（本地可跑）：
  S1 N=1 边界：两形式等价（fixtures n1）
  S2 rope_matrix 正交性（R^T R = I，FP64）
  S3 相对位置性质 R_a^T R_b = R_{b-a}（FP64）
  S4 apply_rope 与显式矩阵 R_pos 逐位一致（FP64）
  S5 权重缺 key → KeyError 显式报错
  S6 N=0 空缓存：CPU 参考实现行为观察（torch 语义返回零向量；合同仍规定
     CUDA 核 N≥1 显式报错——见 S14，两种行为都如实记录）
  S7 大 score 数值稳定性：softmax 减 max 保证有限（不产生 NaN/Inf）
  S8 非有限输入传播：NaN 输入 → 输出 NaN（观察项）

CUDA 部分（--backend cuda，4090 服务器）：
  S9 dtype 合同：fp32 输入 → RuntimeError
  S10 非连续输入 → RuntimeError
  S11 N=0 → RuntimeError
  S12 H 不一致 → RuntimeError
  S13 CPU Tensor → RuntimeError
  S14 splits > N：N=3 + splits=16 空分块仍正确（vs torch 吸收核）
  S15 splits=1 与启发式 splits 结果一致性（容差内，分块归约顺序不同允许舍入差）

用法：
    python smoke_test.py                 # CPU 部分
    python smoke_test.py --backend cuda  # CPU + CUDA 部分（服务器）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from mla_reference import (  # noqa: E402
    apply_rope,
    mla_core_absorbed_bmm,
    mla_decode_absorbed,
    mla_decode_unabsorbed,
    q_projection_stage,
    rope_freqs,
    rope_matrix,
    rotate_q_rope,
)
import mla_config as cfg  # noqa: E402

WEIGHT_NAMES = ("W_QA", "W_QB", "W_KVA", "W_UK", "W_UV", "W_O")
PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    tag = "PASS" if condition else "FAIL"
    print(f"[smoke/{tag}] {name} {detail}")
    (PASSED if condition else FAILED).append(name)


def fixture_arrays() -> dict:
    data = np.load(HERE / "fixtures" / "correctness.npz")
    return {k: data[k] for k in data.files}


def smoke_cpu() -> None:
    arrays = fixture_arrays()
    freqs64 = rope_freqs().double()
    weights32 = {name: torch.from_numpy(arrays[f"weight_{name}"]) for name in WEIGHT_NAMES}

    # S1
    n = 1
    x = torch.from_numpy(arrays["x"])
    c1 = torch.from_numpy(arrays["n1_c_kv"])
    k1 = torch.from_numpy(arrays["n1_k_rope_rot"])
    u = mla_decode_unabsorbed(x, weights32, c1, k1, n, rope_freqs())
    a = mla_decode_absorbed(x, weights32, c1, k1, n, rope_freqs())
    check("S1 N=1 两形式等价", (u["out"] - a["out"]).abs().max().item() < 1e-4,
          f"max={float((u['out'] - a['out']).abs().max()):.3e}")

    # S2
    r = rope_matrix(123.456, freqs64)
    ident = (r.T @ r - torch.eye(64, dtype=torch.float64)).abs().max().item()
    check("S2 R 正交性", ident < 1e-12, f"max={ident:.3e}")

    # S3
    diff = (rope_matrix(13.0, freqs64).T @ rope_matrix(129.0, freqs64)
            - rope_matrix(116.0, freqs64)).abs().max().item()
    check("S3 相对位置性质", diff < 1e-12, f"max={diff:.3e}")

    # S4
    k_raw = torch.from_numpy(arrays["n128_k_rope_raw"]).double()
    pos = torch.arange(128, dtype=torch.float64)
    via_apply = apply_rope(k_raw, pos, freqs64)
    via_matrix = torch.stack([k_raw[t] @ rope_matrix(float(t), freqs64).T for t in range(128)])
    check("S4 apply_rope 与 R_pos 一致", (via_apply - via_matrix).abs().max().item() < 1e-12,
          f"max={float((via_apply - via_matrix).abs().max()):.3e}")

    # S5
    try:
        q_projection_stage(x, {k: v for k, v in weights32.items() if k != "W_QA"})
        check("S5 缺 key 显式报错", False, "未触发 KeyError")
    except KeyError:
        check("S5 缺 key 显式报错", True)

    # S6
    c0 = torch.zeros(0, cfg.KV_LORA_RANK)
    k0 = torch.zeros(0, cfg.QK_ROPE_HEAD_DIM)
    out0 = mla_decode_unabsorbed(x, weights32, c0, k0, 0, rope_freqs())["out"]
    check("S6 N=0 CPU 行为观察（返回零向量，见合同）",
          bool(torch.isfinite(out0).all()) and out0.abs().max().item() == 0.0,
          "torch softmax 对空维返回空张量 → einsum 得零 → out=0；CUDA 核按合同显式拒绝 N=0")

    # S7
    q_nope, q_rope_raw = q_projection_stage(x, weights32)
    q_rope = rotate_q_rope(q_rope_raw, 128, rope_freqs())
    k128 = torch.from_numpy(arrays["n128_k_rope_rot"])
    c128 = torch.from_numpy(arrays["n128_c_kv"])
    core_big = mla_core_absorbed_bmm(q_nope * 1e3, q_rope * 1e3, c128, k128, weights32)
    finite = bool(torch.isfinite(core_big["scores"]).all() and torch.isfinite(core_big["attn"]).all())
    check("S7 大 score 稳定性（softmax 减 max）", finite,
          f"score_max={float(core_big['scores'].abs().max()):.1f}")

    # S8
    core_nan = mla_core_absorbed_bmm(q_nope, q_rope, c128, k128, weights32)
    nan_out = mla_decode_unabsorbed(torch.full_like(x, float("nan")), weights32, c128, k128, 128,
                                    rope_freqs())["out"]
    check("S8 NaN 输入传播（观察项）", bool(torch.isnan(nan_out).all()),
          "NaN 输入 → 输出全 NaN（预期，无数值防线需求）")


def smoke_cuda(device: torch.device) -> None:
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    module = build_ext.get_module()
    arrays = fixture_arrays()
    freqs16 = rope_freqs().to(device=device)
    w16 = {name: torch.from_numpy(arrays[f"weight_{name}"]).to(device=device, dtype=torch.float16)
           for name in WEIGHT_NAMES}
    x16 = torch.from_numpy(arrays["x"]).to(device=device, dtype=torch.float16)
    c16 = torch.from_numpy(arrays["n3_c_kv"]).to(device=device, dtype=torch.float16)
    k16 = torch.from_numpy(arrays["n3_k_rope_rot"]).to(device=device, dtype=torch.float16)
    q_nope, q_rope_raw = q_projection_stage(x16, w16)
    q_rope = rotate_q_rope(q_rope_raw, 3, freqs16)
    q_abs = torch.matmul(q_nope.unsqueeze(1), w16["W_UK"].transpose(1, 2)).squeeze(1)

    def expect_error(name: str, fn) -> None:
        try:
            fn()
            check(name, False, "未触发 RuntimeError")
        except RuntimeError as error:
            check(name, True, f"{type(error).__name__}: {str(error)[:80]}")

    # S9
    expect_error("S9 dtype 合同（fp32 被拒）",
                 lambda: module.mla_absorbed_attention(
                     q_abs.float(), q_rope.float(), c16.float(), k16.float(),
                     cfg.SOFTMAX_SCALE, 0))
    # S10
    c_noncontig = c16.repeat(1, 2)[:, :cfg.KV_LORA_RANK]
    check("S10 输入为非连续切片", not c_noncontig.is_contiguous())
    expect_error("S10 非连续输入被拒",
                 lambda: module.mla_absorbed_attention(
                     q_abs, q_rope, c_noncontig, k16, cfg.SOFTMAX_SCALE, 0))
    # S11
    expect_error("S11 N=0 被拒",
                 lambda: module.mla_absorbed_attention(
                     q_abs, q_rope, torch.zeros(0, cfg.KV_LORA_RANK, device=device, dtype=torch.float16),
                     torch.zeros(0, cfg.QK_ROPE_HEAD_DIM, device=device, dtype=torch.float16),
                     cfg.SOFTMAX_SCALE, 0))
    # S12
    expect_error("S12 H 不一致被拒",
                 lambda: module.mla_absorbed_attention(
                     q_abs[:8], q_rope, c16, k16, cfg.SOFTMAX_SCALE, 0))
    # S13
    expect_error("S13 CPU Tensor 被拒",
                 lambda: module.mla_absorbed_attention(
                     q_abs.cpu(), q_rope.cpu(), c16.cpu(), k16.cpu(), cfg.SOFTMAX_SCALE, 0))

    # S14 splits > N
    core_torch = mla_core_absorbed_bmm(q_nope, q_rope, c16, k16, w16, cfg.SOFTMAX_SCALE)
    s_forced = module.mla_absorbed_attention(q_abs, q_rope, c16, k16, cfg.SOFTMAX_SCALE, 16)
    err = (s_forced.float() - core_torch["s"].float()).abs().max().item()
    check("S14 splits=16 > N=3 空分块正确", err < 5e-3, f"max={err:.3e}")

    # S15 splits=1 vs 启发式
    s_h = module.mla_absorbed_attention(q_abs, q_rope, c16, k16, cfg.SOFTMAX_SCALE, 0)
    s_1 = module.mla_absorbed_attention(q_abs, q_rope, c16, k16, cfg.SOFTMAX_SCALE, 1)
    err = (s_h.float() - s_1.float()).abs().max().item()
    check("S15 splits=1 vs 启发式一致（容差内）", err < 5e-3, f"max={err:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    arguments = parser.parse_args()
    smoke_cpu()
    if arguments.backend == "cuda":
        device = torch.device("cuda")
        smoke_cuda(device)
    print(f"\n[smoke] PASS={len(PASSED)} FAIL={len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print("  FAILED:", name)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
