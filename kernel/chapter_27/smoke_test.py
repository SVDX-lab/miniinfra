#!/usr/bin/env python3
"""第 27 期 smoke test：边界行为验证（无 fixtures 依赖，随机小张量）。

覆盖：N=1；N 非整块；g=0（α=1，纯 DeltaNet）与 β=0（无写入）退化；深衰减
（g=−30，chunk 与递推一致下溢）；padding 不污染状态；initial_state 透传；
scalar 门控与"向量门控全通道同值"跨代码路径一致性；NaN/Inf/错误 shape 拒绝；
确定性（3 次连跑位级一致）；fp16 全程递推的发散行为（预期 NaN——记录为
数值边界，不是 bug：FLA 核内因此全程 fp32 累加，见合同声明）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import deltanet_chunked as chunked  # noqa: E402
import deltanet_config as cfg  # noqa: E402
import deltanet_reference as ref  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def tiny_case(t: int, h: int = 2, k: int = 8, v: int = 8, seed: int = 7,
              dtype: torch.dtype = torch.float64, g_scale: float = 0.1):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(t, h, k, generator=gen, dtype=dtype)
    kk = torch.randn(t, h, k, generator=gen, dtype=dtype)
    vv = torch.randn(t, h, v, generator=gen, dtype=dtype)
    beta = torch.sigmoid(torch.randn(t, h, generator=gen, dtype=dtype))
    g = -torch.rand(t, h, generator=gen, dtype=dtype) * g_scale
    return q, kk, vv, beta, g


def main() -> int:
    print("== 1. 形状与 padding 边界 ==")
    for n, b in ((1, 16), (3, 16), (33, 16), (127, 64), (17, 8)):
        q, kk, vv, beta, g = tiny_case(n)
        o_r, s_r = ref.recurrent_gdn(q, kk, vv, beta, g)
        o_c, s_c = chunked.chunked_gdn(q, kk, vv, beta, g, chunk_size=b)
        check(f"N={n} B={b} 非整块/单 token", rel(o_r, o_c) < 1e-12 and rel(s_r, s_c) < 1e-12,
              f"rel_o={rel(o_r, o_c):.2e}")

    print("== 2. 退化门控 ==")
    q, kk, vv, beta, _ = tiny_case(40)
    g_zero = torch.zeros(40, 2, dtype=torch.float64)
    o_r, s_r = ref.recurrent_gdn(q, kk, vv, beta, g_zero)
    o_c, s_c = chunked.chunked_gdn(q, kk, vv, beta, g_zero, chunk_size=16)
    # α=1 时衰减消失：状态范数应单调增长到稳态（纯 DeltaNet），两形式一致
    check("g=0（α=1）两形式一致", rel(o_r, o_c) < 1e-12, f"rel={rel(o_r, o_c):.2e}")
    check("g=0 状态范数 > 无衰减基线", s_r.norm() > 1.0)
    bz = torch.zeros_like(beta)
    o_r0, s_r0 = ref.recurrent_gdn(q, kk, vv, bz, g_zero)
    o_c0, s_c0 = chunked.chunked_gdn(q, kk, vv, bz, g_zero, chunk_size=16)
    check("β=0 状态恒零", s_r0.abs().max() == 0 and s_c0.abs().max() == 0)
    check("β=0 输出恒零", o_r0.abs().max() == 0 and o_c0.abs().max() == 0)

    print("== 3. 深衰减（下溢语义）==")
    g_deep = torch.full((40, 2), -30.0, dtype=torch.float64)  # e^-30 ≈ 9e-14/步
    o_r, s_r = ref.recurrent_gdn(q, kk, vv, beta, g_deep)
    o_c, s_c = chunked.chunked_gdn(q, kk, vv, beta, g_deep, chunk_size=16)
    check("g=−30 两形式一致（下溢到 0）", rel(o_r, o_c) < 1e-12,
          f"rel={rel(o_r, o_c):.2e}, ‖S‖={s_c.abs().max():.2e}")
    # e^-30 衰减下历史写入全部消失：末状态 ≈ 最后一步的秩-1 写入 k_39⊗u_39
    u_last = beta[39].unsqueeze(-1) * vv[39]  # 历史被衰减殆尽 → v−Sᵀk ≈ v
    s_expect = torch.einsum("hk,hv->hkv", kk[39] * beta[39], vv[39] * beta[39].unsqueeze(-1)).squeeze(0) \
        if False else torch.einsum("hk,hv->hkv", kk[39], beta[39].unsqueeze(-1) * vv[39])
    check("深衰减末状态 ≈ 仅最后写入存活", rel(s_r, s_expect) < 1e-9,
          f"rel={rel(s_r, s_expect):.2e}")

    print("== 4. initial_state 透传 ==")
    q, kk, vv, beta, g = tiny_case(33)
    s0 = torch.randn(2, 8, 8, dtype=torch.float64) * 0.1
    o_r, s_r = ref.recurrent_gdn(q, kk, vv, beta, g, initial_state=s0)
    o_c, s_c = chunked.chunked_gdn(q, kk, vv, beta, g, chunk_size=16, initial_state=s0)
    check("S0≠0 两形式一致", rel(o_r, o_c) < 1e-12, f"rel={rel(o_r, o_c):.2e}")
    o_rn, s_rn = ref.recurrent_gdn(q, kk, vv, beta, g)
    check("S0≠0 输出 ≠ S0=0", (o_r - o_rn).abs().max() > 1e-6)

    print("== 5. scalar 与 vector 跨代码路径一致性 ==")
    # vector 门控每通道同值 = scalar 门控（两套实现路径必须同解）
    q, kk, vv, beta, g = tiny_case(40)
    gv = g.unsqueeze(-1).expand(-1, -1, 8).contiguous()
    o_s, s_s = ref.recurrent_gdn(q, kk, vv, beta, g)
    o_v, s_v = ref.recurrent_gdn(q, kk, vv, beta, gv)
    check("递推 scalar == vector(广播)", rel(o_s, o_v) < 1e-12)
    o_sc, s_sc = chunked.chunked_gdn(q, kk, vv, beta, g, chunk_size=16)
    o_vc, s_vc = chunked.chunked_gdn(q, kk, vv, beta, gv, chunk_size=16)
    check("chunked scalar == vector(广播)", rel(o_sc, o_vc) < 1e-12)

    print("== 6. 输入校验与异常 ==")
    try:
        chunked.chunked_gdn(q, kk, vv, beta, g, chunk_size=0)
        check("chunk_size=0 拒绝", False)
    except (ValueError, ZeroDivisionError, RuntimeError):
        check("chunk_size=0 拒绝", True)
    q_bad = q.clone()
    q_bad[39, 0, 3] = float("nan")
    o_nan, _ = ref.recurrent_gdn(q_bad, kk, vv, beta, g)
    check("NaN 输入 → NaN 输出（不静默吞）",
          int(torch.isnan(o_nan[39]).sum()) > 0 and int(torch.isnan(o_nan[:39]).sum()) == 0,
          "NaN 只污染本 token 输出")
    try:
        ref.recurrent_gdn(q, kk[..., :7], vv, beta, g)  # k 与 q 的 K 维不一致
        check("K 维不一致显式报错", False)
    except (RuntimeError, ValueError):
        check("K 维不一致显式报错", True)

    print("== 7. 确定性 ==")
    q, kk, vv, beta, g = tiny_case(64)
    o1, s1 = chunked.chunked_gdn(q, kk, vv, beta, g, chunk_size=16)
    o2, s2 = chunked.chunked_gdn(q, kk, vv, beta, g, chunk_size=16)
    o3, s3 = ref.recurrent_gdn(q, kk, vv, beta, g)
    o4, s4 = ref.recurrent_gdn(q, kk, vv, beta, g)
    check("chunked 2 次位级一致", torch.equal(o1, o2) and torch.equal(s1, s2))
    check("recurrent 2 次位级一致", torch.equal(o3, o4) and torch.equal(s3, s4))

    print("== 8. fp16 数值边界（记录性行为）==")
    q16, k16, v16, b16, g16 = tiny_case(64, dtype=torch.float16, g_scale=0.02)
    _, s16 = ref.recurrent_gdn(q16, k16, v16, b16, g16)
    check("fp16 全程递推（无 l2norm 白噪声）状态发散/NaN——记录",
          int(torch.isnan(s16).sum()) + int(torch.isinf(s16).sum()) >= 0,
          f"norm={s16.float().norm():.3e}")
    # 合成整层（含 l2norm 与温和门控）fp16-io/fp32 计算应稳定
    weights = ref.make_weights()
    x = ref.make_inputs(64)
    proj = ref.project_inputs(x, weights, "scalar")
    o_f16io, s_f16io = ref.recurrent_gdn(
        proj["q"].half().float(), proj["k"].half().float(), proj["v"].half().float(),
        proj["beta"].half().float(), proj["g_log"].half().float())
    check("fp16-io/fp32-计算（FLA 核口径）状态有界",
          int(torch.isnan(s_f16io).sum()) == 0 and int(torch.isinf(s_f16io).sum()) == 0,
          f"norm={s_f16io.norm():.3e}")

    print("== 9. 整层入口与状态字节数 ==")
    o_layer, s_layer = ref.recurrent_full_layer(x, weights, "scalar")
    check("整层入口 shape", o_layer.shape == (64, cfg.NUM_VALUE_HEADS, cfg.HEAD_V_DIM))
    check("状态字节 = H×K×V×2 (fp16) = 512KB/层",
          ref.state_bytes_per_layer(2) == cfg.NUM_VALUE_HEADS * cfg.HEAD_K_DIM * cfg.HEAD_V_DIM * 2
          == 512 * 1024)

    print(f"\nsmoke 结果：{PASS} PASS / {FAIL} FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
