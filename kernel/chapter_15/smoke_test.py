#!/usr/bin/env python3
"""第 15 期 smoke test（边界行为验证，CPU-only，本地与 S4000 host 双跑）。

覆盖量化路径的边界语义：
1. 全零列保护（scale=1，q 全 0，dequant 仍为 0）；
2. int8 裁剪边界（权重恰好 127×scale / 超界 clip）；
3. 单列/单行权重（K=1、N=1 的 per-channel scale 退化）；
4. 常数权重列（amax=|c|，q=±127 饱和，dequant=sign(c)×amax 精确恢复方向）；
5. 量化-反量化恒等性：dequant(quant(w)) 与 w 的最大误差上界 = scale/2（RNE）；
6. 反量化 GEMM 与直接 GEMM 的误差界 sanity（每项误差 ≤ scale/2 × |a| 的累积）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "implementations" / "cpu"))

from quant_reference import (
    QMAX,
    dequantize_weight,
    gemm_fp32_truth,
    quantize_weight_perchannel,
)

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"SMOKE {name} PASS {detail}")
    else:
        FAIL += 1
        print(f"SMOKE {name} FAIL {detail}")


def main() -> int:
    rng = np.random.default_rng(20261121)

    # 1. 全零列保护
    w = rng.standard_normal((16, 4)).astype(np.float32)
    w[:, 2] = 0.0
    q, s = quantize_weight_perchannel(w)
    wh = dequantize_weight(q, s)
    check(
        "zero_column",
        bool(s[2] == 1.0 and np.all(q[:, 2] == 0) and np.all(wh[:, 2] == 0)),
        f"scale[2]={s[2]}",
    )

    # 2. 裁剪边界性质：per-column amax 归一化下最大比值恒为 127，
    #    clip 到 [-127,127] 是防御性冗余（混合来源权重 / 手工注入极端 q 时才触发）。
    #    这里直接对"手工越界 q"验证 dequant 不越 int8 表示域。
    w = np.zeros((4, 2), dtype=np.float32)
    w[:, 0] = [127.4, -127.4, 1.0, -1.0]
    w[:, 1] = [127.6, 0.5, -0.5, 3.0]
    q, s = quantize_weight_perchannel(w)
    check("amax_normalizes_to_qmax", bool(np.abs(q).max() == QMAX))
    q_manual = np.clip(q.copy(), -127, 127)
    q_manual[0, 1] = 127  # 强制饱和值
    wh_manual = dequantize_weight(q_manual, s)
    check(
        "clip_defensive",
        bool(np.abs(wh_manual).max() <= np.abs(w).max() * (1 + 1.0 / 127) + 1e-6),
        f"dequant max={wh_manual.max():.4f}",
    )

    # 3. K=1 / N=1 退化
    w = rng.standard_normal((1, 7)).astype(np.float32)
    q, s = quantize_weight_perchannel(w)
    wh = dequantize_weight(q, s)
    expected = np.abs(w[0]) / QMAX
    check("k1_scale_direction", bool(np.allclose(s, expected)), f"scale={s[:3]}")
    w = rng.standard_normal((9, 1)).astype(np.float32)
    q, s = quantize_weight_perchannel(w)
    check(
        "n1_single_scale",
        bool(s.shape == (1,) and abs(s[0] - np.abs(w).max() / QMAX) < 1e-9),
    )

    # 4. 常数列方向恢复
    w = np.full((8, 3), -2.5, dtype=np.float32)
    w[:, 1] = 2.5
    q, s = quantize_weight_perchannel(w)
    wh = dequantize_weight(q, s)
    check(
        "const_column_direction",
        bool(np.all(q[:, 0] == -127) and np.allclose(wh[:, 0], -2.5, atol=2.5 / 127)),
        f"dequant[0,0]={wh[0, 0]:.5f}",
    )

    # 5. 恒等误差上界 scale/2
    w = (rng.standard_normal((64, 32)) * 0.05).astype(np.float32)
    q, s = quantize_weight_perchannel(w)
    wh = dequantize_weight(q, s)
    err = np.abs(wh - w)
    bound = (s[None, :] / 2.0) + 1e-12
    check("identity_error_bound", bool(np.all(err <= bound)), f"max err/scale = {(err / s[None, :]).max():.4f}")

    # 6. GEMM 误差界 sanity：每项 |Δ| ≤ scale/2 × |a| → 逐元素累积上界
    a = (rng.standard_normal((4, 64)) * 0.5).astype(np.float16).astype(np.float32)
    y_direct = gemm_fp32_truth(a, w)
    y_deq = gemm_fp32_truth(a, wh)
    err_y = np.abs(y_direct - y_deq)
    # bound[i,j] = Σ_k |a[i,k]| × scale[j]/2（每项权重误差 ≤ scale/2，方向随机不叠加符号）
    bound_y = np.abs(a).sum(axis=1, keepdims=True) * (s / 2.0)[None, :] + 1e-9
    check("gemm_error_bound", bool(np.all(err_y <= bound_y)), f"max_ratio={float((err_y / bound_y).max()):.3f}")

    print(f"{{\"mode\":\"smoke\",\"passed\":{PASS},\"failed\":{FAIL}}}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
