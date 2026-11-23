#!/usr/bin/env python3
"""第 17 期 Smoke Test（不占 GPU，本地/服务器 CPU 通用）。

覆盖：
1. 手算极小例：4×8 · 8×4 整数矩阵 vs 手算期望（位级整数一致）；
2. 字面三重循环 vs naive rows 两层实现一致性（整数域 + 随机小例）；
3. Epilogue 公式：bias 广播、GELU(tanh)/SiLU 已知值、融合 vs 分离一致；
4. 边界：K=1、非整除形状、shape 校验显式报错、一维输入报错；
5. roofline 手算数：GEMM AI、bound 分类、RMSNorm 对照 AI（与 ch02 同式）；
6. tile 配置表自洽：线程数 ≤1024、smem 公式、每配置 reg 分块 ≥2（本期设计约束）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "implementations" / "cpu"))

from naive_gemm import (  # noqa: E402
    TINY_A,
    TINY_B,
    TINY_BIAS,
    TINY_C_EXPECTED,
    add_bias,
    fused_bias_gelu,
    gelu_tanh,
    matmul_naive_rows,
    matmul_numpy,
    matmul_triple_loop,
    silu,
)
from gemm_configs import CONFIGS, accumulator_count, block_dims, smem_bytes  # noqa: E402
from roofline_gemm import (  # noqa: E402
    PLATFORM_SPEC,
    classify_kernel_bound,
    gemm_arithmetic_intensity,
    rmsnorm_contrast,
    theoretical_bytes,
    theoretical_flops,
)

PASS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise SystemExit(f"Smoke Test 失败于: {name}")


def test_hand_computed_example() -> None:
    # 手算核对：C[0,0]=1*1+2*0+0*2+1*1+3*0+0*1+1*2+1*0=4（逐项见 naive_gemm.py 注释）
    triple = matmul_triple_loop(TINY_A, TINY_B)
    rows = matmul_naive_rows(TINY_A, TINY_B)
    check("手算例：字面三重循环 = 手算期望（整数位级）", np.array_equal(triple, TINY_C_EXPECTED))
    check("手算例：naive rows = 手算期望（整数位级）", np.array_equal(rows, TINY_C_EXPECTED))


def test_two_implementations_agree() -> None:
    rng = np.random.default_rng(20261118)
    A = rng.standard_normal((13, 17), dtype=np.float32)
    B = rng.standard_normal((17, 11), dtype=np.float32)
    a_t = matmul_triple_loop(A, B)
    a_r = matmul_naive_rows(A, B)
    # k 升序同序累加：float32 舍入路径一致（triple 内部 float64 兜底，用 allclose）
    check("13×17·17×11：三重循环 vs naive rows 一致", np.allclose(a_t, a_r, atol=1e-4, rtol=1e-5))
    A64 = rng.standard_normal((64, 64), dtype=np.float32)
    B64 = rng.standard_normal((64, 64), dtype=np.float32)
    check("64³：naive rows vs numpy matmul 一致（FP32 容差）",
          np.allclose(matmul_naive_rows(A64, B64), matmul_numpy(A64, B64), atol=2e-4, rtol=1e-4))


def test_epilogue_formulas() -> None:
    c = matmul_naive_rows(TINY_A, TINY_B)
    bias = TINY_BIAS
    expected_bias = c + bias[None, :]
    check("Bias 广播正确", np.array_equal(add_bias(c, bias), expected_bias))
    check("GELU(0)=0", abs(float(gelu_tanh(np.array([0.0], dtype=np.float32))[0])) < 1e-7)
    gelu1 = float(gelu_tanh(np.array([1.0], dtype=np.float32))[0])
    check("GELU(1) tanh 近似 ≈ 0.841192", abs(gelu1 - 0.841192) < 1e-5, f"gelu(1)={gelu1:.6f}")
    silu1 = float(silu(np.array([1.0], dtype=np.float32))[0])
    check("SiLU(1)=sigmoid(1)≈0.731059", abs(silu1 - 0.731059) < 1e-5, f"silu(1)={silu1:.6f}")
    # 融合 vs 分离（numpy 单遍 vs 两遍）
    fused = fused_bias_gelu(c, bias)
    separate = gelu_tanh(add_bias(c, bias))
    check("GELU：融合 vs 分离逐元素一致", np.allclose(fused, separate, atol=1e-7, rtol=1e-6))


def test_edges_and_errors() -> None:
    rng = np.random.default_rng(7)
    A = rng.standard_normal((5, 1), dtype=np.float32)
    B = rng.standard_normal((1, 9), dtype=np.float32)
    check("K=1 边界（列向量外积）", matmul_naive_rows(A, B).shape == (5, 9))
    try:
        matmul_naive_rows(np.ones((4, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32))
        check("内维不匹配显式报错", False)
    except ValueError:
        check("内维不匹配显式报错", True)
    try:
        matmul_naive_rows(np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        check("一维输入显式报错", False)
    except ValueError:
        check("一维输入显式报错", True)
    try:
        add_bias(np.ones((4, 4), dtype=np.float32), np.ones(5, dtype=np.float32))
        check("bias 长度不符显式报错", False)
    except ValueError:
        check("bias 长度不符显式报错", True)
    # 非连续输入（转置视图）应与连续结果一致
    A_full = rng.standard_normal((8, 6), dtype=np.float32)
    At = A_full.T  # (6,8) 非连续
    c_view = matmul_naive_rows(At, np.ones((8, 3), dtype=np.float32))
    c_cont = matmul_naive_rows(np.ascontiguousarray(At), np.ones((8, 3), dtype=np.float32))
    check("非连续输入结果与连续一致", np.allclose(c_view, c_cont, atol=0, rtol=0))


def test_roofline_hand_numbers() -> None:
    ai_256 = gemm_arithmetic_intensity(256, 256, 256, 4)
    check("AI(256³, FP32) = 2*256/12 ≈ 42.67 FLOP/B", abs(ai_256 - 2 * 256 / 12) < 1e-9,
          f"{ai_256:.4f}")
    ai_1024_fp16 = gemm_arithmetic_intensity(1024, 1024, 1024, 2)
    check("AI(1024³, FP16) = 2*1024/6 ≈ 341.33 FLOP/B", abs(ai_1024_fp16 - 2 * 1024 / 6) < 1e-9,
          f"{ai_1024_fp16:.4f}")
    check("FLOPs(256³) = 2*256³", theoretical_flops(256, 256, 256) == 2 * 256**3)
    check("Bytes(256³, FP32) = 3*256²*4", theoretical_bytes(256, 256, 256, 4) == 3 * 256**2 * 4)

    spec = PLATFORM_SPEC["cuda"]
    verdict_1024 = classify_kernel_bound(
        theoretical_bytes(1024, 1024, 1024, 4), theoretical_flops(1024, 1024, 1024),
        spec["peak_bandwidth_gb_s"] * 1e9, spec["peak_fp32_tflops"] * 1e12)
    check("1024³ FP32 @4090 规格峰 → compute-bound",
          verdict_1024["kernel_bound_class"].startswith("compute-bound"),
          f"t_mem={verdict_1024['roofline_t_mem_ms']}ms t_compute={verdict_1024['roofline_t_compute_ms']}ms")
    verdict_256 = classify_kernel_bound(
        theoretical_bytes(256, 256, 256, 4), theoretical_flops(256, 256, 256),
        spec["peak_bandwidth_gb_s"] * 1e9, spec["peak_fp32_tflops"] * 1e12)
    check("256³ FP32 @4090 规格峰 → memory-bound（AI 42.67 < 脊点 82）",
          verdict_256["kernel_bound_class"].startswith("memory-bound"),
          f"class={verdict_256['kernel_bound_class']}")

    rms = rmsnorm_contrast(2048, 1024, 2)
    check("RMSNorm 对照 AI ≈ 1.0005（ch02 同式）", abs(rms["arithmetic_intensity_flop_per_byte"] - 1.0005) < 0.001,
          f"AI={rms['arithmetic_intensity_flop_per_byte']}")


def test_config_table() -> None:
    for config in CONFIGS:
        bx, by = block_dims(config)
        threads = bx * by
        check(f"配置 {config['name']}：线程数 {threads} ≤ 1024", threads <= 1024)
        check(f"配置 {config['name']}：寄存器分块 {accumulator_count(config)} ≥ 4 且整除",
              accumulator_count(config) >= 4 and config["tm"] % config["rm"] == 0
              and config["tn"] % config["rn"] == 0)
        smem = smem_bytes(config)
        expect = (config["tm"] * (config["tk"] + 1) + config["tk"] * (config["tn"] + 1)) * 4
        check(f"配置 {config['name']}：smem {smem}B 公式一致且 < 48KB", smem == expect and smem < 48 * 1024)
    # 抽查手算值：32x32x8 → (32*9 + 8*33)*4 = (288+264)*4 = 2208 B
    check("抽查 smem(32x32x8) = 2208B", smem_bytes(CONFIGS[2]) == 2208)
    # 64x64x8 → (64*9 + 8*65)*4 = (576+520)*4 = 4384 B
    check("抽查 smem(64x64x8) = 4384B", smem_bytes(CONFIGS[8]) == 4384)


def main() -> int:
    test_hand_computed_example()
    test_two_implementations_agree()
    test_epilogue_formulas()
    test_edges_and_errors()
    test_roofline_hand_numbers()
    test_config_table()
    print("[done] Smoke Test 全部通过（CPU-only，未占 GPU）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
