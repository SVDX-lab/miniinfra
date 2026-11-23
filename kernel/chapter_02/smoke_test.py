#!/usr/bin/env python3
"""第 02 期 Smoke Test：CPU-only、不占 GPU 的最低成本检查。

覆盖第 02 期合同中的边界行为条目（§8"非整 Tile、空 Tensor、极端数值和
错误输入行为"的 CPU 侧部分；Kernel 侧行为在服务器上由 compare_correctness.py
的 edge 链覆盖）：
1. 手算 2×4 极小例 vs 朴素串行实现 vs 行内 NumPy（vs torch 组合表达式，可选）；
2. eps 位置：全零输入输出全零（eps 在根号内），不产生 NaN；
3. NaN 按行传播（一行 NaN 不影响另一行）；
4. 极端数值：FP32 路径大数平方（合同记录 FP32 平方溢出阈值 ~1e19）；
5. 串行（自然顺序累加）vs 行内 NumPy（pairwise）的归约顺序差异定量；
6. 错误 Shape / 空 Tensor / 错误 dtype 显式报错；
7. 非连续输入行为记录（CPU 参考接受任意布局；Kernel 合同要求连续，第 06 期）。

torch 为可选依赖（本机无 torch 时跳过 torch 相关检查并显式标注，
torch 检查在服务器 CPU 侧补跑）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from naive_rmsnorm import (  # noqa: E402
    RMS_NORM_EPS,
    hand_computed_example,
    naive_rmsnorm_rowwise,
    naive_rmsnorm_serial,
)

PASS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise SystemExit(f"Smoke Test 失败于: {name}")


def skip(name: str) -> None:
    print(f"[SKIP] {name} — torch 不可用（服务器 CPU 侧补跑）")


def test_hand_computed_example() -> None:
    x, weight, expected = hand_computed_example()
    serial = naive_rmsnorm_serial(x, weight, RMS_NORM_EPS)
    check("手算例 vs 朴素串行实现", np.allclose(serial, expected, atol=1e-5))
    rowwise = naive_rmsnorm_rowwise(x, weight, RMS_NORM_EPS)
    check("手算例 vs 行内 NumPy 实现", np.allclose(rowwise, expected, atol=1e-5))
    if TORCH_AVAILABLE:
        from naive_rmsnorm import composite_rmsnorm_torch
        torch_result = composite_rmsnorm_torch(
            torch.from_numpy(x), torch.from_numpy(weight), RMS_NORM_EPS
        ).numpy()
        check("手算例 vs torch 组合表达式", np.allclose(torch_result, expected, atol=1e-5))
    else:
        skip("手算例 vs torch 组合表达式")


def test_eps_and_zero_input() -> None:
    x = np.zeros((2, 8), dtype=np.float32)
    weight = np.ones(8, dtype=np.float32)
    result = naive_rmsnorm_serial(x, weight, RMS_NORM_EPS)
    check("全零输入输出全零（eps 在根号内，无 NaN）", np.all(result == 0.0))


def test_nan_propagation() -> None:
    x = np.ones((2, 8), dtype=np.float32)
    x[0, 0] = np.float32("nan")
    result = naive_rmsnorm_serial(x, np.ones(8, dtype=np.float32), RMS_NORM_EPS)
    check(
        "NaN 输入按逐行传播（NaN 行整行 NaN，另一行不受影响）",
        bool(np.isnan(result[0]).all()) and bool(np.isfinite(result[1]).all()),
    )


def test_extreme_values() -> None:
    # |x| = 1e4：x^2 = 1e8，FP32 平方和完全可表示；缩放后输出回正常范围。
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((4, 1024)) * 1e4).astype(np.float32)
    result = naive_rmsnorm_serial(x, np.ones(1024, dtype=np.float32), RMS_NORM_EPS)
    check("极端数值（|x|~1e4，FP32 平方不溢出）输出有限", bool(np.isfinite(result).all()))
    # |x| ~ 3e19 > sqrt(FLT_MAX)≈1.8e19：平方和溢出为 +Inf -> scale=1/Inf=0
    # -> 输出全零（与 torch 组合表达式 rsqrt(inf)=0 行为一致；无 NaN/Inf 传播）。
    x_big = np.full((1, 64), 3e19, dtype=np.float32)
    result_big = naive_rmsnorm_serial(x_big, np.ones(64, dtype=np.float32), RMS_NORM_EPS)
    check(
        "FP32 平方溢出（|x|>1.8e19）：平方和=+Inf，scale=0，输出全零（合同如实记录）",
        bool(np.all(result_big == 0.0)) and not bool(np.isnan(result_big).any()),
    )


def test_reduction_order_difference() -> None:
    # 归约顺序差异：串行（自然顺序）vs 行内 NumPy（pairwise）。
    # 这是第 07 期确定性主题的引子素材，量化后写进 fixtures manifest。
    worst = 0.0
    for h in (127, 1024, 1025):
        rng = np.random.default_rng(h)
        x = rng.standard_normal((32, h)).astype(np.float32)
        w = rng.standard_normal(h).astype(np.float32)
        diff = np.abs(naive_rmsnorm_serial(x, w) - naive_rmsnorm_rowwise(x, w)).max()
        worst = max(worst, float(diff))
    check("串行 vs 行内 NumPy（归约顺序差异）max diff < 1e-5", worst < 1e-5, f"worst={worst:.3e}")


def test_invalid_inputs() -> None:
    def expect_error(name: str, fn) -> None:
        try:
            fn()
            check(name, False)
        except ValueError:
            check(name, True)

    expect_error("一维输入显式报错", lambda: naive_rmsnorm_serial(np.ones(8, dtype=np.float32), np.ones(8, dtype=np.float32)))
    expect_error("w 长度不匹配显式报错", lambda: naive_rmsnorm_serial(np.ones((2, 8), dtype=np.float32), np.ones(7, dtype=np.float32)))
    expect_error("H=0 空 Tensor 显式报错", lambda: naive_rmsnorm_serial(np.ones((2, 0), dtype=np.float32), np.ones(0, dtype=np.float32)))
    expect_error(
        "FP64 输入显式报错（真值路径只接受 float32）",
        lambda: naive_rmsnorm_serial(np.ones((2, 8), dtype=np.float64), np.ones(8, dtype=np.float64)),
    )


def test_non_contiguous_behavior() -> None:
    x_nc = np.arange(32, dtype=np.float32).reshape(8, 4).T  # [4, 8] 非连续布局
    result_np = naive_rmsnorm_serial(x_nc, np.ones(8, dtype=np.float32))
    check(
        "非连续输入：CPU 串行参考接受任意布局（NumPy 语义）",
        result_np.shape == (4, 8) and np.isfinite(result_np).all(),
    )
    if TORCH_AVAILABLE:
        from naive_rmsnorm import composite_rmsnorm_torch
        base = torch.randn(8, 4)
        transposed = base.t()  # [4, 8]，非连续
        result = composite_rmsnorm_torch(transposed, torch.ones(8), RMS_NORM_EPS)
        contiguous = composite_rmsnorm_torch(transposed.contiguous(), torch.ones(8), RMS_NORM_EPS)
        check(
            "非连续输入：torch 组合表达式接受任意 stride 且结果一致（Kernel 合同要求连续，第 06 期主题）",
            torch.equal(result, contiguous),
        )
    else:
        skip("非连续输入：torch 组合表达式 stride 行为")


def main() -> int:
    test_hand_computed_example()
    test_eps_and_zero_input()
    test_nan_propagation()
    test_extreme_values()
    test_reduction_order_difference()
    test_invalid_inputs()
    test_non_contiguous_behavior()
    print(f"[done] Smoke Test 通过（CPU-only，不占 GPU；torch 可用={TORCH_AVAILABLE}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
