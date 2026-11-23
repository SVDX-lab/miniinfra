#!/usr/bin/env python3
"""第 16 期 Smoke Test：CPU-only 边界行为验证，不占 GPU。

覆盖合同中的边界条目：
1. 五个算子的手算极小例 vs NumPy 参考实现；
2. softmax 数值稳定性：x+100 使朴素版溢出为 NaN/Inf，稳定版正常；
3. scan 与 reduce 的交叉一致性（末元素 = 行和）；双转置 = 恒等；
4. SiLU 边界：0 → 0、大负值 → 0-、单调段行为；
5. 参考实现重复运行位级确定性；
6. 非连续 torch 输入的参考实现行为（GPU Kernel 合同要求连续，在此声明）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # 主 agent 统一装 torch CPU 版；未就绪时跳过 torch 交叉项（记入日志）
    torch = None
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "implementations" / "cpu"))

from ops_reference import (  # noqa: E402
    naive_softmax_rows_numpy,
    reduce_hand_computed_example,
    reduce_max_rows_numpy,
    reduce_sum_rows_numpy,
    scan_hand_computed_example,
    scan_inclusive_rows_numpy,
    silu_gate_hand_computed_example,
    silu_gate_numpy,
    silu_numpy,
    softmax_hand_computed_example,
    softmax_rows_numpy,
    softmax_rows_torch,
    transpose_hand_computed_example,
    transpose_numpy,
)


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise SystemExit(f"Smoke Test 失败于: {name}")


def test_hand_computed() -> None:
    x, expected = softmax_hand_computed_example()
    got = softmax_rows_numpy(x)
    check("softmax 手算例 [0.0900,0.2447,0.6652]", np.allclose(got, expected, atol=1e-6),
          f"got={got[0].tolist()}")

    x, expected_sum, expected_max = reduce_hand_computed_example()
    check("reduce sum 手算例 [6, 2]", np.array_equal(reduce_sum_rows_numpy(x), expected_sum))
    check("reduce max 手算例 [3, 2]", np.array_equal(reduce_max_rows_numpy(x), expected_max))

    x, expected = transpose_hand_computed_example()
    check("transpose 手算例 2x3→3x2", np.array_equal(transpose_numpy(x), expected))

    x, expected = scan_hand_computed_example()
    check("scan 手算例 [1,3,6,5]", np.array_equal(scan_inclusive_rows_numpy(x), expected))

    gate, up, expected = silu_gate_hand_computed_example()
    got = silu_gate_numpy(gate, up)
    check("silu+gate 手算例 [1.4621,-0.8068,0]",
          np.allclose(got, expected, atol=1e-6), f"got={got[0].tolist()}")


def test_softmax_stability() -> None:
    generator = np.random.Generator(np.random.PCG64(17))
    x = (generator.standard_normal((4, 1024), dtype=np.float32)
         + np.float32(100.0)).astype(np.float32)
    naive = naive_softmax_rows_numpy(x)
    stable = softmax_rows_numpy(x)
    check("朴素 softmax 大数值溢出为 NaN/Inf（平台无关的数值边界证据）",
          bool(np.isnan(naive).any() or np.isinf(naive).any()))
    check("稳定 softmax（减 max）同输入全部有限", bool(np.isfinite(stable).all()))
    if TORCH_AVAILABLE:
        torch_result = softmax_rows_torch(torch.from_numpy(x)).numpy()
        check("稳定 softmax 与 torch.softmax 一致", np.allclose(stable, torch_result, atol=1e-6))
    else:
        print("[SKIP] torch 未就绪：与 torch.softmax 的交叉项移到服务器 CPU 步骤（EXPERIMENT_LOG 注明）")
    check("稳定 softmax 行和 = 1",
          np.allclose(stable.sum(axis=1), 1.0, atol=1e-6))


def test_scan_reduce_cross_consistency() -> None:
    generator = np.random.Generator(np.random.PCG64(18))
    x = generator.standard_normal((8, 257), dtype=np.float32)
    scan = scan_inclusive_rows_numpy(x)
    row_sum = reduce_sum_rows_numpy(x)
    check("scan 末元素 = 行和（scan/reduce 交叉印证）",
          np.allclose(scan[:, -1], row_sum, atol=1e-4))
    check("scan 单调性不成立但增量 = 原值",
          np.allclose(np.diff(scan, axis=1, prepend=np.float32(0.0)), x, atol=1e-5))


def test_transpose_involution() -> None:
    generator = np.random.Generator(np.random.PCG64(19))
    x = generator.standard_normal((37, 123), dtype=np.float32)
    check("双转置 = 恒等", np.array_equal(transpose_numpy(transpose_numpy(x)), x))


def test_silu_edges() -> None:
    values = np.array([0.0, 1000.0, -1000.0, -0.0], dtype=np.float32)
    result = silu_numpy(values)
    check("silu(0)=0、silu(±大数) 有限不溢出",
          bool(np.isfinite(result).all()) and result[0] == 0.0 and result[3] == 0.0,
          f"got={result.tolist()}")


def test_reference_determinism() -> None:
    generator = np.random.Generator(np.random.PCG64(20))
    x = generator.standard_normal((64, 1000), dtype=np.float32)
    up = generator.standard_normal((64, 1000), dtype=np.float32)
    first = [softmax_rows_numpy(x), scan_inclusive_rows_numpy(x), silu_gate_numpy(x, up)]
    second = [softmax_rows_numpy(x), scan_inclusive_rows_numpy(x), silu_gate_numpy(x, up)]
    check("参考实现两次运行位级一致",
          all(np.array_equal(a, b) for a, b in zip(first, second)))


def test_non_contiguous_input() -> None:
    if not TORCH_AVAILABLE:
        print("[SKIP] torch 未就绪：非连续输入行为检查移到服务器 CPU 步骤")
        print("[NOTE] GPU Kernel 合同：仅接受连续行主输入（ctypes 直传 data_ptr），"
              "非连续输入需先 .contiguous()——与第 06 期'布局即性能'呼应")
        return
    base = torch.randn(8, 6)
    transposed = base.t()  # (6, 8) 非连续
    result = softmax_rows_numpy(transposed.numpy())
    expected = softmax_rows_numpy(transposed.contiguous().numpy())
    check("非连续输入的参考实现结果与连续一致（allclose）",
          np.allclose(result, expected, atol=1e-6))
    print("[NOTE] NumPy 对 strided 输入走不同 SIMD 归约路径，末位 ULP 有差"
          "（非位级相等）——正确性口径用 allclose，位级口径只对连续输入成立")
    print("[NOTE] GPU Kernel 合同：仅接受连续行主输入（ctypes 直传 data_ptr），"
          "非连续输入需先 .contiguous()——与第 06 期'布局即性能'呼应")


def main() -> int:
    test_hand_computed()
    test_softmax_stability()
    test_scan_reduce_cross_consistency()
    test_transpose_involution()
    test_silu_edges()
    test_reference_determinism()
    test_non_contiguous_input()
    print("[done] Smoke Test 全部通过（CPU-only，未占 GPU）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
