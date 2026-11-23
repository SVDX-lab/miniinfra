#!/usr/bin/env python3
"""第 07 期 Smoke Test：CPU-only、不占 GPU 的最低成本检查。

覆盖合同（§8 + 第 07 期"确定性声明"条目）的 CPU 侧行为：
1. 手算 2×4 极小例 vs 串行实现（真值锚点自洽）；
2. 串行实现 100 次运行位级一致（确定性参照系的构造保证实测确认）；
3. 求和顺序破缺：serial_l2r vs serial_r2l vs pairwise vs numpy —— 位级差
   实测（FP32 有界漂移 / FP16 卡死 / FP64 淹没）；
4. 全零行输出全零（eps 在根号内，无 NaN）；
5. NaN 按行传播；
6. 错误 Shape / dtype / 空 Tensor 显式报错；
7. atomic 乱序一阶模型冒烟：100 种子 unique 计数（模型本体在
   exp_cpu_sum_order.py）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from serial_rmsnorm import (  # noqa: E402
    RMS_NORM_EPS,
    hand_computed_example,
    naive_rmsnorm_rowwise,
    naive_rmsnorm_serial,
)
from sum_order_lab import (  # noqa: E402
    ORDER_FUNCTIONS,
    shuffle_drift_distribution,
    sum_pairwise,
    sum_serial,
)

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


def main() -> int:
    print("=== 第 07 期 Smoke Test（CPU-only）===")

    # 1. 手算例
    x, w, expected = hand_computed_example()
    y = naive_rmsnorm_serial(x, w, RMS_NORM_EPS)
    check("手算 2x4 例 vs 串行实现", np.allclose(y, expected, atol=1e-6),
          f"max={np.abs(y - expected).max():.2e}")

    # 2. 串行 100 次位级一致
    rng = np.random.default_rng(20260800)
    x2 = rng.standard_normal((16, 1024), dtype=np.float32)
    w2 = rng.standard_normal(1024, dtype=np.float32)
    first = naive_rmsnorm_serial(x2, w2, RMS_NORM_EPS)
    same = sum(int(naive_rmsnorm_serial(x2, w2, RMS_NORM_EPS).tobytes() == first.tobytes())
               for _ in range(99))
    check("串行实现 100 次位级一致", same == 99, f"{same + 1}/100")

    # 3. 求和顺序破缺
    row = rng.standard_normal(1024, dtype=np.float32)
    l2r = sum_serial(row, "fp32")
    r2l = sum_serial(row, "fp32") if False else None
    # 右到左
    acc = np.float32(0.0)
    for j in range(1023, -1, -1):
        acc = np.float32(acc + np.float32(row[j] * row[j]))
    r2l = float(acc)
    pw = sum_pairwise(row, "fp32")
    npv = ORDER_FUNCTIONS["numpy_pairwise"](row, acc_dtype_name="fp32")
    check("serial_l2r != serial_r2l（位级）",
          np.float32(l2r).tobytes() != np.float32(r2l).tobytes(),
          f"l2r={l2r:.4f} r2l={r2l:.4f}")
    check("serial_l2r != pairwise（位级）",
          np.float32(l2r).tobytes() != np.float32(pw).tobytes())
    check("顺序间相对差 < 1e-5（FP32 有界）",
          max(abs(l2r - r2l), abs(l2r - pw), abs(l2r - npv)) / l2r < 1e-5)
    fp16_serial = sum_serial(row, "fp16")
    check("FP16 累加漂移显著大于 FP32（>100x）",
          abs(fp16_serial - l2r) / l2r > 100 * abs(pw - l2r) / l2r,
          f"fp16 rel={abs(fp16_serial - l2r) / l2r:.2e} fp32 order rel={abs(pw - l2r) / l2r:.2e}")

    # 4. 全零行
    x0 = np.zeros((2, 1024), dtype=np.float32)
    y0 = naive_rmsnorm_serial(x0, w2, RMS_NORM_EPS)
    check("全零行输出全零（无 NaN）", np.all(y0 == 0) and not np.isnan(y0).any())

    # 5. NaN 行传播
    x3 = rng.standard_normal((2, 1024), dtype=np.float32)
    x3[0, 0] = np.float32("nan")
    y3 = naive_rmsnorm_serial(x3, w2, RMS_NORM_EPS)
    check("NaN 按行传播（行 0 全 NaN、行 1 不受影响）",
          bool(np.isnan(y3[0]).all()) and not bool(np.isnan(y3[1]).any()))

    # 6. 错误输入
    try:
        naive_rmsnorm_serial(np.zeros((2, 3, 4), dtype=np.float32), w2, RMS_NORM_EPS)
        check("错误 Shape 显式报错", False)
    except ValueError:
        check("错误 Shape 显式报错", True)
    try:
        naive_rmsnorm_serial(np.zeros((2, 4), dtype=np.float64), np.ones(4, np.float64), RMS_NORM_EPS)
        check("错误 dtype 显式报错", False)
    except ValueError:
        check("错误 dtype 显式报错", True)
    try:
        naive_rmsnorm_serial(np.zeros((2, 0), dtype=np.float32), np.zeros(0, np.float32), RMS_NORM_EPS)
        check("H=0 空 Tensor 显式报错", False)
    except ValueError:
        check("H=0 空 Tensor 显式报错", True)

    # 7. 乱序模型冒烟
    truth = float(np.sum(row.astype(np.float64) ** 2))
    dist = shuffle_drift_distribution(row, truth, 100, "fp32", seed_base=20260803)
    check("乱序模型 100 种子产生多个唯一值（顺序不确定）", dist["unique_values"] > 1,
          f"unique={dist['unique_values']}/100 rel_max={dist['rel_error_max']:.2e}")
    check("乱序模型 FP32 漂移有界（<1e-5）", dist["rel_error_max"] < 1e-5)

    print(f"=== Smoke Test 完成：PASS={PASS} FAIL={FAIL} ===")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
