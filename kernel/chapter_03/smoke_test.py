#!/usr/bin/env python3
"""第 03 期冒烟测试（本地 CPU 即可运行）：边界行为与语义合同验证。

覆盖（合同"非整 Tile、空 Tensor、极端数值和错误输入行为"条目）：
1. 手算例（2×4 硬编码，纸面可核）；
2. 全零输入 → 全零输出（eps 在根号内），无 NaN；
3. NaN 按行传播（其他行不受影响）；
4. in-place 语义：融合后 r buffer == t 真值；原 r 的外部引用不受影响
   （验证"调用方传克隆"模式）；out-of-place 变体与 in-place 数学一致；
5. 极端数值：fp32 平方和溢出行为（|t|>~1.8e19 → scale=0 → y 全零）；
6. 一维/r-h 形状不匹配/H=0/FP64 输入 → 显式 ValueError；
7. 分离 vs 融合串行实现位级一致（同一 FP32 累加顺序）。

用法：python smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from fused_add_rmsnorm import (  # noqa: E402
    RMS_NORM_EPS,
    fused_add_rmsnorm_inplace_serial,
    fused_add_rmsnorm_out_of_place_serial,
    hand_computed_example,
    separated_add_rmsnorm_serial,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{status}] {name}" + (f" —— {detail}" if detail else ""))


def main() -> int:
    print("=== 第 03 期冒烟测试（CPU 参考实现，numpy-only）===")

    # 1. 手算例
    r, h, t, y = hand_computed_example()
    expected_t = np.array([[2.0, 2.0, 2.0, 4.0], [0.0, 1.0, 2.0, -1.0]], dtype=np.float32)
    scale0 = np.float32(1.0) / np.float32(np.sqrt(np.float32(7.0) + np.float32(1e-6)))
    scale1 = np.float32(1.0) / np.float32(np.sqrt(np.float32(1.5) + np.float32(1e-6)))
    expected_y0 = (expected_t[0] * scale0).astype(np.float32)
    expected_y1 = (expected_t[1] * scale1).astype(np.float32)
    check("手算例 t = r+h 位级一致", np.array_equal(t, expected_t))
    check("手算例 y 第 1 行", np.allclose(y[0], expected_y0, rtol=1e-6))
    check("手算例 y 第 2 行", np.allclose(y[1], expected_y1, rtol=1e-6))

    # 2. 全零行
    rng = np.random.default_rng(0)
    r2 = rng.standard_normal((3, 256), dtype=np.float32)
    h2 = rng.standard_normal((3, 256), dtype=np.float32)
    r2[1] = 0
    h2[1] = 0
    w2 = rng.standard_normal(256, dtype=np.float32)
    r_clone = r2.copy()
    r_upd, y2 = fused_add_rmsnorm_inplace_serial(r_clone, h2, w2)
    check("双零行 residual=0 且 y=0", bool((r_upd[1] == 0).all() and (y2[1] == 0).all()))
    check("双零行无 NaN", not np.isnan(y2[1]).any())
    check("其他行不受影响", np.isfinite(y2[[0, 2]]).all())

    # 3. NaN 传播语义（合同 nan_semantics）：t/residual 仅在 NaN 元素位置 NaN
    #    （加法逐元素），平方和被污染 → 该行 y 全 NaN；其他行不受影响。
    r3 = rng.standard_normal((2, 64), dtype=np.float32)
    h3 = rng.standard_normal((2, 64), dtype=np.float32)
    r3[0, 0] = np.nan
    r_clone = r3.copy()
    r_upd, y3 = fused_add_rmsnorm_inplace_serial(r_clone, h3, np.ones(64, dtype=np.float32))
    check("NaN 元素位置 residual NaN、其余位置保留有限值",
          bool(np.isnan(r_upd[0, 0]) and np.isfinite(r_upd[0, 1:]).all()))
    check("NaN 行 y 全 NaN（平方和归约污染）", bool(np.isnan(y3[0]).all()))
    check("相邻行 finite", bool(np.isfinite(y3[1]).all()))

    # 4. in-place 语义
    r4 = rng.standard_normal((4, 32), dtype=np.float32)
    h4 = rng.standard_normal((4, 32), dtype=np.float32)
    w4 = rng.standard_normal(32, dtype=np.float32)
    r_orig = r4.copy()
    r_clone = r4.copy()
    r_upd, y4 = fused_add_rmsnorm_inplace_serial(r_clone, h4, w4)
    t_true, y_true = separated_add_rmsnorm_serial(r4, h4, w4)
    check("in-place：传入 buffer 被更新为 t", np.array_equal(r_upd, t_true))
    check("in-place：外部引用的原 r 不受影响", np.array_equal(r4, r_orig))
    check("in-place：y 与分离真值位级一致", np.array_equal(y4, y_true))
    r_ooo, y_ooo = fused_add_rmsnorm_out_of_place_serial(r4, h4, w4)
    check("out-of-place：原 r 保留", np.array_equal(r4, r_orig))
    check("out-of-place：r_out == in-place residual", np.array_equal(r_ooo, r_upd))
    check("out-of-place：y == in-place y", np.array_equal(y_ooo, y4))

    # 5. 极端数值：fp32 平方和溢出（|t| ~ 3e19 > sqrt(FLT_MAX)≈1.8e19）
    big = 3.0e19
    r5 = np.full((1, 16), big, dtype=np.float32)
    h5 = np.zeros((1, 16), dtype=np.float32)
    r_clone = r5.copy()
    r_upd, y5 = fused_add_rmsnorm_inplace_serial(r_clone, h5, np.ones(16, dtype=np.float32))
    check("fp32 平方和溢出：t=±3e19 保留（加法不溢出）", bool(np.isfinite(r_upd).all()))
    check("fp32 平方和溢出：y 全零（scale=1/Inf=0），无 NaN",
          bool((y5 == 0).all() and not np.isnan(y5).any()))

    # 6. 错误输入
    try:
        separated_add_rmsnorm_serial(np.zeros((2, 0), dtype=np.float32),
                                     np.zeros((2, 0), dtype=np.float32),
                                     np.zeros(0, dtype=np.float32))
        check("H=0 显式报错", False)
    except ValueError:
        check("H=0 显式报错", True)
    try:
        separated_add_rmsnorm_serial(np.zeros((2, 8), dtype=np.float32),
                                     np.zeros((3, 8), dtype=np.float32),
                                     np.zeros(8, dtype=np.float32))
        check("r/h 形状不匹配显式报错", False)
    except ValueError:
        check("r/h 形状不匹配显式报错", True)
    try:
        separated_add_rmsnorm_serial(np.zeros((2, 8), dtype=np.float64),
                                     np.zeros((2, 8), dtype=np.float64),
                                     np.zeros(8, dtype=np.float64))
        check("FP64 输入显式报错", False)
    except ValueError:
        check("FP64 输入显式报错", True)

    # 7. 分离 vs 融合位级一致（随机大样本）
    r7 = rng.standard_normal((7, 127), dtype=np.float32)  # 非整 Tile H
    h7 = rng.standard_normal((7, 127), dtype=np.float32)
    w7 = rng.standard_normal(127, dtype=np.float32)
    t_sep, y_sep = separated_add_rmsnorm_serial(r7, h7, w7)
    r_clone = r7.copy()
    r_fus, y_fus = fused_add_rmsnorm_inplace_serial(r_clone, h7, w7)
    check("分离 vs 融合：t/residual 位级一致（H=127 非整 Tile）",
          np.array_equal(t_sep, r_fus))
    check("分离 vs 融合：y 位级一致（H=127 非整 Tile）", np.array_equal(y_sep, y_fus))

    print(f"\n=== 冒烟结果：{PASS} PASS / {FAIL} FAIL ===")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
