"""第 16 期五个算子的 CPU 教学参考实现（真值锚点，NumPy FP32）。

口径与第 01 期一致：CPU FP32 朴素实现是唯一真值锚点，机器无关。
每算子附一版"朴素/不走捷径"实现与一版硬编码手算例（供 smoke_test 对照）。

五个算子对应大纲第 16 期的四类访存/归约形态：
- 行内归约：softmax（数值稳定版）、warp/block reduce（per-row sum/max）；
- 跨行重排：transpose；
- 前缀依赖：inclusive prefix sum/scan（按行独立）；
- 逐元素链：SiLU + Gate Multiply（y = silu(gate) * up）。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. softmax（数值稳定版：减最大值）
# ---------------------------------------------------------------------------


def softmax_rows_numpy(x: np.ndarray) -> np.ndarray:
    """数值稳定 softmax：先减每行最大值再取指数。

    x: (rows, cols) float32 → y: (rows, cols) float32。
    减 max 不改变 softmax 数学结果（分子分母同乘 exp(-m)），
    但把 exp 的自变量压到 [-inf, 0]，避免 FP32 溢出（exp(89) 已超 FP32 最大值）。
    """
    x = np.asarray(x, dtype=np.float32)
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def naive_softmax_rows_numpy(x: np.ndarray) -> np.ndarray:
    """朴素 softmax（不减 max）：smoke test 用于演示大数值时溢出为 NaN/Inf。"""
    x = np.asarray(x, dtype=np.float32)
    exp = np.exp(x, dtype=np.float32)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def softmax_rows_torch(x) -> "object":
    """torch 组合表达式（CPU 对照组，供 smoke/correctness 交叉印证）。"""
    import torch

    return torch.softmax(x.float(), dim=-1)


def softmax_hand_computed_example():
    """手算例：softmax([1,2,3]) = [0.09003, 0.24473, 0.66524]（FP32）。"""
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    e = np.exp(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    expected = (e / e.sum()).astype(np.float32).reshape(1, 3)
    return x, expected


# ---------------------------------------------------------------------------
# 2. per-row 归约（sum / max，warp/block reduce 的语义层）
# ---------------------------------------------------------------------------


def reduce_sum_rows_numpy(x: np.ndarray) -> np.ndarray:
    """每行求和：(rows, cols) → (rows,) float32。

    NumPy 的 sum 是成对（pairwise）树形求和——与 GPU 的二分树归约同构，
    误差特性相近；这正是"树形归约优于串行累加"的语义层印证素材。
    """
    x = np.asarray(x, dtype=np.float32)
    return x.sum(axis=1, dtype=np.float32)


def reduce_max_rows_numpy(x: np.ndarray) -> np.ndarray:
    """每行求最大：(rows, cols) → (rows,) float32。"""
    x = np.asarray(x, dtype=np.float32)
    return x.max(axis=1)


def reduce_hand_computed_example():
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=np.float32)
    return x, np.array([6.0, 1.5], dtype=np.float32), np.array([3.0, 2.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. transpose（跨行重排）
# ---------------------------------------------------------------------------


def transpose_numpy(x: np.ndarray) -> np.ndarray:
    """矩阵转置：(rows, cols) → (cols, rows) float32。"""
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x.T).astype(np.float32)


def transpose_hand_computed_example():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
    return x, expected


# ---------------------------------------------------------------------------
# 4. inclusive prefix sum / scan（按行独立）
# ---------------------------------------------------------------------------


def scan_inclusive_rows_numpy(x: np.ndarray) -> np.ndarray:
    """每行 inclusive scan（前缀和）：y[r, c] = sum(x[r, 0..c])，FP32 逐位累加语义。"""
    x = np.asarray(x, dtype=np.float32)
    return np.cumsum(x, axis=1, dtype=np.float32).astype(np.float32)


def scan_hand_computed_example():
    x = np.array([[1.0, 2.0, 3.0, -1.0]], dtype=np.float32)
    expected = np.array([[1.0, 3.0, 6.0, 5.0]], dtype=np.float32)
    return x, expected


# ---------------------------------------------------------------------------
# 5. SiLU + Gate Multiply（逐元素链融合）
# ---------------------------------------------------------------------------


def silu_numpy(x: np.ndarray) -> np.ndarray:
    """SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))，FP32。"""
    x = np.asarray(x, dtype=np.float32)
    return (x / (np.float32(1.0) + np.exp(-x, dtype=np.float32))).astype(np.float32)


def silu_gate_numpy(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """分离语义（数学上等价于两步计算，但此处单函数表达）：
    y = silu(gate) * up。

    SwiGLU 的 gate 分支：LLM MLP 中 gate/up 投影后的逐元素链。
    """
    return (silu_numpy(gate) * np.asarray(up, dtype=np.float32)).astype(np.float32)


def silu_gate_hand_computed_example():
    """silu(1)=0.731059, silu(-1)=-0.268941, silu(0)=0。"""
    gate = np.array([[1.0, -1.0, 0.0]], dtype=np.float32)
    up = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    expected = np.array([[1.462118, -0.806823, 0.0]], dtype=np.float32)
    return gate, up, expected


OP_LIST = ["softmax", "reduce", "transpose", "scan", "silu_gate"]
