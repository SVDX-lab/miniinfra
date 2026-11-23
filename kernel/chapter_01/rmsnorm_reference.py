"""RMSNorm 参考实现集合（第 01 期真值锚点与被测对象）。

三层实现对应第 01 期合同的三条对照链：

1. naive_rmsnorm_numpy —— 独立的 NumPy 朴素实现，按公式逐语义写成，
   作为 FP32 CPU 真值锚点（与 torch 实现相互独立，避免同源偏差）；
2. composite_rmsnorm_torch —— PyTorch 组合表达式（三后端被测对象，
   与手写 Qwen3 模型内部 RMSNorm 同一语义）；
3. hand_computed_example —— 2×4 手算例（硬编码数值，纸面可核）。

数学定义（eps=1e-6，与 Qwen3 config.json 的 rms_norm_eps 一致）：
    y_i = w_i * x_i / sqrt( mean_j(x_j^2) + eps )
平方和的求均值固定使用 FP32 累加。
"""

from __future__ import annotations

import numpy as np
import torch

RMS_NORM_EPS = 1e-6


def naive_rmsnorm_numpy(x: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS) -> np.ndarray:
    """FP32 NumPy 朴素参考实现。x: [rows, H] float32, weight: [H] float32。"""

    if x.ndim != 2:
        raise ValueError("x 必须是 [rows, H] 二维数组")
    if x.dtype != np.float32 or weight.dtype != np.float32:
        raise ValueError("真值路径只接受 float32 输入")
    # 均方差：FP32 累加（sum 指定 dtype），再 rsqrt、逐元素乘权重。
    mean_square = (x * x).sum(axis=1, dtype=np.float32) / np.float32(x.shape[1])
    normalized = x / np.sqrt(mean_square + np.float32(eps))[:, None]
    return (normalized * weight[None, :]).astype(np.float32)


def composite_rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """PyTorch 组合表达式（三后端 baseline 的被测对象）。

    x: [rows, H]，weight: [H]，dtype 支持 fp32/fp16（fp64 仅限 CPU 实验）。
    平方和均值在 float32 上计算（fp16 输入时先升精度），与手写 Qwen3
    模型内部 RMSNorm 完全同一语义。
    """

    if x.ndim != 2:
        raise ValueError("x 必须是 [rows, H] 二维 Tensor")
    input_dtype = x.dtype
    float_states = x.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def hand_computed_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2×4 手算例：x、weight 与纸面计算得到的期望输出。

    x = [[1, 2, 3, 4], [-1, 0, 1, -2]]，w = [1, 1, 1, 1]，eps = 1e-6。
    第一行：mean(x^2) = (1+4+9+16)/4 = 7.5，
           sqrt(7.5 + 1e-6) ≈ 2.738613，y ≈ x / 2.738613。
    第二行：mean(x^2) = (1+0+1+4)/4 = 1.5，
           sqrt(1.5 + 1e-6) ≈ 1.224745，y ≈ x / 1.224745。
    期望值按公式保留 6 位小数（与 FP32 实测对比时容差 1e-5）。
    """

    x = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, -2.0]], dtype=np.float32)
    weight = np.ones(4, dtype=np.float32)
    expected = np.array(
        [
            [0.365148, 0.730297, 1.095445, 1.460593],
            [-0.816497, 0.0, 0.816497, -1.632993],
        ],
        dtype=np.float32,
    )
    return x, weight, expected
