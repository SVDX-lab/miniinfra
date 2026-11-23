"""第 02 期 CPU 朴素串行 RMSNorm —— 本期真值锚点。

第 02 期的核心问题：同一算法因硬件组织方式不同而有三种长相。
CPU 版回答"算法本身"：逐元素循环、逐行归约，每个步骤都可读、可纸面核对，
同时是 CUDA/MUSA Kernel 数值对照的 FP32 真值锚点。

数学定义（eps=1e-6，与 Qwen3 config.json 的 rms_norm_eps 一致，同第 01 期）：
    y_i = w_i * x_i / sqrt( mean_j(x_j^2) + eps )
平方和与均值固定使用 FP32 累加；串行实现按 j=0..H-1 的自然顺序累加
（与 CUDA Kernel 的 warp 树形归约、NumPy 的 pairwise 归约形成"同一数学、
不同归约顺序"的对照素材，第 07 期确定性主题复用）。

本模块提供：
1. naive_rmsnorm_serial —— 逐行 × 逐元素双重显式循环（真值锚点本体）；
2. naive_rmsnorm_rowwise —— 行循环 + 行内 NumPy 显式算式（大 Shape 的
   快速路径，语义一致、归约顺序不同，仅用于生成大 Shape 期望值时的
   交叉核对，见 generate_fixtures.py）；
3. composite_rmsnorm_torch —— PyTorch 组合表达式（与第 01 期同一语义，
   作为"平台原生路径"的对照对象）；
4. hand_computed_example —— 2×4 手算例（硬编码数值，纸面可核）。
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # torch 为可选依赖（CPU 真值路径 numpy-only）
    TORCH_AVAILABLE = False

RMS_NORM_EPS = 1e-6


def _validate_inputs(x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    weight = np.asarray(weight)
    if x.ndim != 2:
        raise ValueError("x 必须是 [rows, H] 二维数组")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight 必须是 [H] 一维数组且与 x 的 H 一致")
    if x.dtype != np.float32 or weight.dtype != np.float32:
        raise ValueError("真值路径只接受 float32 输入")
    if x.shape[1] == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错（不支持静默处理）")
    return x, weight


def naive_rmsnorm_serial(x: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS) -> np.ndarray:
    """朴素串行参考实现：逐行、行内逐元素累加平方和（FP32），真值锚点。

    x: [rows, H] float32, weight: [H] float32。输出 float32 [rows, H]。
    每行两遍扫描：第一遍累加 sum(x_j^2)（j 按自然顺序），求均方差、rsqrt；
    第二遍逐元素 y_j = x_j * scale * w_j。
    这正是 CUDA Kernel（每 block 一行 + warp 归约）和 MUSA 迁移版的
    串行对应物，代码结构刻意与 Kernel 的两遍扫描一一对应。
    """

    x, weight = _validate_inputs(x, weight)
    rows, hidden = x.shape
    output = np.empty((rows, hidden), dtype=np.float32)
    f32 = np.float32
    for i in range(rows):
        row = x[i]
        # 第一遍：平方和串行累加（FP32，自然顺序）。
        acc = f32(0.0)
        for j in range(hidden):
            v = row[j]
            acc = f32(acc + f32(v * v))
        # 均方差 + eps（eps 在根号内）。
        scale = f32(1.0) / f32(np.sqrt(f32(acc / f32(hidden)) + f32(eps)))
        w_row = weight
        out_row = output[i]
        # 第二遍：归一化并乘权重。
        for j in range(hidden):
            out_row[j] = f32(f32(row[j] * scale) * w_row[j])
    return output


def naive_rmsnorm_rowwise(x: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS) -> np.ndarray:
    """行循环 + 行内 NumPy 算式的可读版（大 Shape 快速路径）。

    与 naive_rmsnorm_serial 数学定义完全一致，仅行内求和交给 NumPy
    （pairwise 归约，FP32 累加）。用于大 Shape fixtures 的生成与两组
    实现之间的交叉核对（归约顺序差异应 <= 1e-6 量级，见 smoke_test）。
    """

    x, weight = _validate_inputs(x, weight)
    mean_square = (x * x).sum(axis=1, dtype=np.float32) / np.float32(x.shape[1])
    normalized = x / np.sqrt(mean_square + np.float32(eps))[:, None]
    return (normalized * weight[None, :]).astype(np.float32)


def composite_rmsnorm_torch(x, weight, eps: float = RMS_NORM_EPS):
    """PyTorch 组合表达式（第 01 期同语义：平台原生路径的对照对象）。

    x: [rows, H]，weight: [H]，dtype 支持 fp32/fp16；平方和均值在
    float32 上计算（fp16 输入先升精度），与手写 Qwen3 模型内部
    RMSNorm 完全同一语义。
    需要本机 torch（本地无 torch 时由服务器侧执行）。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("本机 torch 不可用：composite_rmsnorm_torch 需在服务器侧执行")

    if x.ndim != 2:
        raise ValueError("x 必须是 [rows, H] 二维 Tensor")
    input_dtype = x.dtype
    float_states = x.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def hand_computed_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2×4 手算例：x、weight 与纸面计算得到的期望输出（同第 01 期）。

    x = [[1, 2, 3, 4], [-1, 0, 1, -2]]，w = [1, 1, 1, 1]，eps = 1e-6。
    第一行：mean(x^2) = (1+4+9+16)/4 = 7.5，
           sqrt(7.5 + 1e-6) ≈ 2.738613，y ≈ x / 2.738613。
    第二行：mean(x^2) = (1+0+1+4)/4 = 1.5，
           sqrt(1.5 + 1e-6) ≈ 1.224745，y ≈ x / 1.224745。
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
