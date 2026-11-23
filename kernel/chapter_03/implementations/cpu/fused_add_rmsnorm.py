"""第 03 期 CPU 参考实现：Residual Add + RMSNorm 的分离版与融合版（真值锚点）。

第 03 期核心问题（大纲 §13 行 681–698）：融合减少的不是"计算"，是
**中间结果的写回与读回** 和一次 Kernel Launch。CPU 版回答"算法本身"：

1. separated_add_rmsnorm_serial —— 分离路径：第一遍扫描逐元素 t = r + h
   （物化中间 Tensor t），第二遍复用第 02 期 naive_rmsnorm_serial 对 t 归一化。
   与 GPU 分离路径（add kernel + 第 02 期 rmsnorm kernel，共 2 次 Kernel）
   结构一一对应；
2. fused_add_rmsnorm_inplace_serial —— 融合路径：同一行循环内完成
   t = r + h、平方和归约、rsqrt、权重缩放，并**原位更新 residual**
   （合同语义：residual buffer 调用后内容变为 r + h，参考 vLLM
   fused_add_rmsnorm 的 in-place 语义）。中间 t 只存在于行缓冲
   （对应 GPU 上的寄存器/shared memory），不物化为独立 Tensor；
3. fused_add_rmsnorm_out_of_place_serial —— 融合路径的 out-of-place 变体
   （residual 原样保留、更新值写进新的 r_out）。合同记录两种语义的
   选择：GPU Kernel 采用 in-place（省一次中间写 + 调用方多为 Transformer
   residual 流，更新后的 residual 正是下一层输入）；out-of-place 仅在
   CPU 参考提供，用于语义对照与"调用方需保留原 r 时先克隆"的说明；
4. composite_separated_add_rmsnorm_torch —— PyTorch 组合表达式分离路径
   （平台原生路径对照：框架会对这条链发射多个 Kernel，第 03 期用
   profiler 计数对照"我们手写分离=2 Kernel"）。

数学定义（eps=1e-6，与 Qwen3 config.json 一致，延续第 01/02 期）：
    t_i = r_i + h_i（FP32，单次舍入）
    y_i = w_i * t_i / sqrt( mean_j(t_j^2) + eps )
平方和与均值固定 FP32 累加；串行实现按 j=0..H-1 自然顺序累加
（与第 02 期 Kernel 的 warp 树形归约、NumPy pairwise 归约形成
"同一数学、不同归约顺序"的对照素材，第 07 期主题）。

本模块提供真值锚点：fixtures 的 t（=r+h）与 y（=RMSNorm(t)）由
separated_add_rmsnorm_serial 生成（t 的加法无累加，串行与向量化位级一致）。
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # torch 为可选依赖（CPU 真值路径 numpy-only）
    TORCH_AVAILABLE = False

RMS_NORM_EPS = 1e-6


# ---------------------------------------------------------------------------
# 第 02 期真值实现原样复用（naive_rmsnorm_serial，逐行 × 逐元素自然顺序累加）。
# 复制自 chapter_02/implementations/cpu/naive_rmsnorm.py（协议：不跨期 import）。
# ---------------------------------------------------------------------------
def _validate_pair(r: np.ndarray, h: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r)
    h = np.asarray(h)
    weight = np.asarray(weight)
    if r.ndim != 2 or h.ndim != 2:
        raise ValueError("r/h 必须是 [rows, H] 二维数组")
    if r.shape != h.shape:
        raise ValueError("r 与 h 形状必须一致")
    if weight.ndim != 1 or weight.shape[0] != r.shape[1]:
        raise ValueError("weight 必须是 [H] 一维数组且与 r/h 的 H 一致")
    if r.dtype != np.float32 or h.dtype != np.float32 or weight.dtype != np.float32:
        raise ValueError("真值路径只接受 float32 输入")
    if r.shape[1] == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错（不支持静默处理）")
    return r, h, weight


def naive_rmsnorm_serial(x: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS) -> np.ndarray:
    """第 02 期真值锚点（原样复用）：逐行、行内逐元素 FP32 自然顺序累加。"""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x 必须是 [rows, H] 二维数组")
    if x.dtype != np.float32:
        raise ValueError("真值路径只接受 float32 输入")
    if x.shape[1] == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错（不支持静默处理）")
    rows, hidden = x.shape
    output = np.empty((rows, hidden), dtype=np.float32)
    f32 = np.float32
    for i in range(rows):
        row = x[i]
        acc = f32(0.0)
        for j in range(hidden):
            v = row[j]
            acc = f32(acc + f32(v * v))
        scale = f32(1.0) / f32(np.sqrt(f32(acc / f32(hidden)) + f32(eps)))
        w_row = weight
        out_row = output[i]
        for j in range(hidden):
            out_row[j] = f32(f32(row[j] * scale) * w_row[j])
    return output


# ---------------------------------------------------------------------------
# 1. 分离路径：两个"Kernel"（两遍扫描），中间 t 物化
# ---------------------------------------------------------------------------
def separated_add_rmsnorm_serial(
    r: np.ndarray, h: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """分离路径（真值锚点本体）：先 Add 后 RMSNorm，中间 t 物化为独立数组。

    Pass 1（对应 GPU add kernel）：t_ij = fl32(r_ij + h_ij)，单元素加法
    无累加，串行与 NumPy 向量化位级一致，故 t 用向量化形式生成；
    Pass 2（对应第 02 期 rmsnorm kernel）：naive_rmsnorm_serial(t, w)。

    返回 (t, y)：t 即更新后的 residual（分离语义下 residual 更新值就是
    中间 Tensor t 本身——调用方不需要再写一次 r），y 为归一化输出。
    无 aliasing：r/h 原样保留（合同：分离路径无原位修改）。
    """
    r, h, weight = _validate_pair(r, h, weight)
    t = (r + h).astype(np.float32)          # Pass 1：物化中间 Tensor（写回）
    y = naive_rmsnorm_serial(t, weight, eps)  # Pass 2：读回 t 归一化
    return t, y


# ---------------------------------------------------------------------------
# 2. 融合路径（in-place）：单循环内完成加法+归一化，residual 原位更新
# ---------------------------------------------------------------------------
def fused_add_rmsnorm_inplace_serial(
    r: np.ndarray, h: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """融合路径（合同主语义：in-place）：同一行循环内完成 t=r+h、平方和、
    rsqrt、权重缩放，并原位把 r 更新为 t。

    返回 (r_updated, y)。调用后 r buffer 内容 = t = r+h（T 舍入语义在
    FP32 下恒等）；需要保留原始 r 的调用方在调用前自行克隆
    （与 GPU 融合 Kernel 的 in-place 语义逐条对应：中间 t 不物化为
    独立 Tensor，理论访存 = 读 r + 读 h + 读 w + 写 y + 写 r）。
    """
    r, h, weight = _validate_pair(r, h, weight)
    rows, hidden = r.shape
    y = np.empty((rows, hidden), dtype=np.float32)
    f32 = np.float32
    for i in range(rows):
        row_r = r[i]
        row_h = h[i]
        # 行缓冲 = GPU 上的寄存器/shared memory（中间 t 不物化为独立 Tensor）。
        t_row = np.empty(hidden, dtype=np.float32)
        acc = f32(0.0)
        for j in range(hidden):
            t = f32(row_r[j] + row_h[j])
            t_row[j] = t
            acc = f32(acc + f32(t * t))
        scale = f32(1.0) / f32(np.sqrt(f32(acc / f32(hidden)) + f32(eps)))
        out_row = y[i]
        for j in range(hidden):
            out_row[j] = f32(f32(t_row[j] * scale) * weight[j])
        r[i] = t_row  # 原位更新 residual（合同语义）
    return r, y


# ---------------------------------------------------------------------------
# 3. 融合路径（out-of-place 变体）：residual 原样保留，更新值写新 buffer
# ---------------------------------------------------------------------------
def fused_add_rmsnorm_out_of_place_serial(
    r: np.ndarray, h: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """融合路径 out-of-place 变体（合同记录的替代语义）：返回 (r_out, y)，
    输入 r 原样保留。数学与 in-place 版完全一致；GPU Kernel 未实现此变体
    （调用方需保留原 r 时先克隆再调 in-place Kernel，语义等价），仅 CPU
    参考提供，用于两种 residual 语义的对照与文档。"""
    r, h, weight = _validate_pair(r, h, weight)
    r_out, y = fused_add_rmsnorm_inplace_serial(r.copy(), h, weight, eps)
    return r_out, y


# ---------------------------------------------------------------------------
# 3b. 行内向量化版本（性能口径的 CPU 对照；数学同上，仅组织不同）
# ---------------------------------------------------------------------------
def separated_add_rmsnorm_rowwise(
    r: np.ndarray, h: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """分离路径行内向量化版（与 fused_rowwise 同向量化等级的公平对照）：
    Pass 1 全矩阵向量化 t = r + h（物化完整中间矩阵）；
    Pass 2 全矩阵向量化归一化。与 fused_rowwise 的差异只有"中间 t 是否
    物化为完整矩阵 + 两次全矩阵遍历 vs 单循环"。"""
    r, h, weight = _validate_pair(r, h, weight)
    rows, hidden = r.shape
    t = (r + h).astype(np.float32)  # Pass 1：全矩阵物化
    mean_square = (t * t).sum(axis=1, dtype=np.float32) / np.float32(hidden)
    scale = (np.float32(1.0) / np.sqrt(mean_square + np.float32(eps))).astype(np.float32)
    y = (t * scale[:, None] * weight[None, :]).astype(np.float32)  # Pass 2：读回 t
    return t, y


def fused_add_rmsnorm_inplace_rowwise(
    r: np.ndarray, h: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """融合路径行内向量化版：单层行循环内完成加法、平方和、缩放与原位
    residual 更新（合同 in-place 语义）。行缓冲 t 仍是一次物化——NumPy
    每条语句都写回内存，无法把中间值留在寄存器，这正是"融合收益只有在
    Kernel 层（寄存器/shared memory）才能兑现"的 CPU 侧证据。"""
    r, h, weight = _validate_pair(r, h, weight)
    rows, hidden = r.shape
    y = np.empty((rows, hidden), dtype=np.float32)
    f32 = np.float32
    for i in range(rows):
        t = (r[i] + h[i]).astype(np.float32)       # 行内向量化加法
        acc = f32((t * t).sum(dtype=np.float32))
        scale = f32(1.0) / f32(np.sqrt(f32(acc / f32(hidden)) + f32(eps)))
        y[i] = (t * scale) * weight
        r[i] = t                                    # 原位更新 residual
    return r, y


# ---------------------------------------------------------------------------
# 4. PyTorch 组合表达式分离路径（平台原生对照，profiler 数 Kernel 用）
# ---------------------------------------------------------------------------
def composite_separated_add_rmsnorm_torch(r, h, weight, eps: float = RMS_NORM_EPS):
    """PyTorch 组合表达式分离路径：t = r + h 后接第 01/02 期 composite RMSNorm。

    框架对这条链的 Kernel 发射数远大于 2（add + pow + mean + rsqrt + mul...），
    本期用 profiler 计数对照"手写分离 = 2 Kernel / 融合 = 1 Kernel"。
    需要本机 torch（本地无 torch 时由服务器侧执行）。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("本机 torch 不可用：composite_separated_add_rmsnorm_torch 需在服务器侧执行")
    if r.ndim != 2 or r.shape != h.shape:
        raise ValueError("r/h 必须是同形状 [rows, H] 二维 Tensor")
    input_dtype = r.dtype
    t = r + h
    float_states = t.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    y = weight * normalized.to(input_dtype)
    return t, y


def hand_computed_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2×4 手算例（纸面可核）：r + h 先加，再按第 02 期例归一化。

    r = [[1, 2, 3, 4], [-1, 0, 1, -2]]，h = [[1, 0, -1, 0], [1, 1, 1, 1]]，
    t = r + h = [[2, 2, 2, 4], [0, 1, 2, -1]]，w = [1, 1, 1, 1]，eps = 1e-6。
    第一行：mean(t^2) = (4+4+4+16)/4 = 7，sqrt(7+1e-6) ≈ 2.6457513，
           y ≈ t / 2.6457513。
    第二行：mean(t^2) = (0+1+4+1)/4 = 1.5，sqrt(1.5+1e-6) ≈ 1.2247449。
    """
    r = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, -2.0]], dtype=np.float32)
    h = np.array([[1.0, 0.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    weight = np.ones(4, dtype=np.float32)
    t, y = separated_add_rmsnorm_serial(r, h, weight)
    return r, h, t, y
