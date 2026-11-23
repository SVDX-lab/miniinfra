"""第 14 期 CPU 真值锚点（自第 03 期 implementations/cpu 原样复制的两个函数）。

协议不跨期 import：naive_rmsnorm_serial 与 separated_add_rmsnorm_serial
逐字复制自 chapter_03/implementations/cpu/fused_add_rmsnorm.py（该文件
声明其 RMSNorm 部分复用第 02 期）。真值语义：
    t_ij = fl32(r_ij + h_ij)（单元素加法无累加，向量化与串行位级一致）
    y_i  = w_i * t_i / sqrt( mean_j(t_j^2) + eps )（FP32 自然顺序累加）
"""

from __future__ import annotations

import numpy as np

RMS_NORM_EPS = 1e-6


def _validate_pair(r, h, weight):
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


def separated_add_rmsnorm_serial(
    r: np.ndarray, h: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """分离路径真值锚点（原样复用）：返回 (t, y)。"""
    r, h, weight = _validate_pair(r, h, weight)
    t = (r + h).astype(np.float32)
    y = naive_rmsnorm_serial(t, weight, eps)
    return t, y
