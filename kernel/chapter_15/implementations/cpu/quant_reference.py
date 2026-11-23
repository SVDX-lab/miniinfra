"""第 15 期 CPU 量化/反量化参考实现（真值锚点 + 误差分层基准）。

量化方案：per-channel symmetric INT8 weight-only（W4A16 记大纲附录 A 考点，
不进实验）：
    scale[n]  = max_k |W[k, n]| / 127.0        （按输出通道 = W[K,N] 的列）
    q[k, n]   = clip(round(W[k, n] / scale[n]), -127, 127)   int8
    dequant   = q.astype(fp32) * scale[n]

误差分层（大纲 §9 行 476：量化误差与算子实现误差分开归因）：
    L0  FP32 真值 GEMM                 y32   = A32 @ W32          （锚点）
    La1 仅量化误差（FP32 算术）         ya1   = A32 @ dequant(q)
    La2 仅量化 + FP16 权重存储          ya2   = A16 @ Ŵ16          （引擎参考实现：
                                        物化反量化后走 torch/引擎 FP16 GEMM）
    La3 FP16 无量化对照                 ya3   = A16 @ W16          （隔离
                                        "FP16 存储/累加误差"与"量化误差"）
    Lb  量化 + 自定义 Kernel 实现误差    kernel(int8_fused) vs L0
    Lc  FP16 自定义 Kernel 对照         kernel(fp16) vs L0
  —— (La, Lb, Lc) 三层分开报告；Lb − La 为"kernel 实现误差叠加量"。

GEMM 参考（三版）：
  gemm_fp32_truth        FP32 真值（torch FP32 BLAS；tiny shape 另有字面三重循环）
  gemm_quant_ref_fp32    仅量化路径（FP32 算术，物化/融合在 FP32 算术下数学等价）
  gemm_engine_fp16       引擎参考 FP16 GEMM（torch CPU half matmul，FP32 累加语义
                         由引擎决定——这正是"引擎参考实现"口径的含义）

numpy/torch 双依赖允许：本地 venv torch 2.6.0+cpu 就绪（协议 §3）。
"""

from __future__ import annotations

import numpy as np

QMAX = 127  # symmetric 量化用满 [-127, 127]，不用 -128（保持对称，与 vLLM 对称模式一致）


# ---------------------------------------------------------------------------
# 量化 / 反量化（per-channel symmetric INT8）
# ---------------------------------------------------------------------------
def quantize_weight_perchannel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """W[K, N] -> (q int8 [K, N], scale float32 [N])。

    scale 按"输出通道"（= N 列）计算：weight-only 量化的标准方向——
    反量化乘法在 GEMM 中只依赖输出列号 n，Kernel 内 per-channel scale[n] 广播。
    """
    w32 = np.asarray(w, dtype=np.float64)
    amax = np.max(np.abs(w32), axis=0)  # [N]
    scale = amax / QMAX
    # 全零列保护：scale=1（q 全 0，语义不变），避免除零
    scale = np.where(scale > 0, scale, 1.0)
    q = np.clip(np.round(w32 / scale[None, :]), -QMAX, QMAX).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_weight(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(q int8 [K,N], scale fp32 [N]) -> Ŵ fp32 [K,N]。"""
    return (q.astype(np.float32) * scale[None, :].astype(np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# GEMM 三版参考
# ---------------------------------------------------------------------------
def gemm_fp32_truth(a32: np.ndarray, w32: np.ndarray) -> np.ndarray:
    """FP32 真值 GEMM（torch FP32；导入即用，保证与 GPU 侧比较脚本同口径）。"""
    import torch

    return (torch.from_numpy(a32) @ torch.from_numpy(w32)).numpy()


def gemm_triple_loop(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """字面三重循环（tiny shape 自检用，k 升序 FP32 累加）。"""
    m, k = a.shape
    k2, n = w.shape
    assert k == k2
    out = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for p in range(k):
            av = float(a[i, p])
            for j in range(n):
                out[i, j] += av * float(w[p, j])
    return out.astype(np.float32)


def gemm_quant_ref_fp32(a32: np.ndarray, q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """La1：仅量化误差（FP32 算术）。

    物化路径与融合路径在 FP32 算术下数学等价（先乘 scale 再累加 vs 累加后
    乘 scale 只在 FP16 舍入下才有差别）——本函数即两层共同的 FP32 参考锚点。
    """
    w_hat = dequantize_weight(q, scale)
    return gemm_fp32_truth(a32, w_hat)


def gemm_engine_fp16(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """引擎参考 FP16 GEMM：输入 FP16，输出 FP32（torch CPU half matmul）。

    这就是"仅量化、用引擎参考实现"口径中的引擎实现——自定义 Kernel 的误差
    必须与这一层的差值分开归因（大纲 §9 行 476）。
    """
    import torch

    a16 = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float16))
    w16 = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float16))
    return (a16 @ w16).float().numpy()


def quantize_activation_amax_fp16(a: np.ndarray) -> np.ndarray:
    """激活转 FP16（服务真实路径：激活始终 FP16，本期不做激活量化）。"""
    return np.asarray(a, dtype=np.float16)


# ---------------------------------------------------------------------------
# 一体化误差分层报告（CPU 列 + GPU 列共用同一真值/分层定义）
# ---------------------------------------------------------------------------
def error_layer_report(a32: np.ndarray, w32: np.ndarray) -> dict:
    """生成 L0/La1/La2/La3 各层真值输出与逐层误差字典（fixtures 生成与 CPU 实验共用）。"""
    q, scale = quantize_weight_perchannel(w32)
    a16 = quantize_activation_amax_fp16(a32)
    w_hat = dequantize_weight(q, scale)

    y_truth = gemm_fp32_truth(a32, w32)          # L0
    y_a1 = gemm_quant_ref_fp32(a32, q, scale)    # La1 仅量化（FP32 算术）
    y_a2 = gemm_engine_fp16(a16, w_hat.astype(np.float16))   # La2 量化+FP16 权重存储（引擎）
    y_a3 = gemm_engine_fp16(a16, w32)            # La3 FP16 无量化对照（引擎）

    from sys import path as _sys_path
    from pathlib import Path as _Path
    _sys_path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
    from experiment_utils import comparison_metrics_np  # noqa: E402

    report = {
        "quant_params": {
            "scheme": "per_channel_symmetric_int8",
            "scale_direction": "output_channel (N columns of W[K,N])",
            "qmax": QMAX,
            "scale_shape": list(scale.shape),
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
            "scale_mean": float(scale.mean()),
        },
        "layers": {
            "La1_quant_only_fp32": comparison_metrics_np(y_a1, y_truth),
            "La2_quant_fp16weight_engine": comparison_metrics_np(y_a2, y_truth),
            "La3_fp16_no_quant_engine": comparison_metrics_np(y_a3, y_truth),
        },
        "deltas": {
            # kernel 实现误差叠加量的基准：Lb − La1、Lc − La3 在 compare 阶段计算
            "note": "Lb/Lc 由 GPU 侧 kernel 实测后与 L0 比较，叠加量 = Lb.err − La1.err",
        },
    }
    return {
        "q": q,
        "scale": scale,
        "a16": a16,
        "w_hat": w_hat,
        "y_truth": y_truth,
        "y_a1": y_a1,
        "y_a2": y_a2,
        "y_a3": y_a3,
        "report": report,
    }
