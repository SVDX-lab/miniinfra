"""第 17 期 CPU 朴素 GEMM 参考实现（真值锚点 + 教学版）。

三个层次（同一语义 C = A @ B，行主序）：
1. matmul_triple_loop：字面意义三重循环（纯 Python 标量 FMA），O(M*N*K)。
   只用于极小例（手算核对），1024^3 = 10.7 亿次迭代在 Python 里不可行；
2. matmul_naive_rows：朴素三重循环的"内层交给 numpy"版：i、k 两层 Python
   循环 + 向量行更新 C[i,:] += A[i,k] * B[k,:]。逐条 FMA 的朴素思想不变
   （k 严格升序累加），是真值锚点与 CPU 性能口径的主体；
3. numpy matmul（BLAS）：CPU 侧"厂商库"标尺，衡量朴素版的位置。

Epilogue 参考：add_bias / gelu_tanh / silu，以及 fused_biaAct 的一体版。
GELU 采用 tanh 近似式（与 Kernel 侧一致，逐元素公式同源）：
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""

from __future__ import annotations

import math

import numpy as np

GELU_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
GELU_COEF = 0.044715


def validate_shapes(A: np.ndarray, B: np.ndarray) -> tuple[int, int, int, int]:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"GEMM 要求二维输入：A.ndim={A.ndim}, B.ndim={B.ndim}")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"内维不匹配：A.shape={A.shape}, B.shape={B.shape}")
    return A.shape[0], A.shape[1], B.shape[1], A.shape[1]


def matmul_triple_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """字面三重循环教学版（纯 Python 标量，k 升序累加）。只用于小例核对。"""
    m, k, n, _ = validate_shapes(A, B)
    A_list = A.astype(np.float64).tolist()
    B_list = B.astype(np.float64).tolist()
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        Ai = A_list[i]
        Ci = C[i]
        for p in range(k):
            a = Ai[p]
            if a == 0.0:
                continue
            Bp = B_list[p]
            for j in range(n):
                Ci[j] += a * Bp[j]
    return np.array(C, dtype=np.float32)


def matmul_naive_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """朴素三重循环（i/k 两层循环 + 向量行 FMA），真值锚点主体。

    累加顺序与字面三重循环一致（k 升序），FP32 舍入路径可复现。
    """
    m, k, n, _ = validate_shapes(A, B)
    A32 = A.astype(np.float32, copy=False)
    B32 = B.astype(np.float32, copy=False)
    C = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        Ci = C[i]
        Ai = A32[i]
        for p in range(k):
            Ci += Ai[p] * B32[p]
    return C


def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """numpy/BLAS 版（CPU 侧厂商库标尺）。"""
    m, k, n, _ = validate_shapes(A, B)
    return np.matmul(A.astype(np.float32, copy=False), B.astype(np.float32, copy=False))


def add_bias(C: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Epilogue Bias：C[i,j] += bias[j]（按列广播）。"""
    if bias.shape != (C.shape[1],):
        raise ValueError(f"bias.shape={bias.shape} 与 C 的列数 {C.shape[1]} 不匹配")
    return (C + bias[None, :]).astype(np.float32)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU（tanh 近似式），Kernel 侧同一公式。"""
    return (0.5 * x * (1.0 + np.tanh(GELU_SQRT_2_OVER_PI * (x + GELU_COEF * x**3)))).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish：x * sigmoid(x)。"""
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def fused_bias_gelu(C: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """一体版：bias + GELU 单遍（对应 Kernel epilogue 融合路径的真值）。"""
    return gelu_tanh(add_bias(C, bias))


def fused_bias_silu(C: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return silu(add_bias(C, bias))


# ---- 手算极小例（4×8 · 8×4 = 4×4，整数可手核）----
TINY_A = np.array(
    [
        [1, 2, 0, 1, 3, 0, 1, 1],
        [0, 1, 1, 0, 2, 2, 0, 0],
        [2, 0, 1, 1, 0, 1, 1, 2],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.float32,
)
TINY_B = np.array(
    [
        [1, 0, 2, 1],
        [0, 1, 1, 0],
        [2, 1, 0, 3],
        [1, 1, 1, 0],
        [0, 2, 1, 1],
        [1, 0, 0, 2],
        [2, 1, 3, 0],
        [0, 1, 1, 1],
    ],
    dtype=np.float32,
)
# 手算期望（逐行点积，如 C[0,0]=1*1+2*0+0*2+1*1+3*0+0*1+1*2+1*0=4）：
# C = [[4, 11, 12, 5],
#      [4,  6,  3, 9],
#      [8,  5, 10, 9],
#      [7,  7,  9, 8]]
TINY_C_EXPECTED = np.array(
    [
        [4, 11, 12, 5],
        [4, 6, 3, 9],
        [8, 5, 10, 9],
        [7, 7, 9, 8],
    ],
    dtype=np.float32,
)
TINY_BIAS = np.array([1, 0, -1, 2], dtype=np.float32)
