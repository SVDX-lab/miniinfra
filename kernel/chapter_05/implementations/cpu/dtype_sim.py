"""第 05 期 CPU 模拟路径 —— 三 dtype RMSNorm 与位精确的累加链模拟器。

三个角色（对应 GPU Kernel 的三条编译路径，语义一一对应）：
1. rmsnorm_simulated_kernel(x, w, eps, dtype, acc_dtype)：
   模拟手写 Kernel 的 specialization —— 读 T、转 FP32 计算、写回转 T。
   acc_dtype ∈ {fp32, fp16, bf16} 控制平方和累加链的算术（见下）；
2. rmsnorm_composite_torch(x, w, eps, dtype)：torch 原生组合表达式
   （平台原生路径，CPU 上也支持 fp16/bf16 —— torch CPU 有原生 kernel；
   NumPy 无 bf16，所以模拟路径全部用 torch 张量转 FP32 或位运算模拟）；
3. 位精确累加模拟器（serial_sqsum_fp16 / serial_sqsum_bf16）：
   不依赖任何库的内部累加实现，逐元素模拟目标格式的 IEEE 算术：
   - FP16：NumPy 原生 float16（IEEE 754-2008 binary16，每次 +/* 独立舍入）；
   - BF16：无原生库 → 位运算模拟：bf16 算术 = FP32 精确计算 + RNE 舍回 bf16
     （硬件 BF16 ALU 的标准语义：全精度计算、结果舍入到 8 位有效数字）。
   这两个模拟器是"累加 dtype"实验（实验 4）的 CPU 侧真值：用来在本地
   复现 FP16 累加的溢出（sum > 65504 → inf）与精度衰减（ulp 随量级增大）。

BF16 RNE 舍入公式（uint32 位域）：
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    （0x7FFF 加半 ulp，尾数最低有效位决定 tie 时向偶数舍入；inf/NaN 位型
    经同样公式后仍是 inf/NaN，无需特判 —— 本模块用 random 全量核对。）
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from naive_rmsnorm import RMS_NORM_EPS, composite_rmsnorm_torch  # noqa: F401
except ImportError:  # 独立运行时降级
    RMS_NORM_EPS = 1e-6


# ---------------------------------------------------------------------------
# BF16 位运算模拟（RNE，与 torch bfloat16 转换全量位级一致——见 smoke_test）
# ---------------------------------------------------------------------------
def f32_bits(x: np.ndarray) -> np.ndarray:
    """float32 numpy -> uint32 位型。"""
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def bits_f32(bits: np.ndarray) -> np.ndarray:
    """uint32 位型 -> float32 numpy（bf16 位型左移 16 位即 FP32 位型）。"""
    return np.ascontiguousarray(bits, dtype=np.uint32).view(np.float32)


def round_f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 位型（RNE）。纯整数运算，向量批处理。"""
    u = f32_bits(x)
    lsb = (u >> 16) & 1
    rounded = (u + np.uint32(0x7FFF) + lsb) >> 16
    return rounded.astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """bf16 位型（uint16）-> float32（左移 16 位即 FP32 位型）。"""
    return bits_f32(bits.astype(np.uint32) << np.uint32(16))


def round_f32_to_bf16(x: np.ndarray) -> np.ndarray:
    return bf16_bits_to_f32(round_f32_to_bf16_bits(x))


# ---------------------------------------------------------------------------
# 位精确串行平方和累加链（实验 4 的 CPU 真值模拟器）
# ---------------------------------------------------------------------------
def serial_sqsum_fp32(row: np.ndarray) -> tuple[float, bool]:
    """FP32 自然顺序累加平方和（真值口径，同 ch03 naive_rmsnorm_serial）。"""
    acc = np.float32(0.0)
    for v in row:
        acc = np.float32(acc + np.float32(v * v))
    return float(acc), bool(np.isinf(acc))


def serial_sqsum_fp16(row: np.ndarray) -> tuple[float, bool]:
    """FP16 串行累加平方和：每一步乘法与加法都在 binary16 内舍入。

    NumPy float16 的 +/* 是真正的 IEEE binary16 运算（逐元素慢路径），
    溢出按 IEEE 规则饱和为 inf。返回 (累加值, 是否出现 inf/overflow)。
    """
    acc = np.float16(0.0)
    overflowed = False
    for v in row:
        hv = np.float16(v)
        prod = np.float16(hv * hv)      # 平方在 fp16 内舍入（|v|>~256 时溢出）
        acc = np.float16(acc + prod)    # 累加在 fp16 内舍入
        if np.isinf(acc) or np.isinf(prod):
            overflowed = True
    return float(acc), overflowed


def serial_sqsum_bf16(row: np.ndarray) -> tuple[float, bool]:
    """BF16 串行累加平方和：位运算模拟（全精度计算 + RNE 舍回，标量路径）。"""
    import struct

    def f32_bits_scalar(v: float) -> int:
        return struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]

    def bf16_round_scalar(v: float) -> int:
        u = f32_bits_scalar(v)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF

    def bf16_to_f32_scalar(bits: int) -> float:
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]

    row32 = np.asarray(row, dtype=np.float32)
    acc_bits = 0  # bf16 的 +0.0
    overflowed = False
    for v in row32:
        prod_bits = bf16_round_scalar(np.float32(v) * np.float32(v))   # bf16 乘法（舍入到位型）
        prod_val = bf16_to_f32_scalar(prod_bits)                       # 位型 -> 数值
        acc_f = bf16_to_f32_scalar(acc_bits)                           # acc -> f32
        acc_bits = bf16_round_scalar(np.float32(acc_f) + prod_val)     # bf16 加法（全精度和 + RNE）
        if np.isinf(bf16_to_f32_scalar(acc_bits)):
            overflowed = True
    return bf16_to_f32_scalar(acc_bits), overflowed


SERIAL_SQSUM = {"fp32": serial_sqsum_fp32, "fp16": serial_sqsum_fp16, "bf16": serial_sqsum_bf16}


# ---------------------------------------------------------------------------
# 模拟 Kernel 路径（与 GPU Kernel specialization 一一对应）
# ---------------------------------------------------------------------------
def _to_storage_dtype(x32: np.ndarray, dtype: str) -> np.ndarray:
    """FP32 -> 存储格式 T（模拟 Kernel 输入已在 T 中存放）。"""
    if dtype == "fp32":
        return x32.astype(np.float32)
    if dtype == "fp16":
        return x32.astype(np.float16)
    if dtype == "bf16":
        return round_f32_to_bf16(x32)
    raise ValueError("dtype 只支持 fp32/fp16/bf16")


def _storage_to_f32(x_t: np.ndarray, dtype: str) -> np.ndarray:
    """存储格式 T -> FP32（Kernel Pass 1 读入后转换，值无损：16 位都是 FP32 的子集）。"""
    if dtype == "fp32":
        return x_t.astype(np.float32)
    if dtype == "fp16":
        return x_t.astype(np.float32)
    if dtype == "bf16":
        return x_t.astype(np.float32)
    raise ValueError(dtype)


def rmsnorm_simulated_kernel(x: np.ndarray, w: np.ndarray, eps: float = RMS_NORM_EPS,
                             dtype: str = "fp16", acc_dtype: str = "fp32") -> dict:
    """模拟手写 RMSNorm Kernel 的三条编译路径（数值语义对齐，非性能模拟）。

    路径（与 CUDA/MUSA Kernel 注释严格对应）：
      Pass 1: 读 T -> 转 FP32（to_float）-> 平方与累加在 acc_dtype 算术内完成
      归约:   total -> FP32（溢出时为 inf）-> mean + eps -> rsqrt -> scale（FP32）
      Pass 2: y = x_f32 * scale * w_f32（FP32）-> 一次舍入写回 T（cast_from_float）

    acc_dtype 的含义（实验 4 的核心变量）：
      fp32: 平方、累加都在 FP32（合同默认）；
      fp16: 平方与每步累加都在 FP16（模拟"acc 声明成 half"的实现错误）；
      bf16: 平方与每步累加都在 BF16。
    返回输出与诊断（平方和值、溢出标志）。
    """
    x32 = np.asarray(x, dtype=np.float32)
    w32 = np.asarray(w, dtype=np.float32)
    if x32.ndim != 2 or w32.shape[0] != x32.shape[1]:
        raise ValueError("x [rows,H] 与 w [H] 形状不匹配")
    x_t = _to_storage_dtype(x32, dtype)          # 输入落到存储 dtype
    f = _storage_to_f32(x_t, dtype)              # 读入转 FP32（无损）
    rows, h = x32.shape
    sqsum_fn = SERIAL_SQSUM[acc_dtype]
    y_t = np.empty((rows, h), dtype=np.float32)
    diag = {"rows": [], "acc_dtype": acc_dtype, "dtype": dtype}
    for i in range(rows):
        if acc_dtype == "fp32":
            acc = np.float32(0.0)
            fv = f[i]
            acc = np.sum((fv * fv).astype(np.float32), dtype=np.float32)
            ovf = bool(np.isinf(acc))
        else:
            acc, ovf = sqsum_fn(f[i])
        total_f = np.float32(acc)  # ACC -> FP32（inf 传染：rsqrtf(inf)=0）
        scale = np.float32(1.0) / np.float32(np.sqrt(total_f / np.float32(h) + np.float32(eps)))
        y32 = (f[i] * scale) * w32  # FP32 计算与真值同式
        diag["rows"].append({"sqsum": float(total_f), "overflow": ovf,
                             "scale": float(scale)})
        y_t[i] = y32
    # 写回：一次舍入到 T（与 Kernel cast_from_float<T> 一致）
    y_storage = _to_storage_dtype(y_t, dtype)
    diag["any_overflow"] = any(r["overflow"] for r in diag["rows"])
    return {"y": y_storage.astype(np.float32) if dtype != "fp32" else y_storage,
            "diag": diag}


def rmsnorm_composite_torch_dtype(x_t: torch.Tensor, w_t: torch.Tensor,
                                  eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """torch 组合表达式（参数化 dtype 版，三 dtype 平台原生路径对照）。

    与 ch01/03 composite_rmsnorm_torch 同一公式；本函数直接接收已落在
    目标 dtype 的张量（内部升 FP32 求方差 —— torch 官方 RMSNorm 语义），
    写回输入 dtype。
    """
    input_dtype = x_t.dtype
    float_states = x_t.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return w_t * normalized.to(input_dtype)


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


# ---------------------------------------------------------------------------
# 累加实验构造输入（实验 4；固定 seed，服务器端同规则重建）
# ---------------------------------------------------------------------------
def large_norm_row(h: int, scale: float, seed: int) -> np.ndarray:
    """大范数行：standard_normal × scale，平方和 ≈ H·scale²。"""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(h) * scale).astype(np.float32)


def long_row(h: int, seed: int) -> np.ndarray:
    """长行：standard_normal（平方和 ≈ H，H 增大逼近 FP16 max=65504）。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(h).astype(np.float32)


def wide_dynamic_row(rng: np.random.Generator, h: int) -> np.ndarray:
    """宽动态范围行：一半元素 ~1e4、一半 ~1e-7（bf16 vs fp16 的"更准"构造）。"""
    big = rng.standard_normal(h // 2) * 1e4
    small = rng.standard_normal(h - h // 2) * 1e-7
    row = np.concatenate([big, small])
    rng.shuffle(row)
    return row.astype(np.float32)
