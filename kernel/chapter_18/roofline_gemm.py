"""第 17 期 roofline 计算助手（改造自 chapter_01/performance/roofline.py，口径单源延续）。

与第 01 期的关系：
- bound 分类逻辑（t_mem/t_compute 比值 + 10×/0.1× 阈值）直接沿用 ch02；
- 平台规格峰值（1008/768 GB/s 等）沿用 ch02 PLATFORM_SPEC；
- 实测峰值引用 ch02 measure_peak 结果（4090: D2D 918.8 GB/s、FP32 GEMM
  55.6 TF、FP16 GEMM 171.63 TF；S4000: 711.28 GB/s、12.53 TF、64.55 TF），
  作为"与厂商库比值"的分母口径（主 agent 转达的 ep02 情报）。

GEMM 理论模型（与任务书一致）：
- 理论字节 = (M*K + K*N + M*N) * itemsize（读 A、读 B、写 C 各一遍，
  bias 向量 N*4B 忽略不计并注明）；
- 理论 FLOPs = 2*M*N*K；
- AI = 2MNK / ((MK+KN+MN)*itemsize)。

注意：这是"理想复用"（每个 A/B 元素只从 DRAM 读一次）的算法 AI。
tile 版 kernel 在这些形状下 A+B 工作集 ≤ 8MB ≪ L2（4090 72MB），
tile 间的重读大概率命中 L2，故名义字节 ≈ DRAM 字节的假设成立；
naive（无 tile）kernel 的实际访存远超名义字节（每线程整行/整列重复读），
其 bound 判定要按实测有效带宽对账，EXPERIMENT_LOG 单独分析。
"""

from __future__ import annotations

from typing import Any

# 平台规格（沿用 chapter_01/performance/roofline.py PLATFORM_SPEC）
PLATFORM_SPEC = {
    "cuda": {
        "gpu_name": "NVIDIA GeForce RTX 4090 (SM 8.9 Ada)",
        "peak_bandwidth_gb_s": 1008.0,
        "peak_fp16_tflops": 165.2,
        "peak_fp32_tflops": 82.6,
        "l2_bytes": 72 * 1024 * 1024,
        "spec_source": "chapter_01/performance/roofline.py 同源：Ada 白皮书 + 大纲 §3.3",
    },
    "musa": {
        "gpu_name": "MTT S4000 48GB (第三代 MUSA 架构)",
        "peak_bandwidth_gb_s": 768.0,
        "peak_fp16_tflops": 100.0,
        "peak_fp32_tflops": 25.0,
        "l2_bytes": None,  # 公开资料未确认，实机核对记 None
        "spec_source": "chapter_01/performance/roofline.py 同源：大纲 §3.3 公开规格",
    },
    "cpu": {
        "gpu_name": "WSL2 12 核 x86_64（教学数量级锚点）",
        "peak_bandwidth_gb_s": None,
        "peak_fp16_tflops": None,
        "peak_fp32_tflops": None,
        "l2_bytes": None,
        "spec_source": "CPU 列不计峰值达成率（大纲 §3.4）",
    },
}

# chapter_01/performance 实测峰值（results/peak-measurement-*.json，ep02 情报转达）
MEASURED_PEAK = {
    "cuda": {
        "copy_bandwidth_gb_s": 918.8,
        "fp32_gemm_tflops": 55.6,  # cuBLAS FP32 自测（受功耗墙，规格 82.6 的 67%）
        "fp16_gemm_tflops": 171.63,  # cuBLAS FP16 Tensor Core 自测
        "source": "chapter_01/performance/results/peak-measurement-cuda.json (2026-11-20 新实例)",
    },
    "musa": {
        "copy_bandwidth_gb_s": 711.28,
        "fp32_gemm_tflops": 12.53,
        "fp16_gemm_tflops": 64.55,
        "source": "chapter_01/performance/results/peak-measurement-musa.json (2026-11-20)",
    },
}


def theoretical_bytes(m: int, n: int, k: int, itemsize: int) -> int:
    """GEMM 理论字节：读 A、读 B、写 C 各一遍（bias N*4B 忽略，注明口径）。"""
    return (m * k + k * n + m * n) * itemsize


def theoretical_flops(m: int, n: int, k: int) -> int:
    """GEMM 理论 FLOPs = 2*M*N*K（每 MAC 计 2）。"""
    return 2 * m * n * k


def arithmetic_intensity(flops: int, bytes_value: int) -> float:
    """计算强度（FLOP/Byte）。"""
    return flops / bytes_value


def gemm_arithmetic_intensity(m: int, n: int, k: int, itemsize: int) -> float:
    """GEMM 算法 AI = 2MNK / ((MK+KN+MN)*itemsize)。"""
    return arithmetic_intensity(theoretical_flops(m, n, k), theoretical_bytes(m, n, k, itemsize))


def latency_lower_bound_ms(bytes_value: int, peak_bandwidth_bytes_per_s: float) -> float:
    """访存下限（毫秒）= 理论字节 / 峰值带宽（ch02 同式）。"""
    return bytes_value / peak_bandwidth_bytes_per_s * 1000.0


def compute_lower_bound_ms(flops: int, peak_flops_per_s: float) -> float:
    """算力下限（毫秒）= FLOPs / 峰值算力（ch02 同式）。"""
    return flops / peak_flops_per_s * 1000.0


def ridge_point(peak_flops_per_s: float, peak_bandwidth_bytes_per_s: float) -> float:
    """脊点（FLOP/Byte）= 峰值算力 / 峰值带宽。"""
    return peak_flops_per_s / peak_bandwidth_bytes_per_s


def classify_kernel_bound(
    bytes_value: int,
    flops: int,
    peak_bandwidth_bytes_per_s: float | None,
    peak_flops_per_s: float | None,
) -> dict[str, Any]:
    """ch02 同款 bound 判定：t_mem/t_compute 比值，>1 偏 memory，<1 偏 compute。

    下限取两者较大者（roofline 双约束的包络）：
    bound_ms = max(t_mem, t_compute)；达成率 = bound_ms / 实测。
    """
    if peak_bandwidth_bytes_per_s is None or peak_flops_per_s is None:
        return {"kernel_bound_class": "unavailable", "reason": "缺少平台峰值规格"}
    t_mem_ms = latency_lower_bound_ms(bytes_value, peak_bandwidth_bytes_per_s)
    t_compute_ms = compute_lower_bound_ms(flops, peak_flops_per_s)
    ratio = t_mem_ms / t_compute_ms if t_compute_ms > 0 else float("inf")
    if ratio >= 10.0:
        classification = "memory-bound (deep)"
    elif ratio > 1.0:
        classification = "memory-bound"
    elif ratio < 0.1:
        classification = "compute-bound (deep)"
    elif ratio < 1.0:
        classification = "compute-bound"
    else:
        classification = "balanced"
    return {
        "kernel_bound_class": classification,
        "roofline_t_mem_ms": round(t_mem_ms, 6),
        "roofline_t_compute_ms": round(t_compute_ms, 6),
        "roofline_bound_ms": round(max(t_mem_ms, t_compute_ms), 6),
        "roofline_mem_over_compute": round(ratio, 2),
        "ridge_point_flop_per_byte": round(ridge_point(peak_flops_per_s, peak_bandwidth_bytes_per_s), 1),
    }


def rmsnorm_contrast(rows: int, hidden_size: int, itemsize: int) -> dict[str, Any]:
    """RMSNorm 对照（ch02 口径：融合单遍字节 + 每行 4H+3 FLOPs）。

    用于"GEMM vs RMSNorm bound 为什么不同"的同一张 roofline 表。
    """
    bytes_value = (rows * hidden_size * 2 + hidden_size) * itemsize
    flops = rows * (4 * hidden_size + 3)
    ai = flops / bytes_value
    return {
        "op": "rmsnorm",
        "shape": {"rows": rows, "hidden_size": hidden_size, "itemsize": itemsize},
        "theoretical_bytes": bytes_value,
        "theoretical_flops": flops,
        "arithmetic_intensity_flop_per_byte": round(ai, 4),
    }


def effective_bandwidth_gb_s(bytes_value: int, p50_ms: float) -> float:
    """有效带宽 = 理论字节 / 实测时间。若结果 > 拷贝实测峰值，说明该 kernel
    不可能被 DRAM 访存解释（compute-bound 侧证据，无计数器条件下的对账法）。"""
    if p50_ms <= 0:
        return 0.0
    return bytes_value / (p50_ms / 1000.0) / 1e9


def effective_tflops(flops: int, p50_ms: float) -> float:
    if p50_ms <= 0:
        return 0.0
    return flops / (p50_ms / 1000.0) / 1e12
