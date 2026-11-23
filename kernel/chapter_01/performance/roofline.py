"""第 01 期性能诊断 roofline 计算助手：理论字节、理论 FLOPs、计算强度、延迟下限、bound 判定。

口径约定（与 chapter_01 合同一脉相承）：

1. 融合单遍理论字节（theoretical_bytes，与 ch01 benchmark 同式）：
       bytes = (rows * H * 2 + H) * itemsize
   即"读一遍 x、写一遍 y、读一遍 weight"的理想单 Kernel 访存。
   这是任何实现（含第 03 期手写 Kernel）延迟下限的口径。

2. 理论 FLOPs（composite_rmsnorm_torch 组合表达式，FP32 域逐 Kernel 计数，每行）：
       pow(2): H 乘；mean 求和: H 加；除以 H: 1；加 eps: 1；rsqrt: 1；
       广播乘 rsqrt: H 乘；乘 weight: H 乘（dtype 转换不计 FLOP）
       => flops_per_row = 4H + 3
   这是"组合表达式语义"的 FLOPs，不是手写 Kernel（第 03 期）的。

3. 计算强度 AI = FLOPs / 融合理论字节。RMSNorm 的 AI ≈ 1 FLOP/Byte，
   远低于现代 GPU 脊点（数十~数百 FLOP/Byte）→ 深度 memory-bound。

4. 延迟下限 = 融合理论字节 / 平台峰值带宽。
   注意：延迟达成率（下限/实测）与带宽达成率（有效带宽/峰值）在数学上是
   同一个数（bytes/实测时间/峰值带宽），报告里两个名字都给，教学上分开讲。

5. 组合表达式未融合链路的理论流量（逐 Kernel 逐行字节，FP16 输入）：
   8 个 Kernel 各自把中间结果写回再读回，理论流量 ≈ 9.5 倍融合口径
   （逐行 reads 22H+12 + writes 16H+12 = 38H+24）。
   实际 DRAM 流量以 ncu 实测为准（中间结果可能驻留 L2，见 EXPERIMENT_LOG）。
"""

from __future__ import annotations

from typing import Any

RMS_NORM_EPS = 1e-6
HIDDEN_SIZE = 1024
ROW_COUNTS = [1, 4, 128, 512, 2048]

# 平台规格（大纲 §3.3 与 NVIDIA 官方规格；实机自测见 measure_peak.py）
PLATFORM_SPEC = {
    "cuda": {
        "gpu_name": "NVIDIA GeForce RTX 4090 (SM 8.9 Ada)",
        "peak_bandwidth_gb_s": 1008.0,  # GDDR6X 21 Gbps × 384-bit
        "peak_fp16_tflops": 165.2,  # Tensor Core，稠密（稀疏 330 不计）
        "peak_fp32_tflops": 82.6,  # CUDA Core FP32
        "spec_source": "大纲 §3.2/§3.3 + NVIDIA Ada 白皮书规格；以实机自测佐证",
    },
    "musa": {
        "gpu_name": "MTT S4000 48GB (第三代 MUSA 架构)",
        "peak_bandwidth_gb_s": 768.0,  # GDDR6 384-bit
        "peak_fp16_tflops": 100.0,  # Tensor 算力
        "peak_fp32_tflops": 25.0,
        "spec_source": "大纲 §3.3 公开规格（以官方规格书与实机复核为准）",
    },
    "cpu": {
        "gpu_name": "通用多核 CPU（串行世界的数量级锚点）",
        "peak_bandwidth_gb_s": None,  # 机型相关，不设统一口径
        "peak_fp16_tflops": None,
        "peak_fp32_tflops": None,
        "spec_source": "CPU 列不计峰值带宽达成率（大纲 §3.4：给出串行数量级）",
    },
}


def theoretical_bytes(rows: int, hidden_size: int, itemsize: int) -> int:
    """融合单遍理论字节（第 01 期合同口径）。"""
    return (rows * hidden_size * 2 + hidden_size) * itemsize


def theoretical_flops(rows: int, hidden_size: int) -> int:
    """组合表达式理论 FLOPs：每行 4H + 3（FP32 域，转换不计）。"""
    return rows * (4 * hidden_size + 3)


def arithmetic_intensity(flops: int, theoretical_bytes_value: int) -> float:
    """计算强度（FLOP/Byte），按融合单遍理论字节。"""
    return flops / theoretical_bytes_value


def latency_lower_bound_ms(theoretical_bytes_value: int, peak_bandwidth_bytes_per_s: float) -> float:
    """延迟下限（毫秒）= 理论字节 / 峰值带宽。"""
    return theoretical_bytes_value / peak_bandwidth_bytes_per_s * 1000.0


def compute_lower_bound_ms(flops: int, peak_flops_per_s: float) -> float:
    """纯算力下限（毫秒）= FLOPs / 峰值算力。"""
    return flops / peak_flops_per_s * 1000.0


def ridge_point(peak_flops_per_s: float, peak_bandwidth_bytes_per_s: float) -> float:
    """脊点（FLOP/Byte）= 峰值算力 / 峰值带宽。"""
    return peak_flops_per_s / peak_bandwidth_bytes_per_s


def unfused_chain_bytes_per_row(hidden_size: int) -> dict[str, int]:
    """组合表达式 8 个 Kernel 的逐行理论流量（FP16 输入，FP32 中间结果）。

    返回 {"read": ..., "write": ..., "total": ...}（字节/行）。
    每行 8 个 Kernel：cast_in, pow2, mean, add_eps, rsqrt, broadcast_mul,
    cast_out, mul_weight（详见模块 docstring 与 EXPERIMENT_LOG）。
    """
    h = hidden_size
    reads = 2 * h + 4 * h + 4 * h + 4 + 4 + (4 * h + 4) + 4 * h + 4 * h
    writes = 4 * h + 4 * h + 4 + 4 + 4 + 4 * h + 2 * h + 2 * h
    return {"read": reads, "write": writes, "total": reads + writes}


def unfused_chain_bytes(rows: int, hidden_size: int, itemsize: int) -> int:
    """未融合链路总理论字节（weight 每次调用读一次，读 rows 次 y/w 已含在逐行项；
    weight 2H 只算一次调用级开销）。"""
    per_row = unfused_chain_bytes_per_row(hidden_size)["total"]
    return rows * per_row + hidden_size * itemsize


# ---- bound 判定 ----

def classify_kernel_bound(
    theoretical_bytes_value: int,
    flops: int,
    peak_bandwidth_bytes_per_s: float | None,
    peak_flops_per_s: float | None,
) -> dict[str, Any]:
    """Kernel 层面 roofline 判定：memory-bound / compute-bound / balanced。

    以理论时间比值为准：t_mem = bytes/峰值带宽，t_compute = flops/峰值算力。
    ratio = t_mem / t_compute：>1 偏 memory，<1 偏 compute，>10 记深度 memory-bound。
    """
    if peak_bandwidth_bytes_per_s is None or peak_flops_per_s is None:
        return {"kernel_bound_class": "unavailable", "reason": "缺少平台峰值规格"}
    t_mem_ms = latency_lower_bound_ms(theoretical_bytes_value, peak_bandwidth_bytes_per_s)
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
        "roofline_mem_over_compute": round(ratio, 2),
        "ridge_point_flop_per_byte": round(ridge_point(peak_flops_per_s, peak_bandwidth_bytes_per_s), 1),
    }


def classify_eager_mode(
    eager_p50_ms: float,
    gpu_p50_ms: float | None,
) -> dict[str, Any]:
    """发射层面判定：eager 计时是否被 CPU 发射开销主导（launch-bound 识别方法之一）。

    依据：eager P50（事件对口径，含 CPU enqueue 空隙）vs Graph 回放 P50
    （纯 GPU 执行时间）。eager 显著大于 graph（>2×）即判 launch-dominated。
    """
    if gpu_p50_ms is None or gpu_p50_ms <= 0:
        return {"eager_mode_class": "unknown", "reason": "无 Graph 回放参照"}
    ratio = eager_p50_ms / gpu_p50_ms
    gap_us = (eager_p50_ms - gpu_p50_ms) * 1000.0
    if ratio >= 2.0:
        classification = "launch-dominated (eager 被 CPU 发射主导)"
    elif ratio >= 1.2:
        classification = "partly launch-affected"
    else:
        classification = "gpu-dominated"
    return {
        "eager_mode_class": classification,
        "eager_over_gpu_ratio": round(ratio, 2),
        "launch_gap_us": round(gap_us, 2),
    }


def per_size_launch_bound(tight_loop_us: float, gpu_us: float) -> dict[str, Any]:
    """launch-bound 场景逐尺寸判定：紧循环单次耗时是否被发射吞吐限制。

    tight_loop_us ≈ max(CPU 发射吞吐, GPU 执行)；
    gpu_us（事件口径）是纯 GPU 执行。
    gpu_us < tight_loop_us 的 1/2 即判该尺寸处于 launch-bound 区。
    """
    if tight_loop_us <= 0:
        return {"launch_bound": False}
    is_bound = gpu_us < tight_loop_us / 2.0
    return {
        "launch_bound": bool(is_bound),
        "launch_overhead_us": round(max(tight_loop_us - gpu_us, 0.0), 3),
        "gpu_share_pct": round(min(gpu_us / tight_loop_us, 1.0) * 100.0, 1),
    }
