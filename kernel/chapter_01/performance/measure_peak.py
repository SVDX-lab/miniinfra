#!/usr/bin/env python3
"""第 01 期性能诊断平台峰值自测：显存带宽（D2D 拷贝）与 FP16/FP32 算力（GEMM）。

方法：
- 带宽：大张量 dst.copy_(src)（读 + 写 = 2×bytes），事件计时，多尺寸取最优。
  该口径与理论峰值带宽对齐（拷贝是最小 Kernel，无计算）。
- FP16 算力：M=N=K 的 fp16 GEMM（cuBLAS/MUSA 库路径，Tensor Core），
  FLOPs = 2*M*N*K，多尺寸取最优；FP32 算力：fp32 GEMM（CUDA Core）。
- 结果写入 results/peak-measurement-{backend}.json，供 benchmark.py 读取佐证。

用法：
    python measure_peak.py --backend cuda
    python measure_peak.py --backend musa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_utils import backend_sync, device_environment_record, resolve_device, write_results_json

SCHEMA_VERSION = "kernel-chapter02.peak.v1"

BANDWIDTH_SIZE_MIB = [256, 512, 1024, 2048]
GEMM_FP16_SIZES = [4096, 8192]
GEMM_FP32_SIZES = [4096]


def _backend_sync(module) -> None:
    backend_sync(torch.device("cuda" if module is torch.cuda else "musa"))


def event_timer(module, repeats: int, warmup: int, operation):
    for _ in range(warmup):
        operation()
    _backend_sync(module)
    start_events = []
    end_events = []
    for _ in range(repeats):
        start_events.append(module.Event(enable_timing=True))
        end_events.append(module.Event(enable_timing=True))
    for index in range(repeats):
        start_events[index].record()
        operation()
        end_events[index].record()
    _backend_sync(module)
    samples = [start_events[i].elapsed_time(end_events[i]) for i in range(repeats)]
    ordered = sorted(samples)
    return {
        "repeats": repeats,
        "p50_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def measure_bandwidth(device: torch.device, module) -> dict:
    entries = []
    for size_mib in BANDWIDTH_SIZE_MIB:
        elements = size_mib * 1024 * 1024 // 2  # fp16
        source = torch.randn(elements, device=device, dtype=torch.float16)
        destination = torch.empty_like(source)
        timing = event_timer(module, repeats=15, warmup=3, operation=lambda: destination.copy_(source))
        bytes_moved = elements * 2 * 2  # 读 + 写
        bw = bytes_moved / (timing["p50_ms"] / 1000.0)
        bw_min = bytes_moved / (timing["min_ms"] / 1000.0)
        entries.append({
            "size_mib": size_mib,
            "bytes_moved": bytes_moved,
            "p50_ms": round(timing["p50_ms"], 4),
            "min_ms": round(timing["min_ms"], 4),
            "bandwidth_gb_s_p50": round(bw / 1e9, 2),
            "bandwidth_gb_s_min": round(bw_min / 1e9, 2),
        })
        print(f"[bandwidth/{device.type}] {size_mib} MiB: p50={timing['p50_ms']:.3f}ms "
              f"带宽={entries[-1]['bandwidth_gb_s_p50']}GB/s (min口径 {entries[-1]['bandwidth_gb_s_min']})")
        del source, destination
        module.empty_cache()
    best = max(entries, key=lambda e: e["bandwidth_gb_s_p50"])
    return {
        "method": "D2D copy_（读+写=2x bytes），事件计时，多尺寸取 p50 最优",
        "sizes_mib": BANDWIDTH_SIZE_MIB,
        "per_size": entries,
        "best_gb_s": best["bandwidth_gb_s_p50"],
        "best_size_mib": best["size_mib"],
    }


def measure_gemm(device: torch.device, module, dtype: torch.dtype, sizes: list[int], repeats: int) -> dict:
    label = "fp16" if dtype == torch.float16 else "fp32"
    entries = []
    for size in sizes:
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)
        timing = event_timer(module, repeats=repeats, warmup=3, operation=lambda: torch.mm(a, b))
        flops = 2.0 * size * size * size
        tflops = flops / (timing["p50_ms"] / 1000.0) / 1e12
        entries.append({
            "size": size,
            "p50_ms": round(timing["p50_ms"], 4),
            "min_ms": round(timing["min_ms"], 4),
            "tflops_p50": round(tflops, 2),
            "tflops_min": round(flops / (timing["min_ms"] / 1000.0) / 1e12, 2),
        })
        print(f"[gemm/{device.type}] {label}@{size}x{size}x{size}: p50={timing['p50_ms']:.3f}ms "
              f"tflops={entries[-1]['tflops_p50']}")
        del a, b
        module.empty_cache()
    best = max(entries, key=lambda e: e["tflops_p50"])
    return {"sizes": sizes, "per_size": entries, "best": best["tflops_p50"], "best_size": best["size"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    module = getattr(torch, device.type)

    bandwidth = measure_bandwidth(device, module)
    # 充分 warmup：MUSA 库路径首轮有 JIT/算法选择开销（第 01 期实测注记）
    fp16 = measure_gemm(device, module, torch.float16, GEMM_FP16_SIZES, repeats=10)
    fp32 = measure_gemm(device, module, torch.float32, GEMM_FP32_SIZES, repeats=10)

    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "bandwidth": bandwidth,
        "tflops_fp16": fp16,
        "tflops_fp32": fp32,
        "note": "峰值自测值用于佐证规格并给出实测峰值参照；主达成率口径仍以规格峰值带宽为准",
        "environment": device_environment_record(device),
    }
    output = Path("results") / f"peak-measurement-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
