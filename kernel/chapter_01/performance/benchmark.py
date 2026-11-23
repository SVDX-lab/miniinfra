#!/usr/bin/env python3
"""第 01 期性能诊断三后端 RMSNorm Baseline + roofline 口径测量。

在第 01 期 benchmark 口径（P50/P95、有效带宽、达成率）之上新增：
- 融合理论字节、未融合链路理论流量、理论 FLOPs、计算强度（AI）；
- 延迟下限（理论字节 / 峰值带宽）与延迟/带宽达成率；
- roofline 判定（t_mem vs t_compute、脊点、memory/compute bound）；
- launch-bound 识别（eager P50 vs Graph 回放 P50，launch gap）。

峰值口径：优先读取 results/peak-measurement-{backend}.json（measure_peak.py
实测），缺失时回退规格值并标注 peak_source。主达成率口径 = 有效带宽 / 规格
峰值带宽（与第 01 期可比），同时给出相对实测峰值达成率。

用法示例：
    python benchmark.py --backend cuda
    python benchmark.py --backend musa
    python benchmark.py --backend cpu
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_utils import (
    OperatorTimer,
    backend_sync,
    device_environment_record,
    measure_with_graph,
    resolve_device,
    write_results_json,
)
from roofline import (
    PLATFORM_SPEC,
    RMS_NORM_EPS,
    HIDDEN_SIZE,
    ROW_COUNTS,
    arithmetic_intensity,
    classify_eager_mode,
    classify_kernel_bound,
    compute_lower_bound_ms,
    latency_lower_bound_ms,
    theoretical_bytes,
    theoretical_flops,
    unfused_chain_bytes,
    unfused_chain_bytes_per_row,
)
from rmsnorm_reference import composite_rmsnorm_torch

SCHEMA_VERSION = "kernel-chapter02.benchmark.v1"


def load_peak_measurement(backend: str) -> dict | None:
    path = Path("results") / f"peak-measurement-{backend}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def resolve_peak(backend: str, peak_json: dict | None) -> dict:
    """峰值口径：规格为主（可比性），实测为辅（佐证与上限参照）。"""
    spec = PLATFORM_SPEC[backend]
    resolved = {
        "peak_bandwidth_gb_s": spec["peak_bandwidth_gb_s"],
        "peak_fp16_tflops": spec["peak_fp16_tflops"],
        "peak_fp32_tflops": spec["peak_fp32_tflops"],
        "spec_source": spec["spec_source"],
        "peak_source": "spec",
        "measured_bandwidth_gb_s": None,
        "measured_fp16_tflops": None,
    }
    if peak_json and backend != "cpu":
        resolved["measured_bandwidth_gb_s"] = peak_json.get("bandwidth", {}).get("best_gb_s")
        resolved["measured_fp16_tflops"] = peak_json.get("tflops_fp16", {}).get("best")
        if resolved["measured_bandwidth_gb_s"]:
            resolved["peak_source"] = "spec (实测佐证: %.1f GB/s)" % resolved["measured_bandwidth_gb_s"]
    return resolved


def load_or_build_input(rows: int, hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """优先 fixtures（位级固定），缺失时按同一 seed 规则内联生成（与 ch01 一致）。"""
    fixture = Path("fixtures") / f"x-{rows}x{hidden_size}-fp32.npy"
    if fixture.exists():
        array = np.load(fixture)
        return torch.from_numpy(array).to(device=device, dtype=dtype)
    seed = 20260101 + rows * 100000 + hidden_size
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden_size, generator=generator).to(device=device, dtype=dtype)


def rmsnorm_chain(arguments, device: torch.device, dtype: torch.dtype, peak: dict) -> dict:
    itemsize = torch.tensor([], dtype=dtype).element_size()
    fixture_weight = Path("fixtures") / "weight-fp32.npy"
    if fixture_weight.exists():
        weight = torch.from_numpy(np.load(fixture_weight)).to(device=device, dtype=dtype)
    else:
        seed = 20260101 + 0 * 100000 + HIDDEN_SIZE
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(HIDDEN_SIZE, generator=generator).to(device=device, dtype=dtype)

    peak_bw_bytes = None if peak["peak_bandwidth_gb_s"] is None else peak["peak_bandwidth_gb_s"] * 1e9
    peak_fp16 = None if peak["peak_fp16_tflops"] is None else peak["peak_fp16_tflops"] * 1e12
    peak_fp32 = None if peak["peak_fp32_tflops"] is None else peak["peak_fp32_tflops"] * 1e12
    # FLOPs 按 FP32 域计（组合表达式在 FP32 域计算），算力峰值取 FP32 CUDA Core。
    compute_peak = peak_fp32

    results = []
    for rows in ROW_COUNTS:
        x = load_or_build_input(rows, HIDDEN_SIZE, device, dtype)
        repeats = 200 if rows <= 128 else (100 if rows == 512 else 50)
        timer = OperatorTimer(device, warmup=10, repeats=repeats)
        operation = lambda: composite_rmsnorm_torch(x, weight, RMS_NORM_EPS)  # noqa: E731
        timing = timer.measure(operation)
        timing["mode"] = "eager"

        total_bytes = theoretical_bytes(rows, HIDDEN_SIZE, itemsize)
        flops = theoretical_flops(rows, HIDDEN_SIZE)
        unfused_bytes = unfused_chain_bytes(rows, HIDDEN_SIZE, itemsize)
        entry = {
            "rows": rows,
            "hidden_size": HIDDEN_SIZE,
            "dtype": str(dtype).replace("torch.", ""),
            "theoretical_bytes_fused": total_bytes,
            "theoretical_bytes_unfused_chain": unfused_bytes,
            "theoretical_flops": flops,
        }
        entry.update(timing)

        # Graph Replay 口径：分离 CPU 发射时间与纯 GPU 执行时间（launch-bound 识别）。
        graph_timing = measure_with_graph(device, warmup=3, repeats=repeats, operation=operation)
        entry["graph_replay"] = graph_timing
        effective_source = timing
        if graph_timing and graph_timing["p50_ms"] < timing["p50_ms"]:
            effective_source = graph_timing

        entry["arithmetic_intensity_flop_per_byte"] = round(arithmetic_intensity(flops, total_bytes), 3)

        if peak_bw_bytes is not None:
            lower_bound = latency_lower_bound_ms(total_bytes, peak_bw_bytes)
            entry["latency_lower_bound_ms"] = round(lower_bound, 6)
            entry["eager_mode_class"] = classify_eager_mode(timing["p50_ms"], None if graph_timing is None else graph_timing["p50_ms"])
            entry["kernel_bound_class"] = classify_kernel_bound(total_bytes, flops, peak_bw_bytes, compute_peak)

            # 主口径：best-of（与 ch01 一致，标注 mode）
            effective = total_bytes / (effective_source["p50_ms"] / 1000.0)
            effective_mode = "graph_replay" if (graph_timing is not None and effective_source is graph_timing) else "eager"
            entry["effective_bandwidth_gb_s"] = round(effective / 1e9, 2)
            entry["achievement_ratio_spec"] = round(effective / peak_bw_bytes, 4)
            entry["latency_achievement_ratio"] = entry["achievement_ratio_spec"]  # 数学上同一个数
            entry["bandwidth_mode"] = effective_mode
            if peak["measured_bandwidth_gb_s"]:
                entry["achievement_ratio_measured_peak"] = round(
                    effective / (peak["measured_bandwidth_gb_s"] * 1e9), 4
                )

            # 纯 GPU 执行口径（graph 回放；MUSA 上 graph 可能慢于 eager，如实双记）
            if graph_timing is not None:
                gpu_effective = total_bytes / (graph_timing["p50_ms"] / 1000.0)
                entry["gpu_only_bandwidth_gb_s"] = round(gpu_effective / 1e9, 2)
                entry["gpu_only_achievement_ratio_spec"] = round(
                    gpu_effective / peak_bw_bytes, 4
                )
            # eager 口径
            eager_effective = total_bytes / (timing["p50_ms"] / 1000.0)
            entry["eager_bandwidth_gb_s"] = round(eager_effective / 1e9, 2)
            entry["eager_achievement_ratio_spec"] = round(eager_effective / peak_bw_bytes, 4)

        results.append(entry)
        print(
            f"[rmsnorm/{arguments.backend}] rows={rows:<5} "
            f"eager_p50={timing['p50_ms']:.4f}ms "
            + (
                f"graph_p50={graph_timing['p50_ms']:.4f}ms "
                if graph_timing
                else ""
            )
            + (
                f"下限={entry.get('latency_lower_bound_ms', 0):.4f}ms "
                f"有效带宽={entry.get('effective_bandwidth_gb_s')}GB/s "
                f"达成率={entry.get('achievement_ratio_spec', 0):.2%} "
                f"[{entry.get('eager_mode_class', {}).get('eager_mode_class', 'n/a')}"
                f"|{entry.get('kernel_bound_class', {}).get('kernel_bound_class', 'n/a')}]"
                if "achievement_ratio_spec" in entry
                else "(CPU 不计带宽)"
            )
        )
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "peak_reference": peak,
        "unfused_chain_bytes_per_row_H1024": unfused_chain_bytes_per_row(HIDDEN_SIZE),
        "entries": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--dtype", default=None, choices=["fp32", "fp16"], help="默认：CPU 用 fp32，GPU 用 fp16")
    arguments = parser.parse_args()

    if arguments.dtype is None:
        arguments.dtype = "fp32" if arguments.backend == "cpu" else "fp16"
    dtype = torch.float16 if arguments.dtype == "fp16" else torch.float32

    device = resolve_device(arguments.backend)
    if device.type != "cpu":
        memory_api = getattr(torch, device.type)
        memory_api.empty_cache()
        if hasattr(memory_api, "reset_peak_memory_stats"):
            memory_api.reset_peak_memory_stats()

    peak = resolve_peak(arguments.backend, load_peak_measurement(arguments.backend))
    print(f"[peak/{arguments.backend}] 峰值口径 source={peak['peak_source']} "
          f"spec_bw={peak['peak_bandwidth_gb_s']}GB/s measured_bw={peak['measured_bandwidth_gb_s']}GB/s")

    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "dtype": "float16" if arguments.dtype == "fp16" else "float32",
        "rmsnorm_baseline": rmsnorm_chain(arguments, device, dtype, peak),
        "environment": device_environment_record(device),
    }
    output = Path("results") / f"benchmark-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
