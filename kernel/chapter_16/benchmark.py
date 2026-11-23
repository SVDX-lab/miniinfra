#!/usr/bin/env python3
"""第 16 期性能基准：5 算子 × 变体 × 9 Shape，统一计时口径与达成率。

口径（沿用第 01 期）：
- 计时：OperatorTimer（Warm-up + 事件对/同步边界，P50/P95/min/mean）；
- 有效带宽 = 理论字节数 / P50；达成率 = 有效带宽 / 平台峰值带宽
  （cuda 1008 GB/s、musa 768 GB/s，来源大纲 §3.3，与第 01 期同源）；
- launch-bound 证据：enqueue_time（仅发射不同步的平均 CPU 耗时）与 P50 对比；
- roofline：计算强度 = 模型 FLOP / 理论字节（见 ops_harness.theoretical_*）；
- 自定义 Kernel 与 torch eager 对照组同口径计时（框架调度计入 eager）。

用法：python benchmark.py --backend {cpu,cuda,musa} [--ops softmax,scan]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from experiment_utils import (
    OperatorTimer,
    backend_sync,
    device_environment_record,
    enqueue_time_ms,
    resolve_device,
    write_results_json,
)
from ops_harness import (
    CPU_MODEL_VARIANT,
    CPU_VARIANTS,
    load_binding,
    resolve_model_variant,
    GPU_VARIANTS,
    PEAK_BANDWIDTH_GB_S,
    SHAPES,
    launch_count,
    load_fixtures,
    run_cpu_variant,
    run_gpu_variant,
    run_torch_eager,
    theoretical_bytes,
    theoretical_flops,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "implementations" / "cuda"))
sys.path.insert(0, str(ROOT / "implementations" / "musa"))


def pick_repeats(backend: str, elements: int) -> tuple[int, int]:
    """按规模选 warmup/repeats：小算子多采样（launch-bound 区需要样本量）。"""
    if backend == "cpu":
        if elements <= 131072:
            return 5, 20
        if elements <= 1048576:
            return 3, 10
        return 2, 5
    if elements <= 131072:
        return 20, 300
    if elements <= 1048576:
        return 20, 200
    return 10, 100


def make_entry(backend: str, op: str, variant: str, rows: int, cols: int, timing: dict,
               device: torch.device, model_variant: str | None = None) -> dict:
    # model_name 只用于理论字节/FLOP 查表；结果 JSON 保留调用方传入的变体名
    model_name = resolve_model_variant(op, model_variant or variant)
    bytes_model = theoretical_bytes(rows, cols, model_name, op=op)
    flops_model = theoretical_flops(rows, cols, model_name, op=op)
    entry = {
        "op": op,
        "variant": variant,
        "rows": rows,
        "cols": cols,
        "elements": rows * cols,
        "dtype": "float32",
        "launch_count_per_call": launch_count(backend, op, variant),
        "theoretical_bytes": bytes_model,
        "theoretical_flops": flops_model,
        "compute_intensity_flop_per_byte": round(flops_model / bytes_model, 4) if bytes_model else None,
    }
    entry.update(timing)
    if PEAK_BANDWIDTH_GB_S[backend] is not None:
        effective = bytes_model / (timing["p50_ms"] / 1000.0)
        entry["effective_bandwidth_gb_s"] = round(effective / 1e9, 2)
        entry["peak_bandwidth_gb_s_spec"] = PEAK_BANDWIDTH_GB_S[backend]
        entry["achievement_ratio"] = round(effective / (PEAK_BANDWIDTH_GB_S[backend] * 1e9), 4)
    if device.type != "cpu" and "p50_ms" in timing:
        theoretical_latency_lower_bound_us = bytes_model / (PEAK_BANDWIDTH_GB_S[backend] * 1e9) * 1e6
        entry["theoretical_latency_lower_bound_us"] = round(theoretical_latency_lower_bound_us, 2)
        entry["latency_achievement_ratio"] = round(
            theoretical_latency_lower_bound_us / (timing["p50_ms"] * 1000.0), 4
        )
    return entry


def run_backend(backend: str, ops: list[str], fixtures: dict, manifest: dict) -> dict:
    device = resolve_device(backend)
    results: dict = {
        "schema_version": "kernel-chapter17.benchmark.v1",
        "backend": backend,
        "timing_note": "GPU：事件对逐次记录、末尾一次同步（ch01 口径）；CPU：perf_counter",
        "peak_bandwidth_source": "大纲 §3.3（cuda 1008 / musa 768 GB/s），与第 01 期同源",
        "ops": {},
    }

    binding = None
    if backend != "cpu":
        lib_name = "libops_cuda.so" if backend == "cuda" else "libops_musa.so"
        binding = load_binding(backend, lib_name)

    for op in ops:
        entries: list[dict] = []
        for rows, cols in SHAPES:
            elements = rows * cols
            warmup, repeats = pick_repeats(backend, elements)
            key = f"{rows}x{cols}"

            if backend == "cpu":
                operation = lambda: run_cpu_variant(op, fixtures, rows, cols)  # noqa: E731
                timer = OperatorTimer(device, warmup=warmup, repeats=repeats)
                for variant in CPU_VARIANTS[op]:
                    timing = timer.measure(operation)
                    entries.append(make_entry(backend, op, variant, rows, cols, timing, device,
                                              model_variant=CPU_MODEL_VARIANT[op]))
                # torch CPU 组合表达式对照
                x_t = torch.from_numpy(fixtures[f"x_{key}"])
                up_t = torch.from_numpy(fixtures[f"up_{key}"])
                torch_op = {
                    "softmax": lambda: torch.softmax(x_t, dim=-1),
                    "reduce": lambda: torch.cat([x_t.sum(dim=1), x_t.max(dim=1).values]),
                    "transpose": lambda: x_t.t().contiguous(),
                    "scan": lambda: torch.cumsum(x_t, dim=1),
                    "silu_gate": lambda: torch.nn.functional.silu(x_t) * up_t,
                }[op]
                timing = OperatorTimer(device, warmup=warmup, repeats=repeats).measure(torch_op)
                eager_variant = {
                    "softmax": "torch_softmax", "reduce": "torch_sum_max",
                    "transpose": "torch_t_contiguous", "scan": "torch_cumsum",
                    "silu_gate": "torch_silu_mul",
                }[op]
                entries.append(make_entry(backend, op, eager_variant, rows, cols, timing, device))
                best = min(
                    (entry["p50_ms"] for entry in entries if entry["rows"] == rows and entry["cols"] == cols)
                )
                print(f"[{backend}] {op} {rows}x{cols}: best p50={best:.3f}ms")
                continue

            buffers = {
                "x": torch.empty(rows, cols, dtype=torch.float32, device=device),
                "up": torch.empty(rows, cols, dtype=torch.float32, device=device),
                "y_softmax": torch.empty(rows, cols, dtype=torch.float32, device=device),
                "y_scan": torch.empty(rows, cols, dtype=torch.float32, device=device),
                "y_transpose": torch.empty(cols, rows, dtype=torch.float32, device=device),
                "sum_out": torch.empty(rows, dtype=torch.float32, device=device),
                "max_out": torch.empty(rows, dtype=torch.float32, device=device),
                "tmp": torch.empty(rows, cols, dtype=torch.float32, device=device),
                "y_silu": torch.empty(rows, cols, dtype=torch.float32, device=device),
            }
            buffers["x"].copy_(torch.from_numpy(fixtures[f"x_{key}"]))
            buffers["up"].copy_(torch.from_numpy(fixtures[f"up_{key}"]))
            backend_sync(device)

            timer = OperatorTimer(device, warmup=warmup, repeats=repeats)
            for variant in GPU_VARIANTS[op]:
                operation = lambda variant=variant: run_gpu_variant(  # noqa: E731
                    binding, op, variant, buffers, rows, cols
                )
                timing = timer.measure(operation)
                entry = make_entry(backend, op, variant, rows, cols, timing, device)
                entry["enqueue_avg_ms"] = enqueue_time_ms(device, warmup=5, repeats=repeats,
                                                          operation=operation)
                entries.append(entry)
                bandwidth = entry.get("effective_bandwidth_gb_s")
                achievement = entry.get("achievement_ratio")
                print(
                    f"[{backend}] {op}/{variant} {rows}x{cols}: p50={timing['p50_ms']:.4f}ms "
                    + (f"eff={bandwidth}GB/s 达成率={achievement:.2%}" if bandwidth else "")
                )

            # torch eager 对照组
            eager_variant = {
                "softmax": "torch_softmax", "reduce": "torch_sum_max",
                "transpose": "torch_t_contiguous", "scan": "torch_cumsum",
                "silu_gate": "torch_silu_mul",
            }[op]
            operation = lambda: run_torch_eager(op, buffers)  # noqa: E731
            timing = timer.measure(operation)
            entry = make_entry(backend, op, eager_variant, rows, cols, timing, device)
            entry["enqueue_avg_ms"] = enqueue_time_ms(device, warmup=5, repeats=repeats,
                                                      operation=operation)
            entries.append(entry)
            print(f"[{backend}] {op}/{eager_variant} {rows}x{cols}: p50={timing['p50_ms']:.4f}ms")

            del buffers
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "musa":
                torch.musa.empty_cache()

        results["ops"][op] = {"entries": entries}

    results["environment"] = device_environment_record(device)
    results["fixtures_npz_sha256"] = manifest.get("npz_sha256")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--ops", default="softmax,reduce,transpose,scan,silu_gate")
    arguments = parser.parse_args()
    ops = [name.strip() for name in arguments.ops.split(",") if name.strip()]

    fixtures, manifest = load_fixtures()
    results = run_backend(arguments.backend, ops, fixtures, manifest)

    output = ROOT / "results" / f"benchmark-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(resolve_device(arguments.backend))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
