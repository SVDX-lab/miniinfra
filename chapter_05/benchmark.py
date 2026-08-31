"""Dense 与 Paged KV Cache 的长短请求受控实验。"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from cache_engine import make_request_specs, run_dense_cache, run_paged_cache
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Paged KV Cache 受控实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--short-prompt-length", type=int, default=128)
    parser.add_argument("--long-prompt-length", type=int, default=2048)
    parser.add_argument(
        "--long-prompt-lengths", type=int, nargs="+", default=[128, 512, 2048, 3072]
    )
    parser.add_argument("--initial-short-requests", type=int, default=7)
    parser.add_argument("--followup-short-requests", type=int, default=9)
    parser.add_argument("--short-output", type=int, default=96)
    parser.add_argument("--followup-output", type=int, default=64)
    parser.add_argument("--long-output", type=int, default=8)
    parser.add_argument("--long-arrival-ms", type=float, default=200.0)
    parser.add_argument("--followup-arrival-ms", type=float, default=201.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--block-sizes", type=int, nargs="+", default=[8, 16, 32, 64]
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--modes", nargs="+", choices=("dense", "paged"),
        default=["dense", "paged"],
    )
    parser.add_argument(
        "--suite", choices=("stress", "ratios", "blocks", "all"), default="all"
    )
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def make_sequence(length, vocab_size, salt):
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def build_long_short_specs(args, model, long_length):
    lengths = (
        [args.short_prompt_length] * args.initial_short_requests
        + [long_length]
        + [args.short_prompt_length] * args.followup_short_requests
    )
    budgets = (
        [args.short_output] * args.initial_short_requests
        + [args.long_output]
        + [args.followup_output] * args.followup_short_requests
    )
    arrivals = (
        [0.0] * args.initial_short_requests
        + [args.long_arrival_ms]
        + [args.followup_arrival_ms] * args.followup_short_requests
    )
    sequences = [
        make_sequence(length, model.config.vocab_size, 37 * (index + 1))
        for index, length in enumerate(lengths)
    ]
    return make_request_specs(sequences, budgets, arrivals)


def mean(values):
    return sum(values) / len(values)


SUMMARY_KEYS = [
    "makespan_ms",
    "busy_ms",
    "model_ms",
    "cache_management_ms",
    "cache_management_fraction",
    "request_throughput_per_second",
    "output_token_throughput_per_second",
    "ttft_ms_p95",
    "itl_ms_p50",
    "itl_ms_p95",
    "itl_ms_max",
    "peak_live_request_cache_bytes",
    "peak_pool_cache_bytes",
    "cache_resize_moved_bytes",
    "visited_kv_token_slots",
]


def summarize(mode, runs):
    metrics = [run["metrics"] for run in runs]
    row = {"mode": mode}
    for key in SUMMARY_KEYS:
        row[key + "_mean"] = mean([item[key] for item in metrics])
    row["peak_memory_mib_max"] = max(
        item["peak_memory_bytes"] for item in metrics
    ) / 1024**2
    row["peak_live_request_cache_mib_max"] = max(
        item["peak_live_request_cache_bytes"] for item in metrics
    ) / 1024**2
    row["peak_pool_cache_mib_max"] = max(
        item["peak_pool_cache_bytes"] for item in metrics
    ) / 1024**2
    for key in (
        "block_size", "bytes_per_block", "peak_used_blocks", "pool_blocks",
        "block_allocation_count", "block_reuse_count", "block_release_count",
    ):
        if key in metrics[0]:
            row[key] = metrics[0][key]
    row["repeat_samples"] = [
        {
            "makespan_ms": item["makespan_ms"],
            "output_token_throughput_per_second": item[
                "output_token_throughput_per_second"
            ],
            "itl_ms_p95": item["itl_ms_p95"],
            "peak_memory_mib": item["peak_memory_bytes"] / 1024**2,
        }
        for item in metrics
    ]
    row["sample_event_trace"] = runs[0]["events"]
    return row


def clean_cuda(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_case(args, model, device, name, specs, block_size):
    all_functions = {
        "dense": lambda: run_dense_cache(
            model, specs, args.max_running_requests, -1, device,
            stop_on_eos=False,
        ),
        "paged": lambda: run_paged_cache(
            model, specs, args.max_running_requests, -1, device,
            block_size=block_size, stop_on_eos=False,
        ),
    }
    functions = {mode: all_functions[mode] for mode in args.modes}
    for _ in range(args.warmup):
        for function in functions.values():
            function()
            clean_cuda(device)

    collected = {mode: [] for mode in functions}
    for repeat in range(args.repeats):
        order = ["dense", "paged"] if repeat % 2 == 0 else ["paged", "dense"]
        for mode in order:
            collected[mode].append(functions[mode]())
            clean_cuda(device)
    rows = [summarize(mode, collected[mode]) for mode in functions]
    comparison = None
    by_mode = {row["mode"]: row for row in rows}
    if set(by_mode) == {"dense", "paged"}:
        dense = by_mode["dense"]
        paged = by_mode["paged"]
        comparison = {
            "paged_makespan_speedup": (
                dense["makespan_ms_mean"] / paged["makespan_ms_mean"]
            ),
            "paged_output_throughput_gain": (
                paged["output_token_throughput_per_second_mean"]
                / dense["output_token_throughput_per_second_mean"] - 1
            ),
            "kv_slot_read_reduction": 1 - (
                paged["visited_kv_token_slots_mean"]
                / dense["visited_kv_token_slots_mean"]
            ),
            "peak_pool_cache_reduction": 1 - (
                paged["peak_pool_cache_bytes_mean"]
                / dense["peak_pool_cache_bytes_mean"]
            ),
        }
        print(
            "%s block=%d speedup=%.3fx throughput=%+.1f%% "
            "KV-read=%+.1f%% pool=%+.1f%%"
            % (
                name,
                block_size,
                comparison["paged_makespan_speedup"],
                comparison["paged_output_throughput_gain"] * 100,
                -comparison["kv_slot_read_reduction"] * 100,
                -comparison["peak_pool_cache_reduction"] * 100,
            )
        )
    else:
        row = rows[0]
        print(
            "%s,%s,block=%d,makespan=%.3fms,tokens=%.3f/s,peak=%.1fMiB"
            % (
                name, row["mode"], block_size, row["makespan_ms_mean"],
                row["output_token_throughput_per_second_mean"],
                row["peak_memory_mib_max"],
            )
        )
    return {
        "case": name,
        "block_size": block_size,
        "max_running_requests": args.max_running_requests,
        "request_count": len(specs),
        "prompt_lengths": [len(spec.token_ids) for spec in specs],
        "output_budgets": [spec.max_new_tokens for spec in specs],
        "arrival_times_ms": [spec.arrival_ms for spec in specs],
        "results": rows,
        "comparison": comparison,
    }


def system_environment(device):
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        driver = "unknown"
    cpu = platform.processor()
    if not cpu or cpu.lower() in ("x86_64", "amd64"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
                cpu = next(
                    line.split(":", 1)[1].strip() for line in file
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            cpu = "unknown"
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_gib": properties.total_memory / 1024**3,
        "cpu": cpu,
        "system_memory_gib": (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        ),
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "gpu_driver": driver,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def main():
    args = parse_args()
    positive = [
        args.max_running_requests, args.short_prompt_length,
        args.long_prompt_length, args.initial_short_requests,
        args.short_output, args.followup_output, args.long_output,
        args.block_size, args.repeats,
    ] + args.long_prompt_lengths + args.block_sizes
    if any(value < 1 for value in positive) or args.warmup < 0:
        raise ValueError("长度、数量和 Block Size 必须为正，warmup 不能小于 0")
    if args.initial_short_requests >= args.max_running_requests:
        raise ValueError("初始短请求数必须小于最大运行请求数，为长请求保留槽位")
    if args.long_arrival_ms < 0 or args.followup_arrival_ms < args.long_arrival_ms:
        raise ValueError("到达时间配置无效")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(model_directory, device)
    results = []
    if args.suite in ("stress", "all"):
        specs = build_long_short_specs(args, model, args.long_prompt_length)
        results.append(run_case(
            args, model, device, "long_short_stress", specs, args.block_size
        ))
    if args.suite in ("ratios", "all"):
        for long_length in args.long_prompt_lengths:
            specs = build_long_short_specs(args, model, long_length)
            results.append(run_case(
                args, model, device, "length_ratio_%d" % long_length,
                specs, args.block_size,
            ))
    if args.suite in ("blocks", "all"):
        specs = build_long_short_specs(args, model, args.long_prompt_length)
        for block_size in args.block_sizes:
            results.append(run_case(
                args, model, device, "block_size_%d" % block_size,
                specs, block_size,
            ))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": {
                **system_environment(device),
                "model": args.model,
                "revision": args.revision,
                "dtype": str(next(model.parameters()).dtype),
                "warmup": args.warmup,
                "repeats": args.repeats,
                "decoding": "greedy, forced output budget, EOS disabled",
                "input": "synthetic exact-length token IDs",
                "tokenizer_included": False,
                "network_included": False,
                "arrival_time": "logical timeline; no real sleep",
                "cuda_allocator_config": os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF", "default"
                ),
            },
            "results": results,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)


if __name__ == "__main__":
    main()
