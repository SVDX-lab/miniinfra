"""第 07 期完整 Prefill 与 Chunked Prefill 受控性能实验。"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from engine import percentile, run_engine
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from scheduler import make_request_specs


def parse_args():
    parser = argparse.ArgumentParser(description="Chunked Prefill 受控性能实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--initial-requests", type=int, default=2)
    parser.add_argument("--initial-prompt-length", type=int, default=128)
    parser.add_argument("--initial-output", type=int, default=48)
    parser.add_argument("--long-prompt-length", type=int, default=2048)
    parser.add_argument("--long-output", type=int, default=8)
    parser.add_argument("--long-arrival-ms", type=float, default=200.0)
    parser.add_argument(
        "--token-budgets", type=int, nargs="+", default=[256, 512, 1024]
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def make_sequence(length, vocab_size, salt):
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def build_specs(args, model):
    lengths = [args.initial_prompt_length] * args.initial_requests + [
        args.long_prompt_length
    ]
    outputs = [args.initial_output] * args.initial_requests + [args.long_output]
    arrivals = [0.0] * args.initial_requests + [args.long_arrival_ms]
    sequences = [
        make_sequence(length, model.config.vocab_size, 41 * (index + 1))
        for index, length in enumerate(lengths)
    ]
    return make_request_specs(sequences, outputs, arrivals)


def mean(values):
    return sum(values) / len(values)


def existing_itls(run, initial_requests):
    values = []
    initial_ids = {str(index) for index in range(initial_requests)}
    for request in run["request_metrics"]:
        if request["request_id"] in initial_ids:
            values.extend(
                right - left
                for left, right in zip(
                    request["token_times_ms"], request["token_times_ms"][1:]
                )
            )
    return values


SUMMARY_KEYS = [
    "makespan_ms",
    "output_token_throughput_per_second",
    "itl_ms_p50",
    "itl_ms_p95",
    "itl_ms_max",
    "prefill_iterations",
    "decode_iterations",
    "prefill_logical_tokens",
    "prefill_padded_tokens",
    "prefill_budget_utilization",
    "oversize_prefill_iterations",
    "hard_budget_violations",
    "prefill_chunk_count",
    "max_prefill_tokens_per_iteration",
    "max_prefill_iteration_ms",
    "peak_memory_bytes",
]


def summarize(mode, budget, runs, initial_requests):
    row = {"mode": mode, "token_budget": budget}
    for key in SUMMARY_KEYS:
        row[key + "_mean"] = mean([run["metrics"][key] for run in runs])
    existing_p95 = []
    existing_max = []
    long_ttft = []
    for run in runs:
        itls = existing_itls(run, initial_requests)
        existing_p95.append(percentile(itls, 95))
        existing_max.append(max(itls) if itls else 0.0)
        long_request = next(
            item for item in run["request_metrics"]
            if item["request_id"] == str(initial_requests)
        )
        long_ttft.append(long_request["ttft_ms"])
    row["existing_itl_ms_p95_mean"] = mean(existing_p95)
    row["existing_itl_ms_max_mean"] = mean(existing_max)
    row["long_request_ttft_ms_mean"] = mean(long_ttft)
    row["repeat_samples"] = [
        {
            "makespan_ms": run["metrics"]["makespan_ms"],
            "output_token_throughput_per_second": run["metrics"][
                "output_token_throughput_per_second"
            ],
            "existing_itl_ms_p95": existing_p95[index],
            "existing_itl_ms_max": existing_max[index],
            "long_request_ttft_ms": long_ttft[index],
            "max_prefill_iteration_ms": run["metrics"][
                "max_prefill_iteration_ms"
            ],
            "peak_memory_bytes": run["metrics"]["peak_memory_bytes"],
        }
        for index, run in enumerate(runs)
    ]
    row["sample_prefill_trace"] = [
        {
            "iteration": event["iteration"],
            "admitted": event["admitted"],
            "chunk_ranges": event["chunk_ranges"],
            "scheduled_tokens": event["scheduled_tokens"],
            "total_ms": event["total_ms"],
            "oversize_singleton": event["oversize_singleton"],
        }
        for event in runs[0]["events"] if event["phase"] == "prefill"
    ]
    return row


def clean_cuda(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def relative_change(baseline, candidate):
    return candidate / baseline - 1 if baseline else None


def reduction(baseline, candidate):
    return 1 - candidate / baseline if baseline else None


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
        args.max_running_requests,
        args.initial_prompt_length, args.initial_output,
        args.long_prompt_length, args.long_output,
        args.block_size, args.repeats,
    ] + args.token_budgets
    if any(value < 1 for value in positive) or args.warmup < 0:
        raise ValueError("长度、数量、预算必须为正，warmup 不能小于 0")
    if args.initial_requests < 0:
        raise ValueError("initial_requests 不能小于 0")
    if args.initial_requests >= args.max_running_requests:
        raise ValueError("initial_requests 必须小于 max_running_requests")
    if args.long_arrival_ms < 0:
        raise ValueError("long_arrival_ms 不能小于 0")
    if any(budget < args.max_running_requests for budget in args.token_budgets):
        raise ValueError("Token Budget 不能小于最大运行请求数")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(model_directory, device)
    specs = build_specs(args, model)
    configurations = [
        (mode, budget)
        for budget in args.token_budgets
        for mode in ("full", "chunked")
    ]

    for mode, budget in configurations:
        for _ in range(args.warmup):
            run_engine(
                model, specs, args.max_running_requests, -1, device,
                mode=mode, token_budget=budget,
                block_size=args.block_size, stop_on_eos=False,
            )
            clean_cuda(device)

    collected = {configuration: [] for configuration in configurations}
    for repeat in range(args.repeats):
        order = configurations if repeat % 2 == 0 else list(reversed(configurations))
        for mode, budget in order:
            collected[(mode, budget)].append(run_engine(
                model, specs, args.max_running_requests, -1, device,
                mode=mode, token_budget=budget,
                block_size=args.block_size, stop_on_eos=False,
            ))
            clean_cuda(device)

    rows = [
        summarize(mode, budget, collected[(mode, budget)], args.initial_requests)
        for mode, budget in configurations
    ]
    for budget in args.token_budgets:
        full = next(
            row for row in rows
            if row["mode"] == "full" and row["token_budget"] == budget
        )
        chunked = next(
            row for row in rows
            if row["mode"] == "chunked" and row["token_budget"] == budget
        )
        chunked["comparison_to_full"] = {
            "throughput_change": relative_change(
                full["output_token_throughput_per_second_mean"],
                chunked["output_token_throughput_per_second_mean"],
            ),
            "existing_itl_p95_reduction": reduction(
                full["existing_itl_ms_p95_mean"],
                chunked["existing_itl_ms_p95_mean"],
            ),
            "existing_itl_max_reduction": reduction(
                full["existing_itl_ms_max_mean"],
                chunked["existing_itl_ms_max_mean"],
            ),
            "long_ttft_change": relative_change(
                full["long_request_ttft_ms_mean"],
                chunked["long_request_ttft_ms_mean"],
            ),
            "peak_memory_reduction": reduction(
                full["peak_memory_bytes_mean"], chunked["peak_memory_bytes_mean"],
            ),
        }
    for row in rows:
        print(
            "%s budget=%d throughput=%.2f token/s existing-ITL-p95=%.2f ms "
            "existing-ITL-max=%.2f ms long-TTFT=%.2f ms max-prefill=%.2f ms "
            "oversize=%.1f violations=%.1f"
            % (
                row["mode"], row["token_budget"],
                row["output_token_throughput_per_second_mean"],
                row["existing_itl_ms_p95_mean"],
                row["existing_itl_ms_max_mean"],
                row["long_request_ttft_ms_mean"],
                row["max_prefill_iteration_ms_mean"],
                row["oversize_prefill_iterations_mean"],
                row["hard_budget_violations_mean"],
            )
        )

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
            "workload": {
                "max_running_requests": args.max_running_requests,
                "initial_requests": args.initial_requests,
                "initial_prompt_length": args.initial_prompt_length,
                "initial_output": args.initial_output,
                "long_prompt_length": args.long_prompt_length,
                "long_output": args.long_output,
                "long_arrival_ms": args.long_arrival_ms,
                "token_budgets": args.token_budgets,
                "block_size": args.block_size,
            },
            "results": rows,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)


if __name__ == "__main__":
    main()
