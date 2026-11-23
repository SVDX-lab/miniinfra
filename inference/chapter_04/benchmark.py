"""固定批次与 Continuous Batching 的受控实验。"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from continuous_batching import (
    make_request_specs,
    run_continuous_batching,
    run_fixed_batching,
)
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous Batching 受控实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--short-output", type=int, default=8)
    parser.add_argument("--long-output", type=int, default=32)
    parser.add_argument("--arrival-interval-ms", type=float, default=150.0)
    parser.add_argument("--interference-arrival-ms", type=float, default=250.0)
    parser.add_argument("--interference-prompt-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--suite",
        choices=("slots", "control", "arrivals", "interference", "all"),
        default="all",
    )
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def make_sequence(length, vocab_size, salt):
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def build_specs(model, lengths, budgets, arrivals=None):
    sequences = [
        make_sequence(length, model.config.vocab_size, 23 * (index + 1))
        for index, length in enumerate(lengths)
    ]
    return make_request_specs(sequences, budgets, arrivals)


def mean(values):
    return sum(values) / len(values)


def summarize(mode, runs):
    metrics = [run["metrics"] for run in runs]
    keys = [
        "makespan_ms",
        "busy_ms",
        "model_ms",
        "cache_management_ms",
        "cache_management_fraction",
        "request_throughput_per_second",
        "output_token_throughput_per_second",
        "execution_slot_utilization",
        "running_capacity_utilization",
        "queue_ms_p50",
        "queue_ms_p95",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "end_to_end_ms_p50",
        "end_to_end_ms_p95",
        "itl_ms_p50",
        "itl_ms_p95",
        "itl_ms_max",
        "peak_live_request_cache_bytes",
    ]
    row = {"mode": mode}
    for key in keys:
        row[key + "_mean"] = mean([item[key] for item in metrics])
    row["peak_memory_mib_max"] = max(
        item["peak_memory_bytes"] for item in metrics
    ) / 1024**2
    row["peak_live_request_cache_mib_max"] = max(
        item["peak_live_request_cache_bytes"] for item in metrics
    ) / 1024**2
    row["repeat_samples"] = [
        {
            "makespan_ms": item["makespan_ms"],
            "request_throughput_per_second": item[
                "request_throughput_per_second"
            ],
            "output_token_throughput_per_second": item[
                "output_token_throughput_per_second"
            ],
            "ttft_ms_p95": item["ttft_ms_p95"],
            "itl_ms_p95": item["itl_ms_p95"],
        }
        for item in metrics
    ]
    row["sample_event_trace"] = runs[0]["events"]
    return row


def system_environment(device):
    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        driver = "unknown"
    cpu = platform.processor()
    if not cpu or cpu.lower() in ("x86_64", "amd64"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
                cpu = next(
                    line.split(":", 1)[1].strip()
                    for line in file
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            cpu = "unknown"
    memory_gib = (
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    )
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_gib": properties.total_memory / 1024**3,
        "cpu": cpu,
        "system_memory_gib": memory_gib,
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "gpu_driver": driver,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def run_case(args, model, device, name, specs, max_running=None):
    maximum = max_running or args.max_running_requests
    functions = {
        "fixed": run_fixed_batching,
        "continuous": run_continuous_batching,
    }
    for _ in range(args.warmup):
        for function in functions.values():
            function(model, specs, maximum, -1, device, stop_on_eos=False)

    collected = {name: [] for name in functions}
    for repeat in range(args.repeats):
        order = ["fixed", "continuous"]
        if repeat % 2:
            order.reverse()
        for mode in order:
            collected[mode].append(
                functions[mode](
                    model, specs, maximum, -1, device, stop_on_eos=False
                )
            )
    rows = [summarize(mode, collected[mode]) for mode in functions]
    for row in rows:
        print(
            "%s,%s,makespan=%.3fms,requests=%.3f/s,tokens=%.3f/s,"
            "exec_slots=%.2f%%,ttft_p95=%.3fms,itl_p95=%.3fms,cache=%.2f%%"
            % (
                name,
                row["mode"],
                row["makespan_ms_mean"],
                row["request_throughput_per_second_mean"],
                row["output_token_throughput_per_second_mean"],
                row["execution_slot_utilization_mean"] * 100,
                row["ttft_ms_p95_mean"],
                row["itl_ms_p95_mean"],
                row["cache_management_fraction_mean"] * 100,
            )
        )
    fixed = rows[0]
    continuous = rows[1]
    return {
        "case": name,
        "max_running_requests": maximum,
        "request_count": len(specs),
        "prompt_lengths": [len(spec.token_ids) for spec in specs],
        "output_budgets": [spec.max_new_tokens for spec in specs],
        "arrival_times_ms": [spec.arrival_ms for spec in specs],
        "results": rows,
        "continuous_makespan_speedup": (
            fixed["makespan_ms_mean"] / continuous["makespan_ms_mean"]
        ),
    }


def main():
    args = parse_args()
    positive_integers = [
        args.max_running_requests,
        args.request_count,
        args.prompt_length,
        args.short_output,
        args.long_output,
        args.interference_prompt_length,
        args.repeats,
    ]
    if any(value < 1 for value in positive_integers) or args.warmup < 0:
        raise ValueError("数量和长度必须为正，warmup 不能小于 0")
    if args.arrival_interval_ms < 0 or args.interference_arrival_ms < 0:
        raise ValueError("到达时间不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(model_directory, device)
    results = []
    mixed_budgets = [
        args.short_output if index % 2 == 0 else args.long_output
        for index in range(args.request_count)
    ]
    common_lengths = [args.prompt_length] * args.request_count

    if args.suite in ("slots", "all"):
        results.append(run_case(
            args,
            model,
            device,
            "mixed_output_slots",
            build_specs(model, common_lengths, mixed_budgets),
        ))
    if args.suite in ("control", "all"):
        control_budget = (args.short_output + args.long_output) // 2
        results.append(run_case(
            args,
            model,
            device,
            "uniform_output_control",
            build_specs(
                model,
                common_lengths,
                [control_budget] * args.request_count,
            ),
        ))
    if args.suite in ("arrivals", "all"):
        arrivals = [
            index * args.arrival_interval_ms for index in range(args.request_count)
        ]
        results.append(run_case(
            args,
            model,
            device,
            "staggered_arrivals",
            build_specs(model, common_lengths, mixed_budgets, arrivals),
        ))
    if args.suite in ("interference", "all"):
        lengths = [
            args.prompt_length,
            args.prompt_length,
            args.interference_prompt_length,
        ]
        budgets = [args.long_output, args.long_output, args.short_output]
        arrivals = [0.0, 0.0, args.interference_arrival_ms]
        results.append(run_case(
            args,
            model,
            device,
            "prefill_interference",
            build_specs(model, lengths, budgets, arrivals),
            max_running=max(3, args.max_running_requests),
        ))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        environment = {
            **system_environment(device),
            "model": args.model,
            "revision": args.revision,
            "dtype": str(next(model.parameters()).dtype),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "decoding": "greedy, forced output budget, EOS disabled",
            "input": "synthetic fixed token IDs",
            "tokenizer_included": False,
            "network_included": False,
            "arrival_time": "logical timeline; no real sleep",
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"environment": environment, "results": results},
                file, ensure_ascii=False, indent=2,
            )
        print("JSON 结果已写入:", output_path)


if __name__ == "__main__":
    main()
