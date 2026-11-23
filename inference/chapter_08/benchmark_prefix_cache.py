"""共享 2048 Token 长前缀的 Prefix Cache 受控主实验。"""

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
    parser = argparse.ArgumentParser(description="Prefix Cache 共享长前缀主实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--common-prefix-length", type=int, default=2048)
    parser.add_argument("--private-suffix-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--cache-capacity-blocks", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def make_sequence(length, vocab_size, salt):
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def build_specs(args, model):
    common = make_sequence(args.common_prefix_length, model.config.vocab_size, 101)
    sequences = [
        common + make_sequence(
            args.private_suffix_length, model.config.vocab_size, 1009 + index * 17
        )
        for index in range(args.requests)
    ]
    return make_request_specs(sequences, args.max_new_tokens)


def clean_cuda(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def mean(values):
    return sum(values) / len(values)


def summarize(mode, runs):
    cold = [run["request_metrics"][0]["service_ttft_ms"] for run in runs]
    warm_samples = [
        value
        for run in runs
        for value in [row["service_ttft_ms"] for row in run["request_metrics"][1:]]
    ]
    keys = [
        "makespan_ms", "busy_ms", "output_token_throughput_per_second",
        "executed_prompt_tokens", "prefix_hit_tokens", "prefix_lookup_ms",
        "prefill_iterations", "model_ms", "cache_management_ms",
        "peak_memory_bytes", "peak_used_blocks",
    ]
    row = {"mode": mode}
    for key in keys:
        row[key + "_mean"] = mean([run["metrics"][key] for run in runs])
    row["cold_service_ttft_ms_mean"] = mean(cold)
    row["warm_service_ttft_ms_p50"] = percentile(warm_samples, 50)
    row["warm_service_ttft_ms_p95"] = percentile(warm_samples, 95)
    row["repeat_samples"] = [
        {
            "makespan_ms": run["metrics"]["makespan_ms"],
            "cold_service_ttft_ms": run["request_metrics"][0]["service_ttft_ms"],
            "warm_service_ttft_ms": [
                item["service_ttft_ms"] for item in run["request_metrics"][1:]
            ],
            "executed_prompt_tokens": run["metrics"]["executed_prompt_tokens"],
            "prefix_hit_tokens": run["metrics"]["prefix_hit_tokens"],
            "peak_memory_bytes": run["metrics"]["peak_memory_bytes"],
        }
        for run in runs
    ]
    row["sample_requests"] = [
        {
            "request_id": item["request_id"],
            "prefix_hit_tokens": item["prefix_hit_tokens"],
            "executed_prefill_tokens": item["executed_prefill_tokens"],
            "service_ttft_ms": item["service_ttft_ms"],
        }
        for item in runs[0]["request_metrics"]
    ]
    return row


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
        args.requests, args.common_prefix_length, args.private_suffix_length,
        args.max_new_tokens, args.token_budget, args.block_size,
        args.cache_capacity_blocks, args.repeats,
    ]
    if any(value < 1 for value in positive) or args.warmup < 0:
        raise ValueError("长度、数量、预算必须为正，warmup 不能小于 0")
    required_shared_blocks = args.common_prefix_length // args.block_size
    if args.cache_capacity_blocks < required_shared_blocks:
        raise ValueError("缓存容量不足以容纳完整公共前缀")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(model_directory, device)
    specs = build_specs(args, model)
    namespace = args.model + "@" + args.revision

    def execute(enabled):
        return run_engine(
            model, specs, 1, -1, device,
            token_budget=args.token_budget,
            block_size=args.block_size,
            stop_on_eos=False,
            prefix_cache_enabled=enabled,
            prefix_cache_capacity_blocks=args.cache_capacity_blocks,
            model_namespace=namespace,
        )

    for mode in (False, True):
        for _ in range(args.warmup):
            execute(mode)
            clean_cuda(device)
    collected = {False: [], True: []}
    for repeat in range(args.repeats):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for mode in order:
            collected[mode].append(execute(mode))
            clean_cuda(device)
    disabled = summarize("disabled", collected[False])
    enabled = summarize("enabled", collected[True])
    enabled["comparison_to_disabled"] = {
        "warm_service_ttft_reduction": reduction(
            disabled["warm_service_ttft_ms_p50"],
            enabled["warm_service_ttft_ms_p50"],
        ),
        "makespan_reduction": reduction(
            disabled["makespan_ms_mean"], enabled["makespan_ms_mean"]
        ),
        "executed_prompt_token_reduction": reduction(
            disabled["executed_prompt_tokens_mean"],
            enabled["executed_prompt_tokens_mean"],
        ),
        "throughput_change": (
            enabled["output_token_throughput_per_second_mean"]
            / disabled["output_token_throughput_per_second_mean"] - 1
        ),
    }
    rows = [disabled, enabled]
    for row in rows:
        print(
            "%s cold-service-TTFT=%.2f ms warm-service-TTFT-p50=%.2f ms "
            "executed-prefill=%.0f hit=%.0f makespan=%.2f ms throughput=%.2f token/s"
            % (
                row["mode"], row["cold_service_ttft_ms_mean"],
                row["warm_service_ttft_ms_p50"],
                row["executed_prompt_tokens_mean"], row["prefix_hit_tokens_mean"],
                row["makespan_ms_mean"],
                row["output_token_throughput_per_second_mean"],
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
                "decoding": "greedy, fixed output budget, EOS disabled",
                "input": "synthetic exact-length token IDs",
                "tokenizer_included": False,
                "network_included": False,
                "cuda_allocator_config": os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF", "default"
                ),
            },
            "workload": {
                "requests": args.requests,
                "max_running_requests": 1,
                "common_prefix_length": args.common_prefix_length,
                "private_suffix_length": args.private_suffix_length,
                "max_new_tokens": args.max_new_tokens,
                "token_budget": args.token_budget,
                "block_size": args.block_size,
                "cache_capacity_blocks": args.cache_capacity_blocks,
            },
            "metric_note": (
                "service_ttft_ms 从引擎接纳到首 Token，排除 FCFS 队列等待；"
                "makespan 包含全部顺序请求。"
            ),
            "results": rows,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)


if __name__ == "__main__":
    main()
