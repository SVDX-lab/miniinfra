"""真实 Qwen3-0.6B Chunked Prefill 与 Decode 端到端对照实验。"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys

import torch

from engine import run_engine
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from scheduler import make_request_specs


def parse_args():
    parser = argparse.ArgumentParser(description="Eager/FlashAttention 端到端实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def make_sequence(length, vocab_size):
    values = torch.arange(length, dtype=torch.long)
    return ((values * 7927 + 31) % (vocab_size - 1) + 1).tolist()


def mean(values):
    return sum(values) / len(values)


def environment(device):
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = "unknown"
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_gib": properties.total_memory / 1024**3,
        "compute_capability": "%d.%d" % (
            properties.major, properties.minor
        ),
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "triton": __import__("triton").__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_driver": driver,
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
    }


def summarize(backend, runs):
    keys = [
        "makespan_ms", "busy_ms", "model_ms", "peak_memory_bytes",
        "output_token_throughput_per_second", "prefill_iterations",
        "decode_iterations", "max_prefill_iteration_ms",
    ]
    row = {"backend": backend}
    for key in keys:
        row[key + "_mean"] = mean([run["metrics"][key] for run in runs])
    row["service_ttft_ms_mean"] = mean([
        run["request_metrics"][0]["service_ttft_ms"] for run in runs
    ])
    row["samples"] = [
        {
            "makespan_ms": run["metrics"]["makespan_ms"],
            "model_ms": run["metrics"]["model_ms"],
            "service_ttft_ms": run["request_metrics"][0]["service_ttft_ms"],
            "peak_memory_bytes": run["metrics"]["peak_memory_bytes"],
        }
        for run in runs
    ]
    return row


def main():
    args = parse_args()
    if min(args.prompt_length, args.max_new_tokens, args.token_budget, args.repeats) < 1:
        raise ValueError("长度、预算和 repeats 必须为正")
    if args.warmup < 0:
        raise ValueError("warmup 不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("端到端实验需要 NVIDIA GPU")
    torch.manual_seed(9)
    device = torch.device("cuda")
    directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(directory, device, attention_backend="eager")
    specs = make_request_specs(
        [make_sequence(args.prompt_length, model.config.vocab_size)],
        args.max_new_tokens,
    )

    def execute(backend):
        model.set_attention_backend(backend)
        return run_engine(
            model, specs, 1, -1, device,
            token_budget=args.token_budget, block_size=16,
            stop_on_eos=False, prefix_cache_enabled=False,
            model_namespace=args.model + "@" + args.revision,
        )

    for backend in ("eager", "flash"):
        for _ in range(args.warmup):
            execute(backend)
            gc.collect()
            torch.cuda.empty_cache()
    collected = {"eager": [], "flash": []}
    for repeat in range(args.repeats):
        order = ("eager", "flash") if repeat % 2 == 0 else ("flash", "eager")
        for backend in order:
            collected[backend].append(execute(backend))
            gc.collect()
            torch.cuda.empty_cache()

    eager = summarize("eager", collected["eager"])
    flash = summarize("flash", collected["flash"])
    comparison = {
        "model_time_speedup": eager["model_ms_mean"] / flash["model_ms_mean"],
        "makespan_speedup": eager["makespan_ms_mean"] / flash["makespan_ms_mean"],
        "service_ttft_speedup": (
            eager["service_ttft_ms_mean"] / flash["service_ttft_ms_mean"]
        ),
        "peak_memory_reduction": 1 - (
            flash["peak_memory_bytes_mean"] / eager["peak_memory_bytes_mean"]
        ),
    }
    for row in (eager, flash):
        print(
            "%s model=%.2f ms makespan=%.2f ms TTFT=%.2f ms peak=%.1f MiB"
            % (
                row["backend"], row["model_ms_mean"], row["makespan_ms_mean"],
                row["service_ttft_ms_mean"], row["peak_memory_bytes_mean"] / 1024**2,
            )
        )
    print("model-time speedup: %.2fx" % comparison["model_time_speedup"])
    payload = {
        "environment": {
            **environment(device), "model": args.model, "revision": args.revision,
            "dtype": str(next(model.parameters()).dtype),
        },
        "workload": {
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "token_budget": args.token_budget,
            "block_size": 16,
            "batch": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "input": "synthetic exact-length token IDs",
            "decoding": "greedy, fixed output budget, EOS disabled",
        },
        "results": [eager, flash],
        "comparison": comparison,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

