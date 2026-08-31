"""在固定 12GB GPU 上扫描 dense 与 Paged KV Cache 的并发容量和吞吐。"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from benchmark import build_long_short_specs, mean, system_environment
from cache_engine import run_dense_cache, run_paged_cache
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Paged KV Cache 容量实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[2, 4, 8, 12, 16])
    parser.add_argument("--total-requests", type=int, default=32)
    parser.add_argument("--short-prompt-length", type=int, default=128)
    parser.add_argument("--long-prompt-length", type=int, default=4096)
    parser.add_argument("--short-output", type=int, default=64)
    parser.add_argument("--followup-output", type=int, default=64)
    parser.add_argument("--long-output", type=int, default=8)
    parser.add_argument("--long-arrival-ms", type=float, default=200.0)
    parser.add_argument("--followup-arrival-ms", type=float, default=201.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def clear_after_run(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def main():
    args = parse_args()
    if (
        any(value < 1 for value in args.concurrencies)
        or args.total_requests < 2
        or args.repeats < 1
    ):
        raise ValueError("并发、请求数和重复次数必须为正")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(model_directory, device)
    rows = []

    for concurrency in args.concurrencies:
        if concurrency > args.total_requests:
            raise ValueError("并发数不能超过总请求数")
        args.max_running_requests = concurrency
        args.initial_short_requests = concurrency - 1
        args.followup_short_requests = args.total_requests - concurrency
        specs = build_long_short_specs(args, model, args.long_prompt_length)
        functions = {
            "dense": lambda: run_dense_cache(
                model, specs, concurrency, -1, device, stop_on_eos=False
            ),
            "paged": lambda: run_paged_cache(
                model, specs, concurrency, -1, device,
                block_size=args.block_size, stop_on_eos=False,
            ),
        }
        for mode, function in functions.items():
            samples = []
            error = None
            for _ in range(args.repeats):
                try:
                    samples.append(function()["metrics"])
                except torch.OutOfMemoryError as exception:
                    error = "CUDA OOM: " + str(exception).split(". ")[0]
                    break
                finally:
                    clear_after_run(device)
            if error:
                row = {
                    "mode": mode,
                    "concurrency": concurrency,
                    "status": "oom",
                    "error": error,
                }
                print("%s concurrency=%d OOM" % (mode, concurrency))
            else:
                row = {
                    "mode": mode,
                    "concurrency": concurrency,
                    "status": "success",
                    "makespan_ms_mean": mean([item["makespan_ms"] for item in samples]),
                    "output_token_throughput_per_second_mean": mean([
                        item["output_token_throughput_per_second"] for item in samples
                    ]),
                    "itl_ms_p95_mean": mean([item["itl_ms_p95"] for item in samples]),
                    "peak_memory_mib_max": max(
                        item["peak_memory_bytes"] for item in samples
                    ) / 1024**2,
                    "peak_pool_cache_mib_max": max(
                        item["peak_pool_cache_bytes"] for item in samples
                    ) / 1024**2,
                    "repeat_samples": [
                        {
                            "makespan_ms": item["makespan_ms"],
                            "output_token_throughput_per_second": item[
                                "output_token_throughput_per_second"
                            ],
                            "itl_ms_p95": item["itl_ms_p95"],
                        }
                        for item in samples
                    ],
                }
                print(
                    "%s concurrency=%d tokens=%.2f/s peak=%.1fMiB"
                    % (
                        mode, concurrency,
                        row["output_token_throughput_per_second_mean"],
                        row["peak_memory_mib_max"],
                    )
                )
            rows.append(row)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "environment": {
                        **system_environment(device),
                        "model": args.model,
                        "revision": args.revision,
                        "dtype": str(next(model.parameters()).dtype),
                        "repeats": args.repeats,
                        "block_size": args.block_size,
                        "total_requests": args.total_requests,
                        "short_prompt_length": args.short_prompt_length,
                        "long_prompt_length": args.long_prompt_length,
                        "decoding": "greedy, forced output budget, EOS disabled",
                        "cuda_allocator_config": os.environ.get(
                            "PYTORCH_CUDA_ALLOC_CONF", "default"
                        ),
                    },
                    "results": rows,
                },
                file, ensure_ascii=False, indent=2,
            )
        print("JSON 结果已写入:", output_path)


if __name__ == "__main__":
    main()
