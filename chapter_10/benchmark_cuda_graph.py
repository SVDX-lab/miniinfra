"""CUDA Graph Decode 主实验：受控比较 static_eager 与 cuda_graph。"""

import argparse
import statistics
import time

import torch

from experiment_utils import (
    bytes_to_mib,
    environment_snapshot,
    load_model,
    set_seed,
    synthetic_prompts,
    write_json,
)
from qwen3_model import DEFAULT_MODEL_ID
from static_decode import StaticDecodeRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("eager", "flash"), default="flash")
    parser.add_argument("--capacities", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--capture-warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="benchmark-cuda-graph-results.json")
    return parser.parse_args()


def rounded_bucket(prompt_length, decode_steps, block_size):
    required = prompt_length + decode_steps
    return ((required + block_size - 1) // block_size) * block_size


def percentile(values, percent):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percent / 100 + 0.999) - 1))
    return ordered[index]


def run_sample(runner, prompts, mode, decode_steps):
    runner.prepare_prompts(prompts)
    torch.cuda.synchronize(runner.device)
    torch.cuda.reset_peak_memory_stats(runner.device)
    allocated_before = torch.cuda.memory_allocated(runner.device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    host_start = time.perf_counter()
    for _ in range(decode_steps):
        runner.step(mode)
    host_submit_ms = (time.perf_counter() - host_start) * 1000
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    device_ms = start_event.elapsed_time(end_event)
    peak_bytes = torch.cuda.max_memory_allocated(runner.device)
    checksum = int(runner.output_tokens[:runner.active_count].sum().item())
    return {
        "mode": mode,
        "device_ms": device_ms,
        "wall_ms": wall_ms,
        "host_submit_ms": host_submit_ms,
        "device_tpot_ms": device_ms / decode_steps,
        "wall_tpot_ms": wall_ms / decode_steps,
        "host_submit_per_step_ms": host_submit_ms / decode_steps,
        "output_throughput_tokens_per_second": (
            runner.active_count * decode_steps / (device_ms / 1000)
        ),
        "peak_memory_bytes": peak_bytes,
        "incremental_peak_memory_bytes": max(0, peak_bytes - allocated_before),
        "output_checksum": checksum,
    }


def summarize(samples):
    fields = (
        "device_ms",
        "wall_ms",
        "host_submit_ms",
        "device_tpot_ms",
        "wall_tpot_ms",
        "host_submit_per_step_ms",
        "output_throughput_tokens_per_second",
        "peak_memory_bytes",
        "incremental_peak_memory_bytes",
    )
    result = {"sample_count": len(samples)}
    for field in fields:
        values = [sample[field] for sample in samples]
        result[field + "_mean"] = statistics.mean(values)
        result[field + "_p50"] = statistics.median(values)
        result[field + "_p95"] = percentile(values, 95)
    result["output_checksums"] = [sample["output_checksum"] for sample in samples]
    return result


def benchmark_config(model, args, capacity, prompt_length):
    context_bucket = rounded_bucket(
        prompt_length, args.decode_steps, args.block_size
    )
    runner = StaticDecodeRunner(
        model,
        capacity=capacity,
        context_bucket=context_bucket,
        block_size=args.block_size,
    )
    capture = runner.capture(args.capture_warmup_steps)
    prompts = synthetic_prompts(
        capacity,
        prompt_length,
        model.config.vocab_size,
        seed=args.seed + capacity * 10000 + prompt_length,
    )
    for mode in StaticDecodeRunner.MODES:
        for _ in range(args.warmup):
            run_sample(runner, prompts, mode, args.decode_steps)

    samples = {mode: [] for mode in StaticDecodeRunner.MODES}
    for repeat in range(args.repeats):
        order = StaticDecodeRunner.MODES
        if repeat % 2:
            order = tuple(reversed(order))
        for mode in order:
            samples[mode].append(
                run_sample(runner, prompts, mode, args.decode_steps)
            )
    summaries = {mode: summarize(values) for mode, values in samples.items()}
    eager_tpot = summaries["static_eager"]["device_tpot_ms_mean"]
    graph_tpot = summaries["cuda_graph"]["device_tpot_ms_mean"]
    saved_per_step = eager_tpot - graph_tpot
    break_even = (
        capture["capture_ms"] / saved_per_step if saved_per_step > 0 else None
    )
    result = {
        "capacity": capacity,
        "active_requests": capacity,
        "prompt_length": prompt_length,
        "decode_steps": args.decode_steps,
        "context_bucket": context_bucket,
        "block_size": args.block_size,
        "capture": capture,
        "runner": runner.snapshot(),
        "samples": samples,
        "summary": summaries,
        "comparison": {
            "device_tpot_speedup": eager_tpot / graph_tpot,
            "device_tpot_reduction_percent": (eager_tpot - graph_tpot) / eager_tpot * 100,
            "host_submit_reduction_percent": (
                summaries["static_eager"]["host_submit_per_step_ms_mean"]
                - summaries["cuda_graph"]["host_submit_per_step_ms_mean"]
            ) / summaries["static_eager"]["host_submit_per_step_ms_mean"] * 100,
            "estimated_capture_break_even_decode_steps": break_even,
        },
    }
    print(
        "capacity=%d prompt=%d bucket=%d eager=%.3f ms graph=%.3f ms speedup=%.2fx"
        % (
            capacity,
            prompt_length,
            context_bucket,
            eager_tpot,
            graph_tpot,
            eager_tpot / graph_tpot,
        )
    )
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("性能实验需要 NVIDIA CUDA GPU")
    if args.decode_steps < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("decode_steps/repeats 必须为正，warmup 不能为负")
    set_seed(args.seed)
    model_directory, model = load_model(
        args.model, "cuda", args.dtype, args.attention_backend
    )
    results = []
    for capacity in args.capacities:
        for prompt_length in args.prompt_lengths:
            results.append(
                benchmark_config(model, args, capacity, prompt_length)
            )
            torch.cuda.empty_cache()
    payload = {
        "environment": environment_snapshot(
            model_directory, args.dtype, args.attention_backend
        ),
        "method": {
            "baseline": "static_eager",
            "optimized": "cuda_graph",
            "controlled_variable": "CUDA Kernel 提交方式",
            "prefill_included": False,
            "tokenizer_included": False,
            "network_included": False,
            "capture_included_in_steady_state": False,
            "eos_enabled": False,
            "sampling": "greedy argmax captured in the decode step",
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "config": vars(args),
        "results": results,
    }
    write_json(args.output, payload)
    print("结果已写入", args.output)
    print("Pool/静态 Buffer 以 MiB 表示的辅助检查:")
    for result in results:
        print(
            "  capacity=%d prompt=%d pool=%.1f MiB buffers=%.1f MiB capture=%.1f ms"
            % (
                result["capacity"],
                result["prompt_length"],
                bytes_to_mib(result["runner"]["pool_bytes"]),
                bytes_to_mib(result["runner"]["static_buffer_bytes"]),
                result["capture"]["capture_ms"],
            )
        )


if __name__ == "__main__":
    main()
