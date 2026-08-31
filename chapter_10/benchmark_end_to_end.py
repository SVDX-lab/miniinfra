"""单请求完整 Prefill + Decode 实验，补充 TTFT 与端到端指标。"""

import argparse
import statistics
import time

import torch

from experiment_utils import (
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
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("eager", "flash"), default="flash")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="benchmark-end-to-end-results.json")
    return parser.parse_args()


def run_sample(runner, prompt, mode, max_new_tokens):
    decode_steps = max_new_tokens - 1
    torch.cuda.synchronize(runner.device)
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    first = runner.prepare_prompts([prompt])
    torch.cuda.synchronize(runner.device)
    ttft_ms = (time.perf_counter() - prefill_start) * 1000

    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)
    decode_start.record()
    for _ in range(decode_steps):
        runner.step(mode)
    decode_end.record()
    decode_end.synchronize()
    end_to_end_ms = (time.perf_counter() - total_start) * 1000
    decode_ms = decode_start.elapsed_time(decode_end)
    return {
        "mode": mode,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "tpot_ms": decode_ms / decode_steps if decode_steps else 0.0,
        "end_to_end_ms": end_to_end_ms,
        "output_throughput_tokens_per_second": max_new_tokens / (end_to_end_ms / 1000),
        "first_token": first[0],
        "last_token": int(runner.output_tokens[0].item()),
    }


def summarize(samples):
    result = {"sample_count": len(samples)}
    for field in (
        "ttft_ms", "decode_ms", "tpot_ms", "end_to_end_ms",
        "output_throughput_tokens_per_second",
    ):
        values = [sample[field] for sample in samples]
        result[field + "_mean"] = statistics.mean(values)
        result[field + "_p50"] = statistics.median(values)
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("端到端实验需要 NVIDIA CUDA GPU")
    if args.max_new_tokens < 2:
        raise ValueError("端到端对照至少需要生成 2 个 Token")
    set_seed(args.seed)
    model_directory, model = load_model(
        args.model, "cuda", args.dtype, args.attention_backend
    )
    required = args.prompt_length + args.max_new_tokens - 1
    context_bucket = (
        (required + args.block_size - 1) // args.block_size * args.block_size
    )
    runner = StaticDecodeRunner(
        model, 1, context_bucket, args.block_size
    )
    capture = runner.capture()
    prompt = synthetic_prompts(
        1, args.prompt_length, model.config.vocab_size, args.seed
    )[0]
    for mode in StaticDecodeRunner.MODES:
        for _ in range(args.warmup):
            run_sample(runner, prompt, mode, args.max_new_tokens)
    samples = {mode: [] for mode in StaticDecodeRunner.MODES}
    for repeat in range(args.repeats):
        order = StaticDecodeRunner.MODES
        if repeat % 2:
            order = tuple(reversed(order))
        for mode in order:
            samples[mode].append(
                run_sample(runner, prompt, mode, args.max_new_tokens)
            )
    summary = {mode: summarize(values) for mode, values in samples.items()}
    eager = summary["static_eager"]
    graph = summary["cuda_graph"]
    payload = {
        "environment": environment_snapshot(
            model_directory, args.dtype, args.attention_backend
        ),
        "config": {**vars(args), "context_bucket": context_bucket},
        "method": {
            "request_count": 1,
            "capture_included": False,
            "tokenizer_included": False,
            "model_loading_included": False,
            "network_included": False,
            "eos_enabled": False,
            "sampling": "greedy argmax",
            "ttft_definition": "从开始 Prefill 到首 Token argmax 完成，包含 KV Block 写入",
        },
        "capture": capture,
        "samples": samples,
        "summary": summary,
        "comparison": {
            "ttft_ratio": eager["ttft_ms_mean"] / graph["ttft_ms_mean"],
            "decode_tpot_speedup": eager["tpot_ms_mean"] / graph["tpot_ms_mean"],
            "end_to_end_speedup": eager["end_to_end_ms_mean"] / graph["end_to_end_ms_mean"],
        },
    }
    write_json(args.output, payload)
    print(
        "static_eager: TTFT %.2f ms, TPOT %.2f ms, E2E %.2f ms"
        % (eager["ttft_ms_mean"], eager["tpot_ms_mean"], eager["end_to_end_ms_mean"])
    )
    print(
        "cuda_graph:  TTFT %.2f ms, TPOT %.2f ms, E2E %.2f ms"
        % (graph["ttft_ms_mean"], graph["tpot_ms_mean"], graph["end_to_end_ms_mean"])
    )
    print("结果已写入", args.output)


if __name__ == "__main__":
    main()
