"""真实 Qwen3-0.6B 的 static eager / CUDA Graph 正确性比较。"""

import argparse

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
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("eager", "flash"), default="flash")
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--context-bucket", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="compare-cuda-graph-results.json")
    return parser.parse_args()


def run_steps(runner, prompts, mode, steps):
    first = runner.prepare_prompts(prompts)
    sequences = [[token] for token in first]
    logits = []
    for _ in range(steps):
        runner.step(mode)
        torch.cuda.synchronize(runner.device)
        step_tokens = runner.output_tokens[:len(prompts)].cpu().tolist()
        logits.append(runner.last_logits[:len(prompts)].float().cpu().clone())
        for sequence, token in zip(sequences, step_tokens):
            sequence.append(int(token))
    cache_slices = []
    final_length = len(prompts[0]) + steps
    for slot in range(len(prompts)):
        layers = runner.cache.dense_slot_cache(slot, final_length)
        cache_slices.append([
            (
                key[:, :, len(prompts[slot]):final_length, :].float().cpu(),
                value[:, :, len(prompts[slot]):final_length, :].float().cpu(),
            )
            for key, value in layers
        ])
    return sequences, logits, cache_slices


def maximum_cache_error(left, right):
    maximum = 0.0
    for left_slot, right_slot in zip(left, right):
        for (left_key, left_value), (right_key, right_value) in zip(
            left_slot, right_slot
        ):
            maximum = max(
                maximum,
                float((left_key - right_key).abs().max()),
                float((left_value - right_value).abs().max()),
            )
    return maximum


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("正确性实验需要 NVIDIA CUDA GPU")
    if args.requests > args.capacity:
        raise ValueError("requests 不能超过 capacity")
    if args.prompt_length + args.decode_steps > args.context_bucket:
        raise ValueError("Context Bucket 不能容纳 Prompt 和 Decode")
    set_seed(args.seed)
    model_directory, model = load_model(
        args.model, "cuda", args.dtype, args.attention_backend
    )
    runner = StaticDecodeRunner(
        model,
        capacity=args.capacity,
        context_bucket=args.context_bucket,
        block_size=args.block_size,
    )
    capture = runner.capture()
    prompts = synthetic_prompts(
        args.requests, args.prompt_length, model.config.vocab_size, args.seed
    )

    eager_tokens, eager_logits, eager_cache = run_steps(
        runner, prompts, "static_eager", args.decode_steps
    )
    graph_tokens, graph_logits, graph_cache = run_steps(
        runner, prompts, "cuda_graph", args.decode_steps
    )
    logit_error = max(
        float((left - right).abs().max())
        for left, right in zip(eager_logits, graph_logits)
    )
    cache_error = maximum_cache_error(eager_cache, graph_cache)
    inactive_output = runner.output_tokens[args.requests:].cpu().tolist()
    result = {
        "environment": environment_snapshot(
            model_directory, args.dtype, args.attention_backend
        ),
        "config": vars(args),
        "capture": capture,
        "static_runner": runner.snapshot(),
        "checks": {
            "token_exact_match": eager_tokens == graph_tokens,
            "maximum_logit_absolute_error": logit_error,
            "maximum_decode_kv_absolute_error": cache_error,
            "inactive_slots_are_pad": all(token == 0 for token in inactive_output),
            "eager_tokens": eager_tokens,
            "cuda_graph_tokens": graph_tokens,
        },
    }
    write_json(args.output, result)
    print("Token 完全一致:", result["checks"]["token_exact_match"])
    print("Logits 最大绝对误差: %.8f" % logit_error)
    print("Decode KV 最大绝对误差: %.8f" % cache_error)
    print("非活跃 Slot 保持 Pad:", result["checks"]["inactive_slots_are_pad"])
    print("结果已写入", args.output)
    if not all((
        result["checks"]["token_exact_match"],
        result["checks"]["inactive_slots_are_pad"],
        logit_error == 0.0,
        cache_error == 0.0,
    )):
        raise SystemExit("正确性比较未通过")


if __name__ == "__main__":
    main()
