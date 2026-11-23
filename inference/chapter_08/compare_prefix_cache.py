"""使用真实 Qwen3 权重验证 Prefix Cache 正确性。"""

import argparse
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Prefix Cache 正确性实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--token-budget", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def make_sequence(length, vocab_size, salt):
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def build_sequences(vocab_size):
    aligned = make_sequence(32, vocab_size, 11)
    unaligned = make_sequence(34, vocab_size, 23)
    repeated_middle = make_sequence(16, vocab_size, 37)
    return [
        aligned + make_sequence(5, vocab_size, 41),
        aligned + make_sequence(5, vocab_size, 43),
        unaligned + make_sequence(7, vocab_size, 47),
        unaligned + make_sequence(7, vocab_size, 53),
        make_sequence(16, vocab_size, 59) + repeated_middle + [61],
        make_sequence(16, vocab_size, 67) + repeated_middle + [71],
    ]


def compact_requests(run):
    return [
        {
            "request_id": row["request_id"],
            "prompt_tokens": row["prompt_tokens"],
            "prefix_hit_tokens": row["prefix_hit_tokens"],
            "executed_prefill_tokens": row["executed_prefill_tokens"],
            "service_ttft_ms": row["service_ttft_ms"],
        }
        for row in run["request_metrics"]
    ]


def main():
    args = parse_args()
    if args.block_size < 1 or args.token_budget < 1:
        raise ValueError("Block Size 和 Token Budget 必须为正")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model_directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(model_directory, device, dtype=dtype)
    sequences = build_sequences(model.config.vocab_size)
    specs = make_request_specs(sequences, [4, 4, 4, 4, 2, 2])
    common = dict(
        model=model,
        request_specs=specs,
        max_running_requests=1,
        eos_token_id=-1,
        device=device,
        token_budget=args.token_budget,
        block_size=args.block_size,
        stop_on_eos=False,
        capture_logits=True,
        prefix_cache_capacity_blocks=64,
        model_namespace=args.model + "@" + args.revision,
    )
    disabled = run_engine(prefix_cache_enabled=False, **common)
    enabled = run_engine(prefix_cache_enabled=True, **common)
    tokens_match = disabled["new_token_ids"] == enabled["new_token_ids"]
    logit_errors = {
        request_id: float(torch.max(torch.abs(
            disabled["first_token_logits"][request_id]
            - enabled["first_token_logits"][request_id]
        )).item())
        for request_id in disabled["first_token_logits"]
    }
    hits = [row["prefix_hit_tokens"] for row in enabled["request_metrics"]]
    expected_hits = [0, 32, 0, 32, 0, 0]
    reference_counts_zero = (
        sum(enabled["final_cache_snapshot"]["block_ref_counts"]) == 0
    )
    result = {
        "tokens_match": tokens_match,
        "max_first_token_logit_error": max(logit_errors.values()),
        "per_request_first_token_logit_error": logit_errors,
        "expected_prefix_hit_tokens": expected_hits,
        "actual_prefix_hit_tokens": hits,
        "reference_counts_zero_after_requests": reference_counts_zero,
        "disabled_requests": compact_requests(disabled),
        "enabled_requests": compact_requests(enabled),
        "disabled_token_ids": disabled["new_token_ids"],
        "enabled_token_ids": enabled["new_token_ids"],
        "disabled_metrics": disabled["metrics"],
        "enabled_metrics": enabled["metrics"],
    }
    print("disabled 与 enabled 逐请求 Token 一致:", tokens_match)
    print("首 Token Logits 最大误差: %.8f" % result["max_first_token_logit_error"])
    print("Prefix 命中 Token:", hits)
    print("请求释放后引用计数归零:", reference_counts_zero)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": {
                "gpu": torch.cuda.get_device_name(device),
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "model": args.model,
                "revision": args.revision,
                "dtype": args.dtype,
                "token_budget": args.token_budget,
                "block_size": args.block_size,
                "decoding": "greedy, fixed output budget, EOS disabled",
                "input": "synthetic deterministic token IDs",
            },
            "result": result,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)
    limit = 1e-3 if args.dtype == "float32" else 1.0
    if (
        not tokens_match or hits != expected_hits or not reference_counts_zero
        or result["max_first_token_logit_error"] > limit
    ):
        raise SystemExit("Prefix Cache 正确性检查失败")


if __name__ == "__main__":
    main()
