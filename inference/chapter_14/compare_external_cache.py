"""真实 Qwen3 权重正确性：recompute、cold store 与 warm load 对照。"""

import argparse

import torch

from engine import run_request
from experiment_utils import (
    add_model_arguments,
    cache_service,
    environment_record,
    load_model,
    parse_dtype,
    write_json,
)


def synthetic_tokens(length, vocab_size):
    return [1000 + ((index * 7919 + 17) % (vocab_size - 1000)) for index in range(length)]


def serializable(result):
    return {
        "new_token_ids": result["new_token_ids"],
        "metrics": result["metrics"],
        "final_cache_snapshot": result["final_cache_snapshot"],
    }


def main():
    parser = argparse.ArgumentParser(description="第 14 期外部 KV Cache 正确性实验")
    add_model_arguments(parser)
    parser.add_argument("--prompt-length", type=int, default=513)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--external-chunk-size", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--capacity-mib", type=float, default=1024.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    _, config, model = load_model(args.model, args.revision, device, dtype)
    tokens = synthetic_tokens(args.prompt_length, config.vocab_size)
    common = dict(
        model=model,
        token_ids=tokens,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=-1,
        device=device,
        model_id=args.model,
        revision=args.revision,
        block_size=args.block_size,
        external_chunk_size=args.external_chunk_size,
        token_budget=args.token_budget,
        stop_on_eos=False,
        capture_logits=True,
    )
    baseline = run_request(mode="recompute", **common)
    with cache_service(args.capacity_mib) as (client, _):
        cold = run_request(mode="external", external_client=client, **common)
        warm = run_request(mode="external", external_client=client, **common)
        server_stats = client.stats()

    baseline_logits = baseline["first_token_logits"]
    cold_logits = cold["first_token_logits"]
    warm_logits = warm["first_token_logits"]
    result = {
        "environment": environment_record(args.model, args.revision, device, dtype),
        "config": vars(args),
        "checks": {
            "baseline_cold_tokens_equal": baseline["new_token_ids"] == cold["new_token_ids"],
            "baseline_warm_tokens_equal": baseline["new_token_ids"] == warm["new_token_ids"],
            "baseline_cold_logits_max_abs": float(torch.max(torch.abs(baseline_logits - cold_logits))),
            "baseline_warm_logits_max_abs": float(torch.max(torch.abs(baseline_logits - warm_logits))),
            "cold_hit_tokens": cold["metrics"]["hit_tokens"],
            "warm_hit_tokens": warm["metrics"]["hit_tokens"],
            "warm_executed_prefill_tokens": warm["metrics"]["executed_prefill_tokens"],
            "all_local_blocks_released": all(
                item["final_cache_snapshot"]["used_blocks"] == 0
                for item in (baseline, cold, warm)
            ),
        },
        "baseline": serializable(baseline),
        "cold_external": serializable(cold),
        "warm_external": serializable(warm),
        "server_stats": server_stats,
    }
    write_json(args.output, result)
    print(result["checks"])
    if not all((
        result["checks"]["baseline_cold_tokens_equal"],
        result["checks"]["baseline_warm_tokens_equal"],
        result["checks"]["all_local_blocks_released"],
    )):
        raise SystemExit("正确性验证失败")


if __name__ == "__main__":
    main()
