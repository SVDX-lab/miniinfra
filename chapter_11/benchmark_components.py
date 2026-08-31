"""拆分 Draft 单步、Target 单步与 Target 块验证成本。"""

import argparse

import torch

from experiment_utils import (
    environment_snapshot,
    load_target_and_draft,
    set_seed,
    summarize,
    timed_call,
    write_json,
)
from qwen3_model import DEFAULT_DRAFT_MODEL_ID, DEFAULT_TARGET_MODEL_ID
from qwen3_tokenizer import Qwen3Tokenizer
from speculative_decode import prefill, target_greedy_generate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL_ID)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL_ID)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="请解释推测解码为什么需要验证候选 token。")
    parser.add_argument("--draft-lengths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", default="chapter_11/benchmark-components-results.local.json"
    )
    return parser.parse_args()


def measure_forward(model, cache, input_ids, warmup, repeats, device):
    checkpoint = cache.length

    def call():
        model(input_ids, cache)
        cache.rollback(checkpoint)

    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        _, seconds = timed_call(call, device)
        samples.append(seconds)
    return summarize(samples)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA GPU")
    set_seed(args.seed)
    target_directory, target, draft_directory, draft = load_target_and_draft(
        args.target_model, args.draft_model, args.device, args.dtype
    )
    tokenizer = Qwen3Tokenizer(target_directory)
    prompt_ids = tokenizer.encode_chat_prompt(args.prompt)
    max_query = max(args.draft_lengths) + 1
    capacity = len(prompt_ids) + max_query + 4
    target_cache, _ = prefill(target, prompt_ids, capacity)
    draft_cache, _ = prefill(draft, prompt_ids, capacity)
    reference = target_greedy_generate(
        target, prompt_ids, max_query, eos_token_id=None
    ).token_ids
    target_device = next(target.parameters()).device
    draft_device = next(draft.parameters()).device
    target_single = torch.tensor(
        [reference[:1]], dtype=torch.long, device=target_device
    )
    draft_single = target_single.to(draft_device)
    result = {
        "environment": environment_snapshot(
            target_directory,
            draft_directory,
            args.dtype,
            target_model=args.target_model,
            draft_model=args.draft_model,
        ),
        "config": {
            "prompt": args.prompt,
            "prompt_tokens": len(prompt_ids),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "timing": "wall clock with CUDA synchronization",
        },
        "target_single_token_seconds": measure_forward(
            target,
            target_cache,
            target_single,
            args.warmup,
            args.repeats,
            args.device,
        ),
        "draft_single_token_seconds": measure_forward(
            draft,
            draft_cache,
            draft_single,
            args.warmup,
            args.repeats,
            args.device,
        ),
        "verification": [],
    }
    draft_single_mean = result["draft_single_token_seconds"]["mean"]
    for draft_length in args.draft_lengths:
        query_length = draft_length + 1
        block = torch.tensor(
            [reference[:query_length]], dtype=torch.long, device=target_device
        )
        summary = measure_forward(
            target,
            target_cache,
            block,
            args.warmup,
            args.repeats,
            args.device,
        )
        result["verification"].append(
            {
                "draft_length": draft_length,
                "target_query_tokens": query_length,
                "target_block_seconds": summary,
                "estimated_draft_proposal_seconds": (
                    (draft_length + 1) * draft_single_mean
                ),
                "estimated_round_seconds_without_control": (
                    (draft_length + 1) * draft_single_mean + summary["mean"]
                ),
            }
        )
    write_json(args.output, result)
    print("结果已写入", args.output)
    print(
        "target q=1 %.3f ms | draft q=1 %.3f ms"
        % (
            result["target_single_token_seconds"]["mean"] * 1000,
            result["draft_single_token_seconds"]["mean"] * 1000,
        )
    )
    for item in result["verification"]:
        print(
            "γ=%d target_block=%.3f ms estimated_round=%.3f ms"
            % (
                item["draft_length"],
                item["target_block_seconds"]["mean"] * 1000,
                item["estimated_round_seconds_without_control"] * 1000,
            )
        )


if __name__ == "__main__":
    main()
