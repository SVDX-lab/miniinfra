"""真实 Qwen3 权重的 Target-only 与 Greedy Speculative 正确性实验。"""

import argparse

import torch

from experiment_utils import (
    environment_snapshot,
    load_target_and_draft,
    memory_snapshot,
    model_parameter_bytes,
    set_seed,
    write_json,
)
from qwen3_model import DEFAULT_DRAFT_MODEL_ID, DEFAULT_TARGET_MODEL_ID
from qwen3_tokenizer import Qwen3Tokenizer
from speculative_decode import speculative_greedy_generate, target_greedy_generate


DEFAULT_PROMPTS = [
    "请用三句话解释什么是 KV Cache。",
    "请补全这个 Python 函数：\ndef fibonacci(n):",
    "依次写出五个颜色名称，每行一个。",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL_ID)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL_ID)
    parser.add_argument("--target-revision")
    parser.add_argument("--draft-revision")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--draft-lengths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--respect-eos", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", default="chapter_11/compare-speculative-results.local.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA GPU")
    set_seed(args.seed)
    target_directory, target, draft_directory, draft = load_target_and_draft(
        args.target_model,
        args.draft_model,
        args.device,
        args.dtype,
        target_revision=args.target_revision,
        draft_revision=args.draft_revision,
    )
    tokenizer = Qwen3Tokenizer(target_directory)
    eos_token_id = tokenizer.eos_token_id if args.respect_eos else None
    prompts = DEFAULT_PROMPTS if args.prompts is None else args.prompts
    cases = []
    all_equal = True
    for prompt in prompts:
        prompt_ids = tokenizer.encode_chat_prompt(prompt)
        baseline = target_greedy_generate(
            target,
            prompt_ids,
            args.max_new_tokens,
            eos_token_id=eos_token_id,
        )
        comparisons = []
        for draft_length in args.draft_lengths:
            result = speculative_greedy_generate(
                target,
                draft,
                prompt_ids,
                args.max_new_tokens,
                draft_length,
                eos_token_id=eos_token_id,
            )
            equal = result.token_ids == baseline.token_ids
            cache_lengths_equal = (
                result.target_cache_length == result.draft_cache_length
            )
            all_equal = all_equal and equal and cache_lengths_equal
            comparisons.append(
                {
                    "draft_length": draft_length,
                    "tokens_equal": equal,
                    "cache_lengths_equal": cache_lengths_equal,
                    "target_cache_length": result.target_cache_length,
                    "draft_cache_length": result.draft_cache_length,
                    "stats": result.stats.to_dict(include_rounds=True),
                }
            )
        cases.append(
            {
                "prompt": prompt,
                "prompt_tokens": len(prompt_ids),
                "baseline_token_ids": baseline.token_ids,
                "baseline_text": tokenizer.decode(
                    baseline.token_ids, skip_special_tokens=True
                ),
                "baseline_stats": baseline.stats.to_dict(),
                "comparisons": comparisons,
            }
        )

    payload = {
        "environment": environment_snapshot(
            target_directory,
            draft_directory,
            args.dtype,
            target_model=args.target_model,
            draft_model=args.draft_model,
        ),
        "config": {
            "draft_lengths": args.draft_lengths,
            "max_new_tokens": args.max_new_tokens,
            "respect_eos": args.respect_eos,
            "seed": args.seed,
        },
        "model_memory": {
            "target_parameter_bytes": model_parameter_bytes(target),
            "draft_parameter_bytes": model_parameter_bytes(draft),
            "cuda": memory_snapshot(),
        },
        "all_equal": all_equal,
        "cases": cases,
    }
    write_json(args.output, payload)
    print("结果已写入", args.output)
    print("全部 Token 与 KV 长度检查:", "通过" if all_equal else "失败")
    for case in cases:
        summary = [
            "γ=%d acceptance=%.3f"
            % (
                item["draft_length"],
                item["stats"]["acceptance_rate"],
            )
            for item in case["comparisons"]
        ]
        print("prompt_tokens=%d | %s" % (case["prompt_tokens"], ", ".join(summary)))
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
