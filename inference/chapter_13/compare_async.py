"""真实 Qwen3 权重下验证 sync/async Offloading 的状态与输出正确性。"""

import argparse

import torch

from engine import run_engine
from experiment_utils import collect_environment, save_results
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer
from scheduler import make_request_specs


PROMPTS = (
    "用一句话解释异步传输。",
    "解释 CUDA Stream 和 Event 如何表达计算与数据传输之间的依赖。" * 2,
    "详细说明 KV Cache 换出与换入期间，GPU Block、Pinned CPU Block 和请求"
    "状态的生命周期，并解释为什么提交完成不等于传输完成。" * 4,
)


def parse_args():
    parser = argparse.ArgumentParser(description="sync/async 正确性对照")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), action="append"
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--extra-pool-blocks", type=int, default=6)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def first_divergence(left, right):
    for index, pair in enumerate(zip(left, right)):
        if pair[0] != pair[1]:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def logits_error(reference, candidate):
    count = min(len(reference), len(candidate))
    return max(
        (
            float(torch.max(torch.abs(reference[index] - candidate[index])).item())
            for index in range(count)
        ),
        default=0.0,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("正确性实验需要 NVIDIA GPU")
    device = torch.device("cuda")
    directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(directory)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in PROMPTS]
    sequences += sequences
    prompt_lengths = [len(sequence) for sequence in sequences]
    dtypes = args.dtype or ["float32", "bfloat16"]
    report = {
        "environment": collect_environment(device),
        "prompt_lengths": prompt_lengths,
        "runs": [],
    }
    for dtype_name in dtypes:
        model = load_handwritten_model(directory, device, getattr(torch, dtype_name))
        specs = make_request_specs(sequences, args.max_new_tokens)
        prompt_blocks = sum(
            (length + args.block_size - 1) // args.block_size
            for length in prompt_lengths
        )
        pool_blocks = prompt_blocks + args.extra_pool_blocks
        common = dict(
            max_running_requests=6,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
            token_budget=args.token_budget,
            block_size=args.block_size,
            stop_on_eos=False,
            capture_logits=True,
        )
        reference = run_engine(model, specs, transfer_mode="sync", **common)
        sync = run_engine(
            model, specs, transfer_mode="sync", pool_blocks=pool_blocks, **common
        )
        asynchronous = run_engine(
            model, specs, transfer_mode="async", pool_blocks=pool_blocks, **common
        )
        requests = []
        for index in range(len(sequences)):
            request_id = str(index)
            reference_tokens = reference["new_token_ids"][index]
            requests.append({
                "request_id": request_id,
                "prompt_tokens": prompt_lengths[index],
                "sync_match": sync["new_token_ids"][index] == reference_tokens,
                "async_match": (
                    asynchronous["new_token_ids"][index] == reference_tokens
                ),
                "sync_divergence": first_divergence(
                    reference_tokens, sync["new_token_ids"][index]
                ),
                "async_divergence": first_divergence(
                    reference_tokens, asynchronous["new_token_ids"][index]
                ),
                "sync_max_abs_logit_error": logits_error(
                    reference["token_logits"][request_id],
                    sync["token_logits"][request_id],
                ),
                "async_max_abs_logit_error": logits_error(
                    reference["token_logits"][request_id],
                    asynchronous["token_logits"][request_id],
                ),
            })
        entry = {
            "dtype": dtype_name,
            "pool_blocks": pool_blocks,
            "reference_preemptions": reference["metrics"]["preemption_count"],
            "sync_preemptions": sync["metrics"]["preemption_count"],
            "async_preemptions": asynchronous["metrics"]["preemption_count"],
            "sync_metrics": sync["metrics"],
            "async_metrics": asynchronous["metrics"],
            "requests": requests,
            "sync_all_match": all(row["sync_match"] for row in requests),
            "async_all_match": all(row["async_match"] for row in requests),
        }
        report["runs"].append(entry)
        print(
            "%s: preempt reference/sync/async=%d/%d/%d, token match=%s/%s"
            % (
                dtype_name, entry["reference_preemptions"],
                entry["sync_preemptions"], entry["async_preemptions"],
                entry["sync_all_match"], entry["async_all_match"],
            )
        )
        del reference, sync, asynchronous, model
        torch.cuda.empty_cache()
    if args.output:
        save_results(args.output, report)


if __name__ == "__main__":
    main()
