"""第 11 期自然语言推理入口。"""

import argparse

import torch

from experiment_utils import load_target_and_draft, set_seed, timed_call
from qwen3_model import DEFAULT_DRAFT_MODEL_ID, DEFAULT_TARGET_MODEL_ID
from qwen3_tokenizer import Qwen3Tokenizer
from speculative_decode import speculative_greedy_generate, target_greedy_generate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["target_only", "speculative"], default="speculative")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL_ID)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL_ID)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--draft-length", type=int, default=4)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA GPU")
    set_seed(args.seed)
    target_directory, target, _, draft = load_target_and_draft(
        args.target_model, args.draft_model, args.device, args.dtype
    )
    tokenizer = Qwen3Tokenizer(target_directory)
    prompt_ids = tokenizer.encode_chat_prompt(args.prompt)
    eos = None if args.ignore_eos else tokenizer.eos_token_id
    if args.mode == "target_only":
        call = lambda: target_greedy_generate(
            target, prompt_ids, args.max_new_tokens, eos_token_id=eos
        )
    else:
        call = lambda: speculative_greedy_generate(
            target,
            draft,
            prompt_ids,
            args.max_new_tokens,
            args.draft_length,
            eos_token_id=eos,
        )
    result, seconds = timed_call(call, args.device)
    print(tokenizer.decode(result.token_ids, skip_special_tokens=True))
    print("\n---")
    print("mode:", args.mode)
    print("prompt_tokens:", len(prompt_ids))
    print("generated_tokens:", len(result.token_ids))
    print("elapsed_seconds: %.6f" % seconds)
    if args.mode == "speculative":
        print("acceptance_rate: %.4f" % result.stats.acceptance_rate)
        print(
            "mean_accepted_tokens_per_round: %.4f"
            % result.stats.mean_accepted_tokens_per_round
        )


if __name__ == "__main__":
    main()
