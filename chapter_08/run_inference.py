"""第 08 期独立的自然语言 Prefix Cache 推理入口。"""

import argparse

import torch

from engine import run_engine
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer
from scheduler import make_request_specs


def parse_args():
    parser = argparse.ArgumentParser(description="比较 Prefix Cache 开关")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--cache-capacity-blocks", type=int, default=64)
    parser.add_argument("--mode", choices=("disabled", "enabled", "both"), default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(
        args.max_new_tokens, args.token_budget, args.block_size,
        args.cache_capacity_blocks,
    ) < 1:
        raise ValueError("长度、预算和缓存容量必须为正")
    if not torch.cuda.is_available():
        raise RuntimeError("真实权重推理需要可用的 NVIDIA GPU")
    shared = (
        "请先阅读以下共同背景：Prefix Cache 会复用多个请求相同前缀已经计算出的 "
        "KV Cache，从而减少后续请求的 Prefill 计算。" * 8
    )
    prompts = [
        shared + "\n现在请用一句话概括它的主要收益。",
        shared + "\n现在请用一句话说明它的显存代价。",
    ]
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    model = load_handwritten_model(model_directory, device)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts]
    specs = make_request_specs(sequences, args.max_new_tokens)
    modes = ("disabled", "enabled") if args.mode == "both" else (args.mode,)
    for mode in modes:
        result = run_engine(
            model, specs, 1, tokenizer.eos_token_id, device,
            token_budget=args.token_budget,
            block_size=args.block_size,
            stop_on_eos=True,
            prefix_cache_enabled=(mode == "enabled"),
            prefix_cache_capacity_blocks=args.cache_capacity_blocks,
            model_namespace=args.model + "@" + args.revision,
        )
        print("\n=== Prefix Cache %s ===" % mode)
        for index, token_ids in enumerate(result["new_token_ids"]):
            row = result["request_metrics"][index]
            print("Request %d hit tokens: %d" % (index, row["prefix_hit_tokens"]))
            print("Output:", tokenizer.decode(token_ids))
        print("Executed Prompt tokens:", result["metrics"]["executed_prompt_tokens"])


if __name__ == "__main__":
    main()
