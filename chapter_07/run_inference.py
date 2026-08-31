"""第 07 期自然语言推理入口。"""

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
    parser = argparse.ArgumentParser(description="运行完整/Chunked Prefill 推理")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt", action="append", help="可重复传入")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-running-requests", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mode", choices=("full", "chunked", "both"), default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(
        args.max_new_tokens, args.max_running_requests,
        args.token_budget, args.block_size,
    ) < 1:
        raise ValueError("长度、预算和运行请求数必须为正")
    if args.token_budget < args.max_running_requests:
        raise ValueError("Token Budget 不能小于最大运行请求数")
    if not torch.cuda.is_available():
        raise RuntimeError("真实权重推理需要可用的 NVIDIA GPU")
    prompts = args.prompt or [
        "用一句话解释 Chunked Prefill。",
        "请用两句话说明长 Prompt 为什么会影响其他请求的逐 Token 输出。" * 4,
        "只回答数字：23 乘以 17 等于多少？",
    ]
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    model = load_handwritten_model(model_directory, device)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts]
    specs = make_request_specs(sequences, args.max_new_tokens)
    modes = ("full", "chunked") if args.mode == "both" else (args.mode,)
    for mode in modes:
        result = run_engine(
            model, specs, args.max_running_requests,
            tokenizer.eos_token_id, device,
            mode=mode, token_budget=args.token_budget,
            block_size=args.block_size, stop_on_eos=True,
        )
        print("\n=== %s ===" % mode)
        for prompt, token_ids in zip(prompts, result["new_token_ids"]):
            print("Prompt:", prompt)
            print("Output:", tokenizer.decode(token_ids))
        print("Prefill iterations:", result["metrics"]["prefill_iterations"])
        print("Oversize iterations:", result["metrics"]["oversize_prefill_iterations"])


if __name__ == "__main__":
    main()
