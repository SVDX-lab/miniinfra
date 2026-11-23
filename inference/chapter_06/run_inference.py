"""运行 baseline 与 budgeted 调度策略的自然语言推理对照。"""

import argparse

import torch

from engine import run_scheduler
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer
from scheduler import make_request_specs


def parse_args():
    parser = argparse.ArgumentParser(description="迭代级调度器推理")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--prompts", nargs="+", default=[
            "用一句话解释推理请求调度器。",
            "用一句话解释 Token Budget。",
            "用一句话解释 Prefill 和 Decode 的区别。",
        ],
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-running-requests", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--policy", choices=("baseline", "budgeted", "both"), default="both"
    )
    return parser.parse_args()


def print_result(name, result, tokenizer):
    print("\n[%s]" % name)
    for index, token_ids in enumerate(result["new_token_ids"]):
        print("请求 %d: %s" % (
            index, tokenizer.decode(token_ids, skip_special_tokens=True)
        ))
    metrics = result["metrics"]
    print("Makespan: %.2f ms" % metrics["makespan_ms"])
    print("输出吞吐: %.3f token/s" % metrics["output_token_throughput_per_second"])
    print("TTFT p95: %.2f ms" % metrics["ttft_ms_p95"])
    print("ITL p50/p95: %.2f / %.2f ms" % (
        metrics["itl_ms_p50"], metrics["itl_ms_p95"]
    ))


def main():
    args = parse_args()
    if min(
        args.max_new_tokens, args.max_running_requests,
        args.token_budget, args.block_size,
    ) < 1:
        raise ValueError("长度、并发、预算和 Block Size 必须为正")
    if args.token_budget < args.max_running_requests:
        raise ValueError("Token Budget 不能小于最大运行请求数")
    if not torch.cuda.is_available():
        raise RuntimeError("本程序需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in args.prompts]
    specs = make_request_specs(sequences, args.max_new_tokens)
    model = load_handwritten_model(model_directory, device)
    results = {}
    if args.policy in ("baseline", "both"):
        results["baseline"] = run_scheduler(
            model, specs, args.max_running_requests,
            tokenizer.eos_token_id, device,
            policy="baseline", block_size=args.block_size,
        )
        print_result("baseline", results["baseline"], tokenizer)
    if args.policy in ("budgeted", "both"):
        results["budgeted"] = run_scheduler(
            model, specs, args.max_running_requests,
            tokenizer.eos_token_id, device,
            policy="budgeted", token_budget=args.token_budget,
            block_size=args.block_size,
        )
        print_result("budgeted", results["budgeted"], tokenizer)
    if len(results) == 2:
        print("\n逐请求 Token 一致:", (
            results["baseline"]["new_token_ids"]
            == results["budgeted"]["new_token_ids"]
        ))
    print("计时不包含模型下载、加载、Tokenizer、网络和真实睡眠。")


if __name__ == "__main__":
    main()
