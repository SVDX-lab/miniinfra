"""运行固定批次与 Continuous Batching 自然语言对照。"""

import argparse

import torch

from continuous_batching import (
    make_request_specs,
    run_continuous_batching,
    run_fixed_batching,
)
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous Batching 推理")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "用一句话解释 Prefill。",
            "用一句话解释 Decode。",
            "用一句话解释 KV Cache。",
            "用一句话解释固定批处理。",
        ],
    )
    parser.add_argument(
        "--max-new-tokens", type=int, nargs="+", default=[8, 24, 8, 24]
    )
    parser.add_argument("--max-running-requests", type=int, default=2)
    parser.add_argument(
        "--mode", choices=("fixed", "continuous", "both"), default="both"
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
    print("请求吞吐: %.3f requests/s" % metrics["request_throughput_per_second"])
    print("输出吞吐: %.3f tokens/s" % metrics["output_token_throughput_per_second"])
    print("执行槽位有效率: %.2f%%" % (metrics["execution_slot_utilization"] * 100))
    print("TTFT p50/p95: %.2f / %.2f ms" % (
        metrics["ttft_ms_p50"], metrics["ttft_ms_p95"]
    ))
    print("ITL p50/p95: %.2f / %.2f ms" % (
        metrics["itl_ms_p50"], metrics["itl_ms_p95"]
    ))
    print("Cache 管理耗时占比: %.2f%%" % (
        metrics["cache_management_fraction"] * 100
    ))


def main():
    args = parse_args()
    if args.max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    if len(args.max_new_tokens) == 1:
        budgets = args.max_new_tokens[0]
    elif len(args.max_new_tokens) == len(args.prompts):
        budgets = args.max_new_tokens
    else:
        raise ValueError("输出预算应提供一个值或与 prompts 数量一致")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in args.prompts]
    specs = make_request_specs(sequences, budgets)
    model = load_handwritten_model(model_directory, device)
    results = {}
    if args.mode in ("fixed", "both"):
        results["fixed"] = run_fixed_batching(
            model, specs, args.max_running_requests,
            tokenizer.eos_token_id, device,
        )
        print_result("fixed", results["fixed"], tokenizer)
    if args.mode in ("continuous", "both"):
        results["continuous"] = run_continuous_batching(
            model, specs, args.max_running_requests,
            tokenizer.eos_token_id, device,
        )
        print_result("continuous", results["continuous"], tokenizer)
    if len(results) == 2:
        print("\n逐请求 Token 一致:", (
            results["fixed"]["new_token_ids"]
            == results["continuous"]["new_token_ids"]
        ))
    print("计时不包含模型下载、加载、Tokenizer、网络和真实睡眠。")


if __name__ == "__main__":
    main()
