"""运行 dense baseline 与 Paged KV Cache 自然语言对照。"""

import argparse

import torch

from cache_engine import make_request_specs, run_dense_cache, run_paged_cache
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Paged KV Cache 推理")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--prompts", nargs="+", default=[
            "用一句话解释 Paged KV Cache。",
            "用一句话解释 Block Table。",
            "用一句话解释显存碎片。",
        ],
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-running-requests", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mode", choices=("dense", "paged", "both"), default="both")
    return parser.parse_args()


def print_result(name, result, tokenizer):
    print("\n[%s]" % name)
    for index, token_ids in enumerate(result["new_token_ids"]):
        print("请求 %d: %s" % (
            index, tokenizer.decode(token_ids, skip_special_tokens=True)
        ))
    metrics = result["metrics"]
    print("Makespan: %.2f ms" % metrics["makespan_ms"])
    print("输出吞吐: %.3f tokens/s" % metrics["output_token_throughput_per_second"])
    print("ITL p50/p95: %.2f / %.2f ms" % (
        metrics["itl_ms_p50"], metrics["itl_ms_p95"]
    ))
    print("峰值 Cache Pool: %.2f MiB" % (
        metrics["peak_pool_cache_bytes"] / 1024**2
    ))


def main():
    args = parse_args()
    if args.max_new_tokens < 1 or args.max_running_requests < 1 or args.block_size < 1:
        raise ValueError("长度、并发和 Block Size 必须大于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本程序需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in args.prompts]
    specs = make_request_specs(sequences, args.max_new_tokens)
    model = load_handwritten_model(model_directory, device)
    results = {}
    if args.mode in ("dense", "both"):
        results["dense"] = run_dense_cache(
            model, specs, args.max_running_requests, tokenizer.eos_token_id, device
        )
        print_result("dense", results["dense"], tokenizer)
    if args.mode in ("paged", "both"):
        results["paged"] = run_paged_cache(
            model, specs, args.max_running_requests, tokenizer.eos_token_id, device,
            block_size=args.block_size,
        )
        print_result("paged", results["paged"], tokenizer)
    if len(results) == 2:
        print("\n逐请求 Token 一致:", (
            results["dense"]["new_token_ids"] == results["paged"]["new_token_ids"]
        ))
    print("计时不包含模型下载、加载、Tokenizer、网络和真实睡眠。")


if __name__ == "__main__":
    main()
