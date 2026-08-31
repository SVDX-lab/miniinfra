"""运行固定 Batch，并与逐请求串行 KV Cache baseline 对照。"""

import argparse

import torch

from batch_generation import generate_fixed_batch, generate_serial, synchronize
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="固定批处理推理")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["用一句话解释 Prefill。", "用一句话解释 Decode。"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("serial", "batch", "both"), default="both"
    )
    return parser.parse_args()


def print_result(name, result, tokenizer):
    metrics = result["metrics"]
    print("\n[%s]" % name)
    for index, token_ids in enumerate(result["new_token_ids"]):
        print("请求 %d Token IDs: %s" % (index, token_ids))
        print(
            "请求 %d 输出: %s"
            % (index, tokenizer.decode(token_ids, skip_special_tokens=True))
        )
    print("端到端模型计算延迟: %.2f ms" % metrics["end_to_end_ms"])
    print("请求吞吐: %.3f requests/s" % metrics["request_throughput_per_second"])
    print(
        "输出吞吐: %.3f tokens/s"
        % metrics["output_token_throughput_per_second"]
    )
    if name == "batch":
        print("Prefill 延迟: %.2f ms" % metrics["prefill_ms"])
        print(
            "平均 Decode 迭代延迟: %.2f ms"
            % metrics["average_decode_iteration_ms"]
        )
        print("Padding 有效率: %.2f%%" % (metrics["padding_efficiency"] * 100))
        print("槽位利用率: %.2f%%" % (metrics["slot_utilization"] * 100))
        print("峰值显存: %.2f MiB" % (metrics["peak_memory_bytes"] / 1024**2))
        print("最终 KV Cache: %.2f MiB" % (metrics["cache_bytes"] / 1024**2))


def main():
    args = parse_args()
    if args.max_new_tokens < 1 or args.warmup < 0:
        raise ValueError("输出长度必须大于 0，warmup 不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in args.prompts]
    model = load_handwritten_model(model_directory, device)

    for _ in range(args.warmup):
        generate_fixed_batch(
            model, sequences, min(2, args.max_new_tokens),
            tokenizer.eos_token_id, device,
        )
    synchronize(device)

    results = {}
    if args.mode in ("serial", "both"):
        results["serial"] = generate_serial(
            model, sequences, args.max_new_tokens,
            tokenizer.eos_token_id, device,
        )
        print_result("serial", results["serial"], tokenizer)
    if args.mode in ("batch", "both"):
        results["batch"] = generate_fixed_batch(
            model, sequences, args.max_new_tokens,
            tokenizer.eos_token_id, device,
        )
        print_result("batch", results["batch"], tokenizer)
    if len(results) == 2:
        print("\n[对照]")
        print(
            "逐请求 Token 一致:",
            results["serial"]["new_token_ids"] == results["batch"]["new_token_ids"],
        )
        serial_ms = results["serial"]["metrics"]["end_to_end_ms"]
        batch_ms = results["batch"]["metrics"]["end_to_end_ms"]
        print("Batch makespan 加速比: %.3fx" % (serial_ms / batch_ms))
    print("计时不包含 tokenizer、下载、加载、排队和网络时间。")


if __name__ == "__main__":
    main()
