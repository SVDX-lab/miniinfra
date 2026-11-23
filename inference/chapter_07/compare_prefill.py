"""使用真实 Qwen3 权重比较完整 Prefill 与 Chunked Prefill。"""

import argparse
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Chunked Prefill 正确性实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--token-budget", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def compact_trace(run):
    return [
        {
            "iteration": event["iteration"],
            "phase": event["phase"],
            "admitted": event["admitted"],
            "completed": event["completed"],
            "chunk_ranges": event["chunk_ranges"],
            "scheduled_tokens": event["scheduled_tokens"],
            "oversize_singleton": event["oversize_singleton"],
        }
        for event in run["events"]
    ]


def main():
    args = parse_args()
    if args.token_budget < 4 or args.block_size < 1:
        raise ValueError("Token Budget 至少为 4，Block Size 必须为正")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    prompts = [
        "用一句话解释 Chunked Prefill。",
        "请简要说明长提示词为什么会干扰正在进行的逐 Token 解码。" * 8,
        "Explain prefill and decode in one short sentence.",
        "只回答数字：19 乘以 17 等于多少？",
    ]
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts]
    specs = make_request_specs(sequences, [8, 8, 8, 2], [0.0, 1.0, 1.0, 1.0])
    model = load_handwritten_model(model_directory, device, dtype=dtype)
    common = dict(
        model=model,
        request_specs=specs,
        max_running_requests=4,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        token_budget=args.token_budget,
        block_size=args.block_size,
        stop_on_eos=False,
        capture_logits=True,
    )
    full = run_engine(mode="full", **common)
    chunked = run_engine(mode="chunked", **common)
    tokens_match = full["new_token_ids"] == chunked["new_token_ids"]
    logit_errors = {
        request_id: float(torch.max(torch.abs(
            full["first_token_logits"][request_id]
            - chunked["first_token_logits"][request_id]
        )).item())
        for request_id in full["first_token_logits"]
    }
    result = {
        "tokens_match": tokens_match,
        "max_first_token_logit_error": max(logit_errors.values()),
        "per_request_first_token_logit_error": logit_errors,
        "prompt_token_lengths": [len(sequence) for sequence in sequences],
        "full_token_ids": full["new_token_ids"],
        "chunked_token_ids": chunked["new_token_ids"],
        "full_trace": compact_trace(full),
        "chunked_trace": compact_trace(chunked),
        "full_metrics": full["metrics"],
        "chunked_metrics": chunked["metrics"],
    }
    print("full 与 chunked 逐请求 Token 一致:", tokens_match)
    print("首 Token Logits 最大误差: %.8f" % result["max_first_token_logit_error"])
    print(
        "超预算 iteration: full=%d, chunked=%d"
        % (
            full["metrics"]["oversize_prefill_iterations"],
            chunked["metrics"]["oversize_prefill_iterations"],
        )
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": {
                "gpu": torch.cuda.get_device_name(device),
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "model": args.model,
                "revision": args.revision,
                "dtype": args.dtype,
                "token_budget": args.token_budget,
                "block_size": args.block_size,
                "decoding": "greedy, EOS disabled",
                "thinking": False,
            },
            "result": result,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)
    if args.dtype == "float32" and (
        not tokens_match or result["max_first_token_logit_error"] > 1e-3
    ):
        raise SystemExit("float32 Chunked Prefill 正确性检查失败")


if __name__ == "__main__":
    main()
