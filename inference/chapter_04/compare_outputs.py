"""用真实 Qwen3 权重检查动态 Batch 的请求归属和输出一致性。"""

import argparse
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Continuous Batching 正确性实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    prompts = [
        "用一句话解释 Continuous Batching。",
        "用一句话解释 Prefill。",
        "用一句话解释 Decode。",
        "用一句话解释 KV Cache。",
    ]
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts]
    budgets = [2, args.max_new_tokens, 3, args.max_new_tokens]
    specs = make_request_specs(sequences, budgets)
    model = load_handwritten_model(model_directory, device, dtype=dtype)

    isolated = []
    for sequence, budget in zip(sequences, budgets):
        run = run_fixed_batching(
            model,
            make_request_specs([sequence], budget),
            1,
            tokenizer.eos_token_id,
            device,
            stop_on_eos=False,
        )
        isolated.append(run["new_token_ids"][0])
    fixed = run_fixed_batching(
        model, specs, 2, tokenizer.eos_token_id, device, stop_on_eos=False
    )
    continuous = run_continuous_batching(
        model, specs, 2, tokenizer.eos_token_id, device, stop_on_eos=False
    )
    result = {
        "isolated_token_ids": isolated,
        "fixed_token_ids": fixed["new_token_ids"],
        "continuous_token_ids": continuous["new_token_ids"],
        "fixed_matches_isolated": fixed["new_token_ids"] == isolated,
        "continuous_matches_isolated": continuous["new_token_ids"] == isolated,
        "fixed_matches_continuous": (
            fixed["new_token_ids"] == continuous["new_token_ids"]
        ),
        "continuous_admission_trace": [
            event["admitted"] for event in continuous["events"]
            if event["admitted"]
        ],
    }
    print("fixed 与单请求一致:", result["fixed_matches_isolated"])
    print("continuous 与单请求一致:", result["continuous_matches_isolated"])
    print("fixed 与 continuous 一致:", result["fixed_matches_continuous"])
    print("Continuous admission trace:", result["continuous_admission_trace"])

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
                "max_new_tokens": args.max_new_tokens,
                "decoding": "greedy, EOS disabled",
                "thinking": False,
            },
            "result": result,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)

    if args.dtype == "float32" and not result["continuous_matches_isolated"]:
        raise SystemExit("float32 动态请求归属或输出一致性检查失败")


if __name__ == "__main__":
    main()
