"""使用真实 Qwen3 权重验证 dense 与 Paged KV Cache 的输出一致性。"""

import argparse
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Paged KV Cache 正确性实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens < 2 or any(size < 1 for size in args.block_sizes):
        raise ValueError("生成长度至少为 2，Block Size 必须为正")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    prompts = [
        "用一句话解释 Paged KV Cache。",
        "只回答数字：17 乘以 23 等于多少？",
        "Explain KV cache in one short sentence.",
        "用一句话解释物理块和逻辑块的区别。",
    ]
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts]
    budgets = [2, args.max_new_tokens, 3, args.max_new_tokens]
    arrivals = [0.0, 0.0, 1.0, 2.0]
    specs = make_request_specs(sequences, budgets, arrivals)
    model = load_handwritten_model(model_directory, device, dtype=dtype)

    dense = run_dense_cache(
        model, specs, 2, tokenizer.eos_token_id, device, stop_on_eos=False
    )
    paged_results = {}
    for block_size in args.block_sizes:
        run = run_paged_cache(
            model, specs, 2, tokenizer.eos_token_id, device,
            block_size=block_size, stop_on_eos=False,
        )
        paged_results[str(block_size)] = {
            "token_ids": run["new_token_ids"],
            "matches_dense": run["new_token_ids"] == dense["new_token_ids"],
            "admission_trace": [
                event["admitted"] for event in run["events"] if event["admitted"]
            ],
            "block_reuse_count": run["metrics"]["block_reuse_count"],
        }
    result = {
        "dense_token_ids": dense["new_token_ids"],
        "paged": paged_results,
        "all_match_dense": all(
            item["matches_dense"] for item in paged_results.values()
        ),
    }
    print("全部 Block Size 与 dense Token 一致:", result["all_match_dense"])
    for block_size, item in paged_results.items():
        print(
            "block=%s, match=%s, reuse=%d, admissions=%s"
            % (
                block_size, item["matches_dense"], item["block_reuse_count"],
                item["admission_trace"],
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
                "max_new_tokens": args.max_new_tokens,
                "block_sizes": args.block_sizes,
                "decoding": "greedy, EOS disabled",
                "thinking": False,
            },
            "result": result,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)

    if args.dtype == "float32" and not result["all_match_dense"]:
        raise SystemExit("float32 Paged KV Cache 输出一致性检查失败")


if __name__ == "__main__":
    main()
