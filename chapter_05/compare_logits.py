"""逐 Token 比较 dense 与 Paged KV Cache 的 Logits。"""

import argparse
import json
import math
from pathlib import Path

import torch

from paged_cache import PagedKVCache, paged_decode_forward
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Paged KV Cache Logits 对照")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--logits-atol", type=float)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


@torch.inference_mode()
def compare_prompt(model, tokenizer, prompt, max_new_tokens, block_size, device):
    token_ids = tokenizer.encode_chat_prompt(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    positions = torch.arange(len(token_ids), device=device).unsqueeze(0)
    logits, dense_cache = model(
        input_ids, attention_mask=mask, position_ids=positions, use_cache=True
    )
    next_token = int(torch.argmax(logits[0, -1]).item())
    generated = [next_token]

    maximum_length = len(token_ids) + max_new_tokens - 1
    paged_cache = PagedKVCache(
        model.config,
        block_size,
        math.ceil(maximum_length / block_size),
        device,
        next(model.parameters()).dtype,
    )
    paged_cache.store_prefill("0", [
        (key.clone(), value.clone()) for key, value in dense_cache
    ])
    steps = []
    for step in range(max_new_tokens - 1):
        position = len(token_ids) + step
        current = torch.tensor([[next_token]], dtype=torch.long, device=device)
        position_ids = torch.tensor([[position]], dtype=torch.long, device=device)
        dense_mask = torch.ones(
            (1, position + 1), dtype=torch.bool, device=device
        )
        dense_logits, dense_cache = model(
            current,
            attention_mask=dense_mask,
            position_ids=position_ids,
            past_key_values=dense_cache,
            use_cache=True,
        )
        paged_logits, _ = paged_decode_forward(
            model, current, position_ids, ["0"], paged_cache
        )
        dense_last = dense_logits[0, -1].float()
        paged_last = paged_logits[0, -1].float()
        difference = (dense_last - paged_last).abs()
        dense_token = int(torch.argmax(dense_last).item())
        paged_token = int(torch.argmax(paged_last).item())
        steps.append({
            "step": step + 1,
            "position": position,
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "dense_token_id": dense_token,
            "paged_token_id": paged_token,
            "token_match": dense_token == paged_token,
        })
        next_token = dense_token
        generated.append(next_token)
    return {
        "prompt": prompt,
        "prompt_tokens": len(token_ids),
        "generated_token_ids": generated,
        "all_tokens_match": all(item["token_match"] for item in steps),
        "max_abs_error": max((item["max_abs_error"] for item in steps), default=0.0),
        "steps": steps,
    }


def main():
    args = parse_args()
    if args.max_new_tokens < 2 or args.block_size < 1:
        raise ValueError("生成长度至少为 2，Block Size 必须为正")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    atol = args.logits_atol
    if atol is None:
        atol = 1e-4 if dtype == torch.float32 else 1.0
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    model = load_handwritten_model(model_directory, device, dtype=dtype)
    prompts = [
        "用一句话解释 Paged KV Cache。",
        "Explain block tables in one short sentence.",
    ]
    results = [
        compare_prompt(
            model, tokenizer, prompt, args.max_new_tokens, args.block_size, device
        )
        for prompt in prompts
    ]
    passed = all(
        result["all_tokens_match"] and result["max_abs_error"] <= atol
        for result in results
    )
    print("逐 Token Top-1 一致:", all(r["all_tokens_match"] for r in results))
    print("最大 Logits 绝对误差:", max(r["max_abs_error"] for r in results))
    print("阈值检查:", passed, "atol=", atol)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "environment": {
                        "gpu": torch.cuda.get_device_name(device),
                        "pytorch": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "model": args.model,
                        "revision": args.revision,
                        "dtype": args.dtype,
                        "block_size": args.block_size,
                        "max_new_tokens": args.max_new_tokens,
                        "logits_atol": atol,
                    },
                    "passed": passed,
                    "results": results,
                },
                file, ensure_ascii=False, indent=2,
            )
        print("JSON 结果已写入:", output_path)
    if not passed:
        raise SystemExit("Paged KV Cache Logits 对照失败")


if __name__ == "__main__":
    main()
