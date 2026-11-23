"""逐步比较同一手写模型的 no-cache 与 KV Cache 输出。"""

import argparse

import torch

from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="验证 KV Cache 前后的逐 Token 正确性")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt", default="请用一句话介绍 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="正确性验证默认使用 float32，避免低精度路径差异干扰判断",
    )
    parser.add_argument(
        "--logits-atol",
        type=float,
        default=1e-4,
        help="两条执行路径允许的最大 Logits 绝对误差",
    )
    return parser.parse_args()


@torch.inference_mode()
def compare_step_by_step(model, input_ids, max_new_tokens, eos_token_id):
    generated_ids = input_ids
    cached_input = input_ids
    past_key_values = None
    generated_token_ids = []
    largest_error = 0.0
    mean_error_sum = 0.0

    for step in range(max_new_tokens):
        baseline_logits, _ = model(generated_ids, use_cache=False)
        cached_logits, past_key_values = model(
            cached_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        baseline_last = baseline_logits[:, -1, :]
        cached_last = cached_logits[:, -1, :]
        difference = (baseline_last.float() - cached_last.float()).abs()
        max_error = difference.max().item()
        mean_error = difference.mean().item()
        largest_error = max(largest_error, max_error)
        mean_error_sum += mean_error

        baseline_token = torch.argmax(baseline_last, dim=-1, keepdim=True)
        cached_token = torch.argmax(cached_last, dim=-1, keepdim=True)
        baseline_id = baseline_token.item()
        cached_id = cached_token.item()
        cache_length = past_key_values[0][0].shape[2]
        print(
            "步骤 %02d: no-cache=%d, kv-cache=%d, cache_length=%d, "
            "max_error=%.6f, mean_error=%.6f"
            % (
                step + 1,
                baseline_id,
                cached_id,
                cache_length,
                max_error,
                mean_error,
            )
        )

        if baseline_id != cached_id:
            return {
                "tokens_match": False,
                "first_mismatch_step": step + 1,
                "largest_error": largest_error,
                "average_mean_error": mean_error_sum / (step + 1),
                "token_ids": generated_token_ids,
            }

        generated_token_ids.append(baseline_id)
        generated_ids = torch.cat((generated_ids, baseline_token), dim=1)
        cached_input = baseline_token
        if baseline_id == eos_token_id:
            break

    return {
        "tokens_match": True,
        "first_mismatch_step": None,
        "largest_error": largest_error,
        "average_mean_error": mean_error_sum / len(generated_token_ids),
        "token_ids": generated_token_ids,
    }


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")
    if args.logits_atol < 0:
        raise ValueError("--logits-atol 不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    token_ids = tokenizer.encode_chat_prompt(args.prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    comparison_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = load_handwritten_model(model_directory, device, dtype=comparison_dtype)

    print("Prompt Token 数:", len(token_ids))
    print("正确性对照 dtype:", args.dtype)
    result = compare_step_by_step(
        model,
        input_ids,
        args.max_new_tokens,
        tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(
        result["token_ids"], skip_special_tokens=True
    )
    print("\n逐 Token 一致:", result["tokens_match"])
    print("所有步骤最大 Logits 绝对误差: %.6f" % result["largest_error"])
    print("各步骤平均 Logits 误差的均值: %.6f" % result["average_mean_error"])
    print("生成 Token IDs:", result["token_ids"])
    print("生成文本:", generated_text)

    if not result["tokens_match"]:
        print("首个不一致步骤:", result["first_mismatch_step"])
        raise SystemExit(1)
    if result["largest_error"] > args.logits_atol:
        print("Token 一致，但 Logits 误差超过设定阈值。")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
