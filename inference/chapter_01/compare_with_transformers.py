"""逐步比较手写实现与 Transformers 参考实现。"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    model_dtype_from_config,
    Qwen3Config,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="与 Transformers 对比逐 Token 输出")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument(
        "--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本"
    )
    parser.add_argument("--prompt", default="请用一句话介绍 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--logits-atol",
        type=float,
        default=0.25,
        help="Logits 最大绝对误差阈值",
    )
    return parser.parse_args()


@torch.inference_mode()
def compare_step_by_step(
    handwritten_model,
    reference_model,
    input_ids,
    max_new_tokens,
    eos_token_id,
):
    """使用相同上下文逐步比较 Logits 和 greedy Token。"""

    generated_ids = input_ids
    generated_token_ids = []
    largest_error = 0.0
    error_sum = 0.0

    for step in range(max_new_tokens):
        handwritten_logits = handwritten_model(generated_ids)[:, -1, :]
        reference_output = reference_model(generated_ids, use_cache=False)
        reference_logits = reference_output.logits[:, -1, :]

        # 先转为 float32 再计算误差，避免低精度减法损失信息。
        difference = (
            handwritten_logits.float() - reference_logits.float()
        ).abs()
        max_error = difference.max().item()
        mean_error = difference.mean().item()
        largest_error = max(largest_error, max_error)
        error_sum += mean_error

        handwritten_token = torch.argmax(handwritten_logits, dim=-1)
        reference_token = torch.argmax(reference_logits, dim=-1)
        handwritten_id = handwritten_token.item()
        reference_id = reference_token.item()

        print(
            "步骤 %02d: 手写=%d, Transformers=%d, max_error=%.6f, mean_error=%.6f"
            % (step + 1, handwritten_id, reference_id, max_error, mean_error)
        )

        if handwritten_id != reference_id:
            return {
                "tokens_match": False,
                "first_mismatch_step": step + 1,
                "largest_error": largest_error,
                "average_mean_error": error_sum / (step + 1),
                "token_ids": generated_token_ids,
            }

        generated_token_ids.append(handwritten_id)
        next_token = handwritten_token.unsqueeze(1)
        generated_ids = torch.cat((generated_ids, next_token), dim=1)

        if handwritten_id == eos_token_id:
            break

    return {
        "tokens_match": True,
        "first_mismatch_step": None,
        "largest_error": largest_error,
        "average_mean_error": error_sum / len(generated_token_ids),
        "token_ids": generated_token_ids,
    }


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    handwritten_tokenizer = Qwen3Tokenizer(model_directory)
    reference_tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # 模型对照开始前，先证明手写 Tokenizer 构造了完全相同的输入。
    handwritten_ids = handwritten_tokenizer.encode_chat_prompt(args.prompt)
    messages = [{"role": "user", "content": args.prompt}]
    reference_text = reference_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    reference_ids = reference_tokenizer(
        reference_text,
        add_special_tokens=False,
    ).input_ids
    if handwritten_ids != reference_ids:
        print("手写 Tokenizer 与 Transformers 输入不一致")
        print("手写 Token IDs:", handwritten_ids)
        print("参考 Token IDs:", reference_ids)
        raise SystemExit(3)

    print("Tokenizer 输入一致，共 %d 个 Token" % len(handwritten_ids))
    input_ids = torch.tensor([handwritten_ids], dtype=torch.long, device=device)

    config = Qwen3Config.from_model_directory(model_directory)
    model_dtype = model_dtype_from_config(config)
    handwritten_model = load_handwritten_model(model_directory, device)

    # 强制参考实现使用 eager attention，并关闭 KV Cache，使两边执行条件一致。
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        torch_dtype=model_dtype,
        attn_implementation="eager",
    )
    reference_model.config.use_cache = False
    reference_model = reference_model.to(device).eval()

    result = compare_step_by_step(
        handwritten_model,
        reference_model,
        input_ids,
        args.max_new_tokens,
        handwritten_tokenizer.eos_token_id,
    )

    generated_text = handwritten_tokenizer.decode(
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
