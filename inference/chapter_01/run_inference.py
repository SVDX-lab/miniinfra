"""运行手写 Qwen3-0.6B 单请求推理，并记录原始 baseline。"""

import argparse
import time

import torch

from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="运行手写 Qwen3-0.6B 推理")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument(
        "--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本"
    )
    parser.add_argument("--prompt", default="请用一句话介绍 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1, help="正式计时前的预热次数")
    return parser.parse_args()


def synchronize(device):
    """CUDA 默认异步执行，计时前后必须等待 GPU 完成工作。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_input_ids(tokenizer, prompt, device):
    token_ids = tokenizer.encode_chat_prompt(prompt)
    return torch.tensor([token_ids], dtype=torch.long, device=device)


@torch.inference_mode()
def generate_with_metrics(model, input_ids, max_new_tokens, eos_token_id, device):
    """生成文本，同时分别记录 Prefill 和每轮 Decode 延迟。"""

    generated_ids = input_ids
    new_token_ids = []
    step_times_ms = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(max_new_tokens):
        synchronize(device)
        start_time = time.perf_counter()
        logits = model(generated_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        synchronize(device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        step_times_ms.append(elapsed_ms)
        new_token_ids.append(next_token.item())
        generated_ids = torch.cat((generated_ids, next_token), dim=1)

        if next_token.item() == eos_token_id:
            break

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    # 第一次前向处理完整 Prompt，因此计为 Prefill；后续步骤计为 Decode。
    prefill_ms = step_times_ms[0]
    decode_times = step_times_ms[1:]
    average_decode_ms = sum(decode_times) / len(decode_times) if decode_times else 0.0

    metrics = {
        "prefill_ms": prefill_ms,
        "average_decode_ms": average_decode_ms,
        "end_to_end_ms": sum(step_times_ms),
        "peak_memory_mb": peak_memory_mb,
    }
    return generated_ids, new_token_ids, metrics


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")
    if args.warmup < 0:
        raise ValueError("--warmup 不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    input_ids = build_input_ids(tokenizer, args.prompt, device)
    model = load_handwritten_model(model_directory, device)

    # 预热不计入正式结果，避免首次 CUDA 初始化干扰测量。
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_ids)
    synchronize(device)

    output_ids, new_token_ids, metrics = generate_with_metrics(
        model,
        input_ids,
        args.max_new_tokens,
        tokenizer.eos_token_id,
        device,
    )

    output_token_ids = output_ids[0, input_ids.shape[1] :].tolist()
    generated_text = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    print("Prompt Token 数:", input_ids.shape[1])
    print("生成 Token IDs:", new_token_ids)
    print("生成文本:", generated_text)
    print("Prefill 延迟: %.2f ms" % metrics["prefill_ms"])
    print("平均 Decode 延迟: %.2f ms/token" % metrics["average_decode_ms"])
    print("端到端模型计算延迟: %.2f ms" % metrics["end_to_end_ms"])
    print("峰值已分配显存: %.2f MiB" % metrics["peak_memory_mb"])
    print("说明: 计时不包含 tokenizer、模型下载和模型加载。")


if __name__ == "__main__":
    main()
