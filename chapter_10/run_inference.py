"""使用第 10 期独立 CUDA Graph Decode 执行器完成自然语言推理。"""

import argparse

import torch

from experiment_utils import load_model, set_seed
from qwen3_model import DEFAULT_MODEL_ID
from qwen3_tokenizer import Qwen3Tokenizer
from static_decode import StaticDecodeRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default="请用一句话解释 CUDA Graph。")
    parser.add_argument("--mode", choices=StaticDecodeRunner.MODES, default="cuda_graph")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-bucket", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("eager", "flash"), default="flash")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("自然语言 CUDA Graph 推理需要 NVIDIA CUDA GPU")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须大于 0")
    set_seed(args.seed)
    model_directory, model = load_model(
        args.model, "cuda", args.dtype, args.attention_backend
    )
    tokenizer = Qwen3Tokenizer(model_directory)
    prompt_tokens = tokenizer.encode_chat_prompt(args.prompt)
    if len(prompt_tokens) + args.max_new_tokens > args.context_bucket:
        raise ValueError(
            "Prompt 与输出预算超过 context_bucket，请增大 --context-bucket"
        )
    runner = StaticDecodeRunner(
        model,
        capacity=1,
        context_bucket=args.context_bucket,
        block_size=args.block_size,
        pad_token_id=tokenizer.eos_token_id,
    )
    if args.mode == "cuda_graph":
        capture = runner.capture()
        print("CUDA Graph capture: %.2f ms" % capture["capture_ms"])
    generated = runner.prepare_prompts([prompt_tokens])
    while len(generated) < args.max_new_tokens:
        runner.step(args.mode)
        # 自然语言入口逐 Token 检查 EOS，因此这里有意同步到 CPU；性能主实验关闭 EOS，
        # 不执行这条每步同步路径。
        token = int(runner.output_tokens[0].item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
