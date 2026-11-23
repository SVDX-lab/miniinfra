"""真实 Qwen3-0.6B 的 Eager/FlashAttention 端到端正确性比较。"""

import argparse
import json

import torch

from engine import run_engine
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from scheduler import make_request_specs


def parse_args():
    parser = argparse.ArgumentParser(description="真实模型 Attention 后端正确性比较")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-logits-error", type=float, default=0.5)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def make_sequence(length, vocab_size, salt):
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("真实模型比较需要 NVIDIA GPU")
    torch.manual_seed(9)
    device = torch.device("cuda")
    directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(directory, device, attention_backend="eager")
    sequences = [
        make_sequence(79, model.config.vocab_size, 11),
        make_sequence(173, model.config.vocab_size, 29),
    ]
    specs = make_request_specs(sequences, 4)
    common = dict(
        model=model, request_specs=specs, max_running_requests=2,
        eos_token_id=-1, device=device, token_budget=128, block_size=16,
        stop_on_eos=False, capture_logits=True, prefix_cache_enabled=False,
        model_namespace=args.model + "@" + args.revision,
    )
    model.set_attention_backend("eager")
    eager = run_engine(**common)
    model.set_attention_backend("flash")
    flash = run_engine(**common)

    errors = {}
    for request_id, reference in eager["first_token_logits"].items():
        candidate = flash["first_token_logits"][request_id]
        errors[request_id] = {
            "max_abs_error": (reference - candidate).abs().max().item(),
            "mean_abs_error": (reference - candidate).abs().mean().item(),
        }
    maximum = max(row["max_abs_error"] for row in errors.values())
    tokens_equal = eager["new_token_ids"] == flash["new_token_ids"]
    if not torch.isfinite(
        torch.stack(list(flash["first_token_logits"].values()))
    ).all():
        raise AssertionError("FlashAttention 产生了非有限 Logits")
    payload = {
        "dtype": str(next(model.parameters()).dtype),
        "prompt_lengths": [len(sequence) for sequence in sequences],
        "max_new_tokens": 4,
        "first_token_logits_error": errors,
        "maximum_error": maximum,
        "generated_tokens_equal": tokens_equal,
        "eager_tokens": eager["new_token_ids"],
        "flash_tokens": flash["new_token_ids"],
    }
    print("最大 Logits 误差: %.6f" % maximum)
    print(
        "平均 Logits 误差:",
        {key: round(value["mean_abs_error"], 6) for key, value in errors.items()},
    )
    print("生成 Token 一致:", tokens_equal)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
    if maximum >= args.max_logits_error:
        raise AssertionError("真实模型 Logits 最大误差过大: %g" % maximum)


if __name__ == "__main__":
    main()
