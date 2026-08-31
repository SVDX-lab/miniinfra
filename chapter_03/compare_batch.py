"""检查 Batch Size、排列、同批内容和 Padding 对 Decode 输出的影响。"""

import argparse
import json
from pathlib import Path

import torch

from batch_generation import generate_fixed_batch
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="固定 Batch 数值与请求隔离实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt", default="用一句话解释固定批处理。")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--logits-atol", type=float, default=1e-4)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def synthetic_sequence(length, vocab_size, salt):
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def compare_trace(reference, candidate, target_index):
    step_count = min(
        len(reference["logits_trace"]), len(candidate["logits_trace"])
    )
    largest_error = 0.0
    largest_error_before_divergence = 0.0
    mean_error_sum = 0.0
    matching_steps = 0
    first_mismatch_step = None
    mismatch_margin = None
    per_step = []
    for step in range(step_count):
        reference_logits = reference["logits_trace"][step][0]
        candidate_logits = candidate["logits_trace"][step][target_index]
        difference = (reference_logits - candidate_logits).abs()
        max_error = difference.max().item()
        mean_error = difference.mean().item()
        reference_token = int(reference_logits.argmax().item())
        candidate_token = int(candidate_logits.argmax().item())
        top_two = torch.topk(candidate_logits, k=2).values
        margin = float((top_two[0] - top_two[1]).item())
        matches = reference_token == candidate_token
        matching_steps += int(matches)
        largest_error = max(largest_error, max_error)
        if first_mismatch_step is None:
            largest_error_before_divergence = max(
                largest_error_before_divergence, max_error
            )
        mean_error_sum += mean_error
        if not matches and first_mismatch_step is None:
            first_mismatch_step = step + 1
            mismatch_margin = margin
        per_step.append(
            {
                "step": step + 1,
                "reference_token": reference_token,
                "candidate_token": candidate_token,
                "matches": matches,
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
                "candidate_top1_top2_margin": margin,
            }
        )
    return {
        "steps": step_count,
        "top1_match_rate": matching_steps / step_count if step_count else 0.0,
        "sequence_matches": first_mismatch_step is None,
        "first_mismatch_step": first_mismatch_step,
        "first_mismatch_margin": mismatch_margin,
        "largest_logits_abs_error": largest_error,
        "largest_logits_abs_error_before_divergence": (
            largest_error_before_divergence
        ),
        "average_mean_logits_abs_error": (
            mean_error_sum / step_count if step_count else 0.0
        ),
        "per_step": per_step,
    }


def main():
    args = parse_args()
    if args.max_new_tokens < 1 or args.logits_atol < 0:
        raise ValueError("输出长度必须大于 0，误差阈值不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    target = tokenizer.encode_chat_prompt(args.prompt)
    equal_b = synthetic_sequence(len(target), len(tokenizer.vocab), 11)
    equal_c = synthetic_sequence(len(target), len(tokenizer.vocab), 29)
    equal_d = synthetic_sequence(len(target), len(tokenizer.vocab), 47)
    short = synthetic_sequence(max(1, len(target) // 2), len(tokenizer.vocab), 61)
    long = synthetic_sequence(len(target) + 17, len(tokenizer.vocab), 79)
    variants = [
        ("single", [target], 0),
        ("pair_content_b", [target, equal_b], 0),
        ("pair_content_c", [target, equal_c], 0),
        ("pair_reordered", [equal_b, target], 1),
        ("quad", [target, equal_b, equal_c, equal_d], 0),
        ("quad_reversed", [equal_d, equal_c, equal_b, target], 3),
        ("pair_short_first", [short, target], 1),
        ("pair_long_second", [target, long], 0),
    ]
    comparison_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = load_handwritten_model(model_directory, device, dtype=comparison_dtype)

    reference = generate_fixed_batch(
        model, [target], args.max_new_tokens, tokenizer.eos_token_id,
        device, stop_on_eos=False, capture_logits=True,
    )
    rows = []
    variant_runs = {}
    for name, sequences, target_index in variants:
        candidate = reference if name == "single" else generate_fixed_batch(
            model, sequences, args.max_new_tokens, tokenizer.eos_token_id,
            device, stop_on_eos=False, capture_logits=True,
        )
        variant_runs[name] = candidate
        comparison = compare_trace(reference, candidate, target_index)
        comparison.update(
            {
                "variant": name,
                "batch_size": len(sequences),
                "target_index": target_index,
                "prompt_lengths": [len(sequence) for sequence in sequences],
            }
        )
        rows.append(comparison)
        print(
            "%s: batch=%d, top1=%.2f%%, sequence_match=%s, "
            "first_mismatch=%s, pre_divergence_max_error=%.6f, max_error=%.6f"
            % (
                name,
                len(sequences),
                comparison["top1_match_rate"] * 100,
                comparison["sequence_matches"],
                comparison["first_mismatch_step"],
                comparison["largest_logits_abs_error_before_divergence"],
                comparison["largest_logits_abs_error"],
            )
        )

    isolation_pairs = [
        ("pair_content_isolation", "pair_content_b", "pair_content_c", 0),
        ("quad_order_isolation", "quad", "quad_reversed", 3),
    ]
    for name, reference_name, candidate_name, candidate_index in isolation_pairs:
        comparison = compare_trace(
            variant_runs[reference_name],
            variant_runs[candidate_name],
            candidate_index,
        )
        comparison.update(
            {
                "variant": name,
                "reference_variant": reference_name,
                "candidate_variant": candidate_name,
                "target_index": candidate_index,
            }
        )
        rows.append(comparison)
        print(
            "%s: top1=%.2f%%, sequence_match=%s, max_error=%.6f"
            % (
                name,
                comparison["top1_match_rate"] * 100,
                comparison["sequence_matches"],
                comparison["largest_logits_abs_error"],
            )
        )

    environment = {
        "gpu": torch.cuda.get_device_name(device),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "prompt_tokens": len(target),
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "thinking": False,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"environment": environment, "results": rows},
                file, ensure_ascii=False, indent=2,
            )
        print("JSON 结果已写入:", output_path)

    if args.dtype == "float32":
        failures = [
            row for row in rows
            if not row["sequence_matches"]
            or row["largest_logits_abs_error"] > args.logits_atol
        ]
        if failures:
            print("float32 请求隔离或数值误差检查未通过")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
