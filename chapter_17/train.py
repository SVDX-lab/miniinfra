#!/usr/bin/env python3
"""Run matched short FP32 trajectories for full-conversation and Assistant-only loss."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from assistant_loss import assert_only_labels_differ, encode_conversation
from experiment_utils import aggregate_steps, environment_record, write_json
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import MODEL_ID, MODEL_REVISION, build_optimizer_and_scheduler, load_tokenizer_and_model, memory_ledger, train_step


MODES = ("full_conversation", "assistant_only")


def run_mode(args: argparse.Namespace, mode: str, record: dict[str, object]) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    tokenizer, model = load_tokenizer_and_model(device)
    sample = encode_conversation(tokenizer, record, args.max_length, mode)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        args.learning_rate,
        args.weight_decay,
        args.warmup_steps,
        args.steps,
    )
    torch.cuda.reset_peak_memory_stats(device)
    step_records: list[dict[str, float]] = []
    for step in range(args.steps):
        metrics = train_step(model, optimizer, scheduler, sample, device, args.max_grad_norm)
        metrics["step"] = float(step + 1)
        metrics["source_line"] = float(sample.source_line)
        step_records.append(metrics)
        print(f"mode={mode} step={step + 1} loss={metrics['loss']:.6f}")
    result = {
        "loss_mode": mode,
        "source_line": sample.source_line,
        "physical_tokens": sample.physical_tokens,
        "non_padding_tokens": sample.non_padding_tokens,
        "supervised_tokens": sample.supervised_tokens,
        "supervision_ratio": sample.supervision_ratio,
        "steps": step_records,
        "aggregate": aggregate_steps(step_records),
        "memory": memory_ledger(model, optimizer),
    }
    del optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter17_train.jsonl"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-cursor", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path, default=Path("train-results.local.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("Chapter 17 short training requires one CUDA GPU")

    torch.cuda.set_device(0)
    records = load_jsonl_records(args.data)
    record = records[args.sample_cursor % len(records)]
    tokenizer = Qwen3Tokenizer(resolve_model_directory(MODEL_ID, MODEL_REVISION))
    baseline_sample = encode_conversation(tokenizer, record, args.max_length, "full_conversation")
    variant_sample = encode_conversation(tokenizer, record, args.max_length, "assistant_only")
    assert_only_labels_differ(baseline_sample, variant_sample)
    objectives = {mode: run_mode(args, mode, record) for mode in MODES}
    baseline_steps = objectives["full_conversation"]["steps"]
    variant_steps = objectives["assistant_only"]["steps"]
    differences = [
        {
            "step": index + 1,
            "baseline_loss": baseline_steps[index]["loss"],
            "variant_loss": variant_steps[index]["loss"],
            "variant_minus_baseline": variant_steps[index]["loss"] - baseline_steps[index]["loss"],
        }
        for index in range(args.steps)
    ]

    payload = environment_record()
    payload.update(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "dataset": {
                "repo": DATASET_REPO,
                "revision": DATASET_REVISION,
                "file": DATASET_FILE,
                "sha256": DATASET_SHA256,
                "sample_source_lines": [int(record["source_line"])],
            },
            "dtype": "float32",
            "attention_implementation": "eager",
            "tf32": False,
            "seed": args.seed,
            "max_length": args.max_length,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "global_batch_sequences": 1,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_gradient_norm": args.max_grad_norm,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.steps,
            "objectives": objectives,
            "loss_difference_by_step": differences,
            "only_labels_differ": True,
            "timing_scope": {
                "included": ["synchronous_h2d", "forward", "backward", "gradient_clipping", "optimizer", "scheduler"],
                "excluded": ["jsonl_read", "chat_template", "tokenization", "checkpoint", "gpu_telemetry_query"],
            },
        }
    )
    write_json(args.output, payload)
    print(f"saved matched trajectories to {args.output}")


if __name__ == "__main__":
    main()
