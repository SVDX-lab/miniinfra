#!/usr/bin/env python3
"""Measure the unoptimized Chapter 16 single-sample baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiment_utils import aggregate_steps, environment_record, write_json
from sft_core import (
    DATASET_FILE,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SHA256,
    encode_conversation,
    load_jsonl_records,
    seed_everything,
)
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_optimizer_and_scheduler,
    load_tokenizer_and_model,
    memory_ledger,
    train_step,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter16_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-cursor", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.repeats) < 1:
        parser.error("warmup, steps, and repeats must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required for the benchmark")

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    tokenizer, model = load_tokenizer_and_model(device)
    records = load_jsonl_records(args.data)
    record = records[args.sample_cursor % len(records)]
    sample = encode_conversation(tokenizer, record, args.max_length)
    total_steps = args.warmup + args.steps * args.repeats
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, args.learning_rate, args.weight_decay, 1, total_steps
    )

    for _ in range(args.warmup):
        train_step(model, optimizer, scheduler, sample, device, args.max_grad_norm)
    torch.cuda.reset_peak_memory_stats(device)

    runs: list[dict[str, object]] = []
    all_steps: list[dict[str, float]] = []
    for repeat in range(args.repeats):
        records_for_run = [
            train_step(model, optimizer, scheduler, sample, device, args.max_grad_norm)
            for _ in range(args.steps)
        ]
        for index, metrics in enumerate(records_for_run, 1):
            metrics["repeat"] = float(repeat + 1)
            metrics["step_in_repeat"] = float(index)
        all_steps.extend(records_for_run)
        runs.append({"repeat": repeat + 1, "steps": records_for_run, "aggregate": aggregate_steps(records_for_run)})

    total_seconds = sum(step["step_time_ms"] for step in all_steps) / 1000.0
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
                "sample_source_lines": [sample.source_line],
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
            "sample_cursor": args.sample_cursor,
            "source_line": sample.source_line,
            "warmup_steps": args.warmup,
            "measurement_steps": args.steps,
            "repeats": args.repeats,
            "runs": runs,
            "aggregate_all_steps": aggregate_steps(all_steps),
            "throughput": {
                "physical_tokens_per_second": sample.physical_tokens * len(all_steps) / total_seconds,
                "non_padding_tokens_per_second": sample.non_padding_tokens * len(all_steps) / total_seconds,
                "supervised_tokens_per_second": sample.supervised_tokens * len(all_steps) / total_seconds,
            },
            "tokens_per_step": {
                "physical": sample.physical_tokens,
                "non_padding": sample.non_padding_tokens,
                "supervised": sample.supervised_tokens,
            },
            "memory": memory_ledger(model, optimizer),
            "timing_scope": {
                "included": ["synchronous_h2d", "forward", "backward", "gradient_clipping", "optimizer", "scheduler"],
                "excluded": ["jsonl_read", "chat_template", "tokenization", "checkpoint", "gpu_telemetry_query"],
                "cuda_intervals": "CUDA events",
                "end_to_end": "wall clock with CUDA synchronization at both boundaries",
            },
        }
    )
    write_json(args.output, payload)
    print(
        f"benchmark complete: mean_step_ms={payload['aggregate_all_steps']['step_time_mean_ms']:.3f} "
        f"supervised_tok_s={payload['throughput']['supervised_tokens_per_second']:.2f}"
    )


if __name__ == "__main__":
    main()
