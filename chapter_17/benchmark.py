#!/usr/bin/env python3
"""Benchmark matched full-conversation and Assistant-only FP32 training steps."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from statistics import mean

import torch

from assistant_loss import assert_only_labels_differ, encode_conversation
from experiment_utils import aggregate_steps, environment_record, write_json
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import MODEL_ID, MODEL_REVISION, build_optimizer_and_scheduler, load_tokenizer_and_model, memory_ledger, train_step


MODES = ("full_conversation", "assistant_only")
TIMING_FIELDS = ("data_time_ms", "h2d_time_ms", "forward_time_ms", "backward_time_ms", "optimizer_time_ms", "step_time_ms")


def measured_step(model, optimizer, scheduler, sample, device, max_grad_norm) -> dict[str, float]:
    data_start = time.perf_counter()
    current_sample = sample
    data_ms = (time.perf_counter() - data_start) * 1000.0
    metrics = train_step(model, optimizer, scheduler, current_sample, device, max_grad_norm)
    compute_ms = metrics["step_time_ms"]
    metrics["data_time_ms"] = data_ms
    metrics["compute_time_ms"] = compute_ms
    metrics["step_time_ms"] = compute_ms + data_ms
    seconds = metrics["step_time_ms"] / 1000.0
    metrics["physical_tokens_per_second"] = current_sample.physical_tokens / seconds
    metrics["non_padding_tokens_per_second"] = current_sample.non_padding_tokens / seconds
    metrics["supervised_tokens_per_second"] = current_sample.supervised_tokens / seconds
    return metrics


def run_mode(args, record, mode: str) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    tokenizer, model = load_tokenizer_and_model(device)
    sample = encode_conversation(tokenizer, record, args.max_length, mode)
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
        run_steps = [
            measured_step(model, optimizer, scheduler, sample, device, args.max_grad_norm)
            for _ in range(args.steps)
        ]
        for index, metrics in enumerate(run_steps, 1):
            metrics["repeat"] = float(repeat + 1)
            metrics["step_in_repeat"] = float(index)
        all_steps.extend(run_steps)
        runs.append({"repeat": repeat + 1, "steps": run_steps, "aggregate": aggregate_steps(run_steps)})

    total_seconds = sum(step["step_time_ms"] for step in all_steps) / 1000.0
    result = {
        "loss_mode": mode,
        "source_line": sample.source_line,
        "runs": runs,
        "aggregate_all_steps": aggregate_steps(all_steps),
        "timing_mean_ms": {field: mean(step[field] for step in all_steps) for field in TIMING_FIELDS},
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
        "supervision_ratio": sample.supervision_ratio,
        "memory": memory_ledger(model, optimizer),
    }
    del optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter17_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-cursor", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.repeats) < 1:
        parser.error("warmup, steps, and repeats must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required for the benchmark")

    torch.cuda.set_device(0)
    records = load_jsonl_records(args.data)
    record = records[args.sample_cursor % len(records)]
    tokenizer = Qwen3Tokenizer(resolve_model_directory(MODEL_ID, MODEL_REVISION))
    baseline_sample = encode_conversation(tokenizer, record, args.max_length, "full_conversation")
    variant_sample = encode_conversation(tokenizer, record, args.max_length, "assistant_only")
    assert_only_labels_differ(baseline_sample, variant_sample)

    objectives = {mode: run_mode(args, record, mode) for mode in MODES}
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
            "formal_dtype": "float32",
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
            "warmup_steps": args.warmup,
            "measurement_steps": args.steps,
            "repeats": args.repeats,
            "objectives": objectives,
            "only_labels_differ": True,
            "timing_scope": {
                "included": ["in_memory_data_fetch", "synchronous_h2d", "forward_and_loss", "backward", "gradient_clipping", "optimizer", "scheduler"],
                "excluded": ["jsonl_read", "chat_template", "tokenization", "checkpoint", "gpu_telemetry_query"],
                "communication_time_ms": 0.0,
                "checkpoint_time_ms": 0.0,
                "cuda_intervals": "CUDA events",
                "end_to_end": "wall clock for data fetch plus synchronized compute step",
            },
        }
    )
    write_json(args.output, payload)
    print(
        "benchmark complete: "
        + " ".join(
            f"{mode}={objectives[mode]['throughput']['supervised_tokens_per_second']:.2f}_supervised_tok_s"
            for mode in MODES
        )
    )


if __name__ == "__main__":
    main()
