#!/usr/bin/env python3
"""Scan fixed mini-batch sizes for serial and padded FP32 execution."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from statistics import mean

import torch

from assistant_loss import encode_conversation
from experiment_utils import aggregate_steps, environment_record, write_json
from mini_batch import pad_samples, trim_sample
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import MODEL_ID, MODEL_REVISION, build_optimizer, load_tokenizer_and_model, memory_record, train_group_step


TIMING_FIELDS = ("data_time_ms", "h2d_time_ms", "forward_time_ms", "backward_time_ms", "optimizer_time_ms", "step_time_ms")


class BenchmarkOOM(RuntimeError):
    def __init__(self, allocated_bytes: int, reserved_bytes: int) -> None:
        super().__init__("CUDA out of memory")
        self.allocated_bytes = allocated_bytes
        self.reserved_bytes = reserved_bytes


def _clear() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_configuration(args, batch, execution: str) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    try:
        for _ in range(args.warmup):
            train_group_step(model, optimizer, batch, device, execution, args.reduction, args.max_grad_norm, collect_telemetry=False)
        torch.cuda.reset_peak_memory_stats(device)
        steps: list[dict[str, float]] = []
        for step_index in range(args.steps):
            data_start = time.perf_counter()
            current_batch = batch
            data_ms = (time.perf_counter() - data_start) * 1000
            metrics = train_group_step(model, optimizer, current_batch, device, execution, args.reduction, args.max_grad_norm)
            metrics["data_time_ms"] = data_ms
            metrics["step_time_ms"] += data_ms
            metrics["step"] = float(step_index + 1)
            steps.append(metrics)
        total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
        physical = sum(step["physical_tokens"] for step in steps)
        non_padding = sum(step["non_padding_tokens"] for step in steps)
        supervised = sum(step["supervised_tokens"] for step in steps)
        return {
            "status": "success", "execution": execution, "batch_size": batch.batch_size,
            "source_lines": list(batch.source_lines), "sequence_lengths": list(batch.lengths),
            "tokens_per_step": {"physical": steps[0]["physical_tokens"], "non_padding": batch.non_padding_tokens, "supervised": batch.supervised_tokens, "padding": steps[0]["padding_tokens"]},
            "padding_ratio": steps[0]["padding_ratio"], "steps": steps,
            "aggregate": aggregate_steps(steps),
            "timing_mean_ms": {field: mean(step[field] for step in steps) for field in TIMING_FIELDS},
            "throughput": {"physical_tokens_per_second": physical / total_seconds, "non_padding_tokens_per_second": non_padding / total_seconds, "supervised_tokens_per_second": supervised / total_seconds, "samples_per_second": batch.batch_size * len(steps) / total_seconds},
            "memory": memory_record(model, optimizer),
        }
    except torch.cuda.OutOfMemoryError:
        raise BenchmarkOOM(torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) from None
    finally:
        del optimizer, model
        _clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter18_train.jsonl"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sample-cursor", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--reduction", choices=["token_mean", "sample_mean"], default="token_mean")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.warmup < 1 or args.steps < 1:
        parser.error("batch sizes, warmup, and steps must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = load_jsonl_records(args.data)
    maximum_batch = max(args.batch_sizes)
    selected_records = records[args.sample_cursor : args.sample_cursor + maximum_batch]
    if len(selected_records) < maximum_batch:
        parser.error(f"data needs at least {maximum_batch} records")
    tokenizer = Qwen3Tokenizer(resolve_model_directory(MODEL_ID, MODEL_REVISION))
    samples = [trim_sample(encode_conversation(tokenizer, record, args.max_length, "assistant_only")) for record in selected_records]
    configurations: list[dict[str, object]] = []
    stop_batched_after_oom = False
    for batch_size in args.batch_sizes:
        group = pad_samples(samples[:batch_size], tokenizer.pad_token_id)
        for execution in ("serial", "batched"):
            if execution == "batched" and stop_batched_after_oom:
                configurations.append({"status": "not_run_after_oom", "execution": execution, "batch_size": batch_size})
                continue
            try:
                result = run_configuration(args, group, execution)
                configurations.append(result)
                print(f"batch_size={batch_size} execution={execution} supervised_tok_s={result['throughput']['supervised_tokens_per_second']:.2f}")
            except BenchmarkOOM as error:
                configurations.append({
                    "status": "oom", "execution": execution, "batch_size": batch_size,
                    "failure_stage": "warmup_or_training_step",
                    "allocated_bytes": error.allocated_bytes, "reserved_bytes": error.reserved_bytes,
                    "source_lines": list(group.source_lines), "sequence_lengths": list(group.lengths),
                })
                _clear()
                if execution == "batched":
                    stop_batched_after_oom = True
                print(f"batch_size={batch_size} execution={execution} status=oom")
    successful_batches = [record for record in configurations if record.get("execution") == "batched" and record.get("status") == "success"]
    best = max(successful_batches, key=lambda record: record["throughput"]["supervised_tokens_per_second"])
    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256, "sample_source_lines": [int(record["source_line"]) for record in selected_records]},
        "dtype": "float32", "formal_dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "world_size": 1, "micro_batch_sizes": args.batch_sizes, "gradient_accumulation_steps": 1,
        "reduction": args.reduction, "seed": args.seed, "max_length": args.max_length, "sample_cursor": args.sample_cursor,
        "warmup_steps": args.warmup, "measurement_steps": args.steps, "repeats": 1,
        "configurations": configurations,
        "best_successful_batched_size_by_supervised_tokens_per_second": best["batch_size"],
        "timing_scope": {
            "included": ["in_memory_group_fetch", "synchronous_h2d", "forward_and_backward", "gradient_clipping", "optimizer"],
            "excluded": ["jsonl_read", "chat_template", "tokenization", "checkpoint", "gpu_telemetry_query"],
            "communication_time_ms": 0.0, "checkpoint_time_ms": 0.0,
            "cuda_intervals": "CUDA events", "end_to_end": "synchronized wall clock plus in-memory fetch",
        },
        "input_pipeline": {
            "encoded_samples_residency": "pageable host memory",
            "pinned_memory": False,
            "asynchronous_h2d": False,
            "disk_read_bandwidth": "not measured; JSONL reading is outside the timed region",
        },
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}")


if __name__ == "__main__":
    main()
