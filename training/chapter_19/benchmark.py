#!/usr/bin/env python3
"""Scan shuffle-window sizes for the Chapter 19 length-bucketed sampler."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from statistics import mean

import torch

from experiment_utils import aggregate_steps, environment_record, percentile, write_json
from length_bucket import (
    bucketed_plan,
    batch_length_spreads,
    companion_repeat_ratio,
    project_plan_tokens,
    random_plan,
)
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
    memory_record,
    train_batch_step,
)


TIMING_FIELDS = ("data_time_ms", "h2d_time_ms", "forward_time_ms", "backward_time_ms", "optimizer_time_ms", "step_time_ms")


def make_plan(strategy: str, lengths: list[int], batch_size: int, window: int, seed: int, epoch: int):
    if strategy == "random":
        return random_plan(len(lengths), batch_size, seed, epoch)
    if strategy == "bucketed":
        return bucketed_plan(lengths, batch_size, window, seed, epoch)
    raise ValueError(f"unsupported strategy: {strategy}")


def plan_summary(lengths: list[int], plans: list) -> dict[str, object]:
    projections = [project_plan_tokens(lengths, plan) for plan in plans]
    spreads = [batch_length_spreads(lengths, plan) for plan in plans]
    return {
        "padding_ratio_by_epoch": [projection.padding_ratio for projection in projections],
        "padding_ratio_mean": mean(projection.padding_ratio for projection in projections),
        "physical_tokens_by_epoch": [projection.physical_tokens for projection in projections],
        "non_padding_tokens_by_epoch": [projection.non_padding_tokens for projection in projections],
        "batch_length_spread_mean_by_epoch": [mean(spread) for spread in spreads],
        "work_cv_by_epoch": [projection.work_cv for projection in projections],
        "work_max_mean_ratio_by_epoch": [projection.work_max_mean_ratio for projection in projections],
        "companion_repeat_ratio_epoch1_vs_epoch2": companion_repeat_ratio(plans),
        "plans": [
            {
                "strategy": plan.strategy, "batch_size": plan.batch_size, "window": plan.window,
                "seed": plan.seed, "epoch": plan.epoch, "batches": [list(batch) for batch in plan.batches],
            }
            for plan in plans
        ],
    }


def run_epoch(args, samples, lengths, pad_token_id, plan, warmup_steps: int) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    try:
        for batch_indices in plan.batches[:warmup_steps]:
            host_batch = build_batch(samples, batch_indices, pad_token_id)
            train_batch_step(model, optimizer, host_batch, device, args.max_grad_norm, collect_telemetry=False)
        torch.cuda.reset_peak_memory_stats(device)
        steps: list[dict[str, float]] = []
        for step_index, batch_indices in enumerate(plan.batches[warmup_steps:], start=warmup_steps):
            data_start = time.perf_counter()
            host_batch = build_batch(samples, batch_indices, pad_token_id)
            data_ms = (time.perf_counter() - data_start) * 1000
            metrics = train_batch_step(model, optimizer, host_batch, device, args.max_grad_norm)
            metrics["data_time_ms"] = data_ms
            metrics["step_time_ms"] += data_ms
            metrics["step"] = float(step_index + 1)
            steps.append(metrics)
            del host_batch
        total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
        measured = {
            field: sum(step[field] for step in steps)
            for field in ("physical_tokens", "non_padding_tokens", "supervised_tokens", "padding_tokens", "samples")
        }
        expected_supervised = sum(
            samples[index].supervised_tokens for batch in plan.batches for index in batch
        )
        warmup_supervised = sum(
            samples[index].supervised_tokens for batch in plan.batches[:warmup_steps] for index in batch
        )
        if int(measured["supervised_tokens"]) + warmup_supervised != expected_supervised:
            raise AssertionError(
                f"measured supervised tokens {measured['supervised_tokens']} + warmup {warmup_supervised} "
                f"!= encoded total {expected_supervised}"
            )
        projection_all = project_plan_tokens(lengths, plan)
        warmup_physical = sum(
            len(batch) * max(lengths[index] for index in batch) for batch in plan.batches[:warmup_steps]
        )
        padding_projection_check = int(measured["physical_tokens"]) + warmup_physical == projection_all.physical_tokens
        if not padding_projection_check:
            raise AssertionError(
                f"measured physical tokens {measured['physical_tokens']} + warmup {warmup_physical} "
                f"!= projected {projection_all.physical_tokens}"
            )
        epoch_times = [step["step_time_ms"] for step in steps]
        return {
            "status": "success",
            "warmup_steps": warmup_steps,
            "measured_steps": len(steps),
            "steps": steps,
            "aggregate": aggregate_steps(steps),
            "step_time_p99_ms": percentile([step["step_time_ms"] for step in steps], 0.99),
            "timing_mean_ms": {field: mean(step[field] for step in steps) for field in TIMING_FIELDS},
            "measured_tokens": measured,
            "epoch_supervised_tokens_including_warmup": measured["supervised_tokens"] + warmup_supervised,
            "throughput": {
                "physical_tokens_per_second": measured["physical_tokens"] / total_seconds,
                "non_padding_tokens_per_second": measured["non_padding_tokens"] / total_seconds,
                "supervised_tokens_per_second": measured["supervised_tokens"] / total_seconds,
                "samples_per_second": measured["samples"] / total_seconds,
            },
            "padding_ratio_measured": measured["padding_tokens"] / measured["physical_tokens"],
            "padding_projection_check": padding_projection_check,
            "loss_weighted_by_epoch": sum(step["loss"] * step["supervised_tokens"] for step in steps) / measured["supervised_tokens"],
            "memory": memory_record(model, optimizer),
            "wall_clock_seconds": sum(epoch_times) / 1000,
        }
    except torch.cuda.OutOfMemoryError:
        return {
            "status": "oom",
            "failure_stage": "warmup_or_measured_step",
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter19_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--windows", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0:
        parser.error("batch size must be positive and warmup non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    records = load_jsonl_records(args.data)
    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    samples = encode_samples(tokenizer, records, args.max_length)
    lengths = [sample.non_padding_tokens for sample in samples]
    pad_token_id = tokenizer.pad_token_id
    num_samples = len(samples)
    if num_samples % args.batch_size != 0:
        parser.error(f"sample count {num_samples} must be divisible by batch size for the main scan")
    for window in args.windows:
        if window % args.batch_size or window < args.batch_size:
            parser.error(f"window {window} must be a multiple of batch size {args.batch_size}")

    configurations: list[dict[str, object]] = []
    random_plan_e1 = random_plan(num_samples, args.batch_size, args.seed, 1)
    random_plan_e2 = random_plan(num_samples, args.batch_size, args.seed, 2)
    random_summary = plan_summary(lengths, [random_plan_e1, random_plan_e2])
    random_run = run_epoch(args, samples, lengths, pad_token_id, random_plan_e1, args.warmup)
    configurations.append({
        "strategy": "random", "window": args.batch_size, "num_samples": num_samples,
        "num_batches": random_plan_e1.num_batches,
        "plan_summary": random_summary, "run": random_run,
    })
    if random_run["status"] == "success":
        print(f"random supervised_tok_s={random_run['throughput']['supervised_tokens_per_second']:.2f} padding={random_run['padding_ratio_measured']:.4f}")
    else:
        print(f"random status={random_run['status']} allocated={random_run.get('allocated_bytes', 0) / 2**30:.2f} GiB")

    for window in args.windows:
        plan_e1 = bucketed_plan(lengths, args.batch_size, window, args.seed, 1)
        plan_e2 = bucketed_plan(lengths, args.batch_size, window, args.seed, 2)
        summary = plan_summary(lengths, [plan_e1, plan_e2])
        run = run_epoch(args, samples, lengths, pad_token_id, plan_e1, args.warmup)
        configurations.append({
            "strategy": "bucketed", "window": window, "num_samples": num_samples,
            "num_batches": plan_e1.num_batches, "plan_summary": summary, "run": run,
        })
        if run["status"] == "success":
            print(f"bucketed window={window} supervised_tok_s={run['throughput']['supervised_tokens_per_second']:.2f} padding={run['padding_ratio_measured']:.4f}")
        else:
            print(f"bucketed window={window} status={run['status']}")

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "sample_count": num_samples, "source_lines_prefix": [record["source_line"] for record in records[:5]],
        "dtype": "float32", "formal_dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "world_size": 1, "batch_size": args.batch_size, "windows": args.windows,
        "gradient_accumulation_steps": 1, "reduction": "token_mean",
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "seed": args.seed, "max_length": args.max_length,
        "warmup_steps": args.warmup, "epochs": 1, "repeats": 1,
        "corpus_length_p50": sorted(lengths)[len(lengths) // 2],
        "corpus_length_min": min(lengths), "corpus_length_max": max(lengths),
        "configurations": configurations,
        "timing_scope": {
            "included": ["in_memory_batch_construction", "synchronous_h2d", "forward_and_backward", "gradient_clipping", "optimizer"],
            "excluded": ["jsonl_read", "chat_template", "tokenization", "plan_construction", "checkpoint", "gpu_telemetry_query"],
            "communication_time_ms": 0.0, "checkpoint_time_ms": 0.0,
            "cuda_intervals": "CUDA events", "end_to_end": "synchronized wall clock",
            "warmup_steps_excluded_from_measurement": True,
        },
        "input_pipeline": {
            "encoded_samples_residency": "pageable host memory",
            "pinned_memory": False, "asynchronous_h2d": False,
        },
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}")


if __name__ == "__main__":
    main()
