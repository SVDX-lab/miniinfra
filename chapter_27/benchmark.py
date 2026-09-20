#!/usr/bin/env python3
"""Chapter 27 benchmark: loop versus foreach versus fused AdamW.

Three repeats of one full training epoch per arm, FP32 eager, identical
forward/backward everywhere.  CUDA events split the optimizer window into
``clip_grad_norm_`` and ``optimizer.step()`` so the kernel organization has
its own bucket.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from statistics import fmean

import torch

from experiment_utils import environment_record, percentile, write_json
from fused_optimizer import build_optimizer
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    encode_samples,
    load_tokenizer_and_model,
    make_batch_plan,
    to_device_batch,
    train_batch_step,
)


def run_epoch(args, samples, plan, device, arm) -> dict:
    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay, arm)
    steps = []
    try:
        for warmup_index in range(args.warmup):
            batch = to_device_batch(build_batch(samples, plan[warmup_index], args.pad_token_id), device)
            train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
        torch.cuda.reset_peak_memory_stats(device)
        for step_index, batch_indices in enumerate(plan[args.warmup :], start=1):
            batch = to_device_batch(build_batch(samples, batch_indices, args.pad_token_id), device)
            metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=True)
            metrics["step"] = float(step_index)
            steps.append(metrics)
        total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
        supervised = sum(step["supervised_tokens"] for step in steps)
        step_times = [step["step_time_ms"] for step in steps]
        return {
            "status": "success",
            "arm": arm,
            "warmup_steps": args.warmup,
            "measured_steps": len(steps),
            "steps": steps,
            "aggregate": {
                "step_time_mean_ms": fmean(step_times),
                "step_time_p50_ms": percentile(step_times, 0.5),
                "step_time_p95_ms": percentile(step_times, 0.95),
                "step_time_max_ms": max(step_times),
            },
            "phase_means_ms": {
                "forward": fmean(step["forward_time_ms"] for step in steps),
                "backward": fmean(step["backward_time_ms"] for step in steps),
                "clip": fmean(step["clip_time_ms"] for step in steps),
                "optimizer_step": fmean(step["optimizer_step_time_ms"] for step in steps),
                "optimizer_window": fmean(step["optimizer_time_ms"] for step in steps),
            },
            "throughput": {
                "supervised_tokens_per_second": supervised / total_seconds,
                "physical_tokens_per_second": sum(step["physical_tokens"] for step in steps) / total_seconds,
                "non_padding_tokens_per_second": sum(step["non_padding_tokens"] for step in steps) / total_seconds,
            },
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        }
    except torch.cuda.OutOfMemoryError:
        return {"status": "oom", "arm": arm, "allocated_bytes": torch.cuda.memory_allocated()}
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter27_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them (counted). Same pattern
        # as chapter_24/chapter_39 benchmark.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
        print(
            f"encoded {len(samples)} samples, filtered_no_supervision={filtered_no_supervision}",
            flush=True,
        )
    plan = make_batch_plan(len(samples), args.batch_size, args.seed)

    repeats = []
    for repeat in range(1, args.repeats + 1):
        for arm in ("loop", "foreach", "fused"):
            result = run_epoch(args, samples, plan, device, arm)
            repeats.append({"repeat": repeat, "arm": arm, "result": result})
            if result["status"] == "success":
                print(
                    f"repeat={repeat} {arm}: sup/s={result['throughput']['supervised_tokens_per_second']:.1f} "
                    f"clip={result['phase_means_ms']['clip']:.2f}ms "
                    f"adamw_step={result['phase_means_ms']['optimizer_step']:.2f}ms "
                    f"step_p50={result['aggregate']['step_time_p50_ms']:.1f}ms",
                    flush=True,
                )
            else:
                print(f"repeat={repeat} {arm}: {result['status']}", flush=True)

    def summarize(arm):
        successful = [entry["result"] for entry in repeats if entry["arm"] == arm and entry["result"]["status"] == "success"]
        if not successful:
            # 12GB calibration: a tier where every repeat OOMs must still
            # produce a JSON record instead of crashing the aggregation.
            oom = [entry["result"] for entry in repeats if entry["arm"] == arm and entry["result"]["status"] == "oom"]
            return {
                "successful_repeats": 0,
                "status": "all_oom",
                "oom_allocated_bytes_max": max((entry["allocated_bytes"] for entry in oom), default=None),
            }
        return {
            "successful_repeats": len(successful),
            "step_time_p50_ms_mean": fmean(r["aggregate"]["step_time_p50_ms"] for r in successful),
            "supervised_tokens_per_second_mean": fmean(r["throughput"]["supervised_tokens_per_second"] for r in successful),
            "clip_ms_mean": fmean(r["phase_means_ms"]["clip"] for r in successful),
            "optimizer_step_ms_mean": fmean(r["phase_means_ms"]["optimizer_step"] for r in successful),
            "optimizer_window_ms_mean": fmean(r["phase_means_ms"]["optimizer_window"] for r in successful),
            "peak_allocated_bytes_mean": fmean(r["memory"]["peak_allocated_bytes"] for r in successful),
        }

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32",
        "tf32": False,
        "attention_implementation": "eager",
        "arms": {"loop": "foreach=False fused=False", "foreach": "foreach=True", "fused": "fused=True"},
        "num_samples": len(samples),
        "filtered_no_supervision": filtered_no_supervision,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "repeats_detail": repeats,
        "summary": {arm: summarize(arm) for arm in ("loop", "foreach", "fused")},
        "timing_scope": {
            "clip": "clip_grad_norm_ only (identical in every arm)",
            "optimizer_step": "optimizer.step() only",
            "optimizer_window": "clip plus step - comparable with chapters 07/08 optimizer_time_ms",
            "excluded": ["host padding", "gpu telemetry"],
        },
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
