#!/usr/bin/env python3
"""Chapter 24 benchmark: big-batch steps versus gradient accumulation.

Arms (FP32, eager attention, same corpus and shuffle):

- ``big4``       one padded batch of 4 samples per optimizer step (baseline)
- ``acc_2x2``    B_eff=4 via two 2-sample micro-batches
- ``acc_4x1``    B_eff=4 via four 1-sample micro-batches
- ``acc_4x4``    B_eff=16 via four 4-sample micro-batches (capacity arm)

Throughput is supervised tokens per wall-clock second over the measured
optimizer steps; peak CUDA memory per arm shows what accumulation does *not*
cost.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from statistics import fmean

import torch

from experiment_utils import environment_record, percentile, write_json
from gradient_accumulation import accumulated_step
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
    make_batch_plan,
    make_step_plan,
    split_effective_batch,
    to_device_batch,
    train_batch_step,
)


def move_micros(samples, members, micro_size, pad_token_id, device):
    return [
        to_device_batch(batch, device)
        for batch in split_effective_batch(samples, members, micro_size, pad_token_id)
    ]


def run_epoch(args, samples, device, arm) -> dict:
    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    micro_size = {"big4": None, "acc_2x2": 2, "acc_4x1": 1, "acc_4x4": 4}[arm]
    effective = 16 if arm == "acc_4x4" else 4
    if micro_size is None:
        plan = make_batch_plan(len(samples), 4, args.seed)
    else:
        plan = make_step_plan(len(samples), effective, args.seed)
    steps = []
    try:
        for members in plan[: args.warmup]:
            if micro_size is None:
                batch = to_device_batch(build_batch(samples, members, args.pad_token_id), device)
                train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
            else:
                micros = move_micros(samples, members, micro_size, args.pad_token_id, device)
                accumulated_step(model, optimizer, micros, device, args.max_grad_norm)
        torch.cuda.reset_peak_memory_stats(device)
        for step_index, members in enumerate(plan[args.warmup :], start=1):
            if micro_size is None:
                batch = to_device_batch(build_batch(samples, members, args.pad_token_id), device)
                metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=True)
            else:
                micros = move_micros(samples, members, micro_size, args.pad_token_id, device)
                metrics = accumulated_step(model, optimizer, micros, device, args.max_grad_norm, collect_telemetry=True, telemetry_fn=_telemetry_from_training)
            metrics["step"] = float(step_index)
            steps.append(metrics)
        total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
        supervised = sum(step["supervised_tokens"] for step in steps)
        step_times = [step["step_time_ms"] for step in steps]
        return {
            "status": "success",
            "arm": arm,
            "effective_batch": effective,
            "micro_batch_size": micro_size or 4,
            "warmup_steps": args.warmup,
            "measured_optimizer_steps": len(steps),
            "steps": steps,
            "aggregate": {
                "step_time_mean_ms": fmean(step_times),
                "step_time_p50_ms": percentile(step_times, 0.5),
                "step_time_p95_ms": percentile(step_times, 0.95),
                "step_time_max_ms": max(step_times),
            },
            "phase_means_ms": {
                "optimizer": fmean(step["optimizer_time_ms"] for step in steps),
                "micro_total": fmean(step.get("micro_total_time_ms", 0.0) for step in steps),
            },
            "padding_ratio_mean": fmean(step["padding_ratio"] for step in steps),
            "throughput": {
                "supervised_tokens_per_second": supervised / total_seconds,
                "physical_tokens_per_second": sum(step["physical_tokens"] for step in steps) / total_seconds,
                "non_padding_tokens_per_second": sum(step["non_padding_tokens"] for step in steps) / total_seconds,
            },
            "telemetry_means": {
                "power_watts": fmean(step["power_watts"] for step in steps),
                "temperature_celsius": fmean(step["temperature_celsius"] for step in steps),
                "sm_clock_mhz": fmean(step["sm_clock_mhz"] for step in steps),
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


def _telemetry_from_training(device):
    from training import _telemetry

    return _telemetry(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter24_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
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

    # 12GB calibration: drop trailing samples so the filtered corpus divides
    # into the hardcoded effective-batch groups (16). Same as chapter_24
    # compare_correctness.
    usable = len(samples) - len(samples) % 16
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(f"using {usable} samples, dropped_for_divisibility={dropped_for_divisibility}", flush=True)
    repeats = []
    for repeat in range(1, args.repeats + 1):
        for arm in ("big4", "acc_2x2", "acc_4x1", "acc_4x4"):
            result = run_epoch(args, samples, device, arm)
            repeats.append({"repeat": repeat, "arm": arm, "result": result})
            if result["status"] == "success":
                print(
                    f"repeat={repeat} {arm}: sup/s={result['throughput']['supervised_tokens_per_second']:.1f} "
                    f"step_p50={result['aggregate']['step_time_p50_ms']:.1f}ms "
                    f"pad={result['padding_ratio_mean']:.3f} "
                    f"peak={result['memory']['peak_allocated_bytes'] / 2**30:.2f}GiB",
                    flush=True,
                )
            else:
                print(f"repeat={repeat} {arm}: {result['status']}", flush=True)

    def summarize(arm):
        successful = [entry["result"] for entry in repeats if entry["arm"] == arm and entry["result"]["status"] == "success"]
        # OOM 臂是有效结果而不是错误：成功次数为 0 时如实汇报。
        if not successful:
            oom = [entry["result"] for entry in repeats if entry["arm"] == arm and entry["result"]["status"] == "oom"]
            return {
                "successful_repeats": 0,
                "oom_repeats": len(oom),
                "allocated_at_first_oom_bytes": oom[0]["allocated_bytes"] if oom else None,
            }
        return {
            "successful_repeats": len(successful),
            "step_time_p50_ms_mean": fmean(r["aggregate"]["step_time_p50_ms"] for r in successful),
            "supervised_tokens_per_second_mean": fmean(r["throughput"]["supervised_tokens_per_second"] for r in successful),
            "padding_ratio_mean": fmean(r["padding_ratio_mean"] for r in successful),
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
        "arms": {"big4": "effective_batch=4 big batch", "acc_2x2": "effective_batch=4, 2 micro x 2", "acc_4x1": "effective_batch=4, 4 micro x 1", "acc_4x4": "effective_batch=16, 4 micro x 4"},
        "num_samples": len(samples),
        "filtered_no_supervision": filtered_no_supervision,
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "repeats_detail": repeats,
        "summary": {arm: summarize(arm) for arm in ("big4", "acc_2x2", "acc_4x1", "acc_4x4")},
        "timing_scope": {
            "step_time": "all micro forward/backward plus final clip and optimizer step, wall clock around synchronized events",
            "excluded": ["host padding of future batches", "gpu telemetry query"],
        },
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
