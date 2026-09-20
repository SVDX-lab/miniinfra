#!/usr/bin/env python3
"""Chapter 23 benchmark: FP32 baseline versus BF16 autocast, 3 repeats each.

Each repeat trains one full epoch with the shared step body in one precision
mode.  CUDA events split H2D, forward, backward and clip+optimizer; peak CUDA
memory and GPU telemetry (power/temperature/clock) are recorded per step.
"""

from __future__ import annotations

import argparse
import gc
import resource
from pathlib import Path
from statistics import fmean

import torch

from experiment_utils import environment_record, percentile, write_json
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    PRECISION_MODES,
    build_batch,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
    make_batch_plan,
    train_batch_step,
)


def run_epoch(args, samples, plan, device, mode) -> dict:
    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    steps = []
    try:
        for warmup_index in range(args.warmup):
            batch = build_batch(samples, plan[warmup_index], args.pad_token_id)
            train_batch_step(model, optimizer, batch, device, args.max_grad_norm, mode=mode, collect_telemetry=False)
        torch.cuda.reset_peak_memory_stats(device)
        for step_index, batch_indices in enumerate(plan[args.warmup :], start=1):
            batch = build_batch(samples, batch_indices, args.pad_token_id)
            metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm, mode=mode)
            metrics["step"] = float(step_index)
            steps.append(metrics)
        total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
        supervised = sum(step["supervised_tokens"] for step in steps)
        step_times = [step["step_time_ms"] for step in steps]
        return {
            "status": "success",
            "mode": mode,
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
                "h2d": fmean(step["h2d_time_ms"] for step in steps),
                "forward": fmean(step["forward_time_ms"] for step in steps),
                "backward": fmean(step["backward_time_ms"] for step in steps),
                "optimizer": fmean(step["optimizer_time_ms"] for step in steps),
            },
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
                "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            },
        }
    except torch.cuda.OutOfMemoryError:
        return {"status": "oom", "mode": mode, "allocated_bytes": torch.cuda.memory_allocated()}
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter23_train.jsonl"))
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
    samples = encode_samples(tokenizer, records, args.max_length)
    plan = make_batch_plan(len(samples), args.batch_size, args.seed)

    repeats = []
    for repeat in range(1, args.repeats + 1):
        for mode in PRECISION_MODES:
            result = run_epoch(args, samples, plan, device, mode)
            repeats.append({"repeat": repeat, "mode": mode, "result": result})
            if result["status"] == "success":
                print(
                    f"repeat={repeat} {mode}: sup/s={result['throughput']['supervised_tokens_per_second']:.1f} "
                    f"fwd={result['phase_means_ms']['forward']:.1f}ms bwd={result['phase_means_ms']['backward']:.1f}ms "
                    f"opt={result['phase_means_ms']['optimizer']:.1f}ms "
                    f"peak={result['memory']['peak_allocated_bytes'] / 2**30:.2f}GiB",
                    flush=True,
                )
            else:
                print(f"repeat={repeat} {mode}: {result['status']}", flush=True)

    def summarize(mode):
        successful = [entry["result"] for entry in repeats if entry["mode"] == mode and entry["result"]["status"] == "success"]
        return {
            "successful_repeats": len(successful),
            "step_time_p50_ms_mean": fmean(r["aggregate"]["step_time_p50_ms"] for r in successful),
            "supervised_tokens_per_second_mean": fmean(r["throughput"]["supervised_tokens_per_second"] for r in successful),
            "forward_ms_mean": fmean(r["phase_means_ms"]["forward"] for r in successful),
            "backward_ms_mean": fmean(r["phase_means_ms"]["backward"] for r in successful),
            "optimizer_ms_mean": fmean(r["phase_means_ms"]["optimizer"] for r in successful),
            "peak_allocated_bytes_mean": fmean(r["memory"]["peak_allocated_bytes"] for r in successful),
        }

    summary = {mode: summarize(mode) for mode in PRECISION_MODES}
    speedup = summary["fp32"]["step_time_p50_ms_mean"] / summary["bf16_autocast"]["step_time_p50_ms_mean"]
    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "precision_arms": list(PRECISION_MODES),
        "bf16_policy": {
            "forward_loss": "autocast bfloat16",
            "master_parameters": "float32",
            "gradients": "float32",
            "optimizer_states": "float32",
            "grad_scaler": "none (BF16 does not need one)",
        },
        "tf32": False,
        "attention_implementation": "eager",
        "world_size": 1,
        "batch_size": args.batch_size,
        "num_samples": len(samples),
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "supervised_tokens_per_epoch": sum(
            batch.supervised_tokens for batch in (build_batch(samples, indices, args.pad_token_id) for indices in plan)
        ),
        "repeats_detail": repeats,
        "summary": summary,
        "step_time_speedup_fp32_over_bf16": speedup,
        "timing_scope": {
            "h2d": "host batch to device copy",
            "forward": "forward plus token_mean loss",
            "backward": "loss.backward()",
            "optimizer": "clip_grad_norm_ plus AdamW step",
            "excluded": ["corpus encode", "batch padding on host", "gpu telemetry query"],
        },
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
