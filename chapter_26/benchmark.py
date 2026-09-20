#!/usr/bin/env python3
"""Chapter 26 benchmark: eager versus SDPA math/mem-efficient attention.

Three repeats of one full training epoch per arm, FP32 with TF32 off, same
corpus and shuffle as the other chapters.  The flash backend has no FP32
kernel and cannot consume padding masks, so it is measured only inside the
correctness script's cross-precision arm, never here.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from statistics import fmean

import torch

from experiment_utils import environment_record, percentile, write_json
from sdpa_attention import set_attention_implementation
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
    make_batch_plan,
    to_device_batch,
    train_batch_step,
)


def run_epoch(args, samples, plan, device, arm) -> dict:
    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device, attention=arm)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
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
    parser.add_argument("--data", type=Path, default=Path("data/chapter26_train.jsonl"))
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
        for arm in ("eager", "sdpa_math", "sdpa_mem_efficient"):
            result = run_epoch(args, samples, plan, device, arm)
            repeats.append({"repeat": repeat, "arm": arm, "result": result})
            if result["status"] == "success":
                print(
                    f"repeat={repeat} {arm}: sup/s={result['throughput']['supervised_tokens_per_second']:.1f} "
                    f"fwd={result['phase_means_ms']['forward']:.1f}ms bwd={result['phase_means_ms']['backward']:.1f}ms "
                    f"peak={result['memory']['peak_allocated_bytes'] / 2**30:.2f}GiB",
                    flush=True,
                )
            else:
                print(f"repeat={repeat} {arm}: {result['status']}", flush=True)

    def summarize(arm):
        successful = [entry["result"] for entry in repeats if entry["arm"] == arm and entry["result"]["status"] == "success"]
        return {
            "successful_repeats": len(successful),
            "step_time_p50_ms_mean": fmean(r["aggregate"]["step_time_p50_ms"] for r in successful),
            "supervised_tokens_per_second_mean": fmean(r["throughput"]["supervised_tokens_per_second"] for r in successful),
            "forward_ms_mean": fmean(r["phase_means_ms"]["forward"] for r in successful),
            "backward_ms_mean": fmean(r["phase_means_ms"]["backward"] for r in successful),
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
        "arms": ["eager", "sdpa_math", "sdpa_mem_efficient"],
        "flash_note": "no FP32 kernel and no additive-mask support; measured only in the BF16 cross-precision arm of compare_correctness.py",
        "num_samples": len(samples),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "repeats_detail": repeats,
        "summary": {arm: summarize(arm) for arm in ("eager", "sdpa_math", "sdpa_mem_efficient")},
        "timing_scope": {
            "forward": "forward plus token_mean loss",
            "backward": "loss.backward()",
            "excluded": ["host padding", "gpu telemetry"],
        },
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
