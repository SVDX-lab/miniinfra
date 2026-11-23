#!/usr/bin/env python3
"""Chapter 28 benchmark: eager versus compiled steady-state training.

Arms over the same corpus and shuffle (FP32, eager attention):

- ``eager_natural``  natural variable-length batches (the chapter 07 body)
- ``eager_fixed``    the same batches padded to a fixed 512 tokens
- ``compiled_fixed`` the fixed-shape batches through ``torch.compile``

The compiled arm reports its first step (trace + codegen) separately from the
measured window; steady-state throughput is the comparison that matters, and
the break-even step count is computed in the summary.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from statistics import fmean

import torch

from experiment_utils import environment_record, percentile, write_json
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
    make_batch_plan,
    pad_batch_to,
    to_device_batch,
    train_batch_step,
)


def run_epoch(args, samples, plan, device, arm) -> dict:
    seed_everything(args.seed)
    _, raw_model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(raw_model, args.learning_rate, args.weight_decay)
    if arm == "compiled_fixed":
        model = torch.compile(raw_model)
    else:
        model = raw_model

    def make_batch(index):
        batch = build_batch(samples, plan[index], args.pad_token_id)
        if arm == "eager_natural":
            return to_device_batch(batch, device)
        return to_device_batch(pad_batch_to(batch, args.max_length, args.pad_token_id), device)

    steps = []
    first_step_ms = None
    try:
        compile_start = time.perf_counter()
        train_batch_step(model, optimizer, make_batch(0), device, args.max_grad_norm, collect_telemetry=False)
        first_step_ms = (time.perf_counter() - compile_start) * 1000
        for warmup_index in range(1, args.warmup):
            train_batch_step(model, optimizer, make_batch(warmup_index), device, args.max_grad_norm, collect_telemetry=False)
        torch.cuda.reset_peak_memory_stats(device)
        for step_index, batch_index in enumerate(range(args.warmup, len(plan)), start=1):
            batch = make_batch(batch_index)
            metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=True)
            metrics["step"] = float(step_index)
            steps.append(metrics)
        total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
        supervised = sum(step["supervised_tokens"] for step in steps)
        step_times = [step["step_time_ms"] for step in steps]
        return {
            "status": "success",
            "arm": arm,
            "first_step_ms_including_compile": first_step_ms,
            "warmup_steps": args.warmup,
            "measured_steps": len(steps),
            "steps": steps,
            "aggregate": {
                "step_time_mean_ms": fmean(step_times),
                "step_time_p50_ms": percentile(step_times, 0.5),
                "step_time_p95_ms": percentile(step_times, 0.95),
                "step_time_max_ms": max(step_times),
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
        del model, optimizer, raw_model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter28_train.jsonl"))
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
        for arm in ("eager_natural", "eager_fixed", "compiled_fixed"):
            result = run_epoch(args, samples, plan, device, arm)
            repeats.append({"repeat": repeat, "arm": arm, "result": result})
            if result["status"] == "success":
                print(
                    f"repeat={repeat} {arm}: sup/s={result['throughput']['supervised_tokens_per_second']:.1f} "
                    f"p50={result['aggregate']['step_time_p50_ms']:.1f}ms "
                    f"first_step={result['first_step_ms_including_compile']:.0f}ms "
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
            "first_step_ms_first_repeat": successful[0]["first_step_ms_including_compile"],
            "first_step_ms_last_repeat": successful[-1]["first_step_ms_including_compile"],
            "peak_allocated_bytes_mean": fmean(r["memory"]["peak_allocated_bytes"] for r in successful),
        }

    summary = {arm: summarize(arm) for arm in ("eager_natural", "eager_fixed", "compiled_fixed")}
    eager_steady = summary["eager_fixed"]["step_time_p50_ms_mean"]
    compiled_steady = summary["compiled_fixed"]["step_time_p50_ms_mean"]
    compile_overhead = summary["compiled_fixed"]["first_step_ms_first_repeat"] - compiled_steady
    per_step_saving = eager_steady - compiled_steady
    break_even = compile_overhead / per_step_saving if per_step_saving > 0 else None
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
        "arms": ["eager_natural", "eager_fixed", "compiled_fixed"],
        "num_samples": len(samples),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "repeats_detail": repeats,
        "summary": summary,
        "break_even_analysis": {
            "eager_fixed_p50_ms": eager_steady,
            "compiled_fixed_p50_ms": compiled_steady,
            "per_step_saving_ms": per_step_saving,
            "compile_overhead_ms": compile_overhead,
            "break_even_steps": break_even,
        },
        "timing_scope": {
            "measured_window": "steps after warmup; steady state only",
            "first_step": "reported separately, includes dynamo tracing and inductor codegen on the first repeat",
            "excluded": ["host padding", "gpu telemetry"],
        },
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
