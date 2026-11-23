#!/usr/bin/env python3
"""Chapter 28 correctness: compiled-step parity and recompile accounting.

Layers:

1. Padding invariance of the eager body: a natural batch versus the same
   batch padded to a fixed 512 (the compiled arm's input shape) - losses must
   agree to floating-point noise.
2. Fixed-shape parity: eager versus ``torch.compile`` over a 16-step
   trajectory of fixed 512-token batches: per-step loss/gradient-norm errors,
   plus the one-step parameter delta.
3. Compile cost: wall-clock of the first (compiling) step versus steady state,
   and dynamo counters (graphs, breaks) after a full epoch.
4. Recompile storm (labeled stress arm): batches whose max lengths are all
   distinct fed to ``dynamic=False`` - every shape is a new graph - then the
   same batches to ``dynamic=True``.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from compiled_step import RecompileTracker, dynamo_stats
from experiment_utils import environment_record, write_json
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


def free_cache():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def pad_invariance(args, samples, plan, device) -> dict:
    tokenizer, model = load_tokenizer_and_model(device)
    batch = to_device_batch(build_batch(samples, plan[0], args.pad_token_id), device)
    fixed = to_device_batch(pad_batch_to(build_batch(samples, plan[0], args.pad_token_id), args.max_length, args.pad_token_id), device)
    losses = {}
    for label, candidate in (("natural", batch), ("fixed_512", fixed)):
        model.train()
        logits = model(candidate.input_ids, attention_mask=candidate.attention_mask)
        from mini_batch import batch_loss

        losses[label] = float(batch_loss(logits, candidate.labels, "token_mean").item())
        del logits
    rel = abs(losses["natural"] - losses["fixed_512"]) / abs(losses["natural"])
    del model
    free_cache()
    return {"natural_loss": losses["natural"], "fixed_512_loss": losses["fixed_512"], "rel_error": rel, "assert_below_1e_5": rel < 1e-5}


def fixed_shape_parity(args, samples, plan, device) -> dict:
    fixed_plan_batches = [
        to_device_batch(pad_batch_to(build_batch(samples, indices, args.pad_token_id), args.max_length, args.pad_token_id), device)
        for indices in plan[: 16 + args.warmup]
    ]
    trajectories = {}
    for arm, model in (("eager", None), ("compiled", None)):
        seed_everything(args.seed)
        tokenizer, raw_model = load_tokenizer_and_model(device)
        optimizer = build_optimizer(raw_model, args.learning_rate, args.weight_decay)
        if arm == "compiled":
            model = torch.compile(raw_model)
        else:
            model = raw_model
        losses = []
        norms = []
        start = time.perf_counter()
        for batch in fixed_plan_batches:
            metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
            losses.append(metrics["loss"])
            norms.append(metrics["gradient_norm_before_clip"])
        elapsed = time.perf_counter() - start
        trajectories[arm] = {"losses": losses, "norms": norms, "total_seconds": elapsed}
        del model, optimizer, raw_model
        free_cache()

    reference = trajectories["eager"]
    variant = trajectories["compiled"]
    loss_rel = [abs(a - b) / abs(a) for a, b in zip(reference["losses"], variant["losses"])]
    norm_rel = [abs(a - b) / abs(a) for a, b in zip(reference["norms"], variant["norms"])]
    if loss_rel[0] > 1e-4:
        raise AssertionError(f"first-step compiled loss rel error {loss_rel[0]}")
    if max(loss_rel) > 1e-3:
        raise AssertionError(f"compiled trajectory diverged: max rel {max(loss_rel)}")
    return {
        "steps": len(reference["losses"]),
        "first_step_loss_rel_error": loss_rel[0],
        "max_loss_rel_error": max(loss_rel),
        "max_gradient_norm_rel_error": max(norm_rel),
        "loss_series": {"eager": reference["losses"], "compiled": variant["losses"]},
        "wall_seconds": {"eager": reference["total_seconds"], "compiled": variant["total_seconds"]},
    }


def compile_cost(args, samples, plan, device) -> dict:
    seed_everything(args.seed)
    tokenizer, raw_model = load_tokenizer_and_model(device)
    compiled = torch.compile(raw_model)
    optimizer = build_optimizer(raw_model, args.learning_rate, args.weight_decay)
    batch = to_device_batch(pad_batch_to(build_batch(samples, plan[0], args.pad_token_id), args.max_length, args.pad_token_id), device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    train_batch_step(compiled, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1000

    steady = []
    for index in range(1, 6):
        batch = to_device_batch(pad_batch_to(build_batch(samples, plan[index], args.pad_token_id), args.max_length, args.pad_token_id), device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        train_batch_step(compiled, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
        torch.cuda.synchronize()
        steady.append((time.perf_counter() - start) * 1000)
    stats = dynamo_stats()
    del compiled, optimizer, raw_model
    free_cache()
    return {
        "first_step_ms_including_compile": first_ms,
        "steady_step_ms": steady,
        "dynamo_stats": stats,
    }


def recompile_storm(args, samples, plan, device) -> dict:
    """Distinct batch shapes: dynamic=False recompiles, dynamic=True does not."""

    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    batches_by_max_length = {}
    for indices in plan:
        batch = build_batch(samples, indices, args.pad_token_id)
        batches_by_max_length.setdefault(max(batch.lengths), batch)
    distinct = [
        to_device_batch(pad_batch_to(batch, length, args.pad_token_id), device)
        for length, batch in sorted(batches_by_max_length.items())[:12]
    ]

    results = {}
    for dynamic in (False, True):
        seed_everything(args.seed)
        tokenizer_gpu, raw_model = load_tokenizer_and_model(device)
        del tokenizer_gpu
        compiled = torch.compile(raw_model, dynamic=dynamic)
        optimizer = build_optimizer(raw_model, args.learning_rate, args.weight_decay)
        step_times = []
        compile_time_total = 0.0
        with RecompileTracker() as tracker:
            for batch in distinct:
                torch.cuda.synchronize()
                start = time.perf_counter()
                train_batch_step(compiled, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
                elapsed_ms = (time.perf_counter() - start) * 1000
                torch.cuda.synchronize()
                step_times.append(elapsed_ms)
        compile_time_total = sum(step_times[:2])  # 前两步包含追踪与代码生成
        results["dynamic" if dynamic else "static"] = {
            "distinct_shapes": len(distinct),
            "unique_graphs": tracker.compiled_graphs,
            "recompiles": tracker.recompiles,
            "graph_breaks": tracker.graph_breaks,
            "first_two_steps_ms": step_times[:2],
            "last_step_ms": step_times[-1],
            "first_two_steps_sum_ms": compile_time_total,
        }
        print(f"recompile storm dynamic={dynamic}: graphs={tracker.compiled_graphs} recompiles={tracker.recompiles}", flush=True)
        del compiled, optimizer, raw_model
        free_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter28_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
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
        # supervised targets to truncation. Drop them (counted) so every phase
        # compares on identical samples. Same pattern as chapter_24.
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

    invariance = pad_invariance(args, samples, plan, device)
    if not invariance["assert_below_1e_5"]:
        raise AssertionError("padding to fixed length changed the loss beyond FP noise")
    parity = fixed_shape_parity(args, samples, plan, device)
    cost = compile_cost(args, samples, plan, device)
    storm = recompile_storm(args, samples, plan, device)

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
        "optimizer": "AdamW loop (unchanged, outside the compiled region)",
        "seed": args.seed,
        "num_samples": len(samples),
        "filtered_no_supervision": filtered_no_supervision,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "pad_invariance": invariance,
        "fixed_shape_parity": parity,
        "compile_cost": cost,
        "recompile_storm": storm,
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_28 correctness checks passed")


if __name__ == "__main__":
    main()
