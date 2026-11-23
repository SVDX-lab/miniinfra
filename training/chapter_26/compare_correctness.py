#!/usr/bin/env python3
"""Chapter 26 correctness: kernel parity, cross-precision flash, boundaries.

Layers:

1. Backend census on this GPU: which SDPA backends actually run for FP32 and
   BF16, with an additive mask versus the causal flag.
2. FP32 kernel parity on a real padded batch: loss, gradient norms and
   selected gradients of sdpa_math / sdpa_mem_efficient against eager.
3. FP32 epoch trajectory: eager versus sdpa_mem_efficient, per-step loss
   divergence summary (kernels differ in reduction order, so this is a
   tolerance comparison, not a bitwise one).
4. Cross-precision flash arm (explicitly labeled): on padding-free batches
   with pure-BF16 weights, eager versus sdpa_mem_efficient versus flash.
5. Sequence-length capacity sweep at batch 1: eager versus sdpa_mem_efficient
   forward+backward time and memory until OOM.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from statistics import fmean

import torch

from experiment_utils import environment_record, write_json
from mini_batch import batch_loss
from sdpa_attention import probe_backend_support, set_attention_implementation
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


SELECTED_PARAMETERS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
)


def fresh_model(args, device, dtype=torch.float32):
    from qwen3_model import load_handwritten_model, resolve_model_directory

    seed_everything(args.seed)
    directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    model = load_handwritten_model(directory, device, dtype=dtype)
    return model


def free_cache():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def kernel_parity(args, samples, plan, device) -> dict:
    batch = to_device_batch(build_batch(samples, plan[0], args.pad_token_id), device)
    results = {}
    for arm in ("eager", "sdpa_math", "sdpa_mem_efficient"):
        model = fresh_model(args, device)
        set_attention_implementation(model, arm)
        optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item())
        selected = {
            name: parameter.grad.detach().clone().cpu()
            for name, parameter in model.named_parameters()
            if name in SELECTED_PARAMETERS
        }
        results[arm] = {"loss": float(loss.item()), "grad_norm": grad_norm, "grads": selected}
        del model, optimizer, logits, loss
        free_cache()

    reference = results["eager"]
    report = {"eager": {"loss": reference["loss"], "grad_norm": reference["grad_norm"]}, "comparisons": {}}
    for arm in ("sdpa_math", "sdpa_mem_efficient"):
        arm_loss = results[arm]["loss"]
        loss_rel = abs(arm_loss - reference["loss"]) / abs(reference["loss"])
        norm_rel = abs(results[arm]["grad_norm"] - reference["grad_norm"]) / reference["grad_norm"]
        per_parameter = {}
        for name in SELECTED_PARAMETERS:
            base = reference["grads"][name]
            other = results[arm]["grads"][name]
            error = (base - other).abs()
            significant = base.abs() > (1e-2 * base.abs().max())
            relative_significant = error[significant] / base.abs()[significant]
            cosine = torch.nn.functional.cosine_similarity(
                base.flatten().double(), other.flatten().double(), dim=0
            )
            per_parameter[name] = {
                "max_abs_error": float(error.max()),
                "max_rel_error_significant": float(relative_significant.max()),
                "cosine_similarity": float(cosine),
            }
            if float(cosine) < 0.99999 or float(relative_significant.max()) > 1e-2:
                raise AssertionError(f"{arm} gradient mismatch at {name}: {per_parameter[name]}")
        if loss_rel > 1e-3:
            raise AssertionError(f"{arm} loss mismatch: rel={loss_rel}")
        report["comparisons"][arm] = {
            "loss": arm_loss,
            "loss_rel_error": loss_rel,
            "grad_norm_rel_error": norm_rel,
            "selected_parameters": per_parameter,
        }
    return report


def epoch_trajectory(args, samples, plan, device) -> dict:
    trajectories = {}
    memory = {}
    for arm in ("eager", "sdpa_mem_efficient"):
        model = fresh_model(args, device)
        set_attention_implementation(model, arm)
        optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
        steps = []
        torch.cuda.reset_peak_memory_stats(device)
        for batch_indices in plan:
            batch = to_device_batch(build_batch(samples, batch_indices, args.pad_token_id), device)
            metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
            steps.append({"loss": metrics["loss"], "gradient_norm": metrics["gradient_norm_before_clip"]})
        trajectories[arm] = steps
        memory[arm] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del model, optimizer
        free_cache()

    losses_ref = [step["loss"] for step in trajectories["eager"]]
    losses_arm = [step["loss"] for step in trajectories["sdpa_mem_efficient"]]
    abs_errors = [abs(a - b) for a, b in zip(losses_ref, losses_arm)]
    rel_errors = [abs(a - b) / abs(a) for a, b in zip(losses_ref, losses_arm)]
    first_rel = rel_errors[0]
    if first_rel > 1e-3:
        raise AssertionError(f"first-step loss rel error {first_rel} exceeds FP32 kernel tolerance")
    return {
        "steps": len(losses_ref),
        "first_step_rel_error": first_rel,
        "max_abs_error": max(abs_errors),
        "mean_abs_error": fmean(abs_errors),
        "max_rel_error": max(rel_errors),
        "mean_rel_error": fmean(rel_errors),
        "loss_series": {"eager": losses_ref, "sdpa_mem_efficient": losses_arm},
        "gradient_norm_series": {
            "eager": [step["gradient_norm"] for step in trajectories["eager"]],
            "sdpa_mem_efficient": [step["gradient_norm"] for step in trajectories["sdpa_mem_efficient"]],
        },
        "memory": memory,
    }


def flash_cross_precision(args, samples, plan, device) -> dict:
    """BF16 arms on padding-free batches only: flash cannot mask padding."""

    padding_free = []
    for batch_indices in plan:
        batch = build_batch(samples, batch_indices, args.pad_token_id)
        if bool((batch.attention_mask == 1).all()):
            padding_free.append(to_device_batch(batch, device))
        if len(padding_free) == 8:
            break
    if len(padding_free) < 4:
        raise AssertionError("not enough padding-free batches for the flash arm")

    trajectories = {}
    for arm in ("eager", "sdpa_mem_efficient", "flash"):
        model = fresh_model(args, device, dtype=torch.bfloat16)
        set_attention_implementation(model, arm)
        losses = []
        for batch in padding_free:
            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            loss = batch_loss(logits, batch.labels, "token_mean")
            losses.append(float(loss.item()))
        trajectories[arm] = losses
        del model
        free_cache()

    reference = trajectories["eager"]
    comparisons = {}
    for arm in ("sdpa_mem_efficient", "flash"):
        rel = [abs(a - b) / abs(a) for a, b in zip(reference, trajectories[arm])]
        comparisons[arm] = {
            "batches": len(reference),
            "loss_series": trajectories[arm],
            "max_rel_error_vs_eager_bf16": max(rel),
            "mean_rel_error_vs_eager_bf16": fmean(rel),
        }
        if max(rel) > 5e-2:
            raise AssertionError(f"{arm} BF16 loss mismatch on padding-free batches: {comparisons[arm]}")
    return {"note": "BF16 pure weights, padding-free batches, is_causal semantics; not comparable with FP32 arms", **comparisons}


def sequence_capacity(args, tokenizer, device) -> dict:
    results = {}
    for arm in ("eager", "sdpa_mem_efficient"):
        model = fresh_model(args, device)
        set_attention_implementation(model, arm)
        optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
        arm_results = []
        for sequence_length in (512, 1024, 2048, 3072, 4096, 8192):
            generator = torch.Generator().manual_seed(args.seed)
            input_ids = torch.randint(4, len(tokenizer.vocab), (1, sequence_length), generator=generator)
            labels = input_ids.clone()
            labels[:, :2] = -100
            batch = {
                "input_ids": input_ids.to(device),
                "attention_mask": torch.ones_like(input_ids).to(device),
                "labels": labels.to(device),
            }
            entry = {"sequence_length": sequence_length}
            try:
                torch.cuda.reset_peak_memory_stats(device)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record()
                logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = batch_loss(logits, batch["labels"], "token_mean")
                forward_end = torch.cuda.Event(enable_timing=True)
                forward_end.record()
                loss.backward()
                events[1].record()
                optimizer.step()
                events[2].record()
                torch.cuda.synchronize(device)
                entry.update({
                    "status": "success",
                    "forward_ms": float(events[0].elapsed_time(forward_end)),
                    "backward_ms": float(forward_end.elapsed_time(events[1])),
                    "optimizer_ms": float(events[1].elapsed_time(events[2])),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                })
                del logits, loss
            except torch.cuda.OutOfMemoryError as error:
                entry.update({
                    "status": "oom",
                    "allocated_at_failure_bytes": torch.cuda.memory_allocated(),
                    "reserved_at_failure_bytes": torch.cuda.memory_reserved(),
                    "message": str(error).splitlines()[0][:160],
                })
            arm_results.append(entry)
            print(f"seq={sequence_length} {arm}: {entry['status']}", flush=True)
            free_cache()
            if entry["status"] == "oom":
                break
        del model, optimizer
        free_cache()
        results[arm] = arm_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter26_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
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

    census = probe_backend_support(device)
    parity = kernel_parity(args, samples, plan, device)
    trajectory = epoch_trajectory(args, samples, plan, device)
    flash_arm = flash_cross_precision(args, samples, plan, device)
    capacity = sequence_capacity(args, tokenizer, device)

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32",
        "tf32": False,
        "arms": ["eager", "sdpa_math", "sdpa_mem_efficient", "flash (BF16 cross-precision only)"],
        "seed": args.seed,
        "num_samples": len(samples),
        "filtered_no_supervision": filtered_no_supervision,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "backend_census": census,
        "kernel_parity": parity,
        "epoch_trajectory": trajectory,
        "flash_cross_precision": flash_arm,
        "sequence_capacity_sweep": capacity,
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_26 correctness checks passed")


if __name__ == "__main__":
    main()
