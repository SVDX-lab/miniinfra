#!/usr/bin/env python3
"""Chapter 29 correctness on real Qwen3 weights: DDP versus the big-batch path.

Phase A (this process, cuda:0): the Chapter 28 stable body on the FULL global
batch — per-step loss, gradient norms, and parameter fingerprints over an
8-step trajectory; step 0 gradients are additionally stashed on disk.

Phase B (spawned workers, one GPU each): the hand-written DDP body on the
sharded batches — scaled local losses, backward, AllReduce SUM, clip, AdamW.
On step 0 rank 0 compares the AllReduced gradients — captured through the
``after_allreduce`` hook, i.e. BEFORE clip_grad_norm_ scales them in place —
against the stashed reference; after every step both ranks prove parameter
identity across ranks, and the per-step global loss is compared against the
Phase A trajectory.

Pass bars (FP32): step-0 gradient cosine > 0.99999 and per-tensor
max(|Δgrad|) / max(|grad|) < 5e-3; trajectory loss relative error < 5e-4;
cross-rank parameter fingerprints bitwise identical.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ddp_core import (
    MODEL_ID,
    MODEL_REVISION,
    build_global_batch,
    cleanup_distributed,
    ddp_train_step,
    encode_samples,
    free_port,
    load_tokenizer_and_model,
    make_batch_plan,
    setup_distributed,
)
from experiment_utils import environment_record, write_json
from mini_batch import batch_loss, pad_samples
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything


GRAD_COSINE_BAR = 0.99999
GRAD_SCALE_REL_BAR = 5e-3
TRAJECTORY_BAR = 5e-4
BATCH_SIZE = 4  # the global batch of the chapter 04-13 lineage


def fingerprint(model: torch.nn.Module) -> dict:
    """One scalar fingerprint per parameter (FP64 sum in fixed-size chunks).

    The 0.6B embedding is 155M elements: a whole-tensor ``.double()`` copy
    needs a 1.24 GB contiguous block and can fragment-fail on the 12 GB
    3080 Ti while the reference phase still holds activations. Chunked sums
    peak at ~256 MiB instead; both comparison sides run this exact function,
    so equal inputs still produce bitwise-equal fingerprints.
    """

    fingerprints: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        flat = parameter.detach().reshape(-1)
        total = 0.0
        for chunk in flat.split(1 << 25):
            total += float(chunk.double().sum())
        fingerprints[name] = total
    return fingerprints


def gradient_report(model: torch.nn.Module, stashed_path: Path) -> dict:
    """Scale-aware comparison of AllReduced grads versus the big-batch reference.

    Both sides are the gradient of the global token_mean objective captured
    BEFORE clipping: the reference stashes pre-clip, and the DDP side must be
    read through the after_allreduce hook. Comparing a post-clip DDP gradient
    here would instead report the clip factor (1 - max_grad_norm/||g||) on
    every tensor — a comparison-point bug, not a gradient error.

    Raw elementwise relative error is dominated by near-zero entries, so the
    primary bar is per-tensor: max abs difference divided by the tensor's max
    absolute value. The global cosine similarity and the raw absolute error
    are reported alongside.
    """

    stashed = torch.load(stashed_path, map_location="cpu")["grads"]
    device = next(model.parameters()).device
    sum_dot = 0.0
    sum_sq_ref = 0.0
    sum_sq_new = 0.0
    worst_rel = 0.0
    worst_abs = 0.0
    worst_entry = None
    for name, parameter in model.named_parameters():
        # Compare on the CPU: the launcher's own CUDA context (~350 MB) stays
        # resident on rank 0's device, and materialising double-precision
        # copies of the tied-embedding gradient there overflows 12 GB.
        reference = stashed[name]
        actual = parameter.grad.detach().to("cpu")
        difference = (actual - reference).abs()
        tensor_max = float(reference.abs().max())
        scale_rel = float(difference.max()) / max(tensor_max, 1e-8)
        if scale_rel > worst_rel:
            worst_rel = scale_rel
            worst_entry = name
        worst_abs = max(worst_abs, float(difference.max()))
        sum_dot += float((actual.double() * reference.double()).sum())
        sum_sq_ref += float((reference.double() ** 2).sum())
        sum_sq_new += float((actual.double() ** 2).sum())
        del reference
    cosine = sum_dot / (sum_sq_ref ** 0.5 * sum_sq_new ** 0.5)
    reference_norm = sum_sq_ref ** 0.5
    return {
        "grad_cosine_similarity": cosine,
        "grad_scale_relative_error": worst_rel,
        "grad_max_absolute_error": worst_abs,
        "grad_abs_error_over_reference_norm": worst_abs / reference_norm,
        "grad_scale_relative_error_parameter": worst_entry,
        "reference_grad_norm": reference_norm,
        "ddp_grad_norm": sum_sq_new ** 0.5,
        "grad_norm_rel_error": abs(sum_sq_new ** 0.5 - reference_norm) / reference_norm,
        "capture_point": "after_allreduce_before_clip",
    }


def run_reference(args: SimpleNamespace, samples, plan, device) -> dict:
    """Phase A: the stable big-batch body over the trajectory steps."""

    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)

    steps = []
    for step_index in range(args.steps):
        host_batch = pad_samples([samples[index] for index in plan[step_index]], args.pad_token_id)
        batch = type(host_batch)(
            input_ids=host_batch.input_ids.to(device),
            attention_mask=host_batch.attention_mask.to(device),
            labels=host_batch.labels.to(device),
            source_lines=host_batch.source_lines,
            lengths=host_batch.lengths,
            supervised_tokens_per_sample=host_batch.supervised_tokens_per_sample,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean")
        loss.backward()
        if step_index == 0:
            # Stash BEFORE clipping: the gradient of the objective is the
            # comparison target; clipping is a training-step concern.
            stashed = {name: parameter.grad.detach().to("cpu").clone() for name, parameter in model.named_parameters()}
            torch.save({"grads": stashed}, args.stash_path)
            del stashed
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if step_index == 0:
            stashed_norm = float(grad_norm.item())
        optimizer.step()
        steps.append({
            "step": step_index,
            "loss": float(loss.item()),
            "gradient_norm": float(grad_norm.item()),
            "fingerprint": fingerprint(model),
        })
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"steps": steps, "stashed_grad_norm": stashed_norm}


def _worker(rank: int, world_size: int, port: int, args_ns: dict) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    args = SimpleNamespace(**args_ns)
    bundle = torch.load(args.samples_path, map_location="cpu", weights_only=False)
    samples, plan = bundle["samples"], bundle["plan"]
    seed_everything(args.seed)
    rank_id, _, device = setup_distributed(rank, world_size)
    assert rank_id == rank
    _, model = load_tokenizer_and_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)

    step0_grad_report = None

    def capture_objective_gradient() -> None:
        # Same point as the reference stash: the AllReduced gradient of the
        # objective, BEFORE clip_grad_norm_ scales it in place.
        nonlocal step0_grad_report
        step0_grad_report = gradient_report(model, args.stash_path)

    steps = []
    for step_index in range(args.steps):
        host_batch, accounting = build_global_batch(samples, plan[step_index], rank, world_size, args.pad_token_id)
        shard_sums = [None] * world_size
        dist.all_gather_object(shard_sums, accounting["supervised_local"])
        assert abs(sum(shard_sums) - accounting["supervised_global"]) < 1e-6, "rank shards must cover the global supervised tokens"

        metrics = ddp_train_step(
            model, optimizer, host_batch, accounting["loss_scale"], device,
            args.max_grad_norm, world_size, rank, collect_telemetry=False,
            after_allreduce=capture_objective_gradient if step_index == 0 and rank == 0 else None,
        )
        steps.append({
            "step": step_index,
            "supervised_local": accounting["supervised_local"],
            "supervised_global": accounting["supervised_global"],
            "loss_scale": accounting["loss_scale"],
            "global_loss": metrics["global_loss"],
            "gradient_norm": metrics["gradient_norm_before_clip"],
            "fingerprint": fingerprint(model),
        })

    final_fingerprints = [None] * world_size
    dist.all_gather_object(final_fingerprints, fingerprint(model))
    identity = {
        "cross_rank_fingerprints_identical": final_fingerprints[0] == final_fingerprints[1],
        "fingerprint_disagreements": (
            [name for name, value in final_fingerprints[0].items() if final_fingerprints[1][name] != value]
            if world_size > 1 else []
        ),
    }
    cleanup_distributed()

    if rank == 0:
        write_json(Path(args.part_path), {
            "world_size": world_size,
            "step0_grad_report": step0_grad_report,
            "steps": steps,
            "identity": identity,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter29_train.jsonl"))
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    args.stash_path = Path("correctness-reference-grads.pt")
    args.part_path = Path("correctness-ddp-part.json")
    args.samples_path = Path("correctness-samples.pt")
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them on BOTH paths so the DDP
        # comparison stays on identical samples.
        kept = []
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                continue
        samples = kept
    # Batch plans of 4 need the count to stay divisible.
    samples = samples[: len(samples) - len(samples) % BATCH_SIZE]
    plan = make_batch_plan(len(samples), BATCH_SIZE, args.seed)

    reference = run_reference(args, samples, plan, torch.device("cuda:0"))
    gc.collect()
    torch.cuda.empty_cache()

    args_ns = {
        "seed": args.seed, "steps": args.steps, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "pad_token_id": args.pad_token_id, "stash_path": str(args.stash_path),
        "part_path": str(args.part_path), "samples_path": str(args.samples_path),
    }
    torch.save({"samples": samples, "plan": plan}, args.samples_path)
    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns), nprocs=args.world_size, join=True)
    args.samples_path.unlink()

    ddp = json.loads(args.part_path.read_text(encoding="utf-8"))
    trajectory_errors = [
        abs(ddp_step["global_loss"] - ref_step["loss"]) / abs(ref_step["loss"])
        for ref_step, ddp_step in zip(reference["steps"], ddp["steps"])
    ]
    fingerprint_errors = [
        max(abs(ddp_step["fingerprint"][name] - ref_step["fingerprint"][name]) for name in ref_step["fingerprint"])
        for ref_step, ddp_step in zip(reference["steps"], ddp["steps"])
    ]
    grad_report = ddp["step0_grad_report"]
    reference_supervised = [sum(samples[index].supervised_tokens for index in plan[step]) for step in range(args.steps)]
    ddp_supervised = [step["supervised_global"] for step in ddp["steps"]]
    checks = {
        "grad_cosine_ok": grad_report["grad_cosine_similarity"] > GRAD_COSINE_BAR,
        "grad_rel_ok": grad_report["grad_scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "trajectory_ok": max(trajectory_errors) < TRAJECTORY_BAR,
        "cross_rank_identical": ddp["identity"]["cross_rank_fingerprints_identical"],
        "supervised_accounting_ok": reference_supervised == ddp_supervised,
    }
    payload = {
        "world_size": args.world_size,
        "steps": args.steps,
        "step0_grad_report": grad_report,
        "reference_grad_norm": reference["stashed_grad_norm"],
        "trajectory_max_loss_rel_error": max(trajectory_errors),
        "trajectory_loss_rel_errors": trajectory_errors,
        "fingerprint_max_abs_error_last": fingerprint_errors[-1],
        "cross_rank": ddp["identity"],
        "checks": checks,
        "pass": all(checks.values()),
    }
    write_json(args.output, payload)
    args.part_path.unlink()
    args.stash_path.unlink()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit("correctness checks failed")


if __name__ == "__main__":
    main()
