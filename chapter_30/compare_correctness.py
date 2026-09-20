#!/usr/bin/env python3
"""Chapter 30 correctness on real weights: bucketed AllReduce versus the flat path.

Phase A (this process, cuda:0): the big-batch reference trajectory — the same
Phase A as Chapter 29 (loss, gradient norms, parameter fingerprints; step-0
gradients stashed).

Phase B (worker processes, one GPU each): the bucketed body with hooks. On
step 0 rank 0 compares gradients three ways: bucketed versus the stashed
big-batch reference (tolerance bars), and bucketed versus an in-process flat
post-backward AllReduce (must be bitwise identical — elementwise SUM over two
ranks). Every step must fire every bucket exactly once; the 8-step global
loss trajectory is compared against the reference.

Launcher: mp.spawn on one node (default), or torchrun across nodes when
RANK already exists in the environment.
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

from bucketed_ddp import BucketCoordinator, bucketed_train_step
from ddp_core import (
    MODEL_ID,
    MODEL_REVISION,
    all_reduce_gradients,
    build_global_batch,
    cleanup_distributed,
    ddp_train_step,
    encode_samples,
    free_port,
    load_tokenizer_and_model,
    make_batch_plan,
    setup_distributed,
)
from experiment_utils import write_json
from compare_reference import gradient_report, run_reference
from mini_batch import batch_loss
from sft_core import seed_everything


GRAD_COSINE_BAR = 0.99999
GRAD_SCALE_REL_BAR = 5e-3
TRAJECTORY_BAR = 5e-4
FLAT_PARITY_BAR = 5e-3
BATCH_SIZE = 4
BUCKET_BYTES = 16 * 1024 * 1024


def fingerprint(model: torch.nn.Module) -> dict:
    return {name: float(parameter.detach().double().sum()) for name, parameter in model.named_parameters()}


def batch_plan_for_world_size(sample_count: int, seed: int, world_size: int) -> list:
    """Global batch of 4 sharded over at most 4 ranks; larger worlds get one
    more member per rank so every rank keeps at least one sample."""

    return make_batch_plan(sample_count, max(BATCH_SIZE, world_size), seed)


def build_report(reference: dict, ddp: dict, bucket_bytes: int, steps: int) -> dict:
    """Compare the bucketed DDP trajectory against the Phase A reference."""

    trajectory_errors = [
        abs(ddp_step["global_loss"] - ref_step["loss"]) / abs(ref_step["loss"])
        for ref_step, ddp_step in zip(reference["steps"], ddp["steps"])
    ]
    grad_report = ddp["step0_grad_report"]
    checks = {
        "grad_cosine_ok": grad_report["grad_cosine_similarity"] > GRAD_COSINE_BAR,
        "grad_rel_ok": grad_report["grad_scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "trajectory_ok": max(trajectory_errors) < TRAJECTORY_BAR,
        "cross_rank_identical": ddp["cross_rank_fingerprints_identical"],
    }
    return {
        "world_size": ddp["world_size"],
        "bucket_bytes": bucket_bytes,
        "steps": steps,
        "step0_grad_report": grad_report,
        "trajectory_max_loss_rel_error": max(trajectory_errors),
        "trajectory_loss_rel_errors": trajectory_errors,
        "cross_rank_identical": ddp["cross_rank_fingerprints_identical"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _worker(rank: int, world_size: int, port: int, args_ns: dict, samples_obj: dict | None = None) -> None:
    if "RANK" not in os.environ:
        # mp.spawn path: publish the env:// endpoints for this single node.
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    # Under torchrun the environment already carries RANK/WORLD_SIZE/LOCAL_RANK
    # and the rendezvous endpoint; overriding them here would break multi-node.
    args = SimpleNamespace(**args_ns)
    if samples_obj is not None:
        bundle = samples_obj
    else:
        bundle = torch.load(args.samples_path, map_location="cpu", weights_only=False)
    samples, plan = bundle["samples"], bundle["plan"]
    seed_everything(args.seed)
    rank_id, _, device = setup_distributed(rank, world_size)
    assert rank_id == rank
    _, model = load_tokenizer_and_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)
    coordinator = BucketCoordinator(model, bucket_bytes=args.bucket_bytes, world_size=world_size)

    step0_grad_report = None
    steps = []
    for step_index in range(args.steps):
        host_batch, accounting = build_global_batch(samples, plan[step_index], rank, world_size, args.pad_token_id)
        if step_index == 0:
            # Parity checks on the pristine initial weights, BEFORE the step.
            device_batch = type(host_batch)(
                input_ids=host_batch.input_ids.to(device),
                attention_mask=host_batch.attention_mask.to(device),
                labels=host_batch.labels.to(device),
                source_lines=host_batch.source_lines,
                lengths=host_batch.lengths,
                supervised_tokens_per_sample=host_batch.supervised_tokens_per_sample,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(device_batch.input_ids, attention_mask=device_batch.attention_mask)
            loss = batch_loss(logits, device_batch.labels, "token_mean") * accounting["loss_scale"]
            loss.backward()
            tail = coordinator.wait_and_attach()
            assert tail["buckets_fired"] == len(coordinator.buckets)
            bucketed_grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
            if rank == 0:
                step0_grad_report = gradient_report(model, args.stash_path)
            # The flat parity backward must run WITHOUT the bucket hooks active,
            # otherwise the hooks would swallow the gradients again.
            coordinator.remove_hooks()
            optimizer.zero_grad(set_to_none=True)
            logits = model(device_batch.input_ids, attention_mask=device_batch.attention_mask)
            loss = batch_loss(logits, device_batch.labels, "token_mean") * accounting["loss_scale"]
            loss.backward()
            all_reduce_gradients(model, world_size)
            flat_worst = max(
                float((parameter.grad - bucketed_grads[name]).abs().max())
                for name, parameter in model.named_parameters()
            )
            if rank == 0:
                if world_size == 2:
                    # Two ranks: elementwise SUM has a single order, so the
                    # bucketed buffer must equal the flat AllReduce bitwise.
                    assert flat_worst == 0.0, f"bucketed grads must equal flat AllReduce bitwise, got {flat_worst}"
                else:
                    # Beyond two ranks, bucket buffers and per-parameter
                    # tensors hit different NCCL chunk boundaries, so the SUM
                    # order — not the values — may change; require tolerance.
                    flat_worst_scale_rel = max(
                        float((parameter.grad - bucketed_grads[name]).abs().max()) / max(float(parameter.grad.abs().max()), 1e-8)
                        for name, parameter in model.named_parameters()
                    )
                    assert flat_worst_scale_rel <= FLAT_PARITY_BAR, (
                        f"bucketed grads deviate from the flat AllReduce beyond tolerance: {flat_worst_scale_rel}"
                    )
            del bucketed_grads
            optimizer.zero_grad(set_to_none=True)
            coordinator.install_hooks()

        metrics = bucketed_train_step(
            model, coordinator, optimizer, host_batch, accounting["loss_scale"], device,
            args.max_grad_norm, world_size, rank, collect_telemetry=False,
        )
        assert metrics["buckets_fired"] == len(coordinator.buckets), "every bucket must fire every step"
        steps.append({
            "step": step_index,
            "global_loss": metrics["global_loss"],
            "gradient_norm": metrics["gradient_norm_before_clip"],
            "fingerprint": fingerprint(model),
        })

    final_fingerprints = [None] * world_size
    dist.all_gather_object(final_fingerprints, fingerprint(model))
    identity = all(entry == final_fingerprints[0] for entry in final_fingerprints)
    coordinator.remove_hooks()
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args.part_path), {
            "world_size": world_size,
            "bucket_count": len(coordinator.buckets),
            "bucket_bytes": args.bucket_bytes,
            "step0_grad_report": step0_grad_report,
            "steps": steps,
            "cross_rank_fingerprints_identical": identity,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter30_train.jsonl"))
    parser.add_argument("--world-size", type=int, default=2,
                        help="mp.spawn path only; under torchrun the world size comes from the environment")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bucket-mib", type=float, default=16)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    args.stash_path = Path("correctness-reference-grads.pt")
    args.part_path = Path("correctness-ddp-part.json")
    args.samples_path = Path("correctness-samples.pt")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if "RANK" in os.environ:
        # torchrun path: every rank runs the worker body directly; rank 0 also
        # builds the single-GPU reference and finalizes the report.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
        args.pad_token_id = tokenizer.pad_token_id
        from sft_core import load_jsonl_records

        records = load_jsonl_records(args.data)
        try:
            samples = encode_samples(tokenizer, records, args.max_length)
            filtered_no_supervision = 0
        except ValueError:
            # 12GB calibration: at reduced max_length a few corpus samples lose
            # all supervised targets to truncation. Drop them on every rank so
            # the comparison stays on identical usable samples.
            kept = []
            filtered_no_supervision = 0
            for record in records:
                try:
                    kept.extend(encode_samples(tokenizer, [record], args.max_length))
                except ValueError:
                    filtered_no_supervision += 1
            samples = kept
        # Batch plans shard max(BATCH_SIZE, world_size) members per step.
        usable = len(samples) - len(samples) % max(BATCH_SIZE, world_size)
        dropped_for_divisibility = len(samples) - usable
        samples = samples[:usable]
        if rank == 0:
            print(
                f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
                f"dropped_for_divisibility={dropped_for_divisibility}"
            )
        plan = batch_plan_for_world_size(len(samples), args.seed, world_size)
        reference = None
        if rank == 0:
            reference = run_reference(args, samples, plan, torch.device("cuda:0"))
            gc.collect()
            torch.cuda.empty_cache()
        args_ns = {
            "seed": args.seed, "steps": args.steps, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
            "pad_token_id": args.pad_token_id, "stash_path": str(args.stash_path),
            "part_path": str(args.part_path), "samples_path": str(args.samples_path),
            "bucket_bytes": int(args.bucket_mib * 1024 * 1024),
        }
        _worker(rank, world_size, 0, args_ns, samples_obj={"samples": samples, "plan": plan})
        if rank == 0:
            ddp = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
            payload = build_report(reference, ddp, args_ns["bucket_bytes"], args.steps)
            write_json(args.output, payload)
            Path(args.part_path).unlink()
            args.stash_path.unlink()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if not payload["pass"]:
                raise SystemExit("correctness checks failed")
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    from sft_core import load_jsonl_records

    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them on BOTH paths so the DDP
        # comparison stays on identical samples.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    # Batch plans shard max(BATCH_SIZE, world_size) members per step.
    usable = len(samples) - len(samples) % max(BATCH_SIZE, args.world_size)
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    plan = batch_plan_for_world_size(len(samples), args.seed, args.world_size)

    reference = run_reference(args, samples, plan, torch.device("cuda:0"))
    gc.collect()
    torch.cuda.empty_cache()

    args_ns = {
        "seed": args.seed, "steps": args.steps, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "pad_token_id": args.pad_token_id, "stash_path": str(args.stash_path),
        "part_path": str(args.part_path), "samples_path": str(args.samples_path),
        "bucket_bytes": int(args.bucket_mib * 1024 * 1024),
    }
    torch.save({"samples": samples, "plan": plan}, args.samples_path)
    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns), nprocs=args.world_size, join=True)
    args.samples_path.unlink()

    ddp = json.loads(args.part_path.read_text(encoding="utf-8"))
    payload = build_report(reference, ddp, args_ns["bucket_bytes"], args.steps)
    write_json(args.output, payload)
    args.part_path.unlink()
    args.stash_path.unlink()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit("correctness checks failed")


if __name__ == "__main__":
    main()
