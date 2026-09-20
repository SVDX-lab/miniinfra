#!/usr/bin/env python3
"""Chapter 32 correctness on real weights: ZeRO-1 versus the torch-AdamW path.

Phase A (rank 0 only, its local cuda:0): the big-batch reference trajectory
with the stock torch AdamW — identical to Chapters 14-16.

Phase B (one GPU per rank): the Chapter 29 gradient path (full gradients,
post-backward AllReduce SUM) but the optimizer is the hand-written
ShardedStateAdamW: each rank updates only its own shard of every parameter,
then a chunked AllGather refreshes every shard from its owner. Rank 0 checks
the step-0 gradients and the 8-step trajectory against Phase A with the same
bars as Chapter 29; all ranks must end with bitwise-identical parameters.
The state memory ledger proves each rank allocates only 1/world_size of the
AdamW moments.

Launch with ``python compare_correctness.py`` (mp.spawn, single node) or under
``torchrun``; the global batch stays at BATCH_SIZE, so ``--world-size`` is 2
(default) or 4, and under torchrun it must match ``WORLD_SIZE``.
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
    all_reduce_scalars,
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
from sft_core import seed_everything
from zero1_core import ShardedStateAdamW, all_gather_parameters, memory_ledger


GRAD_COSINE_BAR = 0.99999
GRAD_SCALE_REL_BAR = 5e-3
TRAJECTORY_BAR = 5e-4
BATCH_SIZE = 4


def fingerprint(model: torch.nn.Module) -> dict:
    return {name: float(parameter.detach().double().sum()) for name, parameter in model.named_parameters()}


def assemble_report(args: SimpleNamespace, report: dict, reference: dict) -> dict:
    """Compare the sharded trajectory against the big-batch reference (rank 0)."""

    trajectory_errors = [
        abs(step["global_loss"] - ref["loss"]) / abs(ref["loss"])
        for ref, step in zip(reference["steps"], report["steps"])
    ]
    grad_report = report["step0_grad_report"]
    checks = {
        "grad_cosine_ok": grad_report["grad_cosine_similarity"] > GRAD_COSINE_BAR,
        "grad_rel_ok": grad_report["grad_scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "trajectory_ok": max(trajectory_errors) < TRAJECTORY_BAR,
        "cross_rank_identical": report["cross_rank_fingerprints_identical"],
        "state_shard_ok": abs(report["memory_ledger"]["state_shard_fraction"] - 1.0 / args.world_size) < 0.01,
    }
    return {
        "world_size": args.world_size,
        "steps": args.steps,
        "nnodes": int(os.environ.get("NNODES", "1")),
        "node_rank": int(os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "step0_grad_report": grad_report,
        "trajectory_max_loss_rel_error": max(trajectory_errors),
        "trajectory_loss_rel_errors": trajectory_errors,
        "cross_rank_identical": report["cross_rank_fingerprints_identical"],
        "memory_ledger": report["memory_ledger"],
        "allgather_ms_total": report["allgather_ms_total"],
        "allgather_bytes_per_step": report["allgather_bytes_per_step"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _worker(rank: int, world_size: int, port: int, args_ns: dict, samples_obj=None, plan_obj=None) -> None:
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
        samples, plan = samples_obj, plan_obj
    else:
        bundle = torch.load(args.samples_path, map_location="cpu", weights_only=False)
        samples, plan = bundle["samples"], bundle["plan"]
    seed_everything(args.seed)
    rank_id, _, device = setup_distributed(rank, world_size)
    assert rank_id == rank
    _, model = load_tokenizer_and_model(device)
    optimizer = ShardedStateAdamW(
        list(model.parameters()), world_size, rank,
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay,
    )
    ledger = memory_ledger(list(model.parameters()), optimizer, world_size)
    ledger["measured_allocated_after_state_bytes"] = float(torch.cuda.memory_allocated(device))

    step0_grad_report = None

    def capture_objective_gradient() -> None:
        # Same point as the reference stash: the AllReduced gradient of the
        # objective, BEFORE clip_grad_norm_ scales it in place (Chapter 29's
        # after_allreduce hook). Comparing a post-clip gradient here would
        # instead report the clip factor on every tensor — a comparison-point
        # bug, not a gradient error.
        nonlocal step0_grad_report
        step0_grad_report = gradient_report(model, args.stash_path)

    allgather_ms_total = 0.0
    allgather_bytes_total = 0
    steps = []
    for step_index in range(args.steps):
        from ddp_core import build_global_batch

        host_batch, accounting = build_global_batch(samples, plan[step_index], rank, world_size, args.pad_token_id)
        metrics = ddp_train_step(
            model, optimizer, host_batch, accounting["loss_scale"], device,
            args.max_grad_norm, world_size, rank, collect_telemetry=False,
            after_allreduce=capture_objective_gradient if step_index == 0 and rank == 0 else None,
        )
        elapsed_ms, sent_bytes = all_gather_parameters(list(model.parameters()), world_size, rank)
        allgather_ms_total += elapsed_ms
        allgather_bytes_total += sent_bytes
        global_loss = all_reduce_scalars(metrics["local_loss"], world_size, device)
        steps.append({
            "step": step_index,
            "global_loss": global_loss,
            "gradient_norm": metrics["gradient_norm_before_clip"],
            "fingerprint": fingerprint(model),
        })

    final_fingerprints = [None] * world_size
    dist.all_gather_object(final_fingerprints, fingerprint(model))
    identity = all(other == final_fingerprints[0] for other in final_fingerprints)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args.part_path), {
            "world_size": world_size,
            "step0_grad_report": step0_grad_report,
            "steps": steps,
            "cross_rank_fingerprints_identical": identity,
            "memory_ledger": ledger,
            "allgather_ms_total": allgather_ms_total,
            "allgather_bytes_per_step": allgather_bytes_total / max(args.steps, 1),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter32_train.jsonl"))
    parser.add_argument("--world-size", type=int, default=2, choices=(2, 4))
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    args_ns = {
        "seed": args.seed, "steps": args.steps, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "pad_token_id": args.pad_token_id, "stash_path": str(args.stash_path),
        "part_path": str(args.part_path), "samples_path": str(args.samples_path),
    }

    if "RANK" in os.environ:
        # torchrun path: rank 0 alone builds the Phase A reference on its local
        # cuda:0 (a second full model on the other ranks' cuda:0 would not
        # fit), then every rank runs the sharded worker body directly.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != args.world_size:
            raise SystemExit(f"--world-size {args.world_size} does not match torchrun WORLD_SIZE {world_size}")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
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
        # Batch plans need BATCH_SIZE members per step.
        usable = len(samples) - len(samples) % BATCH_SIZE
        dropped_for_divisibility = len(samples) - usable
        samples = samples[:usable]
        if rank == 0:
            print(
                f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
                f"dropped_for_divisibility={dropped_for_divisibility}"
            )
        plan = make_batch_plan(len(samples), BATCH_SIZE, args.seed)
        reference = None
        if rank == 0:
            reference = run_reference(args, samples, plan, torch.device("cuda:0"))
            gc.collect()
            torch.cuda.empty_cache()
        _worker(rank, world_size, 0, args_ns, samples_obj=samples, plan_obj=plan)
        if rank == 0:
            report = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
            payload = assemble_report(args, report, reference)
            write_json(args.output, payload)
            Path(args.part_path).unlink()
            args.stash_path.unlink()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if not payload["pass"]:
                raise SystemExit("correctness checks failed")
        return

    # mp.spawn path: single node.
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    from sft_core import load_jsonl_records

    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them on BOTH paths so the
        # comparison stays on identical samples.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    # Batch plans need BATCH_SIZE members per step.
    usable = len(samples) - len(samples) % BATCH_SIZE
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    plan = make_batch_plan(len(samples), BATCH_SIZE, args.seed)

    reference = run_reference(args, samples, plan, torch.device("cuda:0"))
    gc.collect()
    torch.cuda.empty_cache()

    torch.save({"samples": samples, "plan": plan}, args.samples_path)
    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns), nprocs=args.world_size, join=True)
    args.samples_path.unlink()

    report = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
    payload = assemble_report(args, report, reference)
    write_json(args.output, payload)
    Path(args.part_path).unlink()
    args.stash_path.unlink()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit("correctness checks failed")


if __name__ == "__main__":
    main()
