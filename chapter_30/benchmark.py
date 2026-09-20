#!/usr/bin/env python3
"""Chapter 30 benchmark: when the gradient AllReduce starts.

All arms share the same model, per-rank batch, and total communication
volume — the ONLY variable is when the AllReduce starts:

- ``flat``          one AllReduce after the whole Backward (Chapter 29 body;
                    fully exposed communication)
- ``bucket_1mib`` / ``bucket_16mib`` / ``bucket_128mib`` / ``bucket_1024mib``
                    buckets of at most that many FP32 bytes, AllReduced
                    asynchronously during Backward

World size 1 runs the flat baseline only; world sizes 2, 4, and 8 run every
arm. The global batch is MEMBERS_PER_RANK * world_size, so every rank always
trains 2 members and world size 2 reproduces the chapter's original 2+2 shard
of a global batch of 4.

Per step the bucketed body reports ``backward_ms`` (includes overlapped
communication), ``grad_allreduce_ms`` (the exposed tail wait only),
``total_comm_ms`` (sum of per-bucket AllReduce durations), and the inferred
hidden time ``total_comm - tail``. The flat arm's ``grad_allreduce_ms`` is
its whole communication, since none of it overlaps.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from bucketed_ddp import BucketCoordinator, bucketed_train_step
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
    parameter_count_bytes,
    setup_distributed,
)
from experiment_utils import environment_record, percentile, write_json
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything


MEMBERS_PER_RANK = 2
FLAT_ARM = "flat"
BUCKET_ARMS = ("bucket_1mib", "bucket_16mib", "bucket_128mib", "bucket_1024mib")
ALL_ARMS = (FLAT_ARM, *BUCKET_ARMS)
ARMS_BY_WORLD_SIZE = {1: (FLAT_ARM,), 2: ALL_ARMS, 4: ALL_ARMS, 8: ALL_ARMS}


def arm_plan(arm: str, sample_count: int, seed: int, world_size: int) -> list:
    """Batch plan for one arm; every arm of a world size shares one plan.

    The global batch is MEMBERS_PER_RANK * world_size, so each rank always
    trains MEMBERS_PER_RANK members and the per-rank gradient volume never
    depends on the world size. World size 2 reproduces the chapter's original
    2+2 shard of a global batch of 4.
    """

    if arm not in ARMS_BY_WORLD_SIZE[world_size]:
        raise ValueError(f"arm {arm} is not defined for world size {world_size}")
    return make_batch_plan(sample_count, MEMBERS_PER_RANK * world_size, seed)


def _worker(rank: int, world_size: int, port: int, args_ns: dict, plans: dict, arms: list, samples_obj: dict | None = None) -> None:
    import os

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
        samples = samples_obj
    else:
        samples = torch.load(args_ns["samples_path"], map_location="cpu", weights_only=False)
    seed_everything(args.seed)
    _, _, device = setup_distributed(rank, world_size)
    _, model = load_tokenizer_and_model(device)

    results = {arm: [] for arm in arms}
    for arm in arms:
        bucket_bytes = 0 if arm == FLAT_ARM else int(arm.removeprefix("bucket_").removesuffix("mib")) * 1024 * 1024
        coordinator = BucketCoordinator(model, bucket_bytes=bucket_bytes, world_size=world_size) if bucket_bytes else None
        plan = plans[FLAT_ARM]
        bucket_total_bytes = sum(bucket.size_bytes for bucket in coordinator.buckets) if coordinator else 0
        for repeat in range(1, args.repeats + 1):
            seed_everything(args.seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)
            try:
                for warmup_index in range(args.warmup):
                    host_batch, accounting = build_global_batch(samples, plan[warmup_index], rank, world_size, args.pad_token_id)
                    if coordinator is None:
                        ddp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
                    else:
                        bucketed_train_step(model, coordinator, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
                torch.cuda.reset_peak_memory_stats(device)
                steps = []
                for step_index in range(args.warmup, len(plan)):
                    host_batch, accounting = build_global_batch(samples, plan[step_index], rank, world_size, args.pad_token_id)
                    if coordinator is None:
                        metrics = ddp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=True)
                    else:
                        metrics = bucketed_train_step(model, coordinator, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=True)
                    index = step_index - args.warmup
                    metrics.update({
                        "step": float(index + 1),
                        "global_supervised_tokens": float(accounting["supervised_global"]),
                        "global_supervised_tokens_per_second": accounting["supervised_global"] / (metrics["step_time_ms"] / 1000),
                    })
                    steps.append(metrics)
                total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
                step_times = [step["step_time_ms"] for step in steps]
                results[arm].append({
                    "repeat": repeat,
                    "status": "success",
                    "bucket_count": len(coordinator.buckets) if coordinator else 0,
                    "bucket_total_bytes": bucket_total_bytes,
                    "measured_steps": len(steps),
                    "steps": steps,
                    "aggregate": {
                        "step_time_mean_ms": fmean(step_times),
                        "step_time_p50_ms": percentile(step_times, 0.5),
                        "step_time_p95_ms": percentile(step_times, 0.95),
                        "step_time_max_ms": max(step_times),
                        "exposed_comm_mean_ms": fmean(step["grad_allreduce_ms"] for step in steps),
                        "total_comm_mean_ms": fmean(step["total_comm_ms"] for step in steps) if coordinator else fmean(step["grad_allreduce_ms"] for step in steps),
                        "backward_mean_ms": fmean(step["backward_time_ms"] for step in steps),
                        "forward_mean_ms": fmean(step["forward_time_ms"] for step in steps),
                        "optimizer_mean_ms": fmean(step["optimizer_time_ms"] for step in steps),
                        "buckets_fired_mean": fmean(step["buckets_fired"] for step in steps) if coordinator else 0.0,
                    },
                    "throughput": {
                        "global_supervised_tokens_per_second": sum(step["global_supervised_tokens"] for step in steps) / total_seconds,
                    },
                    "memory": {
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    },
                    "first_last_loss": {"first": steps[0]["global_loss"], "last": steps[-1]["global_loss"]},
                })
            except torch.cuda.OutOfMemoryError:
                results[arm].append({"repeat": repeat, "status": "oom", "allocated_bytes": torch.cuda.memory_allocated(device)})
            entry = results[arm][-1]
            if rank == 0:
                if entry["status"] == "success":
                    print(f"{arm} repeat={entry['repeat']}: p50 {entry['aggregate']['step_time_p50_ms']:.1f} ms "
                          f"exposed {entry['aggregate']['exposed_comm_mean_ms']:.1f} ms "
                          f"peak {entry['memory']['peak_allocated_bytes'] / 2**30:.2f} GiB", flush=True)
                else:
                    print(f"{arm} repeat={entry['repeat']}: {entry['status']} allocated {entry.get('allocated_bytes', 0) / 2**30:.2f} GiB", flush=True)
            if entry["status"] != "success":
                break
        if coordinator is not None:
            coordinator.remove_hooks()
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args_ns["part_path"]), {"world_size": world_size, "results": results})


def summarize_entries(entries):
    successful = [entry for entry in entries if entry["status"] == "success"]
    if not successful:
        return {"successful_repeats": 0}
    return {
        "successful_repeats": len(successful),
        "step_time_p50_ms_mean": fmean(entry["aggregate"]["step_time_p50_ms"] for entry in successful),
        "exposed_comm_mean_ms": fmean(entry["aggregate"]["exposed_comm_mean_ms"] for entry in successful),
        "total_comm_mean_ms": fmean(entry["aggregate"]["total_comm_mean_ms"] for entry in successful),
        "backward_mean_ms": fmean(entry["aggregate"]["backward_mean_ms"] for entry in successful),
        "optimizer_mean_ms": fmean(entry["aggregate"]["optimizer_mean_ms"] for entry in successful),
        "global_supervised_tokens_per_second_mean": fmean(entry["throughput"]["global_supervised_tokens_per_second"] for entry in successful),
        "peak_allocated_bytes_mean": fmean(entry["memory"]["peak_allocated_bytes"] for entry in successful),
        "peak_reserved_bytes_mean": fmean(entry["memory"]["peak_reserved_bytes"] for entry in successful),
        "bucket_count": successful[0]["bucket_count"],
        "bucket_total_bytes": successful[0]["bucket_total_bytes"],
    }


def compute_scaling(summary):
    """Overlap scaling of every bucket arm against the flat baseline."""

    scaling = {}
    if not summary.get(FLAT_ARM, {}).get("successful_repeats"):
        return scaling
    flat_time = summary[FLAT_ARM]["step_time_p50_ms_mean"]
    flat_exposed = summary[FLAT_ARM]["exposed_comm_mean_ms"]
    scaling["flat_baseline"] = {"step_time_p50_ms": flat_time, "exposed_comm_ms": flat_exposed}
    for arm, entry in summary.items():
        if arm == FLAT_ARM or not entry.get("successful_repeats"):
            continue
        exposed = entry["exposed_comm_mean_ms"]
        total = entry["total_comm_mean_ms"]
        scaling[arm] = {
            "step_time_saving_vs_flat_ms": flat_time - entry["step_time_p50_ms_mean"],
            "step_time_reduction_vs_flat": (flat_time - entry["step_time_p50_ms_mean"]) / flat_time,
            "exposed_comm_reduction_vs_flat_ms": flat_exposed - exposed,
            "inferred_hidden_comm_ms": max(0.0, total - exposed),
            "exposed_share_of_step": exposed / entry["step_time_p50_ms_mean"],
        }
    return scaling


def finalize_output(merged, summary, scaling, args, world_size, sample_count, output_path):
    # Persist the raw measurements first: derived summaries must never be able
    # to destroy them.
    write_json(Path(output_path), {"partial": True, "repeats_detail": merged, "summary_partial": summary})

    gradient_bytes = parameter_count_bytes() * 4  # FP32 gradients all-reduced every step
    import os as _os

    payload = environment_record(world_size=world_size, dp_degree=world_size)
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32",
        "tf32": False,
        "attention_implementation": "eager",
        "arms": sorted(merged.keys()),
        "gradient_allreduce_bytes_per_step": gradient_bytes,
        "num_samples": sample_count,
        "global_batch_size": MEMBERS_PER_RANK * world_size,
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "nnodes": int(_os.environ.get("NNODES", "1")),
        "node_rank": int(_os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in _os.environ else "mp.spawn",
        "repeats_detail": merged,
        "summary": summary,
        "scaling": scaling,
        "timing_scope": {
            "step_time": "wall clock around a synchronized step",
            "backward_ms": "bucketed arms: includes overlapped AllReduces; flat arm: pure backward compute",
            "exposed_comm": "flat arm: the whole post-backward AllReduce; bucketed arms: only the tail wait after Backward returns",
            "total_comm": "sum of per-bucket AllReduce durations (flat arm equals exposed)",
            "inferred_hidden": "total_comm - exposed tail; an upper bound on successfully overlapped communication",
        },
    })
    write_json(Path(output_path), payload)
    print(json.dumps({"summary": summary, "scaling": scaling}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter30_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--only-arm", default=None, choices=ALL_ARMS)
    parser.add_argument("--world-size", type=int, default=2, choices=sorted(ARMS_BY_WORLD_SIZE),
                        help="mp.spawn path only; under torchrun the world size comes from the environment")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import os

    if "RANK" in os.environ:
        # torchrun path: every rank prepares the same deterministic state in
        # memory, then runs the worker body directly. Rank 0 finalizes.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size not in ARMS_BY_WORLD_SIZE:
            raise SystemExit(f"world size {world_size} is not supported; choose one of {sorted(ARMS_BY_WORLD_SIZE)}")
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
            # all arms benchmark the same usable corpus.
            kept = []
            filtered_no_supervision = 0
            for record in records:
                try:
                    kept.extend(encode_samples(tokenizer, [record], args.max_length))
                except ValueError:
                    filtered_no_supervision += 1
            samples = kept
        # The arm plan shards MEMBERS_PER_RANK * world_size members per step.
        usable = len(samples) - len(samples) % (MEMBERS_PER_RANK * world_size)
        dropped_for_divisibility = len(samples) - usable
        samples = samples[:usable]
        if rank == 0:
            print(
                f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
                f"dropped_for_divisibility={dropped_for_divisibility}"
            )
        arms = tuple(arm for arm in ARMS_BY_WORLD_SIZE[world_size] if args.only_arm in (None, arm))
        plans = {FLAT_ARM: arm_plan(FLAT_ARM, len(samples), args.seed, world_size)}

        args_ns = {
            "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm, "pad_token_id": args.pad_token_id,
            "part_path": str(args.output.with_suffix(".part.json")),
            "samples_path": str(args.output.with_suffix(".samples.pt")),
        }
        _worker(rank, world_size, 0, args_ns, plans, list(arms), samples_obj=samples)
        if rank == 0:
            part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
            merged = part["results"]
            summary = {arm: summarize_entries(entries) for arm, entries in merged.items()}
            scaling = compute_scaling(summary)
            finalize_output(merged, summary, scaling, args, world_size, len(samples), args.output)
            Path(args_ns["part_path"]).unlink()
        return

    # mp.spawn path: single node, one process group for one world size.
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
        # supervised targets to truncation. Drop them so every arm benchmarks
        # the same usable corpus.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    # The arm plan shards MEMBERS_PER_RANK * world_size members per step.
    usable = len(samples) - len(samples) % (MEMBERS_PER_RANK * args.world_size)
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    arms = tuple(arm for arm in ARMS_BY_WORLD_SIZE[args.world_size] if args.only_arm in (None, arm))
    plans = {FLAT_ARM: arm_plan(FLAT_ARM, len(samples), args.seed, args.world_size)}

    args_ns = {
        "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "pad_token_id": args.pad_token_id,
        "part_path": str(args.output.with_suffix(".part.json")),
        "samples_path": str(args.output.with_suffix(".samples.pt")),
    }
    torch.save(samples, args_ns["samples_path"])

    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns, plans, list(arms)), nprocs=args.world_size, join=True)
    part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
    merged = part["results"]
    Path(args_ns["part_path"]).unlink()
    Path(args_ns["samples_path"]).unlink()

    summary = {arm: summarize_entries(entries) for arm, entries in merged.items()}
    scaling = compute_scaling(summary)
    finalize_output(merged, summary, scaling, args, args.world_size, len(samples), args.output)


if __name__ == "__main__":
    main()
