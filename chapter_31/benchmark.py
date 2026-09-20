#!/usr/bin/env python3
"""Chapter 31 benchmark: how many AllReduces does one optimizer step need?

Arms (FP32, same corpus and shuffle):

- ``k1``          the Chapter 29 body: one global batch sharded across the
                  ranks, one backward, one AllReduce — the single-micro-step
                  anchor
- ``k2_every``    K=2 micro-steps per optimizer step, AllReduce after every
                  micro-step (baseline of this chapter)
- ``k2_last``     same split, ``no_sync``: one AllReduce after the last
                  micro-step (variant)
- ``k4_every`` / ``k4_last``
                  the same comparison at accumulation depth K=4, one sample
                  per micro-step

World size 1 runs the ``k1`` anchor only; world sizes 2, 4, and 8 run every
arm. Each arm's global batch is its per-rank batch (2 for ``k1``/``k2_*``,
4 for ``k4_*``) times the world size, so world size 2 reproduces the
chapter's original global batches of 4 and 8.

Per optimizer step the script reports the AllReduce count, total AllReduce
bytes, exposed AllReduce time, and the step wall clock. ``k*_last`` should
match ``k1`` in communication volume regardless of K; ``k*_every`` multiplies
it by K.
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
from no_sync_core import accumulate_step, build_micro_batches
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything


ARMS = ("k1", "k2_every", "k2_last", "k4_every", "k4_last")
ARM_PER_RANK_BATCH = {"k1": 2, "k2_every": 2, "k2_last": 2, "k4_every": 4, "k4_last": 4}
ARMS_BY_WORLD_SIZE = {1: ("k1",), 2: ARMS, 4: ARMS, 8: ARMS}


def arm_plan(arm: str, sample_count: int, seed: int, world_size: int) -> list:
    """Batch plan for one arm; the global batch scales with the world size.

    Each arm keeps its per-rank shard fixed — 2 members for ``k1``/``k2_*``
    and 4 for ``k4_*`` — so every micro-batch stays one sample at every world
    size, and world size 2 reproduces the chapter's original global batches
    of 4 and 8.
    """

    if arm not in ARMS_BY_WORLD_SIZE[world_size]:
        raise ValueError(f"arm {arm} is not defined for world size {world_size}")
    return make_batch_plan(sample_count, ARM_PER_RANK_BATCH[arm] * world_size, seed)


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
        if arm == "k1":
            num_micro, plan = 1, plans["k1"]
        else:
            depth, mode = arm.split("_")
            num_micro = int(depth.removeprefix("k"))
            plan = plans[arm]
        for repeat in range(1, args.repeats + 1):
            seed_everything(args.seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)
            try:
                for warmup_index in range(args.warmup):
                    step_args = (plan[warmup_index], args.pad_token_id)
                    if arm == "k1":
                        host_batch, accounting = build_global_batch(samples, step_args[0], rank, world_size, step_args[1])
                        ddp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
                    else:
                        micros, accounting = build_micro_batches(samples, step_args[0], rank, world_size, num_micro, step_args[1])
                        accumulate_step(model, optimizer, micros, accounting["supervised_global"], device, args.max_grad_norm, world_size, rank, sync_mode=mode, collect_telemetry=False)
                torch.cuda.reset_peak_memory_stats(device)
                steps = []
                for step_index in range(args.warmup, len(plan)):
                    if arm == "k1":
                        host_batch, accounting = build_global_batch(samples, plan[step_index], rank, world_size, args.pad_token_id)
                        metrics = ddp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=True)
                        supervised = accounting["supervised_global"]
                    else:
                        micros, accounting = build_micro_batches(samples, plan[step_index], rank, world_size, num_micro, args.pad_token_id)
                        metrics = accumulate_step(model, optimizer, micros, accounting["supervised_global"], device, args.max_grad_norm, world_size, rank, sync_mode=mode, collect_telemetry=True)
                        supervised = accounting["supervised_global"]
                    metrics.update({
                        "step": float(step_index - args.warmup + 1),
                        "global_supervised_tokens": supervised,
                        "global_supervised_tokens_per_second": supervised / (metrics["step_time_ms"] / 1000),
                    })
                    steps.append(metrics)
                total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
                step_times = [step["step_time_ms"] for step in steps]
                results[arm].append({
                    "repeat": repeat,
                    "status": "success",
                    "measured_steps": len(steps),
                    "steps": steps,
                    "aggregate": {
                        "step_time_mean_ms": fmean(step_times),
                        "step_time_p50_ms": percentile(step_times, 0.5),
                        "step_time_p95_ms": percentile(step_times, 0.95),
                        "step_time_max_ms": max(step_times),
                        "allreduce_mean_ms": fmean(step["grad_allreduce_ms"] for step in steps),
                        "allreduce_count_mean": fmean(step["allreduce_count"] for step in steps),
                        "allreduce_bytes_mean": fmean(step["allreduce_bytes_total"] for step in steps),
                        "forward_mean_ms": fmean(step["forward_time_ms"] for step in steps),
                        "backward_mean_ms": fmean(step["backward_time_ms"] for step in steps),
                        "optimizer_mean_ms": fmean(step["optimizer_time_ms"] for step in steps),
                    },
                    "throughput": {
                        "global_supervised_tokens_per_second": sum(step["global_supervised_tokens"] for step in steps) / total_seconds,
                    },
                    "memory": {
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    },
                    "first_last_loss": {"first": steps[0]["local_loss"], "last": steps[-1]["local_loss"]},
                })
            except torch.cuda.OutOfMemoryError:
                results[arm].append({"repeat": repeat, "status": "oom", "allocated_bytes": torch.cuda.memory_allocated(device)})
            entry = results[arm][-1]
            if rank == 0:
                if entry["status"] == "success":
                    print(f"{arm} repeat={entry['repeat']}: p50 {entry['aggregate']['step_time_p50_ms']:.1f} ms "
                          f"allreduce {entry['aggregate']['allreduce_mean_ms']:.1f} ms "
                          f"({entry['aggregate']['allreduce_count_mean']:.0f} calls) "
                          f"peak {entry['memory']['peak_allocated_bytes'] / 2**30:.2f} GiB", flush=True)
                else:
                    print(f"{arm} repeat={entry['repeat']}: {entry['status']} allocated {entry.get('allocated_bytes', 0) / 2**30:.2f} GiB", flush=True)
            if entry["status"] != "success":
                break
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
        "allreduce_mean_ms": fmean(entry["aggregate"]["allreduce_mean_ms"] for entry in successful),
        "allreduce_count_mean": fmean(entry["aggregate"]["allreduce_count_mean"] for entry in successful),
        "allreduce_bytes_mean": fmean(entry["aggregate"]["allreduce_bytes_mean"] for entry in successful),
        "forward_mean_ms": fmean(entry["aggregate"]["forward_mean_ms"] for entry in successful),
        "backward_mean_ms": fmean(entry["aggregate"]["backward_mean_ms"] for entry in successful),
        "optimizer_mean_ms": fmean(entry["aggregate"]["optimizer_mean_ms"] for entry in successful),
        "global_supervised_tokens_per_second_mean": fmean(entry["throughput"]["global_supervised_tokens_per_second"] for entry in successful),
        "peak_allocated_bytes_mean": fmean(entry["memory"]["peak_allocated_bytes"] for entry in successful),
    }


def compute_scaling(summary):
    """Comm/time scaling of every arm against the k1 single-micro-step anchor."""

    scaling = {}
    if not summary.get("k1", {}).get("successful_repeats"):
        return scaling
    k1_time = summary["k1"]["step_time_p50_ms_mean"]
    k1_bytes = summary["k1"]["allreduce_bytes_mean"]
    for arm, entry in summary.items():
        if not entry.get("successful_repeats"):
            continue
        scaling[arm] = {
            "step_time_ratio_vs_k1": entry["step_time_p50_ms_mean"] / k1_time,
            "comm_bytes_ratio_vs_k1": entry["allreduce_bytes_mean"] / k1_bytes,
            "allreduce_count": entry["allreduce_count_mean"],
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
        "global_batch_sizes": {arm: ARM_PER_RANK_BATCH[arm] * world_size for arm in sorted(merged)},
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
            "step_time": "wall clock around a synchronized optimizer step (all micro-steps included)",
            "allreduce_ms": "sum of exposed AllReduce durations within the step (every: num_micro calls; last: one call)",
            "allreduce_bytes": "FP32 gradient bytes moved by AllReduce per optimizer step",
        },
    })
    write_json(Path(output_path), payload)
    print(json.dumps({"summary": summary, "scaling": scaling}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter31_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--only-arm", default=None, choices=ARMS)
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
        arms = tuple(arm for arm in ARMS_BY_WORLD_SIZE[world_size] if args.only_arm in (None, arm))
        # Arms mix per-rank shards of 2 and 4; keep the corpus divisible for
        # every plan actually being run.
        divisor = max(ARM_PER_RANK_BATCH[arm] for arm in arms) * world_size
        usable = len(samples) - len(samples) % divisor
        dropped_for_divisibility = len(samples) - usable
        samples = samples[:usable]
        if rank == 0:
            print(
                f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
                f"dropped_for_divisibility={dropped_for_divisibility}"
            )
        plans = {arm: arm_plan(arm, len(samples), args.seed, world_size) for arm in arms}

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
    arms = tuple(arm for arm in ARMS_BY_WORLD_SIZE[args.world_size] if args.only_arm in (None, arm))
    # Arms mix per-rank shards of 2 and 4; keep the corpus divisible for every
    # plan actually being run.
    divisor = max(ARM_PER_RANK_BATCH[arm] for arm in arms) * args.world_size
    usable = len(samples) - len(samples) % divisor
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    plans = {arm: arm_plan(arm, len(samples), args.seed, args.world_size) for arm in arms}

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
