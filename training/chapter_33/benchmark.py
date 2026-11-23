#!/usr/bin/env python3
"""Chapter 33 benchmark: ZeRO-1 (full gradients) versus ZeRO-2 (Reduce-Scatter).

Arms over the same corpus and shuffle (FP32, eager attention); both keep the
optimizer state sharded (Chapter 32) and parameters fully replicated; the ONLY
difference is the gradient:

- ``zero1_wN``  full gradient on every rank, one flat post-backward AllReduce
                SUM, ShardedStateAdamW, parameter AllGather (Chapter 32 engine)
- ``zero2_wN``  per-parameter Reduce-Scatter straight into persistent gradient
                shard buffers, ShardedGradAdamW, the same parameter AllGather

Both arms run at every world size in ``ARMS_BY_WORLD_SIZE``, and the global
batch of each round is ``PER_RANK_BATCH * world_size`` so the W=2 round
reproduces the chapter's original global-batch-4 sharded 2+2 baseline while
larger rounds keep the per-rank workload fixed. Reduce-Scatter moves the same
per-link bytes as the AllReduce, so the communication time should stay close
to ZeRO-1's at every world size; the gradient memory per rank shrinks with
1/world_size. The per-parameter Reduce-Scatter chain (about 200 small
collectives) is measured honestly — real implementations flatten parameters
into buckets.

Launch with ``python benchmark.py`` (mp.spawn, single node, one process group
per world size) or under ``torchrun`` (single or multi node; every rank
prepares the data deterministically and rank 0 finalizes the JSON).
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
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything
from zero1_core import (
    ShardedStateAdamW,
    all_gather_parameters as all_gather_zero1,
    memory_ledger as ledger_zero1,
)
from zero2_core import ShardedGradAdamW, memory_ledger, zero2_train_step


PER_RANK_BATCH = 2  # the W=2 baseline shards the global batch of 4 as 2+2
ARMS_BY_WORLD_SIZE = {
    1: ("zero1_w1", "zero2_w1"),
    2: ("zero1_w2", "zero2_w2"),
    4: ("zero1_w4", "zero2_w4"),
    8: ("zero1_w8", "zero2_w8"),
}


def arm_engine(arm: str) -> str:
    """The chapter engine behind a world-size-suffixed arm name."""

    return arm.rsplit("_w", 1)[0]


def _telemetry(rank: int) -> dict:
    """Local-GPU power/temperature/clock; same fields as the in-step samplers.

    The zero2 arm samples here instead of inside ``zero2_train_step`` so the
    physical GPU follows LOCAL_RANK on multi-node launches without touching
    the frozen mechanism file.
    """

    import os
    import subprocess

    local_id = int(os.environ.get("LOCAL_RANK", rank))
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={local_id}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


def arm_plan(arm: str, sample_count: int, seed: int, world_size: int) -> list:
    """Global batch plan for one arm; every arm at one world size shares it.

    The per-rank shard stays at ``PER_RANK_BATCH`` samples, so the global
    batch grows to ``PER_RANK_BATCH * world_size`` (weak-scaling style) and
    the zero1-versus-zero2 comparison at each world size is apples to apples.
    """

    if arm_engine(arm) in ("zero1", "zero2"):
        return make_batch_plan(sample_count, PER_RANK_BATCH * world_size, seed)
    raise ValueError(f"unknown arm {arm}")


def package_repeat(arm: str, repeat: int, steps: list, device) -> dict:
    total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
    step_times = [step["step_time_ms"] for step in steps]
    return {
        "repeat": repeat,
        "status": "success",
        "measured_steps": len(steps),
        "steps": steps,
        "aggregate": {
            "step_time_mean_ms": fmean(step_times),
            "step_time_p50_ms": percentile(step_times, 0.5),
            "step_time_p95_ms": percentile(step_times, 0.95),
            "step_time_max_ms": max(step_times),
            "grad_comm_mean_ms": fmean(step["grad_allreduce_ms"] for step in steps),
            "optimizer_mean_ms": fmean(step["optimizer_time_ms"] for step in steps),
            "param_allgather_mean_ms": fmean(step.get("param_allgather_ms", 0.0) for step in steps),
        },
        "throughput": {
            "global_supervised_tokens_per_second": sum(step["global_supervised_tokens"] for step in steps) / total_seconds,
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "first_last_loss": {"first": steps[0]["local_loss"], "last": steps[-1]["local_loss"]},
    }


def _worker(rank: int, world_size: int, port: int, args_ns: dict, plans: dict, arms: list, samples_obj=None) -> None:
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
    ledgers = {}
    for arm in arms:
        plan = plans[arm]
        engine = arm_engine(arm)
        if engine == "zero1":
            optimizer = ShardedStateAdamW(list(model.parameters()), world_size, rank, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
            ledgers[arm] = ledger_zero1(list(model.parameters()), optimizer, world_size)
        else:
            optimizer = ShardedGradAdamW(list(model.parameters()), world_size, rank, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
            ledgers[arm] = memory_ledger(list(model.parameters()), optimizer)
        for repeat in range(1, args.repeats + 1):
            seed_everything(args.seed)
            try:
                for warmup_index in range(args.warmup):
                    host_batch, accounting = build_global_batch(samples, plan[warmup_index], rank, world_size, args.pad_token_id)
                    if engine == "zero1":
                        ddp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
                        all_gather_zero1(list(model.parameters()), world_size, rank)
                    else:
                        zero2_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
                torch.cuda.reset_peak_memory_stats(device)
                steps = []
                for step_index in range(args.warmup, len(plan)):
                    host_batch, accounting = build_global_batch(samples, plan[step_index], rank, world_size, args.pad_token_id)
                    if engine == "zero1":
                        metrics = ddp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=True)
                        allgather_ms, allgather_bytes = all_gather_zero1(list(model.parameters()), world_size, rank)
                        metrics["param_allgather_ms"] = allgather_ms
                        metrics["step_time_ms"] += allgather_ms
                    else:
                        # Telemetry is sampled at the benchmark level so it
                        # follows LOCAL_RANK across nodes; zero2_core stays frozen.
                        metrics = zero2_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
                        if device.type == "cuda":
                            metrics.update(_telemetry(rank))
                    index = step_index - args.warmup
                    metrics.update({
                        "step": float(index + 1),
                        "global_supervised_tokens": accounting["supervised_global"],
                        "global_supervised_tokens_per_second": accounting["supervised_global"] / (metrics["step_time_ms"] / 1000),
                    })
                    steps.append(metrics)
                results[arm].append(package_repeat(arm, repeat, steps, device))
            except torch.cuda.OutOfMemoryError:
                results[arm].append({"repeat": repeat, "status": "oom", "allocated_bytes": torch.cuda.memory_allocated(device)})
            entry = results[arm][-1]
            if rank == 0:
                if entry["status"] == "success":
                    print(f"{arm} repeat={entry['repeat']}: p50 {entry['aggregate']['step_time_p50_ms']:.1f} ms "
                          f"sup/s {entry['throughput']['global_supervised_tokens_per_second']:.1f} "
                          f"peak {entry['memory']['peak_allocated_bytes'] / 2**30:.2f} GiB", flush=True)
                else:
                    print(f"{arm} repeat={entry['repeat']}: {entry['status']} allocated {entry.get('allocated_bytes', 0) / 2**30:.2f} GiB", flush=True)
            if entry["status"] != "success":
                break  # a failing arm is reported once; later repeats cannot recover it
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args_ns["part_path"]), {"world_size": world_size, "results": results, "ledgers": ledgers})


def summarize_entries(entries):
    successful = [entry for entry in entries if entry["status"] == "success"]
    if not successful:
        return {"successful_repeats": 0}
    return {
        "successful_repeats": len(successful),
        "step_time_p50_ms_mean": fmean(entry["aggregate"]["step_time_p50_ms"] for entry in successful),
        "grad_comm_mean_ms": fmean(entry["aggregate"]["grad_comm_mean_ms"] for entry in successful),
        "optimizer_mean_ms": fmean(entry["aggregate"]["optimizer_mean_ms"] for entry in successful),
        "param_allgather_mean_ms": fmean(entry["aggregate"]["param_allgather_mean_ms"] for entry in successful),
        "global_supervised_tokens_per_second_mean": fmean(entry["throughput"]["global_supervised_tokens_per_second"] for entry in successful),
        "peak_allocated_bytes_mean": fmean(entry["memory"]["peak_allocated_bytes"] for entry in successful),
        "peak_reserved_bytes_mean": fmean(entry["memory"]["peak_reserved_bytes"] for entry in successful),
    }


def compute_scaling(summary):
    """ZeRO-2 saving against the same-world-size zero1 baseline arm."""

    scaling = {}
    for world_size in sorted(ARMS_BY_WORLD_SIZE):
        zero1_arm, zero2_arm = f"zero1_w{world_size}", f"zero2_w{world_size}"
        if not (summary.get(zero1_arm, {}).get("successful_repeats") and summary.get(zero2_arm, {}).get("successful_repeats")):
            continue
        scaling[f"zero2_vs_zero1_w{world_size}"] = {
            "step_time_ratio_zero2_over_zero1": summary[zero2_arm]["step_time_p50_ms_mean"] / summary[zero1_arm]["step_time_p50_ms_mean"],
            "grad_comm_time_ratio_zero2_over_zero1": summary[zero2_arm]["grad_comm_mean_ms"] / summary[zero1_arm]["grad_comm_mean_ms"],
            "peak_allocated_ratio_zero2_over_zero1": summary[zero2_arm]["peak_allocated_bytes_mean"] / summary[zero1_arm]["peak_allocated_bytes_mean"],
            "peak_allocated_saving_bytes": summary[zero1_arm]["peak_allocated_bytes_mean"] - summary[zero2_arm]["peak_allocated_bytes_mean"],
        }
    return scaling


def finalize_output(merged, summary, scaling, ledgers, args, world_size, sample_count, output_path):
    # Persist the raw measurements first: derived summaries must never be able
    # to destroy them.
    write_json(Path(output_path), {"partial": True, "repeats_detail": merged, "summary_partial": summary})

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
        "num_samples": sample_count,
        "global_batch_sizes": {
            f"zero1_w{n}": PER_RANK_BATCH * n for n in sorted(ARMS_BY_WORLD_SIZE) if f"zero1_w{n}" in merged
        },
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "parameter_bytes": parameter_count_bytes() * 4,
        "nnodes": int(_os.environ.get("NNODES", "1")),
        "node_rank": int(_os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in _os.environ else "mp.spawn",
        "state_ledgers": ledgers,
        "repeats_detail": merged,
        "summary": summary,
        "scaling": scaling,
        "timing_scope": {
            "step_time": "wall clock of a synchronized optimizer step including its communication segments",
            "grad_comm": "zero1: one flat AllReduce; zero2: the per-parameter Reduce-Scatter chain",
            "param_allgather": "post-step chunked parameter AllGather, event-timed separately and included in step_time",
            "throughput": "global (all-rank) supervised tokens divided by the per-step wall-clock window",
        },
    })
    write_json(Path(output_path), payload)
    print(json.dumps({"summary": summary, "scaling": scaling}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter33_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--only-arm", default=None,
                        choices=tuple(arm for arms in ARMS_BY_WORLD_SIZE.values() for arm in arms))
    parser.add_argument("--world-sizes", type=str, default="all",
                        help="comma list subset of ARMS_BY_WORLD_SIZE keys, e.g. '4' or '2,4'; 'all' runs every arm")
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
            raise SystemExit(f"torchrun WORLD_SIZE {world_size} is not one of {sorted(ARMS_BY_WORLD_SIZE)}")
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
        # Every arm plan shards PER_RANK_BATCH * world_size members per step.
        usable = len(samples) - len(samples) % (PER_RANK_BATCH * world_size)
        dropped_for_divisibility = len(samples) - usable
        samples = samples[:usable]
        if rank == 0:
            print(
                f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
                f"dropped_for_divisibility={dropped_for_divisibility}"
            )
        arms = tuple(arm for arm in ARMS_BY_WORLD_SIZE[world_size] if args.only_arm in (None, arm))
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
            ledgers = part.get("ledgers", {})
            summary = {arm: summarize_entries(entries) for arm, entries in merged.items()}
            scaling = compute_scaling(summary)
            finalize_output(merged, summary, scaling, ledgers, args, world_size, len(samples), args.output)
            Path(args_ns["part_path"]).unlink()
        return

    # mp.spawn path: single node, one process group per world size.
    requested = sorted(ARMS_BY_WORLD_SIZE) if args.world_sizes == "all" else sorted(
        int(value) for value in args.world_sizes.split(",") if value
    )
    unknown = [value for value in requested if value not in ARMS_BY_WORLD_SIZE]
    if unknown:
        parser.error(f"--world-sizes values not supported: {unknown}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < max(requested):
        raise SystemExit(f"the requested world sizes {requested} need {max(requested)} CUDA GPUs on one node")
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
    # One corpus feeds every requested world size, whose arm plans shard
    # PER_RANK_BATCH * world_size members per step; cover the largest one.
    usable = len(samples) - len(samples) % (PER_RANK_BATCH * max(requested))
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )

    args_ns = {
        "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "pad_token_id": args.pad_token_id,
        "part_path": str(args.output.with_suffix(".part.json")),
        "samples_path": str(args.output.with_suffix(".samples.pt")),
    }
    torch.save(samples, args_ns["samples_path"])

    merged: dict[str, list] = {}
    ledgers = {}
    schedule = [
        (world_size, tuple(arm for arm in ARMS_BY_WORLD_SIZE[world_size] if args.only_arm in (None, arm)))
        for world_size in requested
    ]
    for world_size, arms in schedule:
        if not arms:
            continue
        # Arm names carry no world size, so the plan is rebuilt per round.
        plans = {arm: arm_plan(arm, len(samples), args.seed, world_size) for arm in arms}
        port = free_port()
        mp.spawn(_worker, args=(world_size, port, args_ns, plans, list(arms)), nprocs=world_size, join=True)
        part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
        for arm, entries in part["results"].items():
            merged[arm] = entries
        ledgers.update(part.get("ledgers", {}))
    Path(args_ns["part_path"]).unlink()
    Path(args_ns["samples_path"]).unlink()

    summary = {arm: summarize_entries(entries) for arm, entries in merged.items()}
    scaling = compute_scaling(summary)
    finalize_output(merged, summary, scaling, ledgers, args, max(requested), len(samples), args.output)


if __name__ == "__main__":
    main()
