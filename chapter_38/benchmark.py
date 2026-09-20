#!/usr/bin/env python3
"""Chapter 38 benchmark: pure TP (replicated activations) vs Sequence Parallel.

Two arms per world size; the TP degree IS the world size, and the ONLY
variable between the arms is whether the activations BETWEEN the parallel
regions are also sequence-sharded (weight layout, batch plan and optimizer
are identical):

- ``tp_wN``  N GPUs, chapter-36 layout: attention AND MLP sharded, residual
  stream and LayerNorm activations replicated on every rank
- ``sp_wN``  N GPUs, same weight shards PLUS sequence shards across the non-TP
  regions (AllGather / ReduceScatter at every TP-region boundary)

Both arms replicate DATA: every rank of a group runs the SAME global batch of
``PER_RANK_BATCH * tp_size`` samples, so per-step useful tokens grow with the
TP degree (the chapter-35/21 weak-scaling convention).  The honest expectation
for 0.6B at L=512 is that SP trades extra boundary communication for
activation memory on the norm/residual regions; both outcomes are recorded.

Per rank the benchmark reports the usual peak allocated/reserved memory, the
state ledger, the measured saved-for-backward activation bytes (taken once per
arm under ``ActivationLedger``, outside the timed repeats) and the
recorder-timed AllGather/ReduceScatter/AllReduce totals with ring-model byte
ledgers for both arms.

Launch with ``python benchmark.py`` (mp.spawn, single node, one process group
per world size) or under ``torchrun`` (single or multi node; every rank
prepares the data deterministically and rank 0 finalizes the JSON).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from experiment_utils import environment_record, percentile, write_json
from mlp_tp import tp_memory_ledger, tp_train_step
from sequence_parallel import (
    ActivationLedger,
    arm_byte_ledger,
    sp_loss_terms,
    sp_memory_ledger,
    sp_train_step,
)
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything
from tp_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    build_replicated_batch,
    cleanup_distributed,
    encode_samples,
    free_port,
    load_tokenizer,
    load_tokenizer_and_model_sp,
    load_tokenizer_and_model_tp,
    make_batch_plan,
    parameter_count_bytes,
    setup_distributed,
)


PER_RANK_BATCH = 2  # the per-step global batch is PER_RANK_BATCH * tp_size
MAIN_MAX_LENGTH = 512
ARMS_BY_WORLD_SIZE = {
    2: ("tp_w2", "sp_w2"),
    4: ("tp_w4", "sp_w4"),
}


def arm_tp_size(arm: str) -> int:
    """The TP degree encoded in an arm name's ``_wN`` suffix."""

    return int(arm.rsplit("_w", 1)[1])


def arm_is_sp(arm: str) -> bool:
    """Sequence Parallel arms shard the non-TP activations too."""

    return arm.startswith("sp_")


def arm_plan(arm: str, sample_count: int, seed: int, world_size: int) -> list:
    """Global batch plan for one arm; every arm at one world size shares it.

    TP and SP both replicate data, so the plan is identical on all ranks of
    the group and the global batch grows as PER_RANK_BATCH * tp_size.
    """

    if arm_tp_size(arm) != world_size:
        raise ValueError(f"arm {arm} needs world size {arm_tp_size(arm)}, got {world_size}")
    return make_batch_plan(sample_count, PER_RANK_BATCH * world_size, seed)


def measure_saved_activation_bytes(model, samples, plan, args_ns, world_size, rank, is_sp: bool) -> dict:
    """One forward+loss under the ActivationLedger (outside timed repeats)."""

    device = next(model.parameters()).device
    host_batch, _ = build_replicated_batch(samples, plan[0], args_ns["pad_token_id"])
    from mlp_tp import to_device_batch

    batch = to_device_batch(host_batch, device)
    ledger = ActivationLedger(model.parameters())
    with ledger:
        if is_sp:
            sequence_length = host_batch.input_ids.shape[1]
            # Full sequence in; SequenceParallelModel.forward slices its own
            # window internally (same convention as sp_train_step).
            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            loss_sum, count = sp_loss_terms(logits, batch.labels, rank, world_size, sequence_length)
            loss_sum / float(count.clamp(min=1).item())
        else:
            from mini_batch import batch_loss

            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            batch_loss(logits, batch.labels, "token_mean")
    del batch, logits
    return {"saved_bytes": ledger.saved_bytes, "saved_count": ledger.saved_count}


def package_repeat(arm: str, repeat: int, steps: list, device, activation: dict) -> dict:
    total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
    step_times = [step["step_time_ms"] for step in steps]
    aggregate = {
        "step_time_mean_ms": fmean(step_times),
        "step_time_p50_ms": percentile(step_times, 0.5),
        "step_time_p95_ms": percentile(step_times, 0.95),
        "step_time_max_ms": max(step_times),
        "h2d_mean_ms": fmean(step["h2d_time_ms"] for step in steps),
        "forward_compute_mean_ms": fmean(step["forward_time_ms"] for step in steps),
        "backward_compute_mean_ms": fmean(step["backward_time_ms"] for step in steps),
        "optimizer_mean_ms": fmean(step["optimizer_time_ms"] for step in steps),
        "tp_allreduce_forward_mean_ms": fmean(step["tp_allreduce_forward_ms"] for step in steps),
        "tp_allreduce_backward_mean_ms": fmean(step["tp_allreduce_backward_ms"] for step in steps),
        "tp_allreduce_total_mean_ms": fmean(step["tp_allreduce_total_ms"] for step in steps),
        "tp_allreduce_count_per_step": fmean(
            step["tp_allreduce_forward_count"] + step["tp_allreduce_backward_count"] for step in steps
        ),
    }
    if arm_is_sp(arm):
        aggregate.update({
            "sp_boundary_comm_mean_ms": fmean(step["sp_boundary_comm_total_ms"] for step in steps),
            "sp_total_comm_mean_ms": fmean(step["sp_total_comm_ms"] for step in steps),
            "sp_comm_exposed_ratio_mean": fmean(step["sp_comm_exposed_ratio"] for step in steps),
            "sp_allgather_forward_mean_ms": fmean(step["sp_allgather_forward_ms"] for step in steps),
            "sp_reduce_scatter_forward_mean_ms": fmean(step["sp_reduce_scatter_forward_ms"] for step in steps),
            "sp_allgather_backward_mean_ms": fmean(step["sp_allgather_backward_ms"] for step in steps),
            "sp_loss_allreduce_mean_ms": fmean(step["sp_loss_allreduce_ms"] for step in steps),
        })
        for direction in ("forward", "backward"):
            for op in ("allgather", "reduce_scatter"):
                key = f"sp_{op}_{direction}_ring_bytes"
                if key in steps[0]:
                    aggregate[f"{key}_mean"] = fmean(step[key] for step in steps)
    else:
        aggregate["tp_allreduce_exposed_ratio_mean"] = fmean(step["tp_allreduce_exposed_ratio"] for step in steps)
    return {
        "repeat": repeat,
        "status": "success",
        "measured_steps": len(steps),
        "steps": steps,
        "aggregate": aggregate,
        "throughput": {
            "global_supervised_tokens_per_second": sum(step["supervised_tokens"] for step in steps) / total_seconds,
            "global_physical_tokens_per_second": sum(step["physical_tokens"] for step in steps) / total_seconds,
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "saved_activation_bytes": activation,
        "first_last_loss": {"first": steps[0]["global_loss"], "last": steps[-1]["global_loss"]},
    }


def _worker(rank: int, world_size: int, port: int, args_ns: dict, plans: dict, arms: list, samples_obj=None) -> None:
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

    results = {arm: [] for arm in arms}
    ledgers = {}
    activations = {}
    for arm in arms:
        # Each arm owns a DIFFERENT activation layout (sequence shards or not),
        # so the model is built and torn down per arm; weight shards match.
        is_sp = arm_is_sp(arm)
        if is_sp:
            _, model = load_tokenizer_and_model_sp(device, rank, world_size)
        else:
            _, model = load_tokenizer_and_model_tp(device, rank, world_size, attention_parallel=True)
        plan = plans[arm]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
            eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False,
        )
        for repeat in range(1, args.repeats + 1):
            seed_everything(args.seed)
            try:
                for warmup_index in range(args.warmup):
                    host_batch, accounting = build_replicated_batch(samples, plan[warmup_index], args.pad_token_id)
                    if is_sp:
                        sp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
                    else:
                        tp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
                # AdamW states materialize at the first step, so the ledger is
                # captured after warm-up; peak counters reset for measurement.
                if repeat == 1:
                    ledgers[arm] = (sp_memory_ledger if is_sp else tp_memory_ledger)(model, optimizer, world_size)
                    ledgers[arm]["measured_allocated_after_state_bytes"] = float(torch.cuda.memory_allocated(device))
                    activations[arm] = measure_saved_activation_bytes(
                        model, samples, plan, args_ns, world_size, rank, is_sp
                    )
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                steps = []
                for step_index in range(args.warmup, len(plan)):
                    host_batch, accounting = build_replicated_batch(samples, plan[step_index], args.pad_token_id)
                    # collect_telemetry samples power/temperature on LOCAL_RANK's GPU.
                    if is_sp:
                        metrics = sp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=True)
                    else:
                        metrics = tp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=True)
                    index = step_index - args.warmup
                    metrics.update({
                        "step": float(index + 1),
                        "global_supervised_tokens": accounting["supervised_global"],
                        "global_supervised_tokens_per_second": accounting["supervised_global"] / (metrics["step_time_ms"] / 1000),
                    })
                    steps.append(metrics)
                results[arm].append(package_repeat(arm, repeat, steps, device, activations[arm]))
            except torch.cuda.OutOfMemoryError:
                results[arm].append({"repeat": repeat, "status": "oom", "allocated_bytes": torch.cuda.memory_allocated(device)})
            entry = results[arm][-1]
            if rank == 0:
                if entry["status"] == "success":
                    comm_text = (
                        f"sp-comm {entry['aggregate']['sp_total_comm_mean_ms']:.1f} ms"
                        if arm_is_sp(arm) else
                        f"tp-comm {entry['aggregate']['tp_allreduce_total_mean_ms']:.1f} ms"
                    )
                    print(f"{arm} repeat={entry['repeat']}: p50 {entry['aggregate']['step_time_p50_ms']:.1f} ms "
                          f"{comm_text} "
                          f"sup/s {entry['throughput']['global_supervised_tokens_per_second']:.1f} "
                          f"peak {entry['memory']['peak_allocated_bytes'] / 2**30:.2f} GiB "
                          f"saved-act {entry['saved_activation_bytes']['saved_bytes'] / 2**20:.1f} MiB", flush=True)
                else:
                    print(f"{arm} repeat={entry['repeat']}: {entry['status']} allocated {entry.get('allocated_bytes', 0) / 2**30:.2f} GiB", flush=True)
            if entry["status"] != "success":
                break  # a failing arm is reported once; later repeats cannot recover it
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args_ns["part_path"]), {
            "world_size": world_size, "results": results, "ledgers": ledgers, "activations": activations,
        })


def summarize_entries(entries):
    successful = [entry for entry in entries if entry["status"] == "success"]
    if not successful:
        return {"successful_repeats": 0}
    summary = {
        "successful_repeats": len(successful),
        "step_time_p50_ms_mean": fmean(entry["aggregate"]["step_time_p50_ms"] for entry in successful),
        "forward_compute_mean_ms": fmean(entry["aggregate"]["forward_compute_mean_ms"] for entry in successful),
        "backward_compute_mean_ms": fmean(entry["aggregate"]["backward_compute_mean_ms"] for entry in successful),
        "optimizer_mean_ms": fmean(entry["aggregate"]["optimizer_mean_ms"] for entry in successful),
        "tp_allreduce_total_mean_ms": fmean(entry["aggregate"]["tp_allreduce_total_mean_ms"] for entry in successful),
        "global_supervised_tokens_per_second_mean": fmean(entry["throughput"]["global_supervised_tokens_per_second"] for entry in successful),
        "peak_allocated_bytes_mean": fmean(entry["memory"]["peak_allocated_bytes"] for entry in successful),
        "peak_reserved_bytes_mean": fmean(entry["memory"]["peak_reserved_bytes"] for entry in successful),
        "saved_activation_bytes_mean": fmean(entry["saved_activation_bytes"]["saved_bytes"] for entry in successful),
    }
    first = successful[0]["aggregate"]
    for key, value in first.items():
        if key not in summary and key.endswith("_mean"):
            summary[key] = fmean(entry["aggregate"][key] for entry in successful)
    return summary


def compute_scaling(summary):
    """The SP arm against the pure-TP arm at the SAME world size."""

    scaling = {}
    for world_size in sorted(ARMS_BY_WORLD_SIZE):
        baseline_arm, variant_arm = f"tp_w{world_size}", f"sp_w{world_size}"
        if not (summary.get(baseline_arm, {}).get("successful_repeats") and summary.get(variant_arm, {}).get("successful_repeats")):
            continue
        scaling[f"sp{world_size}_vs_tp_w{world_size}"] = {
            "step_time_ratio_sp_over_tp": summary[variant_arm]["step_time_p50_ms_mean"] / summary[baseline_arm]["step_time_p50_ms_mean"],
            "throughput_ratio_sp_over_tp": summary[variant_arm]["global_supervised_tokens_per_second_mean"] / summary[baseline_arm]["global_supervised_tokens_per_second_mean"],
            "peak_allocated_ratio_sp_over_tp": summary[variant_arm]["peak_allocated_bytes_mean"] / summary[baseline_arm]["peak_allocated_bytes_mean"],
            "saved_activation_ratio_sp_over_tp": summary[variant_arm]["saved_activation_bytes_mean"] / max(summary[baseline_arm]["saved_activation_bytes_mean"], 1.0),
            "extra_boundary_comm_ms": summary[variant_arm].get("sp_boundary_comm_mean_ms", 0.0),
            "sp_total_comm_ms": summary[variant_arm].get("sp_total_comm_mean_ms", 0.0),
            "tp_total_comm_ms": summary[baseline_arm]["tp_allreduce_total_mean_ms"],
        }
    return scaling


def finalize_output(merged, summary, scaling, ledgers, activations, args, world_size, sample_count, output_path):
    # Persist the raw measurements first: derived summaries must never be able
    # to destroy them.
    write_json(Path(output_path), {"partial": True, "repeats_detail": merged, "summary_partial": summary})

    payload = environment_record(world_size=world_size, dp_degree=1, tp_degree=world_size)
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
        "arm_meaning": {
            "tp_wN": "chapter-36 layout: attention AND MLP sharded; residual stream and LayerNorm activations replicated",
            "sp_wN": "same weight shards PLUS sequence shards across the non-TP regions (this chapter's variant)",
        },
        "qwen3_head_layout": {
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "per_rank_q_heads": {str(tp): 32 // tp for tp in sorted(ARMS_BY_WORLD_SIZE)},
            "per_rank_kv_heads": {str(tp): 8 // tp for tp in sorted(ARMS_BY_WORLD_SIZE)},
        },
        "num_samples": sample_count,
        "global_batch_sizes": {
            arm: PER_RANK_BATCH * arm_tp_size(arm) for arm in sorted(merged.keys())
        },
        "batch_rule": "TP and SP replicate data: every rank of a group runs the SAME batch of PER_RANK_BATCH * tp_size samples; throughput counts the batch once, not per-rank multiplicity",
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "parameter_bytes_full_model": parameter_count_bytes() * 4,
        "nnodes": int(os.environ.get("NNODES", "1")),
        "node_rank": int(os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "state_ledgers": ledgers,
        "saved_activation_bytes_by_rank": activations,
        "repeats_detail": merged,
        "summary": summary,
        "scaling": scaling,
        "analytic_byte_ledger": arm_byte_ledger(
            PER_RANK_BATCH * max(sorted(ARMS_BY_WORLD_SIZE)), args.max_length, 1024, world_size, 28
        ) if merged else {},
        "timing_scope": {
            "step_time": "wall clock of a synchronized optimizer step",
            "tp/sp collectives": "event-timed per collective and summed per step (SP: forward AllGather+ReduceScatter per block, backward f-AllReduce per block, sync hooks, and the ḡ AllGathers; TP: forward/backward AllReduces); they sit INSIDE the forward/backward windows and are fully exposed on the critical path",
            "forward/backward compute": "the forward/backward event windows with the recorded collective time subtracted",
            "optimizer": "exact-norm gradient clipping (one scalar AllReduce) plus AdamW",
            "saved_activation_bytes": "measured once per arm via saved_tensors_hooks with parameter storages excluded; not taken during timed repeats",
            "byte ledgers": "ring-model analytic per-rank volumes, not wire measurements",
            "throughput": "global supervised tokens (batch replicated across the TP group) divided by the per-step wall-clock window",
        },
    })
    write_json(Path(output_path), payload)
    print(json.dumps({"summary": summary, "scaling": scaling}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter38_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=MAIN_MAX_LENGTH)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--only-arm", default=None,
                        choices=tuple(arm for arms in ARMS_BY_WORLD_SIZE.values() for arm in arms))
    parser.add_argument("--world-sizes", type=str, default="2",
                        help="comma list subset of ARMS_BY_WORLD_SIZE keys, e.g. '2' or '2,4'; default '2' (the main experiment)")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
        tokenizer = load_tokenizer(torch.device("cpu"))
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
            activations = part.get("activations", {})
            summary = {arm: summarize_entries(entries) for arm, entries in merged.items()}
            scaling = compute_scaling(summary)
            finalize_output(merged, summary, scaling, ledgers, activations, args, world_size, len(samples), args.output)
            Path(args_ns["part_path"]).unlink()
        return

    # mp.spawn path: single node, one process group per world size.
    requested = sorted(int(value) for value in args.world_sizes.split(",") if value)
    unknown = [value for value in requested if value not in ARMS_BY_WORLD_SIZE]
    if unknown:
        parser.error(f"--world-sizes values not supported: {unknown}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < max(requested):
        raise SystemExit(f"the requested world sizes {requested} need {max(requested)} CUDA GPUs on one node")
    tokenizer = load_tokenizer(torch.device("cpu"))
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
    activations = {}
    schedule = [
        (world_size, tuple(arm for arm in ARMS_BY_WORLD_SIZE[world_size] if args.only_arm in (None, arm)))
        for world_size in requested
    ]
    for world_size, arms in schedule:
        if not arms:
            continue
        plans = {arm: arm_plan(arm, len(samples), args.seed, world_size) for arm in arms}
        port = free_port()
        mp.spawn(_worker, args=(world_size, port, args_ns, plans, list(arms)), nprocs=world_size, join=True)
        part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
        for arm, entries in part["results"].items():
            merged[arm] = entries
        ledgers.update(part.get("ledgers", {}))
        activations.update(part.get("activations", {}))
    Path(args_ns["part_path"]).unlink()
    Path(args_ns["samples_path"]).unlink()

    summary = {arm: summarize_entries(entries) for arm, entries in merged.items()}
    scaling = compute_scaling(summary)
    finalize_output(merged, summary, scaling, ledgers, activations, args, max(requested), len(samples), args.output)


if __name__ == "__main__":
    main()
