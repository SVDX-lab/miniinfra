#!/usr/bin/env python3
"""Chapter 37 benchmark: attention+MLP TP (chapter-36 layout) versus
vocab+attention+MLP TP.

Two arms per world size; the TP degree IS the world size, and the ONLY
variable between the arms is whether the Embedding/LM Head vocab dimension is
also sharded:

- ``attn_tp_wN``  N GPUs, attention AND MLP sharded, embedding/LM head and
                  their FULL-VOCAB logits replicated (chapter-36 layout)
- ``vocab_tp_wN`` N GPUs, vocab ALSO sharded N-way: lookup masks out-of-range
                  rows behind one embedding AllReduce, the LM head produces
                  SHARDED logits ``[batch, seq, 151936/N]`` consumed by the
                  distributed cross entropy (three ``[token]``-sized forward
                  collectives, none in its backward), and the head input
                  gradient is AllReduced once per step

For Qwen3-0.6B the vocab axis is the dominant remaining activation: with
batch 2 x seq 512 the full logits are ~622 MiB of FP32 per rank, so the
``attn_tp`` arm pays them in full while ``vocab_tp`` divides them by the TP
degree — both arms record ``local_logits_bytes`` vs ``full_logits_bytes``.
0.6B may still favor the replicated head at tp=2; a negative optimization is
a valid outcome here.  Both arms replicate DATA: every rank of a group runs
the SAME global batch of ``PER_RANK_BATCH * tp_size`` samples.

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
from vocab_tp import tp_memory_ledger_all, tp_train_step
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything
from tp_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    build_replicated_batch,
    cleanup_distributed,
    encode_samples,
    free_port,
    load_tokenizer,
    load_tokenizer_and_model_tp,
    make_batch_plan,
    parameter_count_bytes,
    setup_distributed,
)


PER_RANK_BATCH = 2  # the per-step global batch is PER_RANK_BATCH * tp_size
ARMS_BY_WORLD_SIZE = {
    2: ("attn_tp_w2", "vocab_tp_w2"),
    4: ("attn_tp_w4", "vocab_tp_w4"),
    8: ("attn_tp_w8", "vocab_tp_w8"),
}


def arm_tp_size(arm: str) -> int:
    """The TP degree encoded in an arm name's ``_wN`` suffix."""

    return int(arm.rsplit("_w", 1)[1])


def arm_vocab_parallel(arm: str) -> bool:
    """vocab_tp arms shard the vocab too; attn_tp arms keep chapter-36 layout."""

    return arm.startswith("vocab_tp")


def arm_plan(arm: str, sample_count: int, seed: int, world_size: int) -> list:
    """Global batch plan for one arm; every arm at one world size shares it.

    TP does not shard data, so the plan is identical on all ranks of the
    group and the global batch grows as PER_RANK_BATCH * tp_size.
    """

    if arm_tp_size(arm) != world_size:
        raise ValueError(f"arm {arm} needs world size {arm_tp_size(arm)}, got {world_size}")
    return make_batch_plan(sample_count, PER_RANK_BATCH * world_size, seed)


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
            "h2d_mean_ms": fmean(step["h2d_time_ms"] for step in steps),
            "forward_compute_mean_ms": fmean(step["forward_time_ms"] for step in steps),
            "backward_compute_mean_ms": fmean(step["backward_time_ms"] for step in steps),
            "optimizer_mean_ms": fmean(step["optimizer_time_ms"] for step in steps),
            "tp_allreduce_forward_mean_ms": fmean(step["tp_allreduce_forward_ms"] for step in steps),
            "tp_allreduce_backward_mean_ms": fmean(step["tp_allreduce_backward_ms"] for step in steps),
            "tp_allreduce_total_mean_ms": fmean(step["tp_allreduce_total_ms"] for step in steps),
            "tp_allreduce_exposed_ratio_mean": fmean(step["tp_allreduce_exposed_ratio"] for step in steps),
            "tp_allreduce_count_per_step": fmean(
                step["tp_allreduce_forward_count"] + step["tp_allreduce_backward_count"] for step in steps
            ),
            "local_logits_bytes_mean": fmean(step["local_logits_bytes"] for step in steps),
            "full_logits_bytes_mean": fmean(step["full_logits_bytes"] for step in steps),
        },
        "throughput": {
            "global_supervised_tokens_per_second": sum(step["supervised_tokens"] for step in steps) / total_seconds,
            "global_physical_tokens_per_second": sum(step["physical_tokens"] for step in steps) / total_seconds,
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "first_last_loss": {"first": steps[0]["local_loss"], "last": steps[-1]["local_loss"]},
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
    for arm in arms:
        # Each arm owns a DIFFERENT weight layout (vocab sharded or not), so
        # the model is built and torn down per arm.
        _, model = load_tokenizer_and_model_tp(
            device, rank, world_size, vocab_parallel=arm_vocab_parallel(arm)
        )
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
                    tp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False, vocab_parallel=arm_vocab_parallel(arm))
                # AdamW states materialize at the first step, so the ledger is
                # captured after warm-up; peak counters reset for measurement.
                if repeat == 1:
                    ledgers[arm] = tp_memory_ledger_all(model, optimizer, world_size)
                    ledgers[arm]["measured_allocated_after_state_bytes"] = float(torch.cuda.memory_allocated(device))
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                steps = []
                for step_index in range(args.warmup, len(plan)):
                    host_batch, accounting = build_replicated_batch(samples, plan[step_index], args.pad_token_id)
                    # collect_telemetry samples power/temperature on LOCAL_RANK's GPU.
                    metrics = tp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=True, vocab_parallel=arm_vocab_parallel(arm))
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
                          f"tp-comm {entry['aggregate']['tp_allreduce_total_mean_ms']:.1f} ms "
                          f"({entry['aggregate']['tp_allreduce_exposed_ratio_mean'] * 100:.1f}% exposed) "
                          f"sup/s {entry['throughput']['global_supervised_tokens_per_second']:.1f} "
                          f"peak {entry['memory']['peak_allocated_bytes'] / 2**30:.2f} GiB", flush=True)
                else:
                    print(f"{arm} repeat={entry['repeat']}: {entry['status']} allocated {entry.get('allocated_bytes', 0) / 2**30:.2f} GiB", flush=True)
            if entry["status"] != "success":
                break  # a failing arm is reported once; later repeats cannot recover it
        del optimizer, model
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
        "forward_compute_mean_ms": fmean(entry["aggregate"]["forward_compute_mean_ms"] for entry in successful),
        "backward_compute_mean_ms": fmean(entry["aggregate"]["backward_compute_mean_ms"] for entry in successful),
        "optimizer_mean_ms": fmean(entry["aggregate"]["optimizer_mean_ms"] for entry in successful),
        "tp_allreduce_forward_mean_ms": fmean(entry["aggregate"]["tp_allreduce_forward_mean_ms"] for entry in successful),
        "tp_allreduce_backward_mean_ms": fmean(entry["aggregate"]["tp_allreduce_backward_mean_ms"] for entry in successful),
        "tp_allreduce_total_mean_ms": fmean(entry["aggregate"]["tp_allreduce_total_mean_ms"] for entry in successful),
        "tp_allreduce_exposed_ratio_mean": fmean(entry["aggregate"]["tp_allreduce_exposed_ratio_mean"] for entry in successful),
        "tp_allreduce_count_per_step_mean": fmean(entry["aggregate"]["tp_allreduce_count_per_step"] for entry in successful),
        "local_logits_bytes_mean": fmean(entry["aggregate"]["local_logits_bytes_mean"] for entry in successful),
        "full_logits_bytes_mean": fmean(entry["aggregate"]["full_logits_bytes_mean"] for entry in successful),
        "global_supervised_tokens_per_second_mean": fmean(entry["throughput"]["global_supervised_tokens_per_second"] for entry in successful),
        "peak_allocated_bytes_mean": fmean(entry["memory"]["peak_allocated_bytes"] for entry in successful),
        "peak_reserved_bytes_mean": fmean(entry["memory"]["peak_reserved_bytes"] for entry in successful),
    }


def compute_scaling(summary):
    """Each vocab-TP arm against the attn_tp arm at the SAME world size."""

    scaling = {}
    for world_size in sorted(ARMS_BY_WORLD_SIZE):
        baseline_arm, variant_arm = f"attn_tp_w{world_size}", f"vocab_tp_w{world_size}"
        if not (summary.get(baseline_arm, {}).get("successful_repeats") and summary.get(variant_arm, {}).get("successful_repeats")):
            continue
        scaling[f"vocab_tp{world_size}_vs_attn_tp_w{world_size}"] = {
            "step_time_ratio_vocab_tp_over_attn_tp": summary[variant_arm]["step_time_p50_ms_mean"] / summary[baseline_arm]["step_time_p50_ms_mean"],
            "throughput_ratio_vocab_tp_over_attn_tp": summary[variant_arm]["global_supervised_tokens_per_second_mean"] / summary[baseline_arm]["global_supervised_tokens_per_second_mean"],
            "peak_allocated_ratio_vocab_tp_over_attn_tp": summary[variant_arm]["peak_allocated_bytes_mean"] / summary[baseline_arm]["peak_allocated_bytes_mean"],
            "logits_bytes_ratio_vocab_over_full": summary[variant_arm]["local_logits_bytes_mean"] / max(summary[variant_arm]["full_logits_bytes_mean"], 1.0),
            "extra_tp_allreduce_ms": summary[variant_arm]["tp_allreduce_total_mean_ms"] - summary[baseline_arm]["tp_allreduce_total_mean_ms"],
            "tp_allreduce_exposed_ratio": summary[variant_arm]["tp_allreduce_exposed_ratio_mean"],
        }
    return scaling


def finalize_output(merged, summary, scaling, ledgers, args, world_size, sample_count, output_path):
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
            "attn_tp_wN": "attention AND MLP sharded (chapter-36 layout); embedding/LM head + full logits replicated",
            "vocab_tp_wN": "vocab, attention AND MLP sharded; sharded logits + distributed cross entropy",
        },
        "vocab_layout": {
            "vocab_size": 151936,
            "local_vocab_by_tp": {str(tp): 151936 // tp for tp in sorted(ARMS_BY_WORLD_SIZE)},
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
        "batch_rule": "TP replicates data: every rank of a group runs the SAME batch of PER_RANK_BATCH * tp_size samples; throughput counts the batch once, not per-rank multiplicity",
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "parameter_bytes_full_model": parameter_count_bytes() * 4,
        "nnodes": int(os.environ.get("NNODES", "1")),
        "node_rank": int(os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "state_ledgers": ledgers,
        "repeats_detail": merged,
        "summary": summary,
        "scaling": scaling,
        "timing_scope": {
            "step_time": "wall clock of a synchronized optimizer step",
            "tp_allreduce_forward/backward": "event-timed per AllReduce and summed per step (attention o_proj + MLP down_proj + embedding g + 3 vocab-CE scalars on the forward side of the vocab arm; block-input gradients, q/k norm syncs and the LM head input gradient on the backward side); they sit INSIDE the forward/backward windows and are fully exposed on the critical path",
            "forward/backward compute": "the forward/backward event windows with the TP AllReduce time subtracted",
            "optimizer": "exact-norm gradient clipping (one scalar AllReduce) plus AdamW",
            "throughput": "global supervised tokens (batch replicated across the TP group) divided by the per-step wall-clock window",
        },
    })
    write_json(Path(output_path), payload)
    print(json.dumps({"summary": summary, "scaling": scaling}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter37_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--only-arm", default=None,
                        choices=tuple(arm for arms in ARMS_BY_WORLD_SIZE.values() for arm in arms))
    parser.add_argument("--world-sizes", type=str, default="all",
                        help="comma list subset of ARMS_BY_WORLD_SIZE keys, e.g. '2' or '2,4'; 'all' runs every arm")
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
    Path(args_ns["part_path"]).unlink()
    Path(args_ns["samples_path"]).unlink()

    summary = {arm: summarize_entries(entries) for arm, entries in merged.items()}
    scaling = compute_scaling(summary)
    finalize_output(merged, summary, scaling, ledgers, args, max(requested), len(samples), args.output)


if __name__ == "__main__":
    main()
