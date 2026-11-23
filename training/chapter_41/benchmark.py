#!/usr/bin/env python3
"""Chapter 41 benchmark: non-interleaved 1F1B vs Interleaved 1F1B.

Two arms; the ONLY variable is how many VIRTUAL stages each physical rank
owns (the placement table and the schedule over it):

- ``onefoone_ppN``     one stage per rank (chapter-40 non-interleaved 1F1B;
                       total stages = N)
- ``interleaved_ppN``  two virtual stages per rank (the paper's interleaved
                       layout; total stages = 2N, every rank holds the same
                       layer count in two non-contiguous chunks)

Micro-batch scan {4, 8}: the interleaved schedule processes micro-batches in
groups of the physical pipeline size, so ``num_micro % world_size == 0``
holds for the whole scan.  With ``samples_per_micro = 2 * world_size`` (the
chapters' weak-scaling convention of 2 samples per rank) the per-step global
batch is ``num_micro * 2 * world_size`` for BOTH arms — identical workload,
only the stage graph differs.  Per-rank layer count is also identical across
arms (e.g. W=4: 7 layers), so compute per step is matched by construction.

Per (arm, micro) the benchmark records, per rank: step time, per virtual
stage forward/backward event spans, P2P time/bytes/count (interleaved flies
the same ``[b, s, h]`` payload over ``2N - 1`` stage edges instead of
``N - 1``), the schedule's theoretical in-flight count, per-step bubble
ratio, peak allocated/reserved memory, and the global supervised-token
throughput.  The last measured step's full event timeline is kept.

Launch with ``python benchmark.py`` (mp.spawn, single node) or under
``torchrun`` (single or multi node; every rank prepares the data
deterministically and rank 0 finalizes the JSON).
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
import torch.distributed as dist
import torch.multiprocessing as mp

from experiment_utils import environment_record, percentile, write_json
from interleaved import load_pipeline_model, pp_train_step, split_virtual_stages
from pp_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    build_micro_batches,
    cleanup_distributed,
    encode_samples,
    free_port,
    gpu_telemetry,
    load_tokenizer,
    make_batch_plan,
    micro_dicts,
    parameter_count_bytes,
    setup_distributed,
)
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything


SAMPLES_PER_RANK_PER_MICRO = 2  # samples_per_micro = SAMPLES_PER_RANK_PER_MICRO * world_size
MICRO_BATCHES = (4, 8)
MAIN_MAX_LENGTH = 512
INTERLEAVED_VPP = 2  # virtual stages per physical rank for the interleaved arm
ARMS_BY_WORLD_SIZE = {
    2: ("onefoone_pp2", "interleaved_pp2"),
    4: ("onefoone_pp4", "interleaved_pp4"),
}


def arm_vpp(arm: str) -> int:
    """Virtual stages per rank: 1 for the baseline, INTERLEAVED_VPP otherwise."""

    return INTERLEAVED_VPP if arm.startswith("interleaved_") else 1


def arm_mode(arm: str) -> str:
    """The schedule this arm runs ('onef1b' or 'interleaved')."""

    return "interleaved" if arm.startswith("interleaved_") else "onef1b"


def arm_plan_members(sample_count: int, num_micro: int, world_size: int, seed: int) -> list:
    global_batch = num_micro * SAMPLES_PER_RANK_PER_MICRO * world_size
    return make_batch_plan(sample_count, global_batch, seed)


def package_repeat(arm: str, num_micro: int, repeat: int, steps: list, device,
                   last_timeline: list) -> dict:
    total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
    step_times = [step["step_time_ms"] for step in steps]
    return {
        "repeat": repeat,
        "num_micro": num_micro,
        "status": "success",
        "measured_steps": len(steps),
        "steps": steps,
        "aggregate": {
            "step_time_mean_ms": fmean(step_times),
            "step_time_p50_ms": percentile(step_times, 0.5),
            "step_time_p95_ms": percentile(step_times, 0.95),
            "step_time_max_ms": max(step_times),
            "bubble_ratio_mean": fmean(step["bubble"]["bubble_ratio"] for step in steps),
            "p2p_count_mean": fmean(step["p2p_count"] for step in steps),
            "p2p_time_s_mean": fmean(step["p2p_time_s"] for step in steps),
            "p2p_bytes_mean": fmean(step["p2p_bytes"] for step in steps),
            "tied_exchange_bytes_mean": fmean(step["tied_exchange_bytes"] for step in steps),
            "in_flight_micros_mean": fmean(step["in_flight_micros"] for step in steps),
            "gradient_norm_mean": fmean(step["gradient_norm_before_clip"] for step in steps),
        },
        "throughput": {
            "global_supervised_tokens_per_second": sum(step["supervised_tokens"] for step in steps) / total_seconds,
            "global_physical_tokens_per_second": sum(step["physical_tokens"] for step in steps) / total_seconds,
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "first_last_loss": {"first": steps[0]["global_loss"], "last": steps[-1]["global_loss"]},
        "last_step_timeline": last_timeline,
    }


def _worker(rank: int, world_size: int, port: int, args_ns: dict, arms: list, micro_list: list, samples_obj=None) -> None:
    if "RANK" not in os.environ:
        # mp.spawn path: publish the env:// endpoints for this single node.
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    # Under torchrun the environment already carries RANK/WORLD_SIZE/LOCAL_RANK.
    args = SimpleNamespace(**args_ns)
    if samples_obj is not None:
        samples = samples_obj
    else:
        samples = torch.load(args_ns["samples_path"], map_location="cpu", weights_only=False)
    seed_everything(args.seed)
    _, _, device = setup_distributed(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    results = {arm: {} for arm in arms}
    stage_states = {}
    for arm in arms:
        mode = arm_mode(arm)
        vpp = arm_vpp(arm)
        # The two arms own DIFFERENT chunk structures (1 vs vpp virtual stages
        # per rank), so weights and optimizer are built per arm; the per-rank
        # layer count stays identical across arms.
        chunks = load_pipeline_model(MODEL_ID, device, rank, world_size, vpp, dtype=torch.float32)
        optimizer = torch.optim.AdamW(
            [parameter for chunk in chunks for parameter in chunk.parameters()],
            lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay,
            foreach=False, fused=False,
        )
        stage_states[arm] = {
            "rank": rank,
            "vpp": vpp,
            "chunk_layer_ranges": [f"{chunk.layer_range[0]}:{chunk.layer_range[1]}" for chunk in chunks],
            "chunk_parameter_bytes": [
                float(sum(p.numel() * p.element_size() for p in chunk.parameters())) for chunk in chunks
            ],
        }
        for num_micro in micro_list:
            plan = arm_plan_members(len(samples), num_micro, world_size, args.seed)
            entries = []
            for repeat in range(1, args.repeats + 1):
                seed_everything(args.seed)
                try:
                    step_index = 0
                    steps = []
                    last_timeline: list = []
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    for members in plan:
                        micro_batches = build_micro_batches(samples, members, num_micro, args.pad_token_id)
                        supervised_global = float(sum(batch.supervised_tokens for batch in micro_batches))
                        micros = micro_dicts(micro_batches, supervised_global)
                        warmup = step_index < args.warmup
                        metrics = pp_train_step(chunks, optimizer, micros, device, mode, args.max_grad_norm)
                        if not warmup:
                            if device.type == "cuda":
                                metrics.update(gpu_telemetry(rank))
                            metrics.update({
                                "step": float(step_index - args.warmup),
                                "global_supervised_tokens": metrics["supervised_tokens"],
                            })
                            last_timeline = metrics.pop("timeline")
                            steps.append(metrics)
                        else:
                            metrics.pop("timeline", None)
                        step_index += 1
                    entry = package_repeat(arm, num_micro, repeat, steps, device, last_timeline)
                    # Per-rank peaks for this (arm, micro) block: gather once.
                    peaks = [None] * world_size
                    dist.all_gather_object(peaks, entry["memory"])
                    entry["per_rank_memory"] = peaks
                    entries.append(entry)
                except torch.cuda.OutOfMemoryError:
                    entries.append({"repeat": repeat, "num_micro": num_micro, "status": "oom",
                                    "allocated_bytes": torch.cuda.memory_allocated(device)})
                entry = entries[-1]
                if rank == 0:
                    if entry["status"] == "success":
                        print(f"{arm} micro={num_micro} repeat={entry['repeat']}: "
                              f"p50 {entry['aggregate']['step_time_p50_ms']:.1f} ms "
                              f"bubble {entry['aggregate']['bubble_ratio_mean'] * 100:.1f}% "
                              f"p2p {entry['aggregate']['p2p_time_s_mean'] * 1000:.1f} ms "
                              f"x{entry['aggregate']['p2p_count_mean']:.0f} "
                              f"in-flight {entry['aggregate']['in_flight_micros_mean']:.0f} "
                              f"sup/s {entry['throughput']['global_supervised_tokens_per_second']:.1f} "
                              f"peak {entry['memory']['peak_allocated_bytes'] / 2**30:.2f} GiB", flush=True)
                    else:
                        print(f"{arm} micro={num_micro} repeat={entry['repeat']}: {entry['status']} "
                              f"allocated {entry.get('allocated_bytes', 0) / 2**30:.2f} GiB", flush=True)
                if entry["status"] != "success":
                    break
            results[arm][str(num_micro)] = entries
        del chunks, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    own_state = {"rank": rank, "by_arm": stage_states}
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args_ns["part_path"]), {
            "world_size": world_size, "results": results, "stage_state": own_state,
        })


def summarize_entries(entries):
    successful = [entry for entry in entries if entry["status"] == "success"]
    if not successful:
        return {"successful_repeats": 0}
    summary = {
        "successful_repeats": len(successful),
        "step_time_p50_ms_mean": fmean(entry["aggregate"]["step_time_p50_ms"] for entry in successful),
        "step_time_p95_ms_mean": fmean(entry["aggregate"]["step_time_p95_ms"] for entry in successful),
        "bubble_ratio_mean": fmean(entry["aggregate"]["bubble_ratio_mean"] for entry in successful),
        "p2p_count_mean": fmean(entry["aggregate"]["p2p_count_mean"] for entry in successful),
        "p2p_time_s_mean": fmean(entry["aggregate"]["p2p_time_s_mean"] for entry in successful),
        "p2p_bytes_mean": fmean(entry["aggregate"]["p2p_bytes_mean"] for entry in successful),
        "tied_exchange_bytes_mean": fmean(entry["aggregate"]["tied_exchange_bytes_mean"] for entry in successful),
        "in_flight_micros_mean": fmean(entry["aggregate"]["in_flight_micros_mean"] for entry in successful),
        "global_supervised_tokens_per_second_mean": fmean(entry["throughput"]["global_supervised_tokens_per_second"] for entry in successful),
        "peak_allocated_bytes_mean": fmean(entry["memory"]["peak_allocated_bytes"] for entry in successful),
        "peak_reserved_bytes_mean": fmean(entry["memory"]["peak_reserved_bytes"] for entry in successful),
    }
    return summary


def summarize_by_micro(arm_entries: dict) -> dict:
    return {micro: summarize_entries(entries) for micro, entries in arm_entries.items()}


def compute_scaling(summary) -> dict:
    """Interleaved against non-interleaved 1F1B at the SAME world size and
    micro count (strong scaling: identical global workload per comparison)."""

    scaling = {}
    for world_size in sorted(ARMS_BY_WORLD_SIZE):
        onefoone_arm, interleaved_arm = f"onefoone_pp{world_size}", f"interleaved_pp{world_size}"
        onefoone_by_micro = summary.get(onefoone_arm, {})
        interleaved_by_micro = summary.get(interleaved_arm, {})
        for micro in sorted(set(onefoone_by_micro) & set(interleaved_by_micro), key=int):
            onefoone_summary, interleaved_summary = onefoone_by_micro[micro], interleaved_by_micro[micro]
            if not (onefoone_summary.get("successful_repeats") and interleaved_summary.get("successful_repeats")):
                continue
            scaling[f"interleaved_over_onefoone_pp{world_size}_micro{micro}"] = {
                "step_time_ratio_interleaved_over_onefoone": interleaved_summary["step_time_p50_ms_mean"] / onefoone_summary["step_time_p50_ms_mean"],
                "throughput_ratio_interleaved_over_onefoone": interleaved_summary["global_supervised_tokens_per_second_mean"] / onefoone_summary["global_supervised_tokens_per_second_mean"],
                "peak_allocated_ratio_interleaved_over_onefoone": interleaved_summary["peak_allocated_bytes_mean"] / max(onefoone_summary["peak_allocated_bytes_mean"], 1.0),
                "bubble_ratio_onefoone": onefoone_summary["bubble_ratio_mean"],
                "bubble_ratio_interleaved": interleaved_summary["bubble_ratio_mean"],
                "in_flight_micros_onefoone": onefoone_summary["in_flight_micros_mean"],
                "in_flight_micros_interleaved": interleaved_summary["in_flight_micros_mean"],
                "p2p_count_onefoone": onefoone_summary["p2p_count_mean"],
                "p2p_count_interleaved": interleaved_summary["p2p_count_mean"],
                "p2p_time_s_onefoone": onefoone_summary["p2p_time_s_mean"],
                "p2p_time_s_interleaved": interleaved_summary["p2p_time_s_mean"],
            }
    return scaling


def finalize_output(merged, summary, scaling, stage_states, args, world_size, sample_count, micro_list, output_path):
    write_json(Path(output_path), {"partial": True, "repeats_detail": merged, "summary_partial": summary})

    payload = environment_record(world_size=world_size, dp_degree=1, pp_degree=world_size)
    total_stages = world_size * INTERLEAVED_VPP
    layer_ranges, split_meta = split_virtual_stages(28, total_stages)
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
            "onefoone_ppN": f"non-interleaved 1F1B baseline (chapter 25): one stage per rank, total stages = N",
            "interleaved_ppN": f"interleaved 1F1B (this chapter's variant): {INTERLEAVED_VPP} non-contiguous virtual stages per rank, total stages = {INTERLEAVED_VPP}*N",
        },
        "interleaved_vpp": INTERLEAVED_VPP,
        "stage_split_interleaved": {
            "total_virtual_stages": total_stages,
            "stage_layer_counts": split_meta["stage_layer_counts"],
            "uneven": split_meta["uneven"],
            "layer_ranges": [f"{begin}:{end}" for begin, end in layer_ranges],
            "stage_state_by_world": stage_states,
        },
        "num_samples": sample_count,
        "micro_batch_scan": micro_list,
        "global_batch_sizes": {
            arm: {micro: int(micro) * SAMPLES_PER_RANK_PER_MICRO * world_size for micro in merged[arm]}
            for arm in sorted(merged.keys())
        },
        "batch_rule": "every stage processes every micro-batch; per-step global batch = num_micro * 2 * world_size samples for BOTH arms (2 samples per rank per micro, the chapters' weak-scaling convention); per-rank layer count is identical across arms",
        "max_length": args.max_length,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "parameter_bytes_full_model": parameter_count_bytes() * 4,
        "nnodes": int(os.environ.get("NNODES", "1")),
        "node_rank": int(os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "repeats_detail": merged,
        "summary": summary,
        "scaling": scaling,
        "timing_scope": {
            "step_time": "wall clock of a synchronized pipeline step (all micro-batch forwards+backwards, tied-gradient P2P, clip, AdamW)",
            "bubble_ratio": "1 - compute-busy/window from THIS rank's CUDA-event timeline (forward/backward spans; P2P reported separately)",
            "p2p": "send/recv event spans per (virtual stage, micro-batch) (activations, input gradients, tied-gradient exchange), bytes = payload numel; the count field separates message count from wait time",
            "in_flight_micros": "theoretical peak number of alive activation graphs per rank from the schedule formula",
            "timeline": "the last measured step's full per-(chunk, micro) event timeline is stored for recomputing bubble numbers",
            "throughput": "global supervised tokens (the batch each stage fully processes) divided by the per-step wall-clock window",
        },
    })
    write_json(Path(output_path), payload)
    print(json.dumps({"summary": summary, "scaling": scaling}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter41_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=MAIN_MAX_LENGTH)
    parser.add_argument("--micro-batches", type=str, default="all",
                        help="comma list subset of MICRO_BATCHES, e.g. '4'; 'all' runs the full scan")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--only-arm", default=None,
                        choices=tuple(arm for arms in ARMS_BY_WORLD_SIZE.values() for arm in arms))
    parser.add_argument("--world-sizes", type=str, default="4",
                        help="comma list subset of ARMS_BY_WORLD_SIZE keys, e.g. '2' or '2,4'; default '4' (the interleaved main experiment)")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    micro_list = sorted(MICRO_BATCHES) if args.micro_batches == "all" else sorted(
        int(value) for value in args.micro_batches.split(",") if value
    )
    unknown_micro = [value for value in micro_list if value not in MICRO_BATCHES]
    if unknown_micro:
        parser.error(f"--micro-batches values not supported: {unknown_micro}")

    if "RANK" in os.environ:
        # torchrun path: every rank prepares the same deterministic state in
        # memory, then runs the worker body directly. Rank 0 finalizes.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size not in ARMS_BY_WORLD_SIZE:
            raise SystemExit(f"torchrun WORLD_SIZE {world_size} is not one of {sorted(ARMS_BY_WORLD_SIZE)}")
        for num_micro in micro_list:
            if num_micro % world_size != 0:
                raise SystemExit(
                    f"--micro-batches {num_micro} is not divisible by the physical pipeline size {world_size} "
                    "(the interleaved schedule processes micro-batches in groups of the pipeline size)"
                )
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        tokenizer = load_tokenizer(torch.device("cpu"))
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
        # Arm plans shard num_micro * SAMPLES_PER_RANK_PER_MICRO * world_size
        # members per step; cover the largest micro-batch in the scan.
        divisor = SAMPLES_PER_RANK_PER_MICRO * max(micro_list) * world_size
        usable = len(samples) - len(samples) % divisor
        dropped_for_divisibility = len(samples) - usable
        samples = samples[:usable]
        if rank == 0:
            print(
                f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
                f"dropped_for_divisibility={dropped_for_divisibility}"
            )
        arms = tuple(arm for arm in ARMS_BY_WORLD_SIZE[world_size] if args.only_arm in (None, arm))
        args_ns = {
            "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm, "pad_token_id": tokenizer.pad_token_id,
            "part_path": str(args.output.with_suffix(".part.json")),
            "samples_path": str(args.output.with_suffix(".samples.pt")),
        }
        _worker(rank, world_size, 0, args_ns, list(arms), micro_list, samples_obj=samples)
        if rank == 0:
            part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
            merged = part["results"]
            summary = {arm: summarize_by_micro(entries) for arm, entries in merged.items()}
            scaling = compute_scaling(summary)
            finalize_output(merged, summary, scaling, {world_size: part.get("stage_state", {})}, args, world_size, len(samples), micro_list, args.output)
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
    # One corpus feeds every requested world size; arm plans shard
    # num_micro * SAMPLES_PER_RANK_PER_MICRO * world_size members per step.
    divisor = SAMPLES_PER_RANK_PER_MICRO * max(micro_list) * max(requested)
    usable = len(samples) - len(samples) % divisor
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    args_ns = {
        "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "pad_token_id": tokenizer.pad_token_id,
        "part_path": str(args.output.with_suffix(".part.json")),
        "samples_path": str(args.output.with_suffix(".samples.pt")),
    }
    torch.save(samples, args_ns["samples_path"])

    merged: dict[str, dict] = {}
    stage_states = {}
    schedule = [
        (world_size, tuple(arm for arm in ARMS_BY_WORLD_SIZE[world_size] if args.only_arm in (None, arm)))
        for world_size in requested
    ]
    try:
        for world_size, arms in schedule:
            if not arms:
                continue
            port = free_port()
            mp.spawn(_worker, args=(world_size, port, args_ns, list(arms), micro_list), nprocs=world_size, join=True)
            part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
            for arm, entries in part["results"].items():
                merged[arm] = entries
            stage_states[world_size] = part.get("stage_state", {})
    finally:
        Path(args_ns["samples_path"]).unlink(missing_ok=True)
        Path(args_ns["part_path"]).unlink(missing_ok=True)

    summary = {arm: summarize_by_micro(entries) for arm, entries in merged.items()}
    scaling = compute_scaling(summary)
    finalize_output(merged, summary, scaling, stage_states, args, max(requested), len(samples), micro_list, args.output)


if __name__ == "__main__":
    main()
