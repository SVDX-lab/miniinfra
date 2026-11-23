#!/usr/bin/env python3
"""Chapter 29 benchmark: data-parallel scaling of the full SFT step.

Arms over the same corpus and shuffle (FP32, eager attention):

- ``strong_dp1``  global batch 4 on one GPU — the Chapter 28 body, per-step
  global effective tokens fixed
- ``strong_dp2``  the same global batch sharded 2+2 across two GPUs — Strong
  Scaling: the only change is the number of replicas
- ``weak_dp2``    global batch 8 sharded 4+4 — Weak Scaling: per-rank batch
  fixed at 4, global work doubles

Each repeat rebuilds the optimizer and re-measures a fresh window after four
warm-up steps. The gradient AllReduce is fully exposed in this chapter, so
``grad_allreduce_ms`` is the exposed communication on the critical path.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
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
    to_device_batch,
    free_port,
    load_tokenizer_and_model,
    make_batch_plan,
    parameter_count_bytes,
    setup_distributed,
)
from experiment_utils import environment_record, percentile, write_json
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything


BASE_BATCH = 4
ARMS_BY_WORLD_SIZE = {
    1: ("strong_dp1", "strong_dp1_acc2"),
    2: ("strong_dp2", "weak_dp2"),
    4: ("strong_dp4", "weak_dp4"),
    8: ("strong_dp8", "weak_dp8"),
}


def _filter_encodable(tokenizer, records, max_length):
    """12GB calibration: at reduced max_length a few corpus samples lose all
    supervised targets to truncation; drop them so every arm sees the same set."""

    kept = []
    for record in records:
        try:
            kept.extend(encode_samples(tokenizer, [record], max_length))
        except ValueError:
            continue
    return kept


def arm_plan(arm: str, sample_count: int, seed: int) -> list:
    """Global batch plan for one arm; dpN arms keep the strong/weak contract.

    ``strong_dpN`` fixes the GLOBAL batch at BASE_BATCH and shards it over N
    ranks; ``weak_dpN`` fixes the PER-RANK batch at BASE_BATCH, so the global
    batch grows to BASE_BATCH * N.
    """

    if arm.startswith("strong_dp"):
        return make_batch_plan(sample_count, BASE_BATCH, seed)
    if arm.startswith("weak_dp"):
        return make_batch_plan(sample_count, BASE_BATCH * int(arm[len("weak_dp"):]), seed)
    raise ValueError(f"unknown arm {arm}")



def global_accounting(samples, plan, world_size):
    """Host-side per-step global token accounting (shard padding included)."""

    rows = []
    for members in plan:
        supervised = sum(samples[index].supervised_tokens for index in members)
        non_padding = sum(samples[index].non_padding_tokens for index in members)
        physical = 0
        for rank in range(world_size):
            shard = members[rank::world_size]
            longest = max(samples[index].non_padding_tokens for index in shard)
            physical += longest * len(shard)
        rows.append({"supervised": float(supervised), "non_padding": float(non_padding), "physical": float(physical)})
    return rows


def _telemetry(rank: int) -> dict:
    import os

    local_id = int(os.environ.get("LOCAL_RANK", rank))
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={local_id}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


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
            "grad_allreduce_mean_ms": fmean(step["grad_allreduce_ms"] for step in steps),
            "grad_allreduce_p50_ms": percentile([step["grad_allreduce_ms"] for step in steps], 0.5),
            "scalar_sync_mean_ms": fmean(step["scalar_sync_ms"] for step in steps),
            "forward_mean_ms": fmean(step["forward_time_ms"] for step in steps),
            "backward_mean_ms": fmean(step["backward_time_ms"] for step in steps),
            "optimizer_mean_ms": fmean(step["optimizer_time_ms"] for step in steps),
        },
        "throughput": {
            "global_supervised_tokens_per_second": sum(step["global_supervised_tokens"] for step in steps) / total_seconds,
            "global_physical_tokens_per_second": sum(step["global_physical_tokens"] for step in steps) / total_seconds,
            "global_non_padding_tokens_per_second": sum(step["global_non_padding_tokens"] for step in steps) / total_seconds,
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "first_last_loss": {"first": steps[0]["global_loss"], "last": steps[-1]["global_loss"]},
    }


def accumulate_step_single_gpu(model, optimizer, samples, members, pad_token_id, device, max_grad_norm, num_micro=2):
    """Single-GPU anchor step: 2x2 gradient accumulation (the Chapter 24 mechanism).

    On 22 GB the one-shot big batch does not fit (Chapter 28 measured 21.57 GiB
    peak allocated on a 24 GB 4090 D), so the single-GPU anchor accumulates two
    micro-batches of two samples — mathematically equal to the global token_mean
    objective under the same supervised-token weighting.
    """

    from mini_batch import batch_loss, pad_samples

    optimizer.zero_grad(set_to_none=True)
    supervised_global = sum(samples[index].supervised_tokens for index in members)
    size = len(members) // num_micro
    forward_ms = 0.0
    backward_ms = 0.0
    for micro in range(num_micro):
        group = members[micro * size : (micro + 1) * size]
        host_batch = pad_samples([samples[index] for index in group], pad_token_id)
        supervised_micro = sum(samples[index].supervised_tokens for index in group)
        batch = to_device_batch(host_batch, device)
        torch.cuda.synchronize(device)
        seg_start = torch.cuda.Event(enable_timing=True)
        seg_mid = torch.cuda.Event(enable_timing=True)
        seg_end = torch.cuda.Event(enable_timing=True)
        seg_start.record()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean") * (supervised_micro / supervised_global)
        seg_mid.record()
        loss.backward()
        seg_end.record()
        torch.cuda.synchronize(device)
        forward_ms += float(seg_start.elapsed_time(seg_mid))
        backward_ms += float(seg_mid.elapsed_time(seg_end))
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    optimizer_start.record()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer_end.record()
    torch.cuda.synchronize(device)
    return forward_ms, backward_ms, float(optimizer_start.elapsed_time(optimizer_end)), float(grad_norm.item())


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
        plan = plans[arm]
        accounting = global_accounting(samples, plan, world_size)
        for repeat in range(1, args.repeats + 1):
            seed_everything(args.seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)
            try:
                steps = []
                if arm == "strong_dp1_acc2":
                    for warmup_index in range(args.warmup):
                        accumulate_step_single_gpu(model, optimizer, samples, plan[warmup_index], args.pad_token_id, device, args.max_grad_norm)
                    torch.cuda.reset_peak_memory_stats(device)
                    for step_index in range(args.warmup, len(plan)):
                        torch.cuda.synchronize(device)
                        wall_start = time.perf_counter()
                        forward_ms, backward_ms, optimizer_ms, grad_norm = accumulate_step_single_gpu(model, optimizer, samples, plan[step_index], args.pad_token_id, device, args.max_grad_norm)
                        step_ms = (time.perf_counter() - wall_start) * 1000
                        index = step_index - args.warmup
                        metrics = {
                            "h2d_time_ms": 0.0,
                            "forward_time_ms": forward_ms,
                            "backward_time_ms": backward_ms,
                            "grad_allreduce_ms": 0.0,
                            "scalar_sync_ms": 0.0,
                            "optimizer_time_ms": optimizer_ms,
                            "step_time_ms": step_ms,
                            "global_loss": 0.0,
                            "gradient_norm_before_clip": grad_norm,
                            "global_supervised_tokens": accounting[index]["supervised"],
                            "global_non_padding_tokens": accounting[index]["non_padding"],
                            "global_physical_tokens": accounting[index]["physical"],
                            "physical_tokens_per_second": accounting[index]["physical"] / (step_ms / 1000),
                            "non_padding_tokens_per_second": accounting[index]["non_padding"] / (step_ms / 1000),
                            "supervised_tokens_per_second": accounting[index]["supervised"] / (step_ms / 1000),
                            "global_supervised_tokens_per_second": accounting[index]["supervised"] / (step_ms / 1000),
                            "global_physical_tokens_per_second": accounting[index]["physical"] / (step_ms / 1000),
                        }
                        if device.type == "cuda":
                            metrics.update(_telemetry(rank))
                        steps.append(metrics)
                else:
                    for warmup_index in range(args.warmup):
                        host_batch, accounting_step = build_global_batch(samples, plan[warmup_index], rank, world_size, args.pad_token_id)
                        ddp_train_step(model, optimizer, host_batch, accounting_step["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
                    torch.cuda.reset_peak_memory_stats(device)
                    for step_index in range(args.warmup, len(plan)):
                        host_batch, accounting_step = build_global_batch(samples, plan[step_index], rank, world_size, args.pad_token_id)
                        metrics = ddp_train_step(model, optimizer, host_batch, accounting_step["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=True)
                        index = step_index - args.warmup
                        metrics.update({
                            "step": float(index + 1),
                            "global_supervised_tokens": accounting[index]["supervised"],
                            "global_non_padding_tokens": accounting[index]["non_padding"],
                            "global_physical_tokens": accounting[index]["physical"],
                            "global_supervised_tokens_per_second": accounting[index]["supervised"] / (metrics["step_time_ms"] / 1000),
                            "global_physical_tokens_per_second": accounting[index]["physical"] / (metrics["step_time_ms"] / 1000),
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
    cleanup_distributed()
    if rank == 0:
        from experiment_utils import write_json

        write_json(Path(args_ns["part_path"]), {"world_size": world_size, "results": results})


def summarize_entries(entries):
    successful = [entry for entry in entries if entry["status"] == "success"]
    if not successful:
        return {"successful_repeats": 0}
    return {
        "successful_repeats": len(successful),
        "step_time_p50_ms_mean": fmean(entry["aggregate"]["step_time_p50_ms"] for entry in successful),
        "grad_allreduce_p50_ms_mean": fmean(entry["aggregate"]["grad_allreduce_p50_ms"] for entry in successful),
        "grad_allreduce_mean_ms_mean": fmean(entry["aggregate"]["grad_allreduce_mean_ms"] for entry in successful),
        "scalar_sync_mean_ms": fmean(entry["aggregate"]["scalar_sync_mean_ms"] for entry in successful),
        "forward_mean_ms": fmean(entry["aggregate"]["forward_mean_ms"] for entry in successful),
        "backward_mean_ms": fmean(entry["aggregate"]["backward_mean_ms"] for entry in successful),
        "optimizer_mean_ms": fmean(entry["aggregate"]["optimizer_mean_ms"] for entry in successful),
        "global_supervised_tokens_per_second_mean": fmean(entry["throughput"]["global_supervised_tokens_per_second"] for entry in successful),
        "global_physical_tokens_per_second_mean": fmean(entry["throughput"]["global_physical_tokens_per_second"] for entry in successful),
        "peak_allocated_bytes_mean": fmean(entry["memory"]["peak_allocated_bytes"] for entry in successful),
        "peak_reserved_bytes_mean": fmean(entry["memory"]["peak_reserved_bytes"] for entry in successful),
    }


def compute_scaling(summary):
    """Strong/weak scaling of every measured dpN arm against the dp1 anchor."""

    scaling = {}
    anchor = "strong_dp1_acc2" if summary.get("strong_dp1_acc2", {}).get("successful_repeats") else "strong_dp1"
    summary["dp1_anchor_arm"] = anchor
    if not summary.get(anchor, {}).get("successful_repeats"):
        return scaling
    world_sizes = sorted(
        int(arm[len("strong_dp"):])
        for arm in summary
        if arm.startswith("strong_dp") and arm[len("strong_dp"):].isdigit() and int(arm[len("strong_dp"):]) > 1
    )
    for n in world_sizes:
        strong_arm, weak_arm = f"strong_dp{n}", f"weak_dp{n}"
        if summary.get(strong_arm, {}).get("successful_repeats"):
            t1 = summary[anchor]["step_time_p50_ms_mean"]
            t2 = summary[strong_arm]["step_time_p50_ms_mean"]
            th1 = summary[anchor]["global_supervised_tokens_per_second_mean"]
            th2 = summary[strong_arm]["global_supervised_tokens_per_second_mean"]
            scaling[f"strong_dp{n}"] = {
                "dp1_anchor_arm": anchor,
                "step_time_ratio_dp1_over_dpn": t1 / t2,
                "strong_scaling_efficiency": (t1 / t2) / n,
                "throughput_ratio_dpn_over_dp1": th2 / th1,
                "exposed_comm_share": summary[strong_arm]["grad_allreduce_mean_ms_mean"] / summary[strong_arm]["step_time_p50_ms_mean"],
            }
        if summary.get(weak_arm, {}).get("successful_repeats"):
            scaling[f"weak_dp{n}"] = {
                "dp1_anchor_arm": anchor,
                "per_step_time_ratio_dpn_over_dp1": summary[weak_arm]["step_time_p50_ms_mean"] / summary[anchor]["step_time_p50_ms_mean"],
                "global_throughput_ratio_dpn_over_dp1": summary[weak_arm]["global_supervised_tokens_per_second_mean"] / summary[anchor]["global_supervised_tokens_per_second_mean"],
                "exposed_comm_share": summary[weak_arm]["grad_allreduce_mean_ms_mean"] / summary[weak_arm]["step_time_p50_ms_mean"],
            }
    return scaling


def finalize_output(merged, summary, scaling, args, world_size, sample_count, output_path):
    # Persist the raw measurements first: derived summaries must never be able
    # to destroy them.
    write_json(Path(output_path), {"partial": True, "repeats_detail": merged, "summary_partial": summary})

    gradient_bytes = parameter_count_bytes() * 4  # FP32 gradients all-reduced every step
    for arm in summary:
        digits = arm[len("strong_dp"):] if arm.startswith("strong_dp") else arm[len("weak_dp"):] if arm.startswith("weak_dp") else ""
        if digits.isdigit() and int(digits) > 1 and summary[arm].get("successful_repeats"):
            summary[arm]["gradient_allreduce_algbw_gb_per_s"] = (
                gradient_bytes / (summary[arm]["grad_allreduce_mean_ms_mean"] / 1000) / 1e9
            )
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
        "dp1_anchor_arm": summary.get("dp1_anchor_arm"),
        "gradient_allreduce_bytes_per_step": gradient_bytes,
        "num_samples": sample_count,
        "global_batch_sizes": {
            f"strong_dp{n}": BASE_BATCH for n in (1, 2, 4, 8) if f"strong_dp{n}" in merged
        } | {
            f"weak_dp{n}": BASE_BATCH * n for n in (2, 4, 8) if f"weak_dp{n}" in merged
        },
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
            "step_time": "wall clock around a synchronized step: h2d + forward + backward + exposed gradient AllReduce + scalar sync + clip/optimizer",
            "grad_allreduce_ms": "post-backward AllReduce SUM over all parameter gradients; fully exposed on the critical path in this chapter",
            "throughput": "global (all-rank) token accounting divided by the per-step wall-clock window",
        },
    })
    write_json(Path(output_path), payload)
    print(json.dumps({"summary": summary, "scaling": scaling}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter29_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--only-arm", choices=("strong_dp1", "strong_dp1_acc2", "strong_dp2", "weak_dp2", "strong_dp4", "weak_dp4", "strong_dp8", "weak_dp8"), default=None)
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
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
        args.pad_token_id = tokenizer.pad_token_id
        from sft_core import load_jsonl_records

        records = load_jsonl_records(args.data)
        try:
            samples = encode_samples(tokenizer, records, args.max_length)
        except ValueError:
            samples = _filter_encodable(tokenizer, records, args.max_length)
        samples = samples[: len(samples) - len(samples) % BASE_BATCH]
        arms = tuple(arm for arm in ARMS_BY_WORLD_SIZE[world_size] if args.only_arm in (None, arm))
        plans = {arm: arm_plan(arm, len(samples), args.seed) for arm in arms}
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
    except ValueError:
        samples = _filter_encodable(tokenizer, records, args.max_length)
    samples = samples[: len(samples) - len(samples) % BASE_BATCH]

    args_ns = {
        "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "pad_token_id": args.pad_token_id,
        "part_path": str(args.output.with_suffix(".part.json")),
        "samples_path": str(args.output.with_suffix(".samples.pt")),
    }
    torch.save(samples, args_ns["samples_path"])

    all_arms = [arm for world_size in requested for arm in ARMS_BY_WORLD_SIZE[world_size]]
    if args.only_arm:
        all_arms = [arm for arm in all_arms if arm == args.only_arm]
    plans = {arm: arm_plan(arm, len(samples), args.seed) for arm in all_arms}

    merged: dict[int, dict] = {}
    schedule = [(world_size, tuple(arm for arm in ARMS_BY_WORLD_SIZE[world_size] if arm in plans))
                for world_size in requested]
    for world_size, arms in schedule:
        if not arms:
            continue
        port = free_port()
        mp.spawn(_worker, args=(world_size, port, args_ns, plans, list(arms)), nprocs=world_size, join=True)
        part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
        for arm, entries in part["results"].items():
            merged[arm] = entries
    Path(args_ns["part_path"]).unlink()
    Path(args_ns["samples_path"]).unlink()

    summary = {arm: summarize_entries(entries) for arm, entries in merged.items()}
    scaling = compute_scaling(summary)
    finalize_output(merged, summary, scaling, args, max(requested), len(samples), args.output)


if __name__ == "__main__":
    main()
