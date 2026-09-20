#!/usr/bin/env python3
"""Chapter 35 capacity probe for Qwen3-0.6B on 12 GB GPUs.

Runs two real optimizer steps of full-precision SFT for each
(tp_size, micro_batch_per_rank, max_length) configuration and reports success
or the OOM point.  This is a boundary scan, not a benchmark: numbers recorded
here are capacity facts, not throughput.

TP replicates data, so the per-step global batch equals the per-rank micro
batch (there is no data-parallel split inside the TP group); capacity
pressure comes from micro-batch, sequence length, and the TP degree's effect
on per-rank parameter/activation memory.

Launch with ``python capacity_probe.py`` (mp.spawn, single node; the grid
walks tp sizes 1/2/4 in separate process-group rounds) or under ``torchrun``
(one world size per launch; ``--tp-sizes`` must match ``WORLD_SIZE``).  Every
rank walks the same grid on its own GPU, OOM entries keep the probe alive,
and rank 0 writes the JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

from experiment_utils import environment_record, write_json
from mlp_tp import tp_memory_ledger, tp_train_step
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import seed_everything
from tp_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    build_replicated_batch,
    cleanup_distributed,
    encode_samples,
    free_port,
    load_tokenizer_and_model_tp,
    make_batch_plan,
    setup_distributed,
)


def build_configs(args: argparse.Namespace) -> list[dict]:
    """The (tp, micro-batch, sequence) grid; TP replicates the batch."""

    return [
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tp_size": tp_size,
            "micro_batch_per_rank": micro_batch,
            "global_batch": micro_batch,
            "max_length": max_length,
            "steps": args.steps,
        }
        for tp_size in sorted(set(args.tp_sizes))
        for max_length in sorted(set(args.max_lengths))
        for micro_batch in sorted(set(args.micro_batch_sizes))
    ]


def finalize_probe(part: dict, world_size: int, output_path: Path, args: argparse.Namespace) -> None:
    """Rank 0 assembles the probe payload from the raw grid records."""

    payload = environment_record(world_size=world_size, dp_degree=1, tp_degree=world_size)
    payload.update({
        "dtype": "float32",
        "tf32": False,
        "attention_implementation": "eager",
        "capacity_axis": ["tp_size", "micro_batch_per_rank", "max_length"],
        "global_batch_rule": "micro_batch_per_rank (TP replicates the batch; no data-parallel split inside the TP group)",
        "nnodes": int(os.environ.get("NNODES", "1")),
        "node_rank": int(os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if os.environ.get("RANK") else "mp.spawn",
        "requested_tp_sizes": sorted(set(args.tp_sizes)),
        "configs": part["configs"],
        "total_memory_bytes_per_gpu": torch.cuda.get_device_properties(0).total_memory,
    })
    write_json(output_path, payload)
    print(f"saved capacity probe to {output_path}")


def _worker(rank: int, world_size: int, port: int, args_ns: dict, configs: list, samples_obj=None) -> None:
    if "RANK" not in os.environ:
        # mp.spawn path: publish the env:// endpoints for this single node.
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    # Under torchrun the environment already carries RANK/WORLD_SIZE/LOCAL_RANK.
    if samples_obj is not None:
        samples_by_length = samples_obj
    else:
        samples_by_length = torch.load(args_ns["samples_path"], map_location="cpu", weights_only=False)
    seed_everything(args_ns["seed"])
    _, _, device = setup_distributed(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = False

    results = []
    for config in configs:
        samples = samples_by_length[config["max_length"]]
        entry = dict(config)
        stage = "model_load"
        try:
            _, model = load_tokenizer_and_model_tp(device, rank, world_size)
            stage = "optimizer_init"
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args_ns["learning_rate"], betas=(0.9, 0.95),
                eps=1e-8, weight_decay=0.1, foreach=False, fused=False,
            )
            plan = make_batch_plan(len(samples), config["global_batch"], args_ns["seed"])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            entry["allocated_after_model_bytes"] = float(torch.cuda.memory_allocated(device))
            entry["state_ledger"] = tp_memory_ledger(model, None, world_size)
            losses = []
            for step_index in range(config["steps"]):
                stage = f"step_{step_index}"
                host_batch, accounting = build_replicated_batch(samples, plan[step_index], args_ns["pad_token_id"])
                metrics = tp_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, 1.0, world_size, rank, collect_telemetry=False)
                losses.append(metrics["global_loss"])
            entry.update({
                "status": "success",
                "losses": losses,
                "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(device)),
            })
            del optimizer, model
        except torch.cuda.OutOfMemoryError as error:
            entry.update({
                "status": "oom",
                "failure_stage": stage,
                "oom_message": str(error).split("\n")[0][:200],
                "allocated_at_failure_bytes": float(torch.cuda.memory_allocated(device)),
                "reserved_at_failure_bytes": float(torch.cuda.memory_reserved(device)),
            })
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        results.append(entry)
        if rank == 0:
            print(f"tp{config['tp_size']} b{config['global_batch']} L{config['max_length']}: {entry['status']}"
                  + (f" peak {entry.get('peak_allocated_bytes', 0) / 2**30:.2f} GiB" if entry['status'] == 'success' else ""), flush=True)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args_ns["part_path"]), {"world_size": world_size, "configs": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter35_train.jsonl"))
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--micro-batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=Path("capacity-results.local.json"))
    args = parser.parse_args()
    if any(value <= 0 for value in args.max_lengths + args.micro_batch_sizes + args.tp_sizes):
        parser.error("max lengths, micro-batch sizes and tp sizes must be positive")
    if args.steps <= 0:
        parser.error("steps must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    tokenizer = Qwen3Tokenizer(resolve_model_directory(MODEL_ID, MODEL_REVISION))
    args_ns = {
        "seed": args.seed, "learning_rate": 2e-5, "pad_token_id": tokenizer.pad_token_id,
        "part_path": str(args.output.with_suffix(".part.json")),
        "samples_path": str(args.output.with_suffix(".samples.pt")),
    }
    configs = build_configs(args)

    import os

    if "RANK" in os.environ:
        # torchrun path: one world size per launch; only the matching tp
        # slice of the grid runs. Every rank walks the same configurations.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size not in args.tp_sizes:
            raise SystemExit(f"torchrun WORLD_SIZE {world_size} is not in --tp-sizes {sorted(set(args.tp_sizes))}")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        from sft_core import load_jsonl_records

        records = load_jsonl_records(args.data)
        samples_by_length = {}
        for max_length in sorted(set(args.max_lengths)):
            try:
                samples_by_length[max_length] = encode_samples(tokenizer, records, max_length)
                filtered_no_supervision = 0
            except ValueError:
                # 12GB calibration: at reduced max_length a few corpus samples
                # lose all supervised targets to truncation. Drop them on every
                # rank so the grid walks identical usable corpora.
                kept = []
                filtered_no_supervision = 0
                for record in records:
                    try:
                        kept.extend(encode_samples(tokenizer, [record], max_length))
                    except ValueError:
                        filtered_no_supervision += 1
                samples_by_length[max_length] = kept
            # TP replicates the batch: every plan holds micro_batch members.
            divisor = max(args.micro_batch_sizes)
            usable = len(samples_by_length[max_length]) - len(samples_by_length[max_length]) % divisor
            dropped_for_divisibility = len(samples_by_length[max_length]) - usable
            samples_by_length[max_length] = samples_by_length[max_length][:usable]
            if rank == 0:
                print(f"max_length={max_length}: encoded {usable} samples, "
                      f"filtered_no_supervision={filtered_no_supervision}, "
                      f"dropped_for_divisibility={dropped_for_divisibility}")
        matching = [config for config in configs if config["tp_size"] == world_size]
        _worker(rank, world_size, 0, args_ns, matching, samples_obj=samples_by_length)
        if rank == 0:
            part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
            finalize_probe(part, world_size, args.output, args)
            Path(args_ns["part_path"]).unlink()
        return

    # mp.spawn path: single node; one process-group round per tp size.
    from sft_core import load_jsonl_records

    if torch.cuda.device_count() < max(args.tp_sizes):
        raise SystemExit(f"the requested tp sizes {sorted(set(args.tp_sizes))} need {max(args.tp_sizes)} CUDA GPUs on one node")
    records = load_jsonl_records(args.data)
    samples_by_length = {}
    for max_length in sorted(set(args.max_lengths)):
        try:
            samples_by_length[max_length] = encode_samples(tokenizer, records, max_length)
            filtered_no_supervision = 0
        except ValueError:
            # 12GB calibration: at reduced max_length a few corpus samples lose
            # all supervised targets to truncation. Drop them so the grid walks
            # usable corpora.
            kept = []
            filtered_no_supervision = 0
            for record in records:
                try:
                    kept.extend(encode_samples(tokenizer, [record], max_length))
                except ValueError:
                    filtered_no_supervision += 1
            samples_by_length[max_length] = kept
        # TP replicates the batch: every plan holds micro_batch members.
        divisor = max(args.micro_batch_sizes)
        usable = len(samples_by_length[max_length]) - len(samples_by_length[max_length]) % divisor
        dropped_for_divisibility = len(samples_by_length[max_length]) - usable
        samples_by_length[max_length] = samples_by_length[max_length][:usable]
        print(f"max_length={max_length}: encoded {usable} samples, "
              f"filtered_no_supervision={filtered_no_supervision}, "
              f"dropped_for_divisibility={dropped_for_divisibility}")
    samples_path = Path(args_ns["samples_path"])
    torch.save(samples_by_length, samples_path)

    merged: list[dict] = []
    try:
        for tp_size in sorted(set(args.tp_sizes)):
            matching = [config for config in configs if config["tp_size"] == tp_size]
            port = free_port()
            mp.spawn(_worker, args=(tp_size, port, args_ns, matching), nprocs=tp_size, join=True)
            part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
            merged.extend(part["configs"])
    finally:
        samples_path.unlink(missing_ok=True)
        Path(args_ns["part_path"]).unlink(missing_ok=True)

    finalize_probe({"configs": merged}, max(args.tp_sizes), args.output, args)


if __name__ == "__main__":
    main()
