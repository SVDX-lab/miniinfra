#!/usr/bin/env python3
"""Chapter 34 capacity probe for Qwen3-0.6B on 12 GB GPUs.

Runs two real optimizer steps of full-precision SFT for each
(engine, micro-batch, sequence) configuration and reports success or the OOM
point. This is a boundary scan, not a benchmark: numbers recorded here are
capacity facts, not throughput.

Engines share the Chapter 29 batch path; model size stays fixed, and capacity
pressure comes from micro-batch and sequence length only:

- ``zero2``  full parameters + Reduce-Scatter gradients + sharded state
             (Chapter 33 engine) — the chapter baseline
- ``fsdp``   parameters also sharded (FSDP FULL_SHARD) — the chapter variant

The global batch of every configuration is ``micro_batch_per_rank *
world_size``. The hypothesis under test: the ZeRO-3 admission boundary (the
largest (micro-batch, sequence) that fits) sits BEYOND ZeRO-2's on 12 GB.

Launch with ``python capacity_probe.py`` (mp.spawn, single node) or under
``torchrun``; under torchrun ``--world-size`` must match ``WORLD_SIZE``, every
rank walks the same grid on its own GPU, OOM entries keep the probe alive, and
rank 0 writes the JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.multiprocessing as mp

from ddp_core import (
    MODEL_ID,
    MODEL_REVISION,
    build_global_batch,
    cleanup_distributed,
    encode_samples,
    free_port,
    make_batch_plan,
    setup_distributed,
)
from experiment_utils import environment_record, write_json
from fsdp_train import fsdp_train_step, wrap_model_fsdp
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import seed_everything
from zero2_core import ShardedGradAdamW, zero2_train_step


def load_tokenizer_and_model_by_id(model_id: str, revision: str, device: torch.device):
    from qwen3_model import load_handwritten_model, resolve_model_directory

    directory = resolve_model_directory(model_id, revision)
    return Qwen3Tokenizer(directory), load_handwritten_model(directory, device, dtype=torch.float32)


def build_configs(args: argparse.Namespace) -> list[dict]:
    """The (engine, micro-batch, sequence) grid; the global batch follows W."""

    return [
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "engine": engine,
            "micro_batch_per_rank": micro_batch,
            "global_batch": micro_batch * args.world_size,
            "max_length": max_length,
            "steps": args.steps,
        }
        for max_length in sorted(set(args.max_lengths))
        for micro_batch in sorted(set(args.micro_batch_sizes))
        for engine in ("zero2", "fsdp")
    ]


def finalize_probe(part: dict, world_size: int, output_path: Path) -> None:
    """Rank 0 assembles the probe payload from the raw grid records."""

    import os as _os

    payload = environment_record(world_size=world_size, dp_degree=world_size)
    payload.update({
        "dtype": "float32",
        "tf32": False,
        "attention_implementation": "eager",
        "capacity_axis": ["micro_batch_per_rank", "max_length"],
        "engines": ["zero2", "fsdp"],
        "global_batch_rule": "micro_batch_per_rank * world_size",
        "nnodes": int(_os.environ.get("NNODES", "1")),
        "node_rank": int(_os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in _os.environ else "mp.spawn",
        "configs": part["configs"],
        "total_memory_bytes_per_gpu": torch.cuda.get_device_properties(0).total_memory,
    })
    write_json(output_path, payload)
    print(f"saved capacity probe to {output_path}")


def _worker(rank: int, world_size: int, port: int, args_ns: dict, configs: list, samples_obj=None) -> None:
    import os

    if "RANK" not in os.environ:
        # mp.spawn path: publish the env:// endpoints for this single node.
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    # Under torchrun the environment already carries RANK/WORLD_SIZE/LOCAL_RANK
    # and the rendezvous endpoint; overriding them here would break multi-node.
    if samples_obj is not None:
        samples_by_length = samples_obj
    else:
        samples_by_length = torch.load(args_ns["samples_path"], map_location="cpu", weights_only=False)
    seed_everything(args_ns["seed"])
    _, _, device = setup_distributed(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = False

    results = []
    for config in configs:
        model_id = config["model_id"]
        revision = config["revision"]
        engine = config["engine"]
        samples = samples_by_length[config["max_length"]]
        entry = dict(config)
        stage = "model_load"
        try:
            tokenizer, model = load_tokenizer_and_model_by_id(model_id, revision, device)
            stage = "optimizer_init"
            if engine == "zero2":
                optimizer = ShardedGradAdamW(list(model.parameters()), world_size, rank, lr=args_ns["learning_rate"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
                train_step = zero2_train_step
            elif engine == "fsdp":
                fsdp_model = wrap_model_fsdp(model, device)
                optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=args_ns["learning_rate"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)
            else:
                raise ValueError(f"unknown engine {engine}")
            plan = make_batch_plan(len(samples), config["global_batch"], args_ns["seed"])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            entry["allocated_after_model_bytes"] = float(torch.cuda.memory_allocated(device))
            losses = []
            for step_index in range(config["steps"]):
                stage = f"step_{step_index}"
                host_batch, accounting = build_global_batch(samples, plan[step_index], rank, world_size, args_ns["pad_token_id"])
                if engine == "zero2":
                    metrics = zero2_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, 1.0, world_size, rank, collect_telemetry=False)
                else:
                    metrics = fsdp_train_step(fsdp_model, optimizer, host_batch, accounting["loss_scale"], device, 1.0, world_size, rank, collect_telemetry=False)
                losses.append(metrics["local_loss"])
            entry.update({
                "status": "success",
                "losses": losses,
                "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(device)),
            })
            if engine == "zero2":
                del optimizer, model
            else:
                del optimizer, fsdp_model, model
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
            print(f"{model_id} {engine} b{config['global_batch']} L{config['max_length']}: {entry['status']}"
                  + (f" peak {entry.get('peak_allocated_bytes', 0) / 2**30:.2f} GiB" if entry['status'] == 'success' else ""), flush=True)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args_ns["part_path"]), {"world_size": world_size, "configs": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter34_train.jsonl"))
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--micro-batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--world-size", type=int, default=2, choices=(2, 4, 8))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output", type=Path, default=Path("capacity-results.local.json"))
    args = parser.parse_args()
    if any(value <= 0 for value in args.max_lengths + args.micro_batch_sizes):
        parser.error("max lengths and micro-batch sizes must be positive")
    if args.steps <= 0:
        parser.error("steps must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    tokenizer, _ = load_tokenizer_and_model_by_id(MODEL_ID, MODEL_REVISION, torch.device("cpu"))
    args_ns = {
        "seed": args.seed, "learning_rate": 2e-5, "pad_token_id": tokenizer.pad_token_id,
        "part_path": str(args.output.with_suffix(".part.json")),
        "samples_path": str(args.output.with_suffix(".samples.pt")),
    }
    configs = build_configs(args)

    import os

    if "RANK" in os.environ:
        # torchrun path: every rank walks the same grid with its env-assigned
        # rank; OOM entries keep the probe alive and rank 0 finalizes.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != args.world_size:
            raise SystemExit(f"--world-size {args.world_size} does not match torchrun WORLD_SIZE {world_size}")
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
            # The grid runs global batches of micro_batch * world_size members.
            divisor = max(args.micro_batch_sizes) * args.world_size
            usable = len(samples_by_length[max_length]) - len(samples_by_length[max_length]) % divisor
            dropped_for_divisibility = len(samples_by_length[max_length]) - usable
            samples_by_length[max_length] = samples_by_length[max_length][:usable]
            if rank == 0:
                print(f"max_length={max_length}: encoded {usable} samples, "
                      f"filtered_no_supervision={filtered_no_supervision}, "
                      f"dropped_for_divisibility={dropped_for_divisibility}")
        _worker(rank, world_size, 0, args_ns, configs, samples_obj=samples_by_length)
        if rank == 0:
            part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
            finalize_probe(part, world_size, args.output)
            Path(args_ns["part_path"]).unlink()
        return

    # mp.spawn path: single node, one process group for the whole grid.
    if torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    from sft_core import load_jsonl_records

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
        # The grid runs global batches of micro_batch * world_size members.
        divisor = max(args.micro_batch_sizes) * args.world_size
        usable = len(samples_by_length[max_length]) - len(samples_by_length[max_length]) % divisor
        dropped_for_divisibility = len(samples_by_length[max_length]) - usable
        samples_by_length[max_length] = samples_by_length[max_length][:usable]
        print(f"max_length={max_length}: encoded {usable} samples, "
              f"filtered_no_supervision={filtered_no_supervision}, "
              f"dropped_for_divisibility={dropped_for_divisibility}")
    samples_path = Path(args_ns["samples_path"])
    torch.save(samples_by_length, samples_path)

    port = free_port()
    try:
        mp.spawn(_worker, args=(args.world_size, port, args_ns, configs), nprocs=args.world_size, join=True)
    finally:
        samples_path.unlink(missing_ok=True)

    part = json.loads(Path(args_ns["part_path"]).read_text(encoding="utf-8"))
    finalize_probe(part, args.world_size, args.output)
    Path(args_ns["part_path"]).unlink()


if __name__ == "__main__":
    main()
