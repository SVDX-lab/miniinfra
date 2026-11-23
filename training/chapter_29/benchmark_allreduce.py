#!/usr/bin/env python3
"""Chapter 29 NCCL AllReduce microbenchmark over the real interconnect.

Explains the communication numbers of the main experiment; it does not
replace them. Sizes cover the FP32 gradient volume of Qwen3-0.6B (about
2.38 GB) and its fractions. Every size is warmed up and then timed with CUDA
events over ``--iters`` iterations; algbw is tensor-bytes over time and
busbw multiplies by ``2 * (W - 1) / W`` (the per-link volume ratio of a ring).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import fmean, median

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ddp_core import cleanup_distributed, free_port, setup_distributed
from experiment_utils import environment_record, write_json


SIZES_MIB = [1, 16, 64, 256, 1024, 2360]


def _worker(rank: int, world_size: int, port: int, args_ns: dict) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    args = argparse.Namespace(**args_ns)
    _, _, device = setup_distributed(rank, world_size)
    results = []
    for size_mib in SIZES_MIB:
        count = size_mib * 1024 * 1024 // 4
        tensor = torch.randn(count, device=device)
        for _ in range(args.warmup):
            dist.all_reduce(tensor)
        torch.cuda.synchronize(device)
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters):
            start.record()
            dist.all_reduce(tensor)
            end.record()
            torch.cuda.synchronize(device)
            times.append(float(start.elapsed_time(end)))
        tensor_bytes = count * 4
        mean_ms = fmean(times)
        results.append({
            "size_mib": size_mib,
            "bytes": tensor_bytes,
            "mean_ms": mean_ms,
            "p50_ms": median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "algbw_gb_per_s": tensor_bytes / (mean_ms / 1000) / 1e9,
            "busbw_gb_per_s": 2 * (world_size - 1) / world_size * tensor_bytes / (mean_ms / 1000) / 1e9,
        })
        if rank == 0:
            print(f"{size_mib} MiB: p50 {results[-1]['p50_ms']:.3f} ms  busbw {results[-1]['busbw_gb_per_s']:.2f} GB/s", flush=True)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args.output), {"world_size": world_size, "sizes": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("allreduce-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    payload = environment_record(world_size=args.world_size, dp_degree=args.world_size)
    args_ns = {"warmup": args.warmup, "iters": args.iters, "output": str(args.output)}
    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns), nprocs=args.world_size, join=True)
    saved = json.loads(args.output.read_text(encoding="utf-8"))
    saved["environment"] = {key: payload[key] for key in ("schema_version", "gpus", "gpu_topology", "nccl_version", "torch", "environment_variables")}
    write_json(args.output, saved)
    print(f"saved allreduce microbenchmark to {args.output}")


if __name__ == "__main__":
    main()
