#!/usr/bin/env python3
"""Chapter 43 benchmark: checkpoint save/restore throughput, centralized vs sharded.

Launch under torchrun (the save I/O is per-rank in the sharded mode):

    torchrun --nproc-per-node 4 benchmark_checkpoint.py                 # 4-rank save+load
    torchrun --nproc-per-node 2 benchmark_checkpoint.py \
        --resume-from checkpoints/step-00001/sharded --reference ...    # 2-rank resharded load

Measured per mode (centralized / sharded / dcp):
- SAVE: wall-clock window bounded by barriers, bytes written, GB/s;
- LOAD: fresh model + optimizer restored, wall-clock, GB/s;
- one DDP training step (mean of 3) so the save time / training time ratio
  is a measured number, not an estimate.

Per-mode file sizes come from the file system AFTER the run (sum of files in
the mode's directory).  GPU numbers stay 未实测 until this script has run on
the RTX 3080 Ti machines; `--resume-from` reuses a checkpoint produced by
`compare_checkpoint.py` or `train_worker.py` for the resharded-load timing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from checkpointing import (
    dcp_available,
    dcp_load_checkpoint,
    dcp_save_checkpoint,
    detect_mode,
    load_for_resume,
    save_centralized,
    save_sharded,
)
from ddp_core import (
    build_global_batch,
    cleanup_distributed,
    ddp_train_step,
    free_port,
    setup_distributed,
)
from experiment_utils import environment_record, write_json
from train_worker import build_batch_plan, build_optimizer_and_scheduler


def directory_bytes(path: Path) -> float:
    if path.is_file():
        return float(path.stat().st_size)
    return float(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _worker(rank: int, world_size: int, port: int, args_ns: dict) -> None:
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    args = SimpleNamespace(**args_ns)
    use_cuda = torch.cuda.is_available() and not args.tiny
    backend = "nccl" if use_cuda else "gloo"
    rank, world_size, device = setup_distributed(rank, world_size, backend=backend)
    try:
        _run(rank, world_size, device, backend, args)
    finally:
        cleanup_distributed()


def _run(rank: int, world_size: int, device: torch.device, backend: str, args: SimpleNamespace) -> None:
    from sft_core import seed_everything

    seed_everything(args.seed)
    if args.tiny:
        from train_worker import TINY_CONFIG, synthetic_samples

        torch.manual_seed(args.seed)
        from qwen3_model import Qwen3ForCausalLM

        model = Qwen3ForCausalLM(TINY_CONFIG).to(device)
        samples = synthetic_samples()
        pad_token_id = 0
    else:
        from ddp_core import load_tokenizer_and_model

        tokenizer, model = load_tokenizer_and_model(device)
        pad_token_id = tokenizer.pad_token_id
        from sft_core import load_jsonl_records

        records = load_jsonl_records(Path(args.data))
        from ddp_core import encode_samples

        try:
            samples = encode_samples(tokenizer, records, args.max_length)
            filtered_no_supervision = 0
        except ValueError:
            # 12GB calibration: at reduced max_length a few corpus samples lose
            # all supervised targets to truncation. Drop them so every rank
            # encodes the same usable corpus.
            kept = []
            filtered_no_supervision = 0
            for record in records:
                try:
                    kept.extend(encode_samples(tokenizer, [record], args.max_length))
                except ValueError:
                    filtered_no_supervision += 1
            samples = kept
        # Every batch of the plan holds global_batch members.
        usable = len(samples) - len(samples) % args.global_batch
        dropped_for_divisibility = len(samples) - usable
        samples = samples[:usable]
        print(
            f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
            f"dropped_for_divisibility={dropped_for_divisibility}"
        )

    optimizer, scheduler = build_optimizer_and_scheduler(model, args.learning_rate,
                                                         args.warmup_steps, args.steps)
    plan = build_batch_plan(len(samples), args.global_batch, args.seed)
    progress = {"global_step": 1, "consumed_supervised_tokens": 0.0, "consumed_samples": args.global_batch}
    sampler = {"plan_seed": args.seed, "global_batch_sequences": float(args.global_batch),
               "plan_batches": float(len(plan)), "cursor": 1}
    config = {"model_id": "tiny-synthetic" if args.tiny else "Qwen/Qwen3-0.6B-Base",
              "benchmark": True, "global_batch_sequences": float(args.global_batch)}
    topology = {"world_size": world_size, "dp_size": world_size, "tp_size": 1, "pp_size": 1,
                "backend": backend, "rank": rank}
    common = (model, optimizer, scheduler, progress, sampler, config, topology)

    results: dict[str, dict] = {}

    if args.resume_from:
        # Resharded-load timing only: the checkpoint was produced elsewhere.
        mode = args.resume_mode if args.resume_mode != "auto" else detect_mode(Path(args.resume_from))
        dist.barrier()
        begin = time.perf_counter()
        payload = load_for_resume(Path(args.resume_from), model, optimizer, scheduler, device,
                                  rank, world_size, mode)
        dist.barrier()
        seconds = time.perf_counter() - begin
        results["load"] = {"mode": mode, "seconds": seconds,
                           "bytes": directory_bytes(Path(args.resume_from)),
                           "resharded": bool(payload.get("resharded"))}
    else:
        # One training step (mean of 3) for the save/training-time ratio.
        members = plan[1]
        step_times = []
        for _ in range(3):
            batch, accounting = build_global_batch(samples, members, rank, world_size, pad_token_id)
            metrics = ddp_train_step(model, optimizer, batch, accounting["loss_scale"], device,
                                     args.max_grad_norm, world_size, rank, collect_telemetry=False)
            step_times.append(metrics["step_time_ms"])
        results["train_step_ms_mean"] = mean(step_times)

        target = Path(args.workdir)
        if "centralized" in args.modes:
            results["save_centralized"] = save_centralized(target / "centralized", *common,
                                                           rank=rank, world_size=world_size)
        if "sharded" in args.modes:
            results["save_sharded"] = save_sharded(target / "sharded", *common,
                                                   rank=rank, world_size=world_size)
        if "dcp" in args.modes:
            if not dcp_available():
                raise SystemExit("torch.distributed.checkpoint is not available in this torch build")
            results["save_dcp"] = dcp_save_checkpoint(target / "dcp", model, optimizer)

        # Load phase measures a cold resume: a fresh process carries no stale
        # parameter gradients from prior training.  Drop them here so the
        # 12 GB cards are not charged ~2.4 GiB of bookkeeping that a resumed
        # run never holds (E3 node1: otherwise optimizer.load_state_dict
        # OOMs around 11.1 GiB = params + stale grads + AdamW state).
        model.zero_grad(set_to_none=True)
        for mode in args.modes:
            fresh_optimizer, fresh_scheduler = build_optimizer_and_scheduler(
                model, args.learning_rate, args.warmup_steps, args.steps)
            source = target / mode
            dist.barrier()
            begin = time.perf_counter()
            if mode == "dcp":
                dcp_load_checkpoint(source, model, fresh_optimizer)
            else:
                load_for_resume(source, model, fresh_optimizer, fresh_scheduler, device,
                                rank, world_size, mode)
            dist.barrier()
            results[f"load_{mode}"] = {
                "mode": mode, "seconds": time.perf_counter() - begin,
                "bytes": directory_bytes(source),
            }
            del fresh_optimizer, fresh_scheduler

    if device.type == "cuda":
        results["peak_memory_bytes"] = {
            "allocated": float(torch.cuda.max_memory_allocated()),
            "reserved": float(torch.cuda.max_memory_reserved()),
        }
    record = {"rank": rank, "results": results}
    records = [None] * world_size
    dist.all_gather_object(records, record)
    if rank == 0:
        write_json(Path(args.part_path), {"world_size": world_size, "ranks": records})


def _finalize(args: SimpleNamespace, args_ns: dict) -> None:
    part = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
    rank0 = next(rank["results"] for rank in part["ranks"] if rank["rank"] == 0)

    def with_throughput(entry: dict) -> dict:
        enriched = dict(entry)
        if "seconds" in entry and entry.get("bytes"):
            enriched["throughput_bytes_per_second"] = entry["bytes"] / entry["seconds"]
        return enriched

    modes_report = {}
    if "train_step_ms_mean" in rank0:
        modes_report["train_step_ms_mean"] = rank0["train_step_ms_mean"]
    for key in ("save_centralized", "save_sharded", "save_dcp", "load", "load_centralized",
                "load_sharded", "load_dcp"):
        if key in rank0:
            modes_report[key] = with_throughput(rank0[key])
    if "train_step_ms_mean" in rank0 and "save_centralized" in rank0 and "save_sharded" in rank0:
        modes_report["save_over_train_ratio_centralized"] = (
            rank0["save_centralized"]["seconds"] / (rank0["train_step_ms_mean"] / 1000))
        modes_report["save_over_train_ratio_sharded"] = (
            rank0["save_sharded"]["seconds"] / (rank0["train_step_ms_mean"] / 1000))

    payload = environment_record(world_size=part["world_size"], dp_degree=part["world_size"])
    payload.update({
        "scenario": "resharded-load" if args.resume_from else "save-and-load",
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "resume_world_size": part["world_size"] if args.resume_from else None,
        "tiny": args.tiny,
        "modes_report": modes_report,
        "per_rank": part["ranks"],
        "notes": [
            "所有数字在 RTX 3080 Ti 实测前均为未实测。",
            "GB/s 按模式目录的磁盘字节数 / barrier 界定的墙钟时间；centralized 的写盘只在 rank 0，其它 rank 在 barrier 等待。",
            "DDP 各分片是可互换的完整副本：sharded 不省字节，只并行 I/O；压缩形态见第 34 期 ZeRO/FSDP。",
        ],
    })
    write_json(args.output, payload)
    Path(args.part_path).unlink()
    print(json.dumps(modes_report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--tiny", action="store_true", help="tiny synthetic model instead of Qwen3-0.6B")
    parser.add_argument("--data", type=Path, default=Path("data/chapter43_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--global-batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--modes", default="centralized,sharded,dcp")
    parser.add_argument("--workdir", type=Path, default=Path("checkpoint-bench"))
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="only time a (possibly resharded) load of this checkpoint")
    parser.add_argument("--resume-mode", default="auto", choices=("auto", "centralized", "sharded", "dcp"))
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("checkpoint-benchmark.local.json"))
    args = parser.parse_args()
    args.modes = [mode for mode in args.modes.split(",") if mode]
    args.part_path = Path("checkpoint-bench.local.part.json")
    args_ns = {
        "tiny": bool(args.tiny), "data": str(args.data), "max_length": args.max_length,
        "global_batch": args.global_batch, "steps": args.steps, "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate, "max_grad_norm": args.max_grad_norm,
        "seed": args.seed, "modes": args.modes, "workdir": str(args.workdir),
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "resume_mode": args.resume_mode, "part_path": str(args.part_path),
    }

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        _worker(rank, world_size, 0, args_ns)
        if rank == 0:
            _finalize(args, args_ns)
        return

    if not args.tiny and (not torch.cuda.is_available() or torch.cuda.device_count() < args.nproc_per_node):
        raise SystemExit(f"real-weight benchmarks need {args.nproc_per_node} CUDA GPUs (or pass --tiny)")
    port = free_port()
    mp.spawn(_worker, args=(args.nproc_per_node, port, args_ns), nprocs=args.nproc_per_node, join=True)
    _finalize(args, args_ns)
    if not args.keep and args.resume_from is None:
        shutil.rmtree(args.workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
