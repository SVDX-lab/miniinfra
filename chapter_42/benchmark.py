#!/usr/bin/env python3
"""Chapter 42 benchmark: one hybrid arm per invocation, warm-up + measured steps.

Launch ONE arm at a time (the world size is fixed by the arm):

    torchrun --nproc-per-node 4 benchmark.py --arm a2_tp2pp2        # single node
    torchrun --nnodes 2 --node-rank R --nproc-per-node 4 \
        --master-addr ... --master-port ... benchmark.py --arm d_tp2pp2dp2

Every rank loads its PP stage (TP-sharded), runs ``--warmup`` discarded steps
and ``--steps`` measured steps of ``hybrid_train_step`` over consecutive
batches of the deterministic plan, and reports: step time, the three
communication segments (TP AllReduce inside forward/backward, PP P2P from
the timeline, DP AllReduce of every parameter gradient), the tied exchange,
the pipeline bubble, per-rank peak allocated/reserved memory, GPU telemetry,
and the state-ownership ledger.  The last measured step's full per-micro
timeline is written so the bubble can be recomputed from raw events.

The GLOBAL batch is ``dp_size x num_micro x samples_per_micro`` — the
pipeline micro loop IS the gradient accumulation, so global batch =
micro x accumulation x DP with accumulation := num_micro per DP rank.  All
arms are compared at the same global batch only if the caller scales
``--samples-per-micro`` accordingly; by default every arm keeps the SAME
per-DP-rank workload (strong-scaling comparisons must fix the global work:
pass ``--samples-per-micro`` multiplied by the DP ratio, which the README
documents per pair).  GPU numbers stay 未实测 until this script runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiment_utils import environment_record, percentile, write_json
from hybrid_parallel import arm_topology, build_parallel_groups, hybrid_train_step, state_ledger
from hybrid_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    cleanup_distributed,
    encode_samples,
    free_port,
    gpu_telemetry,
    load_tokenizer,
    make_batch_plan,
    node_mapping_record,
    setup_distributed,
    shard_micro_batches,
    micro_dicts,
)
from hybrid_parallel import load_hybrid_stage
from sft_core import load_jsonl_records, seed_everything


def step_aggregate(steps: list[dict]) -> dict:
    """Aggregate the raw per-step records (recomputable from the JSON)."""

    times = [step["step_time_ms"] for step in steps]
    aggregate = {
        "step_time_mean_ms": mean(times),
        "step_time_p50_ms": median(times),
        "step_time_p95_ms": percentile(times, 0.95),
        "step_time_max_ms": max(times),
        "supervised_tokens_per_second_mean": mean(step["supervised_tokens_per_second"] for step in steps),
        "physical_tokens_per_second_mean": mean(step["physical_tokens_per_second"] for step in steps),
        "tp_allreduce_total_mean_ms": mean(step["tp_total_ms"] for step in steps),
        "dp_allreduce_mean_ms": mean(step["dp_total_ms"] for step in steps),
        "p2p_time_mean_s": mean(step["p2p_time_s"] for step in steps),
        "p2p_bytes_total": sum(step["p2p_bytes"] for step in steps),
        "bubble_ratio_mean": mean(step["bubble"]["bubble_ratio"] for step in steps),
        "loss_first": steps[0]["global_loss"],
        "loss_last": steps[-1]["global_loss"],
    }
    return aggregate


def _worker(rank: int, world_size: int, port: int, args_ns: dict, encoded_meta) -> None:
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    args = SimpleNamespace(**args_ns)
    seed_everything(args.seed)
    rank_id, _, device = setup_distributed(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    topology = arm_topology(args.arm)
    assert topology.world_size == world_size

    groups = build_parallel_groups(topology)
    require_alignment = args.arm == "d_tp2pp2dp2" and not args.allow_any_node_mapping
    nodes = node_mapping_record(topology, require_alignment)
    membership = groups["membership"]

    samples, plan = encoded_meta
    pad_token_id = load_tokenizer(torch.device("cpu")).pad_token_id
    stage = load_hybrid_stage(MODEL_ID, device, stage_index=membership["pp_rank"],
                              num_stages=topology.pp_size, tp_rank=membership["tp_rank"],
                              tp_size=topology.tp_size, tp_group=groups["tp_group"],
                              dtype=torch.float32)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)

    def step_micros(step_index: int) -> list[dict]:
        members = plan[step_index]
        micros, accounting = shard_micro_batches(samples, members, topology,
                                                 membership["dp_rank"], args.num_micro, pad_token_id)
        micro_ds = micro_dicts(micros, accounting["supervised_global"])
        for micro in micro_ds:
            for key in ("input_ids", "attention_mask", "labels"):
                micro[key] = micro[key].to(device)
        return micro_ds

    total_steps = args.warmup + args.steps
    measured: list[dict] = []
    last_timeline: list[dict] = []
    metrics = None
    oom_step = None
    oom_message = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step_index in range(total_steps):
        try:
            micro_ds = step_micros(step_index)
            metrics = hybrid_train_step([stage], optimizer, micro_ds, device, groups, args.max_grad_norm)
        except torch.OutOfMemoryError as exc:
            oom_step, oom_message = step_index, str(exc).splitlines()[0]
            break
        if device.type == "cuda" and step_index >= args.warmup:
            metrics.update(gpu_telemetry(rank))
        metrics["kind"] = "warmup" if step_index < args.warmup else "measured"
        metrics["global_batch_sequences"] = float(topology.dp_size * args.num_micro * args.samples_per_micro)
        if step_index >= args.warmup:
            measured.append(metrics)
            last_timeline = metrics["timeline"]
    if metrics is not None:
        del metrics

    record = {
        "rank": rank,
        "membership": membership,
        "memory": {
            "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0.0,
            "peak_reserved_bytes": float(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0.0,
        },
        "ledger": state_ledger([stage]),
        "steps": [
            {key: value for key, value in step.items() if key not in ("timeline", "stage_split")}
            for step in measured
        ],
        "bubble_last_step": measured[-1]["bubble"] if measured else {},
        "last_step_timeline": last_timeline,
        "status": "oom" if oom_message is not None else "success",
        "oom_step": oom_step,
        "oom_message": oom_message,
        "oom_memory_allocated_bytes": (float(torch.cuda.memory_allocated())
                                       if device.type == "cuda" and oom_message is not None else None),
    }
    del stage, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    records = [None] * world_size
    dist.all_gather_object(records, record)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args.part_path), {"arm": args.arm, "nodes": nodes, "ranks": records})


def _finalize(args: SimpleNamespace, args_ns: dict) -> None:
    part = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
    topology = arm_topology(args.arm)
    ranks = part["ranks"]
    payload = environment_record(world_size=topology.world_size, dp_degree=topology.dp_size,
                                 tp_degree=topology.tp_size, pp_degree=topology.pp_size)
    payload.update({
        "arm": args.arm,
        "topology": topology.describe(),
        "nodes": part["nodes"],
        "schedule": "gpipe",
        "num_micro": args.num_micro,
        "samples_per_micro": args.samples_per_micro,
        "gradient_accumulation_per_dp_rank": args.num_micro,
        "global_batch_sequences": topology.dp_size * args.num_micro * args.samples_per_micro,
        "global_batch_formula": "dp_size * num_micro * samples_per_micro (= micro x accumulation x DP)",
        "warmup": args.warmup,
        "measured_steps": args.steps,
        "max_length": args.max_length,
        "dtype": "float32",
        "tf32": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "nnodes": int(os.environ.get("NNODES", "1")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "status": "oom" if any(record.get("status") == "oom" for record in ranks) else "success",
        "per_rank": [
            {
                "rank": record["rank"],
                "membership": record["membership"],
                "memory": record["memory"],
                "ledger": record["ledger"],
                "status": record.get("status", "success"),
                "oom_step": record.get("oom_step"),
                "oom_message": record.get("oom_message"),
                "aggregate": step_aggregate(record["steps"]) if record["steps"] else {},
            }
            for record in ranks
        ],
        "last_step_timeline_rank0": next(record["last_step_timeline"] for record in ranks if record["rank"] == 0),
        "bubble_last_step_rank0": next(record["bubble_last_step"] for record in ranks if record["rank"] == 0),
        "notes": [
            "所有数字在 RTX 3080 Ti 实测前均为未实测；本脚本只负责在真实硬件上产生它们。",
            "global_loss 已按 DP 求和、TP 去重；吞吐以全局有效监督 Token/s 为准。",
            "跨节点段（实验 D 的 DP AllReduce）无法与节点内段直接分段时，以 world size、节点映射与 nvidia-smi topo 说明口径。",
        ],
    })
    write_json(args.output, payload)
    Path(args.part_path).unlink()
    if payload["status"] == "oom":
        print(json.dumps({"arm": args.arm, "status": "oom",
                          "oom_steps": [record.get("oom_step") for record in payload["per_rank"]]},
                         ensure_ascii=False, indent=2))
        return
    print(json.dumps({
        "arm": args.arm,
        "per_rank_mean_step_ms": [round(record["aggregate"]["step_time_mean_ms"], 3) for record in payload["per_rank"]],
        "per_rank_supervised_tps": [round(record["aggregate"]["supervised_tokens_per_second_mean"], 1) for record in payload["per_rank"]],
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter42_train.jsonl"))
    parser.add_argument("--arm", required=True, choices=sorted([
        "a1_tp2", "a2_tp2pp2", "b2_tp2dp2", "c1_pp2", "c2_pp2dp2", "d_tp2pp2dp2"]))
    parser.add_argument("--world-size", type=int, default=None,
                        help="must equal tp*pp*dp of the arm; defaults to the arm's world size")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--num-micro", type=int, default=4)
    parser.add_argument("--samples-per-micro", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--allow-any-node-mapping", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()

    topology = arm_topology(args.arm)
    expected_world = topology.world_size
    if args.world_size is None:
        args.world_size = expected_world
    if args.world_size != expected_world:
        raise SystemExit(f"arm {args.arm} needs world size {expected_world}, got --world-size {args.world_size}")

    tokenizer = load_tokenizer(torch.device("cpu"))
    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them so every rank encodes the
        # same usable corpus.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    global_batch = topology.dp_size * args.num_micro * args.samples_per_micro
    # Every measured step consumes one full global batch.
    usable = len(samples) - len(samples) % global_batch
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    if global_batch > len(samples):
        raise SystemExit(f"global batch {global_batch} exceeds the corpus ({len(samples)})")
    plan = make_batch_plan(len(samples), global_batch, args.seed)[: args.warmup + args.steps]
    encoded_meta = (samples, plan)

    args_ns = {
        "seed": args.seed, "warmup": args.warmup, "steps": args.steps,
        "num_micro": args.num_micro, "samples_per_micro": args.samples_per_micro,
        "max_length": args.max_length, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "arm": args.arm, "world_size": args.world_size,
        "allow_any_node_mapping": args.allow_any_node_mapping,
        "part_path": str(Path("benchmark-hybrid.local.part.json")),
    }
    args.part_path = Path(args_ns["part_path"])

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != args.world_size:
            raise SystemExit(f"--world-size {args.world_size} does not match torchrun WORLD_SIZE {world_size}")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        _worker(rank, world_size, 0, args_ns, encoded_meta)
        if rank == 0:
            _finalize(args, args_ns)
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required (or run under torchrun)")
    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns, encoded_meta), nprocs=args.world_size, join=True)
    _finalize(args, args_ns)


if __name__ == "__main__":
    main()
