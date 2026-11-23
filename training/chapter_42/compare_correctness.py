#!/usr/bin/env python3
"""Chapter 42 correctness on real Qwen3-0.6B weights: hybrid arms vs single GPU.

Phase A (one process, cuda:0): the replicated FP32 reference — the FULL model
consumes the whole global batch of one optimizer step (dp_size * num_micro
micro-batches of the SAME deterministic plan), accumulating micro by micro
with the supervised-token scales; records the loss trajectory over ``--steps``
steps (clip + AdamW each) and the post-step parameter state.

Phase B (one GPU per rank, one arm's 3-D layout): every rank builds its PP
stage with TP-sharded layers, verifies the 3-D group membership and the DP
data sharding, then runs ``hybrid_train_step`` for the same number of steps.
Comparisons per rank, per step: the global loss against the reference
trajectory (DP-weighted, TP-replicated, PP-stage-split — the SAME global
token-mean objective); the gradient norm against the reference norm; after
the LAST step, the full-model parameter checksum of THIS DP replica against
the reference checksum (TP shards gathered in rank order, tie counted
twice), plus the tied Embedding/LM-Head consistency across the exchange.

Launch with ``python compare_correctness.py --arm ...`` (mp.spawn, single
node, ``--world-size`` must match the arm) or under ``torchrun`` (single or
dual node; ``WORLD_SIZE`` must match the arm; for the dual-node arm D the
node-per-replica mapping is asserted).  Bars are tight: the reference
consumes identical micro shapes, so differences come from FP32 reduction
order only.  GPU numbers in the chapter documents stay ``未实测`` until this
script has actually run.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from assistant_loss import IGNORE_INDEX
from experiment_utils import environment_record, write_json
from hybrid_parallel import (
    arm_topology,
    build_parallel_groups,
    hybrid_train_step,
    load_hybrid_stage,
    state_ledger,
)
from hybrid_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    cleanup_distributed,
    encode_samples,
    free_port,
    global_batch_members,
    load_tokenizer,
    node_mapping_record,
    resolve_model_directory,
    setup_distributed,
    shard_micro_batches,
    micro_dicts,
)
from mlp_tp import gather_tensor_parallel_parameters
from qwen3_model import load_handwritten_model
from sft_core import load_jsonl_records, seed_everything


LOSS_REL_BAR = 1e-5
TRAJECTORY_BAR = 5e-4
PARAM_SCALE_REL_BAR = 1e-3
NORM_REL_BAR = 5e-3


def build_step_micros(args: SimpleNamespace, pad_token_id: int):
    """The deterministic data plan shared by reference and hybrid ranks."""

    tokenizer = load_tokenizer(torch.device("cpu"))
    records = load_jsonl_records(args.data)
    topology = arm_topology(args.arm)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them so reference and hybrid
        # ranks stay on identical usable samples.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    # The global batch of one step is dp_size * num_micro * samples_per_micro.
    divisor = topology.dp_size * args.num_micro * args.samples_per_micro
    usable = len(samples) - len(samples) % divisor
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    members = global_batch_members(len(samples), topology, args.num_micro,
                                   args.samples_per_micro, args.seed)
    return samples, members, topology


def checksum_state_dict(state: dict) -> float:
    total = 0.0
    for value in state.values():
        total += float(value.detach().double().pow(2).sum())
    return total


def grad_squared_sum_chunked(parameters, device) -> torch.Tensor:
    """Full-gradient squared sum in float64, chunked on device.

    The reference keeps the whole FP32 state (~10 GB) on one 12GB GPU; a
    whole-tensor ``grad.double()`` of the embedding row adds a transient
    1.2 GiB and OOMs. Chunking only changes the summation order (~1e-16
    relative), far inside the trajectory bars below.
    """

    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    chunk = 1 << 22
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        flat = grad.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk):
            total_sq += flat[start:start + chunk].double().pow(2).sum()
    return total_sq


def run_reference(args: SimpleNamespace, samples, members, topology, device) -> dict:
    """Phase A: full-model reference trajectory over the global batch."""

    from mini_batch import batch_loss, pad_samples

    pad_token_id = load_tokenizer(torch.device("cpu")).pad_token_id
    reference_micro_count = topology.dp_size * args.num_micro
    full = pad_samples([samples[index] for index in members], pad_token_id)
    if full.batch_size % reference_micro_count != 0:
        raise SystemExit("global batch does not split into reference micros")
    supervised_global = float(full.supervised_tokens)
    per_micro = full.batch_size // reference_micro_count

    model = load_handwritten_model(resolve_model_directory(MODEL_ID, MODEL_REVISION),
                                   device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)

    def micro_pass() -> float:
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for index in range(reference_micro_count):
            rows = slice(index * per_micro, (index + 1) * per_micro)
            input_ids = full.input_ids[rows].to(device)
            attention_mask = full.attention_mask[rows].to(device)
            labels = full.labels[rows].to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            supervised = float((labels[:, 1:] != IGNORE_INDEX).sum())
            loss = batch_loss(logits, labels, "token_mean") * (supervised / supervised_global)
            loss.backward()
            total += float(loss.detach())
        return total

    steps = []
    for step_index in range(args.steps):
        loss = micro_pass()
        total_sq = grad_squared_sum_chunked(model.parameters(), device)
        grad_norm = torch.sqrt(total_sq)
        clip_coef = args.max_grad_norm / (float(grad_norm.item()) + 1e-6)
        if clip_coef < 1.0:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_coef)
        optimizer.step()
        steps.append({"step": step_index, "loss": loss, "gradient_norm": float(grad_norm.item())})

    final_state = {name: parameter.detach().to("cpu").clone()
                   for name, parameter in model.state_dict().items()}
    checksum = checksum_state_dict(final_state)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"steps": steps, "checksum": checksum}


def _worker(rank: int, world_size: int, port: int, args_ns: dict, samples_meta, members) -> None:
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
    assert topology.world_size == world_size, (topology.world_size, world_size)

    groups = build_parallel_groups(topology)
    membership = groups["membership"]
    require_alignment = args.arm == "d_tp2pp2dp2" and not args.allow_any_node_mapping
    nodes = node_mapping_record(topology, require_alignment)

    dp_rank = membership["dp_rank"]
    samples, _unused_members, _topology = samples_meta
    micros, accounting = shard_micro_batches(samples, tuple(members), topology, dp_rank,
                                             args.num_micro, load_tokenizer(torch.device("cpu")).pad_token_id)
    micro_ds = micro_dicts(micros, accounting["supervised_global"])
    for micro in micro_ds:
        for key in ("input_ids", "attention_mask", "labels"):
            micro[key] = micro[key].to(device)

    stage = load_hybrid_stage(MODEL_ID, device, stage_index=membership["pp_rank"],
                              num_stages=topology.pp_size, tp_rank=membership["tp_rank"],
                              tp_size=topology.tp_size, tp_group=groups["tp_group"],
                              dtype=torch.float32)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)

    reference = torch.load(args.stash_path, map_location="cpu", weights_only=False)
    trajectory = []
    for step_index in range(args.steps):
        metrics = hybrid_train_step([stage], optimizer, micro_ds, device, groups, args.max_grad_norm)
        trajectory.append({
            "step": step_index,
            "loss": metrics["global_loss"],
            "loss_rel_error": abs(metrics["global_loss"] - reference["steps"][step_index]["loss"])
            / abs(reference["steps"][step_index]["loss"]),
            "gradient_norm": metrics["gradient_norm_before_clip"],
            "gradient_norm_rel_error": abs(metrics["gradient_norm_before_clip"]
                                           - reference["steps"][step_index]["gradient_norm"])
            / reference["steps"][step_index]["gradient_norm"],
            "step_time_ms": metrics["step_time_ms"],
            "p2p_count": metrics["p2p_count"],
            "tp_allreduce_forward_count": metrics["tp_allreduce_forward_count"],
            "dp_allreduce_count": metrics["dp_allreduce_count"],
        })

    # Full-model checksum of THIS DP replica (TP shards gathered in rank order).
    gathered_state = gather_tensor_parallel_parameters(stage, topology.tp_size, group=groups["tp_group"])
    replica_checksum = checksum_state_dict(gathered_state)
    if stage.tied_in_stage:
        replica_checksum += checksum_state_dict({"lm_head": stage.lm_head.weight})
    # Tied probe: the Embedding (first stage) and the LM Head (last stage)
    # must hold IDENTICAL values after the exchange + DP sync; middle ranks
    # (pp >= 3) send nothing.
    if stage.is_first:
        tied_probe = checksum_state_dict({"w": stage.embed_tokens.weight})
    elif stage.is_last:
        tied_probe = checksum_state_dict({"w": stage.lm_head.weight})
    else:
        tied_probe = None
    del gathered_state
    gc.collect()
    torch.cuda.empty_cache()

    checksums = [None] * world_size
    dist.all_gather_object(checksums, {
        "dp": dp_rank, "pp": membership["pp_rank"], "checksum": replica_checksum,
        "tied_probe": tied_probe,
    })
    my_replica = [item for item in checksums if item["dp"] == dp_rank]
    # Every (dp, pp) unit reports its TP-GATHERED stage checksum once per TP
    # rank (the gathered values are bitwise identical inside a TP group);
    # summing all members counts each stage tp_size times.
    replica_checksum_total = sum(item["checksum"] for item in my_replica) / topology.tp_size
    replicas_identical = len({
        round(sum(item["checksum"] for item in checksums if item["dp"] == other), 6)
        for other in range(topology.dp_size)
    }) == 1
    tied_values = {round(item["tied_probe"], 6) for item in checksums if item["tied_probe"] is not None}
    tied_consistent = (len(tied_values) == 1) if topology.pp_size >= 2 else True
    param_error = abs(replica_checksum_total - reference["checksum"]) / reference["checksum"]

    worker_report = {
        "rank": rank,
        "membership": membership,
        "stage_layer_range": f"{stage.layer_range[0]}:{stage.layer_range[1]}",
        "local_members": list(accounting["local_members"]),
        "supervised_local": accounting["supervised_local"],
        "supervised_global": accounting["supervised_global"],
        "trajectory": trajectory,
        "replica_checksum": replica_checksum_total,
        "replica_checksum_rel_error": param_error,
        "replicas_identical": replicas_identical,
        "tied_consistent": tied_consistent,
        "ledger": state_ledger([stage]),
    }
    del stage, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    reports = [None] * world_size
    dist.all_gather_object(reports, worker_report)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args.part_path), {
            "arm": args.arm,
            "world_size": world_size,
            "nodes": nodes,
            "topology": topology.describe(),
            "ranks": reports,
        })


def _assemble_report(args: SimpleNamespace, part: dict, reference: dict, topology) -> dict:
    ranks = part["ranks"]
    checks: dict[str, bool] = {}
    trajectory_errors = []
    norm_errors = []
    for rank in ranks:
        trajectory_errors.extend(step["loss_rel_error"] for step in rank["trajectory"])
        norm_errors.extend(step["gradient_norm_rel_error"] for step in rank["trajectory"])
    checks["loss_trajectory_ok"] = max(trajectory_errors) < TRAJECTORY_BAR
    checks["gradient_norm_ok"] = max(norm_errors) < NORM_REL_BAR
    checks["replica_checksum_ok"] = all(
        rank["replica_checksum_rel_error"] < PARAM_SCALE_REL_BAR for rank in ranks
    )
    checks["replicas_identical"] = all(rank["replicas_identical"] for rank in ranks) or topology.dp_size == 1
    checks["tied_consistent"] = all(rank["tied_consistent"] for rank in ranks) or topology.pp_size == 1
    checks["membership_ok"] = all(
        (rank["membership"]["dp_rank"], rank["membership"]["pp_rank"], rank["membership"]["tp_rank"])
        == topology.decompose(rank["rank"]) for rank in ranks
    )
    checks["stage_ranges_ok"] = all(
        rank["stage_layer_range"] == _expected_range(topology, rank["rank"]) for rank in ranks
    )
    return {
        "schema_note": {
            "equivalence": "Every arm computes the SAME global token-mean objective as the single-GPU reference: DP scales by supervised tokens, TP replicates, PP splits stages; bars cover FP32 reduction order only",
            "bars": {
                "loss_trajectory_rel": TRAJECTORY_BAR,
                "gradient_norm_rel": NORM_REL_BAR,
                "param_checksum_rel": PARAM_SCALE_REL_BAR,
            },
        },
        "arm": args.arm,
        "topology": topology.describe(),
        "nodes": part["nodes"],
        "dp_degree": topology.dp_size,
        "tp_degree": topology.tp_size,
        "pp_degree": topology.pp_size,
        "world_size": topology.world_size,
        "steps": args.steps,
        "num_micro": args.num_micro,
        "samples_per_micro": args.samples_per_micro,
        "max_length": args.max_length,
        "global_batch_sequences": topology.dp_size * args.num_micro * args.samples_per_micro,
        "nnodes": int(os.environ.get("NNODES", "1")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "reference_steps": reference["steps"],
        "reference_checksum": reference["checksum"],
        "per_rank": [
            {
                "rank": rank["rank"],
                "membership": rank["membership"],
                "stage_layer_range": rank["stage_layer_range"],
                "supervised_local": rank["supervised_local"],
                "replica_checksum_rel_error": rank["replica_checksum_rel_error"],
                "trajectory_max_loss_rel_error": max(step["loss_rel_error"] for step in rank["trajectory"]),
                "ledger": rank["ledger"],
            }
            for rank in ranks
        ],
        "trajectory_max_loss_rel_error": max(trajectory_errors),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _expected_range(topology, rank: int) -> str:
    from hybrid_parallel import split_layers

    _ranges, meta = split_layers(28, topology.pp_size)
    dp_rank, pp_rank, _tp_rank = topology.decompose(rank)
    begin = sum(meta["stage_layer_counts"][:pp_rank])
    return f"{begin}:{begin + meta['stage_layer_counts'][pp_rank]}"


def _finalize(args: SimpleNamespace, args_ns: dict) -> None:
    part = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
    reference = torch.load(args.stash_path, map_location="cpu", weights_only=False)
    topology = arm_topology(args.arm)
    payload = environment_record(world_size=topology.world_size, dp_degree=topology.dp_size,
                                 tp_degree=topology.tp_size, pp_degree=topology.pp_size)
    payload.update(_assemble_report(args, part, reference, topology))
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "float32",
        "tf32": False,
        "attention_implementation": "eager",
        "data_file": str(args.data),
    })
    write_json(args.output, payload)
    Path(args.part_path).unlink()
    args.stash_path.unlink()
    print(json.dumps({
        key: payload[key]
        for key in ("arm", "checks", "pass", "trajectory_max_loss_rel_error")
    }, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit("correctness checks failed")


def _wait_for_file(path: Path) -> None:
    while not path.exists():
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter42_train.jsonl"))
    parser.add_argument("--arm", default="a2_tp2pp2", choices=sorted([
        "a1_tp2", "a2_tp2pp2", "b2_tp2dp2", "c1_pp2", "c2_pp2dp2", "d_tp2pp2dp2"]))
    parser.add_argument("--world-size", type=int, default=None,
                        help="must equal tp*pp*dp of the arm; defaults to the arm's world size")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-micro", type=int, default=4)
    parser.add_argument("--samples-per-micro", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--allow-any-node-mapping", action="store_true",
                        help="allow a dual-node layout that is not one TP x PP replica per node")
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()

    topology = arm_topology(args.arm)
    expected_world = topology.world_size
    if args.world_size is None:
        args.world_size = expected_world
    if args.world_size != expected_world:
        raise SystemExit(f"arm {args.arm} needs world size {expected_world}, got --world-size {args.world_size}")
    args.stash_path = Path("correctness-hybrid-reference.pt")
    args.part_path = Path("correctness-hybrid.local.part.json")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args_ns = {
        "seed": args.seed, "steps": args.steps, "num_micro": args.num_micro,
        "samples_per_micro": args.samples_per_micro, "max_length": args.max_length,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "data": str(args.data), "arm": args.arm,
        "world_size": args.world_size,
        "allow_any_node_mapping": args.allow_any_node_mapping,
        "stash_path": str(args.stash_path), "part_path": str(args.part_path),
    }

    samples, members, _topology = build_step_micros(args, load_tokenizer(torch.device("cpu")).pad_token_id)
    samples_meta = (samples, members, topology)

    def atomic_torch_save(obj, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, temporary)
        os.replace(temporary, path)

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != args.world_size:
            raise SystemExit(f"--world-size {args.world_size} does not match torchrun WORLD_SIZE {world_size}")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        if rank == 0:
            device0 = torch.device("cuda:0")
            reference = run_reference(args, samples, members, topology, device0)
            atomic_torch_save(reference, args.stash_path)
            del reference
            gc.collect()
            torch.cuda.empty_cache()
        _wait_for_file(args.stash_path)
        _worker(rank, world_size, 0, args_ns, samples_meta, members)
        if rank == 0:
            _finalize(args, args_ns)
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required (or run under torchrun)")
    reference = run_reference(args, samples, members, topology, torch.device("cuda:0"))
    atomic_torch_save(reference, args.stash_path)
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns, samples_meta, members),
             nprocs=args.world_size, join=True)
    _finalize(args, args_ns)


if __name__ == "__main__":
    main()
