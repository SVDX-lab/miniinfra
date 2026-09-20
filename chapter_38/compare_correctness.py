#!/usr/bin/env python3
"""Chapter 38 correctness on real Qwen3-0.6B weights: Sequence Parallel vs the
chapter-36 pure-TP baseline (replicated activations).

Every rank runs BOTH layouts back to back on the same fixed batch plan, so the
comparison is the chapter's single variable directly — same weights, same
batch, same optimizer, only the activation layout and boundary collectives
differ:

- Phase A (baseline, ``tp_wN``): the chapter-36 attention+MLP TP model.  Step 0
  stashes every named parameter gradient (CPU), the forward loss and the
  measured saved-for-backward bytes; the trajectory runs through
  ``tp_train_step``; final parameters are reassembled with
  ``gather_tensor_parallel_parameters``.
- Phase B (variant, ``sp_wN``): the Sequence Parallel model from
  ``load_handwritten_model_sp``.  The same step-0 capture, the trajectory
  through ``sp_train_step``, the same reassembly — then, per rank: loss,
  every shard gradient, every replicated gradient (LayerNorms, Embedding/LM
  Head — the weights whose gradients Sequence Parallel turns into partial
  sums), the trajectory, the final parameters, and the saved activation bytes
  are compared against the stashed baseline values.

Launch with ``python compare_correctness.py`` (mp.spawn, single node,
``--world-size`` 2 or 4) or under ``torchrun`` (``--world-size`` must match
``WORLD_SIZE``).  FP32 AllReduce/AllGather reorders reductions, so checks use
tight tolerance bars, not bitwise equality, and the report distinguishes
mathematical equivalence from tolerance-level agreement.
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

from attention_tp import gqa_head_layout, validate_gqa_layout
from experiment_utils import environment_record, write_json
from mlp_tp import (
    gather_tensor_parallel_parameters,
    tp_memory_ledger,
    tp_train_step,
)
from sequence_parallel import (
    ActivationLedger,
    arm_byte_ledger,
    sp_memory_ledger,
    sp_train_step,
)
from sft_core import seed_everything
from tp_runtime import (
    build_replicated_batch,
    cleanup_distributed,
    encode_samples,
    free_port,
    load_tokenizer,
    load_tokenizer_and_model_sp,
    load_tokenizer_and_model_tp,
    make_batch_plan,
    setup_distributed,
)


LOSS_REL_BAR = 1e-5
GRAD_SCALE_REL_BAR = 5e-3
TRAJECTORY_BAR = 5e-4
PARAM_SCALE_REL_BAR = 1e-3
REPLICATED_SUM_REL_BAR = 1e-9
HIDDEN_SIZE_QWEN3 = 1024
NUM_LAYERS_QWEN3 = 28


def tensor_report(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Scale-aware comparison: max abs diff over max abs reference, plus a
    cosine check for near-zero-dominated tensors."""

    difference = (actual.double() - reference.double()).abs()
    reference_max = float(reference.double().abs().max())
    actual_max = float(actual.double().abs().max())
    dot = float((actual.double() * reference.double()).sum())
    cosine = dot / (max(float((reference.double() ** 2).sum()), 1e-30) ** 0.5 * max(actual_max ** 2, 1e-30) ** 0.5)
    return {
        "scale_relative_error": float(difference.max()) / max(reference_max, 1e-8),
        "max_absolute_error": float(difference.max()),
        "cosine_similarity": cosine,
        "reference_abs_max": reference_max,
    }


def _forward_saved_bytes(model, host_batch, device, tp_size, rank, is_sp: bool) -> dict:
    """One forward+loss under the ActivationLedger (no timing interference)."""

    from mini_batch import batch_loss
    from sequence_parallel import sp_loss_terms

    batch = model_batch_on_device(host_batch, device)
    ledger = ActivationLedger(model.parameters())
    with ledger:
        if is_sp:
            sequence_length = host_batch.input_ids.shape[1]
            # SequenceParallelModel.forward applies sequence_shard_bounds to
            # the FULL sequence it receives (same convention as sp_train_step);
            # pre-slicing here would shard the sequence twice.
            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            loss_sum, count = sp_loss_terms(logits, batch.labels, rank, tp_size, sequence_length)
            loss = loss_sum / float(count.clamp(min=1).item())
        else:
            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            loss = batch_loss(logits, batch.labels, "token_mean")
    return {"saved_bytes": ledger.saved_bytes, "saved_count": ledger.saved_count, "loss": float(loss.item())}


def model_batch_on_device(host_batch, device):
    from mlp_tp import to_device_batch

    return to_device_batch(host_batch, device)


def run_phase(args: SimpleNamespace, samples, plan, device, rank: int, world_size: int, is_sp: bool) -> dict:
    """One full layout pass: step-0 capture, trajectory, final parameters."""

    if is_sp:
        _, model = load_tokenizer_and_model_sp(device, rank, world_size)
    else:
        _, model = load_tokenizer_and_model_tp(device, rank, world_size, attention_parallel=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
        eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False,
    )
    ledger = (sp_memory_ledger if is_sp else tp_memory_ledger)(model, optimizer, world_size)
    ledger["measured_allocated_after_state_bytes"] = float(torch.cuda.memory_allocated(device))

    config = model.config
    layout = gqa_head_layout(rank, world_size, config.num_attention_heads, config.num_key_value_heads)

    # Step 0 on a dedicated backward: capture every named gradient BEFORE any
    # clipping (the gradient of the objective is the comparison target).
    host_batch, accounting = build_replicated_batch(samples, plan[0], args.pad_token_id)
    batch = model_batch_on_device(host_batch, device)
    optimizer.zero_grad(set_to_none=True)
    saved = _forward_saved_bytes(model, host_batch, device, world_size, rank, is_sp)
    # Re-run forward+backward WITH gradients (the ledger pass built no graph).
    if is_sp:
        from mini_batch import batch_loss
        from sequence_parallel import sp_loss_terms, sp_global_loss_scale

        sequence_length = host_batch.input_ids.shape[1]
        # Full sequence in; the model slices its own window internally
        # (same convention as sp_train_step).
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss_sum, count_local = sp_loss_terms(logits, batch.labels, rank, world_size, sequence_length)
        loss, global_loss, _ = sp_global_loss_scale(loss_sum, count_local, world_size, None, None)
        loss.backward()
        loss_value = global_loss
    else:
        from mini_batch import batch_loss

        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean") * accounting["loss_scale"]
        loss.backward()
        loss_value = float(loss.item())
    step0_loss = loss_value
    grads = {
        name: parameter.grad.detach().to("cpu").clone()
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    optimizer.zero_grad(set_to_none=True)
    del batch, logits, loss
    gc.collect()
    torch.cuda.empty_cache()

    steps = []
    for step_index in range(args.steps):
        host_batch, step_accounting = build_replicated_batch(samples, plan[step_index], args.pad_token_id)
        if is_sp:
            metrics = sp_train_step(
                model, optimizer, host_batch, step_accounting["loss_scale"], device,
                args.max_grad_norm, world_size, rank, collect_telemetry=False,
            )
        else:
            metrics = tp_train_step(
                model, optimizer, host_batch, step_accounting["loss_scale"], device,
                args.max_grad_norm, world_size, rank, collect_telemetry=False,
            )
        steps.append({
            "step": step_index,
            "global_loss": metrics["global_loss"],
            "gradient_norm": metrics["gradient_norm_before_clip"],
        })

    # Release grads + optimizer state BEFORE reassembling the full state for
    # the cross-rank comparison: on 12GB cards the 0.6B FP32 trainer sits at
    # ~9.6 GiB with grads and moments resident, and the gather needs headroom.
    # This is a correctness-only path after the last optimizer step; training
    # semantics are untouched.
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    final_state = {name: value.to("cpu") for name, value in gather_tensor_parallel_parameters(model, world_size).items()}

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "step0_loss": step0_loss,
        "step0_saved_activation_bytes": float(saved["saved_bytes"]),
        "step0_saved_activation_count": float(saved["saved_count"]),
        "grads": grads,
        "steps": steps,
        "final_state": final_state,
        "memory_ledger": ledger,
        "layout": layout,
    }


def _worker(rank: int, world_size: int, port: int, args_ns: dict, samples, plan) -> None:
    if "RANK" not in os.environ:
        # mp.spawn path: publish the env:// endpoints for this single node.
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    # Under torchrun the environment already carries RANK/WORLD_SIZE/LOCAL_RANK.
    args = SimpleNamespace(**args_ns)
    seed_everything(args.seed)
    _, _, device = setup_distributed(rank, world_size)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    baseline = run_phase(args, samples, plan, device, rank, world_size, is_sp=False)
    variant = run_phase(args, samples, plan, device, rank, world_size, is_sp=True)

    # ---- per-rank comparisons: variant (SP) against the stashed baseline ----
    # Both arms share the same weight layout, so every name compares directly;
    # classification (shard vs replicated) is by name, because the replicated
    # set (norms, Embedding/LM Head) is exactly what SP turns into partial
    # sums that the sync hooks must restore.
    def is_replicated(name: str) -> bool:
        return not (
            ".self_attn.q_proj." in name or ".self_attn.k_proj." in name or ".self_attn.v_proj." in name
            or ".self_attn.o_proj." in name or ".mlp.gate_proj." in name or ".mlp.up_proj." in name
            or ".mlp.down_proj." in name
        )

    step0 = {
        "loss_rel_error": abs(variant["step0_loss"] - baseline["step0_loss"]) / abs(baseline["step0_loss"]),
        "baseline_loss_step0": baseline["step0_loss"],
        "sp_loss_step0": variant["step0_loss"],
        "gradient_worst": None,
        "replicated_worst": None,
        "replicated_count": 0,
        "shard_count": 0,
    }
    worst = None
    worst_replicated = None
    replicated_count = 0
    shard_count = 0
    replicated_sums = []
    for name, sp_grad in variant["grads"].items():
        baseline_grad = baseline["grads"][name]
        # Both stashes live on the CPU; compare there so the check needs no
        # GPU memory after the models are gone.
        report = tensor_report(sp_grad, baseline_grad)
        report["parameter"] = name
        if is_replicated(name):
            replicated_count += 1
            replicated_sums.append(float(sp_grad.double().sum()))
            if worst_replicated is None or report["scale_relative_error"] > worst_replicated["scale_relative_error"]:
                worst_replicated = report
        else:
            shard_count += 1
            if worst is None or report["scale_relative_error"] > worst["scale_relative_error"]:
                worst = report
    step0["gradient_worst"] = worst
    step0["replicated_worst"] = worst_replicated
    step0["replicated_count"] = replicated_count
    step0["shard_count"] = shard_count

    trajectory_errors = [
        abs(sp_step["global_loss"] - tp_step["global_loss"]) / abs(tp_step["global_loss"])
        for sp_step, tp_step in zip(variant["steps"], baseline["steps"])
    ]
    param_report = []
    param_worst = None
    for name in sorted(variant["final_state"]):
        report = tensor_report(variant["final_state"][name], baseline["final_state"][name])
        report["parameter"] = name
        param_report.append(report)
        if param_worst is None or report["scale_relative_error"] > param_worst["scale_relative_error"]:
            param_worst = report

    # Cross-rank identity of the SP replicated gradients (post-sync): the
    # sums must agree to float64 exactness WITHOUT any extra communication.
    all_replicated_sums = [None] * world_size
    dist.all_gather_object(all_replicated_sums, replicated_sums)
    worst_cross = 0.0
    for other_rank, other_sums in enumerate(all_replicated_sums):
        for own, other in zip(replicated_sums, other_sums):
            worst_cross = max(worst_cross, abs(own - other) / max(1.0, abs(own)))

    sp_bytes = variant["step0_saved_activation_bytes"]
    tp_bytes = baseline["step0_saved_activation_bytes"]

    worker_report = {
        "rank": rank,
        "step0": step0,
        "replicated_grad_cross_rank_max_rel": worst_cross,
        "trajectory_max_loss_rel_error": max(trajectory_errors),
        "trajectory_loss_rel_errors": trajectory_errors,
        "param_worst": param_worst,
        "saved_activation_bytes": {"tp": tp_bytes, "sp": sp_bytes, "ratio_sp_over_tp": sp_bytes / max(tp_bytes, 1.0)},
        "memory_ledger": {"tp": baseline["memory_ledger"], "sp": variant["memory_ledger"]},
        "gqa_layout": variant["layout"],
        "gqa_mapping_valid": validate_gqa_layout(variant["layout"]),
        "steps": variant["steps"],
    }

    del baseline, variant
    gc.collect()
    torch.cuda.empty_cache()

    reports = [None] * world_size
    dist.all_gather_object(reports, worker_report)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args_ns["part_path"]), {"world_size": world_size, "ranks": reports})


def assemble_report(args: SimpleNamespace, part: dict, args_ns: dict) -> dict:
    ranks = part["ranks"]
    rank0 = ranks[0]
    checks = {
        "forward_loss_rel_ok": rank0["step0"]["loss_rel_error"] < LOSS_REL_BAR,
        "cross_rank_loss_ok": all(
            abs(rank["step0"]["sp_loss_step0"] - ranks[0]["step0"]["sp_loss_step0"]) / abs(rank0["step0"]["sp_loss_step0"]) < 1e-6
            for rank in ranks
        ),
        "gqa_mapping_ok": all(rank["gqa_mapping_valid"] for rank in ranks),
        "shard_grad_ok": rank0["step0"]["gradient_worst"]["scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "replicated_grad_ok": rank0["step0"]["replicated_worst"]["scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "replicated_grads_identical": max(
            rank["replicated_grad_cross_rank_max_rel"] for rank in ranks
        ) < REPLICATED_SUM_REL_BAR,
        "trajectory_ok": max(rank["trajectory_max_loss_rel_error"] for rank in ranks) < TRAJECTORY_BAR,
        "final_params_ok": rank0["param_worst"]["scale_relative_error"] < PARAM_SCALE_REL_BAR,
        "activation_saved_ok": all(rank["saved_activation_bytes"]["sp"] < rank["saved_activation_bytes"]["tp"] for rank in ranks),
        "shard_fraction_ok": abs(
            rank0["memory_ledger"]["sp"]["shard_parameter_bytes_local"]
            / (rank0["memory_ledger"]["sp"]["shard_parameter_bytes_full"] + 1e-9) - 1.0 / args.world_size
        ) < 0.01,
    }
    return {
        "schema_note": {
            "equivalence": "Sequence Parallel rearranges activations and boundary collectives only; FP32 collectives reorder reductions, so checks use tight tolerance bars, not bitwise equality",
            "bars": {
                "loss_rel": LOSS_REL_BAR,
                "grad_scale_rel": GRAD_SCALE_REL_BAR,
                "trajectory_rel": TRAJECTORY_BAR,
                "param_scale_rel": PARAM_SCALE_REL_BAR,
            },
        },
        "world_size": args.world_size,
        "tp_size": args.world_size,
        "sequence_parallel": True,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "nnodes": int(os.environ.get("NNODES", "1")),
        "node_rank": int(os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "per_rank_saved_activation_bytes": [
            {"rank": rank["rank"], **rank["saved_activation_bytes"]} for rank in ranks
        ],
        "per_rank_step0": {
            "loss_rel_error": [rank["step0"]["loss_rel_error"] for rank in ranks],
            "shard_gradient_worst": [rank["step0"]["gradient_worst"] for rank in ranks],
            "replicated_gradient_worst": [rank["step0"]["replicated_worst"] for rank in ranks],
            "replicated_gradient_count": [rank["step0"]["replicated_count"] for rank in ranks],
            "shard_gradient_count": [rank["step0"]["shard_count"] for rank in ranks],
        },
        "trajectory_max_loss_rel_error": max(rank["trajectory_max_loss_rel_error"] for rank in ranks),
        "param_worst": rank0["param_worst"],
        "analytic_byte_ledger": arm_byte_ledger(
            args.batch_size, args.max_length, HIDDEN_SIZE_QWEN3, args.world_size, NUM_LAYERS_QWEN3
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _finalize(args: SimpleNamespace, args_ns: dict) -> None:
    part = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
    payload = environment_record(world_size=args.world_size, dp_degree=1, tp_degree=args.world_size)
    payload.update(assemble_report(args, part, args_ns))
    payload.update({
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "model_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "dtype": "float32",
        "tf32": False,
        "attention_implementation": "eager",
        "data_file": str(args.data),
    })
    write_json(args.output, payload)
    Path(args.part_path).unlink()
    print(json.dumps({
        key: payload[key]
        for key in ("checks", "pass", "trajectory_max_loss_rel_error", "per_rank_saved_activation_bytes")
    }, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit("correctness checks failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter38_train.jsonl"))
    parser.add_argument("--world-size", type=int, default=2, choices=(2, 4),
                        help="tp_size equals the world size in this chapter")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    args.part_path = Path("correctness-sp-part.json")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = load_tokenizer(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    args_ns = {
        "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
        "max_length": args.max_length, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "pad_token_id": args.pad_token_id, "part_path": str(args.part_path),
    }
    from sft_core import load_jsonl_records

    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them on BOTH paths so the
        # TP/SP comparison stays on identical samples.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    # Batch plans need args.batch_size members per step.
    usable = len(samples) - len(samples) % args.batch_size
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    plan = make_batch_plan(len(samples), args.batch_size, args.seed)

    if "RANK" in os.environ:
        # torchrun path: every rank runs both phases directly on its GPU.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != args.world_size:
            raise SystemExit(f"--world-size {args.world_size} does not match torchrun WORLD_SIZE {world_size}")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        _worker(rank, world_size, 0, args_ns, samples, plan)
        if rank == 0:
            _finalize(args, args_ns)
        return

    # mp.spawn path: single node.
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns, samples, plan), nprocs=args.world_size, join=True)
    _finalize(args, args_ns)


if __name__ == "__main__":
    main()
