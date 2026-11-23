#!/usr/bin/env python3
"""Chapter 35 correctness on real Qwen3-0.6B weights: MLP TP versus replicated.

Phase A (launcher process, cuda:0): the replicated FP32 reference — full
model, plain AdamW, plain ``clip_grad_norm_`` — over the same fixed batch
plan.  Step 0 stashes the reference MLP gradients, the embedding input
gradient, the full SwiGLU intermediates of two layers, and a few replicated
gradients; the final step stashes the final parameters.

Phase B (one GPU per rank): the MLP-TP model from ``load_handwritten_model_tp``
runs the same batch on every rank (TP replicates data).  Step 0 compares —
per rank — forward loss, the rank's SwiGLU intermediate shard, the input
gradient, every MLP shard gradient against the matching slice of the
reference gradient, and the replicated gradients' cross-rank identity (the
gradient-ownership proof of the module docstring in ``mlp_tp.py``).  The
trajectory runs through ``tp_train_step``; the final parameters are
reassembled from the shards and compared against the reference.

Launch with ``python compare_correctness.py`` (mp.spawn, single node,
``--world-size`` 2 or 4) or under ``torchrun`` (``--world-size`` must match
``WORLD_SIZE``).  Bars are NOT relaxed: FP32 AllReduce changes the reduction
association, so equivalence is expected within tight tolerances and the
report distinguishes mathematical equivalence from tolerance-level agreement.
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

from experiment_utils import environment_record, write_json
from mini_batch import batch_loss
from mlp_tp import (
    gather_tensor_parallel_parameters,
    split_local_parameters,
    to_device_batch,
    tp_memory_ledger,
    tp_train_step,
)
from sft_core import seed_everything
from tp_runtime import (
    build_replicated_batch,
    cleanup_distributed,
    encode_samples,
    free_port,
    load_tokenizer,
    load_tokenizer_and_model_tp,
    make_batch_plan,
    setup_distributed,
)


LOSS_REL_BAR = 1e-5          # red line: forward loss vs the FP32 reference
GRAD_COSINE_BAR = 0.99999
GRAD_SCALE_REL_BAR = 5e-3
TRAJECTORY_BAR = 5e-4
PARAM_SCALE_REL_BAR = 1e-3
REPLICATED_SUM_REL_BAR = 1e-9
INTERMEDIATE_LAYERS = (0, -1)
OWNERSHIP_KEYS = ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.norm.weight")


def tensor_report(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Scale-aware comparison: raw elementwise relative error is dominated by
    near-zero entries, so the primary bar is max abs diff over max abs ref."""

    # On 12GB GPUs whole-tensor float64 copies of the 155M-element embedding
    # (about 1.2 GiB each) do not fit next to the resident training state, so
    # walk both tensors in flat float64 chunks; every reported quantity is
    # additive (or a max) over chunks and therefore unchanged.
    chunk = 1 << 23  # 8M elements: ~32 MB fp32 / ~64 MB fp64 per side
    actual_flat = actual.detach().reshape(-1)
    reference_flat = reference.reshape(-1)
    difference_max = 0.0
    reference_max = 0.0
    actual_max = 0.0
    dot = 0.0
    reference_sq = 0.0
    for start in range(0, actual_flat.numel(), chunk):
        actual_chunk = actual_flat[start:start + chunk].double()
        reference_chunk = reference_flat[start:start + chunk].double()
        difference_max = max(difference_max, float((actual_chunk - reference_chunk).abs().max()))
        reference_max = max(reference_max, float(reference_chunk.abs().max()))
        actual_max = max(actual_max, float(actual_chunk.abs().max()))
        dot += float((actual_chunk * reference_chunk).sum())
        reference_sq += float((reference_chunk * reference_chunk).sum())
    cosine = dot / (max(reference_sq, 1e-30) ** 0.5 * max(actual_max ** 2, 1e-30) ** 0.5)
    return {
        "scale_relative_error": difference_max / max(reference_max, 1e-8),
        "max_absolute_error": difference_max,
        "cosine_similarity": cosine,
        "reference_abs_max": reference_max,
    }


def _chunked_double_sum(tensor: torch.Tensor) -> float:
    """float64 total of a tensor without materializing a full double copy."""

    total = 0.0
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), 1 << 23):
        total += float(flat[start:start + (1 << 23)].double().sum())
    return total


def _capture_hooks(model, layers: tuple[int, ...]):
    """Embedding output-gradient hook + down_proj input (SwiGLU output) hooks."""

    captured: dict = {"input_grad": None, "intermediates": {}}
    handles = []

    def backward_hook(module, grad_input, grad_output):
        if captured["input_grad"] is None:
            captured["input_grad"] = grad_output[0].detach().clone()

    handles.append(model.model.embed_tokens.register_full_backward_hook(backward_hook))
    for layer_index in layers:
        module = model.model.layers[layer_index].mlp.down_proj

        def forward_hook(mod, args, output, index=layer_index):
            if index not in captured["intermediates"]:
                captured["intermediates"][index] = args[0].detach().clone()

        handles.append(module.register_forward_hook(forward_hook))
    return captured, handles


def run_reference(args: SimpleNamespace, samples, plan, device) -> dict:
    """Phase A: the replicated reference trajectory with torch AdamW."""

    seed_everything(args.seed)
    from qwen3_model import load_handwritten_model
    from tp_runtime import MODEL_ID, MODEL_REVISION, resolve_model_directory

    directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    model = load_handwritten_model(directory, device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)

    steps = []
    mlp_grads: dict = {}
    input_grad = None
    intermediates: dict = {}
    ownership_grads: dict = {}
    ref_loss0 = None
    final_params: dict = {}
    for step_index in range(args.steps):
        host_batch, accounting = build_replicated_batch(samples, plan[step_index], args.pad_token_id)
        batch = to_device_batch(host_batch, device)
        captured, handles = (None, [])
        if step_index == 0:
            captured, handles = _capture_hooks(model, INTERMEDIATE_LAYERS)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean") * accounting["loss_scale"]
        loss.backward()
        if step_index == 0:
            ref_loss0 = float(loss.item())
            input_grad = captured["input_grad"]
            intermediates = captured["intermediates"]
            for handle in handles:
                handle.remove()
            # Stash BEFORE clipping: the gradient of the objective is the
            # comparison target; clipping is a training-step concern.
            for name, parameter in model.named_parameters():
                if ".mlp." in name and parameter.grad is not None:
                    mlp_grads[name] = parameter.grad.detach().to("cpu").clone()
                if name in OWNERSHIP_KEYS and parameter.grad is not None:
                    ownership_grads[name] = parameter.grad.detach().to("cpu").clone()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        steps.append({
            "step": step_index,
            "loss": float(loss.item()),
            "gradient_norm": float(grad_norm.item()),
        })
    final_params = {name: parameter.detach().to("cpu").clone() for name, parameter in model.named_parameters()}
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "steps": steps,
        "ref_loss0": ref_loss0,
        "mlp_grads": mlp_grads,
        "input_grad": input_grad,
        "intermediates": intermediates,
        "ownership_grads": ownership_grads,
        "final_params": final_params,
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
    rank_id, _, device = setup_distributed(rank, world_size)
    assert rank_id == rank
    _, model = load_tokenizer_and_model_tp(device, rank, world_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)
    ledger = tp_memory_ledger(model, optimizer, world_size)
    ledger["measured_allocated_after_state_bytes"] = float(torch.cuda.memory_allocated(device))

    reference = torch.load(args.stash_path, map_location="cpu", weights_only=False)
    slice_ = model.model.layers[0].mlp.intermediate_slice()
    local_intermediate = slice_.stop - slice_.start

    step0 = {
        "loss_rel_error": None,
        "input_gradient": None,
        "mlp_intermediates": {},
        "mlp_weight_gradients": [],
        "mlp_weight_gradient_worst": None,
        "ownership_vs_reference": {},
        "replicated_grad_cross_rank_max_rel": None,
        "shard_concat_layer0": {},
    }

    # Step-0 comparisons on a dedicated backward (same batch, same weights).
    host_batch, accounting = build_replicated_batch(samples, plan[0], args.pad_token_id)
    batch = to_device_batch(host_batch, device)
    captured, handles = _capture_hooks(model, INTERMEDIATE_LAYERS)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.input_ids, attention_mask=batch.attention_mask)
    loss = batch_loss(logits, batch.labels, "token_mean") * accounting["loss_scale"]
    loss.backward()
    step0["loss_rel_error"] = abs(float(loss.item()) - reference["ref_loss0"]) / abs(reference["ref_loss0"])
    step0["local_loss_step0"] = float(loss.item())
    for handle in handles:
        handle.remove()

    losses_across_ranks = [None] * world_size
    dist.all_gather_object(losses_across_ranks, float(loss.item()))
    step0["cross_rank_loss_max_abs"] = max(losses_across_ranks) - min(losses_across_ranks)

    step0["input_gradient"] = tensor_report(captured["input_grad"], reference["input_grad"].to(device))
    for layer_index, shard_h in captured["intermediates"].items():
        reference_h = reference["intermediates"][layer_index].to(device)
        step0["mlp_intermediates"][layer_index] = tensor_report(
            shard_h, reference_h[:, :, slice_.start : slice_.stop]
        )

    # Shard gradients vs the matching slices of the full reference gradient.
    named = dict(model.named_parameters())
    worst = None
    for layer_index in range(len(model.model.layers)):
        for param_name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            full_name = f"model.layers.{layer_index}.mlp.{param_name}"
            parameter = named[full_name]
            reference_grad = reference["mlp_grads"][full_name].to(device)
            if param_name == "down_proj.weight":
                reference_slice = reference_grad[:, slice_.start : slice_.stop]
            else:
                reference_slice = reference_grad[slice_.start : slice_.stop, :]
            report = tensor_report(parameter.grad, reference_slice)
            report.update({"layer": layer_index, "parameter": param_name})
            step0["mlp_weight_gradients"].append(report)
            if worst is None or report["scale_relative_error"] > worst["scale_relative_error"]:
                worst = report
    step0["mlp_weight_gradient_worst"] = worst

    # Ownership class 3: replicated gradients must equal the FULL reference
    # gradient on every rank WITHOUT cross-rank communication — an AllReduce
    # SUM here would show up as tp_size times the reference.
    for key in OWNERSHIP_KEYS:
        step0["ownership_vs_reference"][key] = tensor_report(named[key].grad, reference["ownership_grads"][key].to(device))
    replicated, shard = split_local_parameters(model)
    replicated_sums = [None] * world_size
    dist.all_gather_object(replicated_sums, [_chunked_double_sum(p.grad) for p in replicated])
    worst_cross = 0.0
    for other_rank, other_sums in enumerate(replicated_sums):
        for own, other in zip((_chunked_double_sum(p.grad) for p in replicated), other_sums):
            worst_cross = max(worst_cross, abs(own - other) / max(1.0, abs(own)))
    step0["replicated_grad_cross_rank_max_rel"] = worst_cross

    # Layer-0 shards must concatenate into the full reference gradient.
    for param_name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        full_name = f"model.layers.0.mlp.{param_name}"
        parameter = named[full_name]
        buffer = [torch.empty_like(parameter.grad) for _ in range(world_size)]
        dist.all_gather(buffer, parameter.grad.contiguous())
        dim = 1 if param_name == "down_proj.weight" else 0
        step0["shard_concat_layer0"][param_name] = tensor_report(
            torch.cat(buffer, dim=dim), reference["mlp_grads"][full_name].to(device)
        )
    optimizer.zero_grad(set_to_none=True)
    del captured, logits, loss
    gc.collect()
    torch.cuda.empty_cache()

    # Training trajectory through the real step function.
    steps = []
    for step_index in range(args.steps):
        host_batch, step_accounting = build_replicated_batch(samples, plan[step_index], args.pad_token_id)
        metrics = tp_train_step(
            model, optimizer, host_batch, step_accounting["loss_scale"], device,
            args.max_grad_norm, world_size, rank, collect_telemetry=False,
        )
        steps.append({
            "step": step_index,
            "global_loss": metrics["global_loss"],
            "gradient_norm": metrics["gradient_norm_before_clip"],
            "tp_allreduce_forward_ms": metrics["tp_allreduce_forward_ms"],
            "tp_allreduce_backward_ms": metrics["tp_allreduce_backward_ms"],
        })

    final_state = gather_tensor_parallel_parameters(model, world_size)
    param_report = []
    param_worst = None
    for name, parameter in full_state_items(final_state):
        reference_parameter = reference["final_params"][name].to(device)
        report = tensor_report(parameter, reference_parameter)
        report["parameter"] = name
        param_report.append(report)
        if param_worst is None or report["scale_relative_error"] > param_worst["scale_relative_error"]:
            param_worst = report
    del final_state, reference
    gc.collect()
    torch.cuda.empty_cache()

    worker_report = {
        "rank": rank,
        "step0": step0,
        "local_intermediate": local_intermediate,
        "steps": steps,
        "param_worst": param_worst,
        "memory_ledger": ledger,
    }
    reports = [None] * world_size
    dist.all_gather_object(reports, worker_report)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args.part_path), {"world_size": world_size, "ranks": reports})


def full_state_items(state: dict):
    for name in sorted(state):
        yield name, state[name]


def assemble_report(args: SimpleNamespace, part: dict, reference: dict) -> dict:
    ranks = part["ranks"]
    rank0 = ranks[0]
    trajectory_errors = [
        abs(step["global_loss"] - ref["loss"]) / abs(ref["loss"])
        for ref, step in zip(reference["steps"], rank0["steps"])
    ]
    input_grad = rank0["step0"]["input_gradient"]
    checks = {
        "forward_loss_rel_ok": rank0["step0"]["loss_rel_error"] < LOSS_REL_BAR,
        "cross_rank_loss_identical": rank0["step0"]["cross_rank_loss_max_abs"] < 1e-6,
        "input_gradient_cosine_ok": input_grad["cosine_similarity"] > GRAD_COSINE_BAR,
        "input_gradient_scale_ok": input_grad["scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "mlp_intermediate_ok": all(
            report["scale_relative_error"] < GRAD_SCALE_REL_BAR
            for report in rank0["step0"]["mlp_intermediates"].values()
        ),
        "mlp_weight_grad_ok": rank0["step0"]["mlp_weight_gradient_worst"]["scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "ownership_vs_reference_ok": all(
            report["scale_relative_error"] < GRAD_SCALE_REL_BAR
            for report in rank0["step0"]["ownership_vs_reference"].values()
        ),
        "replicated_grads_identical": rank0["step0"]["replicated_grad_cross_rank_max_rel"] < REPLICATED_SUM_REL_BAR,
        "shard_concat_ok": all(
            report["scale_relative_error"] < GRAD_SCALE_REL_BAR
            for report in rank0["step0"]["shard_concat_layer0"].values()
        ),
        "trajectory_ok": max(trajectory_errors) < TRAJECTORY_BAR,
        "final_params_ok": rank0["param_worst"]["scale_relative_error"] < PARAM_SCALE_REL_BAR,
        "shard_fraction_ok": abs(
            rank0["memory_ledger"]["shard_parameter_bytes_local"]
            / (rank0["memory_ledger"]["shard_parameter_bytes_full"] + 1e-9) - 1.0 / args.world_size
        ) < 0.01,
    }
    return {
        "schema_note": {
            "equivalence": "TP path is mathematically equivalent to the replicated path; FP32 AllReduce reorders reductions, so checks use tight tolerance bars, not bitwise equality",
            "bars": {
                "loss_rel": LOSS_REL_BAR,
                "grad_cosine": GRAD_COSINE_BAR,
                "grad_scale_rel": GRAD_SCALE_REL_BAR,
                "trajectory_rel": TRAJECTORY_BAR,
                "param_scale_rel": PARAM_SCALE_REL_BAR,
                "replicated_sum_rel": REPLICATED_SUM_REL_BAR,
            },
        },
        "world_size": args.world_size,
        "tp_size": args.world_size,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "nnodes": int(os.environ.get("NNODES", "1")),
        "node_rank": int(os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "step0": rank0["step0"],
        "per_rank_step0_loss_rel_error": [rank["step0"]["loss_rel_error"] for rank in ranks],
        "trajectory_max_loss_rel_error": max(trajectory_errors),
        "trajectory_loss_rel_errors": trajectory_errors,
        "param_worst": rank0["param_worst"],
        "memory_ledger": rank0["memory_ledger"],
        "tp_allreduce_ms_per_step": {
            "forward_mean": fmean(step["tp_allreduce_forward_ms"] for step in rank0["steps"]),
            "backward_mean": fmean(step["tp_allreduce_backward_ms"] for step in rank0["steps"]),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter35_train.jsonl"))
    parser.add_argument("--world-size", type=int, default=2, choices=(2, 4),
                        help="tp_size equals the world size in this chapter")
    parser.add_argument("--tp-size", type=int, default=None, choices=(2, 4),
                        help="alias for --world-size (the TP degree IS the world size here)")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    if args.tp_size is not None:
        if args.tp_size != args.world_size:
            raise SystemExit("--tp-size is an alias for --world-size; they must agree")
        args.world_size = args.tp_size
    args.stash_path = Path("correctness-reference-grads.pt")
    args.part_path = Path("correctness-tp-part.json")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = load_tokenizer(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    args_ns = {
        "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
        "max_length": args.max_length, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "pad_token_id": args.pad_token_id, "stash_path": str(args.stash_path),
        "part_path": str(args.part_path),
    }
    from sft_core import load_jsonl_records

    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them on BOTH paths so the TP
        # comparison stays on identical samples.
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

    import os

    if "RANK" in os.environ:
        # torchrun path: rank 0 alone builds the Phase A reference on its
        # local cuda:0 (stashing it for the other ranks), then every rank
        # runs the TP worker body directly.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != args.world_size:
            raise SystemExit(f"--world-size {args.world_size} does not match torchrun WORLD_SIZE {world_size}")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        if rank == 0:
            reference = run_reference(args, samples, plan, torch.device("cuda:0"))
            atomic_torch_save(reference, args.stash_path)
            del reference
            gc.collect()
            torch.cuda.empty_cache()
        # No process group exists at launcher level under torchrun; the other
        # ranks wait for the atomically-renamed stash file instead.
        _wait_for_file(args.stash_path)
        _worker(rank, world_size, 0, args_ns, samples, plan)
        if rank == 0:
            _finalize(args, args_ns)
        return

    # mp.spawn path: single node.
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    reference = run_reference(args, samples, plan, torch.device("cuda:0"))
    atomic_torch_save(reference, args.stash_path)
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns, samples, plan), nprocs=args.world_size, join=True)
    _finalize(args, args_ns)


def atomic_torch_save(obj, path: Path) -> None:
    """torch.save into a temp file, then an atomic rename: readers never see
    a half-written stash."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def _wait_for_file(path: Path) -> None:
    import time

    while not path.exists():
        time.sleep(1.0)


def _finalize(args: SimpleNamespace, args_ns: dict) -> None:
    part = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
    reference = torch.load(args.stash_path, map_location="cpu", weights_only=False)
    payload = environment_record(world_size=args.world_size, dp_degree=1, tp_degree=args.world_size)
    payload.update(assemble_report(args, part, reference))
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
    args.stash_path.unlink()
    print(json.dumps({key: payload[key] for key in ("checks", "pass", "trajectory_max_loss_rel_error")}, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit("correctness checks failed")


if __name__ == "__main__":
    main()
