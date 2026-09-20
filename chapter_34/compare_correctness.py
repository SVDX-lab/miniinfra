#!/usr/bin/env python3
"""Chapter 34 correctness on real weights: ZeRO-2 baseline versus FSDP.

Phase A (rank 0 only, its local cuda:0): the big-batch reference trajectory
with torch AdamW — identical to Chapters 14-18 — with step-0 gradients
stashed for the baseline gradient bars.

Phase B (one GPU per rank): the ZeRO-2 BASELINE flow, unchanged from
Chapter 33: sharded batches, per-parameter Reduce-Scatter, ShardedGradAdamW,
parameter AllGather. Step 0 keeps the bitwise shard checks (shard == AllReduce
slice), the full-rank parameter fingerprints are recorded after every step,
and rank 0 stashes the full parameter tensors after step 1 and after the last
step for tensor-level comparison.

Phase C (same ranks, baseline freed first): the ZeRO-3/FSDP VARIANT — the
handwritten Qwen3 model wrapped with FSDP FULL_SHARD (per DecoderLayer units,
BACKWARD_PRE prefetch, use_orig_params=True for the tied embedding). The
local loss is multiplied by world_size to cancel FSDP's averaged
Reduce-Scatter, so both paths optimize the SAME global token_mean objective.
After steps 1 and the last step, FSDP's FULL_STATE_DICT is gathered and
compared tensor-by-tensor against the baseline stash (both FP32; different
reduction orders may fork a few ULPs — tolerance bars, not bitwise). Finally
the FSDP model+optimizer state is saved and restored into a fresh FSDP model,
which replays the next step: its loss and parameters must match the
uninterrupted run.

Launch with ``python compare_correctness.py`` (mp.spawn, single node) or under
``torchrun``; the global batch stays at BATCH_SIZE, ``--world-size`` is 2
(default) or 4, and under torchrun it must match ``WORLD_SIZE``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ddp_core import (
    MODEL_ID,
    MODEL_REVISION,
    all_reduce_gradients,
    all_reduce_scalars,
    cleanup_distributed,
    encode_samples,
    free_port,
    load_tokenizer_and_model,
    make_batch_plan,
    setup_distributed,
    to_device_batch,
)
from experiment_utils import write_json
from compare_reference import gradient_report, run_reference
from fsdp_train import (
    fsdp_train_step,
    fsdp_theoretical_ledger,
    get_full_state_dicts,
    load_full_state_dicts,
    wrap_model_fsdp,
)
from mini_batch import batch_loss
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, parameter_bytes, seed_everything
from zero2_core import (
    ShardedGradAdamW,
    all_gather_parameters,
    memory_ledger,
    reduce_scatter_gradients,
    zero2_train_step,
)


GRAD_COSINE_BAR = 0.99999
GRAD_SCALE_REL_BAR = 5e-3
TRAJECTORY_BAR = 5e-4
STEP0_LOSS_BAR = 1e-6
GRAD_NORM_REL_BAR = 5e-3
PARAM_SCALE_REL_BAR = 1e-3
RESUME_PARAM_ABS_BAR = 1e-7
BATCH_SIZE = 4


def fingerprint(model: torch.nn.Module) -> dict:
    return {name: float(parameter.detach().double().sum()) for name, parameter in model.named_parameters()}


def scale_rel_report(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict:
    """Worst per-tensor scale-relative difference, like compare_reference.gradient_report."""

    worst_rel = 0.0
    worst_entry = None
    total_abs = 0.0
    for name, tensor in actual.items():
        if name not in reference:
            # FSDP's FULL_STATE_DICT lists the tied lm_head.weight as its own
            # entry while the baseline snapshot comes from named_parameters(),
            # which keeps only model.embed_tokens.weight for the tied pair.
            # Skipping the alias loses no coverage because the shared tensor
            # is compared under its embedding name.
            continue
        reference_tensor = reference[name]
        difference = (tensor.to(reference_tensor.dtype) - reference_tensor).abs()
        scale = float(reference_tensor.abs().max())
        relative = float(difference.max()) / max(scale, 1e-8)
        if relative > worst_rel:
            worst_rel = relative
            worst_entry = name
        total_abs += float(difference.max())
    return {
        "max_scale_relative_error": worst_rel,
        "max_scale_relative_error_parameter": worst_entry,
        "sum_of_max_absolute_errors": total_abs,
    }


def _run_baseline_phase(args: SimpleNamespace, samples, plan, device, rank: int, world_size: int) -> dict:
    """Phase B body: the Chapter 33 ZeRO-2 flow, plus state stashes for Phase C."""

    _, model = load_tokenizer_and_model(device)
    optimizer = ShardedGradAdamW(
        list(model.parameters()), world_size, rank,
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay,
    )
    ledger = memory_ledger(list(model.parameters()), optimizer)
    ledger["measured_allocated_after_state_bytes"] = float(torch.cuda.memory_allocated(device))

    step0_grad_report = None
    shard_bitwise_ok = True
    steps = []
    for step_index in range(args.steps):
        from ddp_core import build_global_batch

        host_batch, accounting = build_global_batch(samples, plan[step_index], rank, world_size, args.pad_token_id)
        if step_index == 0:
            device_batch = to_device_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(device_batch.input_ids, attention_mask=device_batch.attention_mask)
            loss = batch_loss(logits, device_batch.labels, "token_mean") * accounting["loss_scale"]
            loss.backward()
            local_grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
            reduce_scatter_gradients(list(model.parameters()), optimizer, world_size)
            # Owner shard must equal the AllReduced global gradient's slice, bitwise.
            for (name, parameter), grad_shard, (start, end) in zip(model.named_parameters(), optimizer.grad_shards, optimizer.bounds):
                reference = local_grads[name].to(device).view(-1)
                dist.all_reduce(reference, op=dist.ReduceOp.SUM)
                if not torch.equal(grad_shard[: end - start], reference[start:end]):
                    shard_bitwise_ok = False
            del local_grads
            # Global gradient report against the big-batch reference.
            optimizer.zero_grad(set_to_none=True)
            logits = model(device_batch.input_ids, attention_mask=device_batch.attention_mask)
            loss = batch_loss(logits, device_batch.labels, "token_mean") * accounting["loss_scale"]
            loss.backward()
            all_reduce_gradients(model, world_size)
            if rank == 0:
                step0_grad_report = gradient_report(model, args.stash_path)
            optimizer.zero_grad(set_to_none=True)

        metrics = zero2_train_step(model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
        # zero2_core already folds loss_scale into metrics["local_loss"] (and
        # fsdp_train_step divides its world-size factor back out), so the sum
        # over ranks IS the global token_mean; scaling again would report
        # sum(local_loss * scale^2) and read about half of the reference loss.
        weighted_global_loss = all_reduce_scalars(metrics["local_loss"], world_size, device)
        steps.append({
            "step": step_index,
            "global_loss": weighted_global_loss,
            "gradient_norm": metrics["gradient_norm_before_clip"],
            "fingerprint": fingerprint(model),
        })
        if step_index in (1, args.steps - 1) and rank == 0:
            torch.save(
                {name: parameter.detach().to("cpu").clone() for name, parameter in model.named_parameters()},
                args.baseline_state_path(step_index),
            )

    final_fingerprints = [None] * world_size
    dist.all_gather_object(final_fingerprints, fingerprint(model))
    identity = all(other == final_fingerprints[0] for other in final_fingerprints)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "step0_grad_report": step0_grad_report,
        "shard_bitwise_ok": shard_bitwise_ok,
        "steps": steps,
        "cross_rank_fingerprints_identical": identity,
        "memory_ledger": ledger,
    }


def _run_fsdp_phase(args: SimpleNamespace, samples, plan, device, rank: int, world_size: int) -> dict:
    """Phase C body: FSDP FULL_SHARD over the same plan, then save/resume.

    All tensor-level comparisons against the stashed baseline states happen
    HERE on rank 0, so only scalar reports cross the part-JSON boundary.
    """

    from fsdp_train import set_full_state_dict_type

    _, model = load_tokenizer_and_model(device)
    numel = parameter_bytes(model) // 4
    fsdp_model = wrap_model_fsdp(model, device)
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False,
    )
    ledger = fsdp_theoretical_ledger(numel, world_size)
    # torch AdamW allocates its state lazily at the FIRST step, so this is the
    # pre-training residency; the training peak is measured_peak_bytes below.
    ledger["measured_allocated_after_wrap_bytes"] = float(torch.cuda.memory_allocated(device))

    from ddp_core import build_global_batch

    set_full_state_dict_type(fsdp_model)
    torch.cuda.reset_peak_memory_stats(device)
    steps = []
    state_after_step1 = None
    saved_state = None
    saved_optim_state = None
    for step_index in range(args.steps):
        host_batch, accounting = build_global_batch(samples, plan[step_index], rank, world_size, args.pad_token_id)
        metrics = fsdp_train_step(fsdp_model, optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
        weighted_global_loss = all_reduce_scalars(metrics["local_loss"], world_size, device)
        steps.append({
            "step": step_index,
            "global_loss": weighted_global_loss,
            "gradient_norm": metrics["gradient_norm_before_clip"],
        })
        if step_index == 1:
            # Collective on every rank; rank 0 keeps the tensors for comparison.
            state_after_step1 = dict(fsdp_model.state_dict())
        if step_index == args.steps - 2:
            # Save BEFORE the final step; the final step is the resume replay.
            saved_state, saved_optim_state = get_full_state_dicts(fsdp_model, optimizer)

    # Continuous final state (full state dict on every rank, CPU tensors).
    final_state = dict(fsdp_model.state_dict())

    # Resume: fresh FSDP model + fresh optimizer, restore, replay the final step.
    del optimizer, fsdp_model, model
    gc.collect()
    torch.cuda.empty_cache()
    _, resumed_raw = load_tokenizer_and_model(device)
    resumed_fsdp = wrap_model_fsdp(resumed_raw, device)
    resumed_optimizer = torch.optim.AdamW(
        resumed_fsdp.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False,
    )
    load_full_state_dicts(resumed_fsdp, resumed_optimizer, saved_state, saved_optim_state)
    host_batch, accounting = build_global_batch(samples, plan[args.steps - 1], rank, world_size, args.pad_token_id)
    metrics = fsdp_train_step(resumed_fsdp, resumed_optimizer, host_batch, accounting["loss_scale"], device, args.max_grad_norm, world_size, rank, collect_telemetry=False)
    resumed_loss = all_reduce_scalars(metrics["local_loss"], world_size, device)
    set_full_state_dict_type(resumed_fsdp)
    resumed_state = dict(resumed_fsdp.state_dict())
    del resumed_optimizer, resumed_fsdp, resumed_raw
    gc.collect()
    torch.cuda.empty_cache()

    step1_checks = None
    final_checks = None
    resume_checks = None
    state_dict_names_ok = None
    final_checksum = float(sum(value.double().sum() for value in final_state.values()))
    if rank == 0:
        step1_checks = scale_rel_report(
            state_after_step1, torch.load(args.baseline_state_path(1), map_location="cpu", weights_only=False),
        )
        baseline_final = torch.load(args.baseline_state_path(args.steps - 1), map_location="cpu", weights_only=False)
        final_checks = scale_rel_report(final_state, baseline_final)
        state_dict_names_ok = set(final_state) - {"lm_head.weight"} == set(baseline_final)
        del baseline_final
        resume_diffs = [
            float((resumed_state[name] - final_state[name]).abs().max())
            for name in final_state
        ]
        resume_checks = {
            "max_absolute_error": max(resume_diffs),
            "bitwise_identical": all(diff == 0.0 for diff in resume_diffs),
        }
    del state_after_step1, final_state, resumed_state, saved_state, saved_optim_state
    gc.collect()

    # FULL_STATE_DICT with rank0_only=False gives every rank the same full
    # state: the gathered checksums must be identical across ranks.
    checksums = [None] * world_size
    dist.all_gather_object(checksums, final_checksum)
    return {
        "steps": steps,
        "resume_loss": resumed_loss,
        "continuous_next_step_loss": steps[-1]["global_loss"],
        "step1_param_checks": step1_checks,
        "final_param_checks": final_checks,
        "resume_param_checks": resume_checks,
        "state_dict_names_ok": state_dict_names_ok,
        "final_state_checksums_identical_across_ranks": len(set(checksums)) == 1,
        "memory_ledger": ledger,
        "measured_peak_bytes": float(torch.cuda.max_memory_allocated(device)),
    }


def _worker(rank: int, world_size: int, port: int, args_ns: dict, samples_obj=None, plan_obj=None) -> None:
    if "RANK" not in os.environ:
        # mp.spawn path: publish the env:// endpoints for this single node.
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    # Under torchrun the environment already carries RANK/WORLD_SIZE/LOCAL_RANK
    # and the rendezvous endpoint; overriding them here would break multi-node.
    args = SimpleNamespace(**args_ns)
    args.baseline_state_path = baseline_state_path
    if samples_obj is not None:
        samples, plan = samples_obj, plan_obj
    else:
        bundle = torch.load(args.samples_path, map_location="cpu", weights_only=False)
        samples, plan = bundle["samples"], bundle["plan"]
    seed_everything(args.seed)
    rank_id, _, device = setup_distributed(rank, world_size)
    assert rank_id == rank

    baseline = _run_baseline_phase(args, samples, plan, device, rank, world_size)
    variant = _run_fsdp_phase(args, samples, plan, device, rank, world_size)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args.part_path), {
            "world_size": world_size,
            "baseline": baseline,
            "variant": variant,
        })


def baseline_state_path(step_index: int) -> Path:
    """Where Phase B stashes the full baseline parameters for Phase C."""

    return Path(f"correctness-zero2-state-step{step_index}.pt")


def assemble_report(args: SimpleNamespace, report: dict, reference: dict) -> dict:
    """All bars, evaluated once, on rank 0."""

    baseline = report["baseline"]
    variant = report["variant"]
    world_size = report["world_size"]

    trajectory_errors_baseline = [
        abs(step["global_loss"] - ref["loss"]) / abs(ref["loss"])
        for ref, step in zip(reference["steps"], baseline["steps"])
    ]
    baseline_losses = {step["step"]: step["global_loss"] for step in baseline["steps"]}
    variant_losses = {step["step"]: step["global_loss"] for step in variant["steps"]}
    trajectory_errors_variant = [
        abs(variant_losses[step] - baseline_losses[step]) / abs(baseline_losses[step])
        for step in sorted(baseline_losses)
    ]
    grad_report = baseline["step0_grad_report"]
    baseline_grad_norm = baseline["steps"][0]["gradient_norm"]
    fsdp_grad_norm = variant["steps"][0]["gradient_norm"]
    step0_loss_rel = abs(variant_losses[0] - baseline_losses[0]) / abs(baseline_losses[0])
    grad_norm_rel = abs(fsdp_grad_norm - baseline_grad_norm) / abs(baseline_grad_norm)

    resume_loss_rel = abs(variant["resume_loss"] - variant["continuous_next_step_loss"]) / abs(variant["continuous_next_step_loss"])
    resume_checks = variant["resume_param_checks"]

    ledger = baseline["memory_ledger"]
    checks = {
        "baseline_grad_cosine_ok": grad_report["grad_cosine_similarity"] > GRAD_COSINE_BAR,
        "baseline_grad_rel_ok": grad_report["grad_scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "baseline_trajectory_ok": max(trajectory_errors_baseline) < TRAJECTORY_BAR,
        "baseline_cross_rank_identical": baseline["cross_rank_fingerprints_identical"],
        "baseline_shard_bitwise_ok": baseline["shard_bitwise_ok"],
        "baseline_grad_shard_ok": abs(ledger["grad_shard_fraction"] - 1.0 / world_size) < 0.01,
        "baseline_state_shard_ok": abs(ledger["state_shard_fraction"] - 1.0 / world_size) < 0.01,
        "fsdp_step0_loss_ok": step0_loss_rel < STEP0_LOSS_BAR,
        "fsdp_grad_norm_ok": grad_norm_rel < GRAD_NORM_REL_BAR,
        "fsdp_trajectory_ok": max(trajectory_errors_variant) < TRAJECTORY_BAR,
        "fsdp_step1_params_ok": variant["step1_param_checks"]["max_scale_relative_error"] < PARAM_SCALE_REL_BAR,
        "fsdp_final_params_ok": variant["final_param_checks"]["max_scale_relative_error"] < PARAM_SCALE_REL_BAR,
        "fsdp_state_dict_names_ok": variant["state_dict_names_ok"],
        "fsdp_cross_rank_consistent": variant["final_state_checksums_identical_across_ranks"],
        "resume_loss_ok": resume_loss_rel < STEP0_LOSS_BAR,
        "resume_params_ok": resume_checks["max_absolute_error"] < RESUME_PARAM_ABS_BAR,
    }
    return {
        "world_size": world_size,
        "steps": args.steps,
        "nnodes": int(os.environ.get("NNODES", "1")),
        "node_rank": int(os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "step0_grad_report": grad_report,
        "baseline_global_losses": baseline_losses,
        "fsdp_global_losses": variant_losses,
        "baseline_trajectory_max_loss_rel_error": max(trajectory_errors_baseline),
        "fsdp_vs_baseline_max_loss_rel_error": max(trajectory_errors_variant),
        "step0_global_loss_rel_error": step0_loss_rel,
        "baseline_step0_grad_norm": baseline_grad_norm,
        "fsdp_step0_grad_norm": fsdp_grad_norm,
        "fsdp_step0_grad_norm_rel_error": grad_norm_rel,
        "fsdp_step1_params": variant["step1_param_checks"],
        "fsdp_final_params": variant["final_param_checks"],
        "resume": {
            "loss_rel_error": resume_loss_rel,
            "continuous_next_step_loss": variant["continuous_next_step_loss"],
            "resumed_step_loss": variant["resume_loss"],
            "params": resume_checks,
        },
        "baseline_memory_ledger": ledger,
        "fsdp_memory_ledger": variant["memory_ledger"],
        "fsdp_measured_peak_bytes": variant["measured_peak_bytes"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter34_train.jsonl"))
    parser.add_argument("--world-size", type=int, default=2, choices=(2, 4))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    args.stash_path = Path("correctness-reference-grads.pt")
    args.part_path = Path("correctness-zero3-part.json")
    args.samples_path = Path("correctness-samples.pt")
    args.baseline_state_path = baseline_state_path
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    args_ns = {
        "seed": args.seed, "steps": args.steps, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "pad_token_id": args.pad_token_id, "stash_path": str(args.stash_path),
        "part_path": str(args.part_path), "samples_path": str(args.samples_path),
    }

    if "RANK" in os.environ:
        # torchrun path: rank 0 alone builds the Phase A reference on its local
        # cuda:0 (a second full model on the other ranks' cuda:0 would not
        # fit), then every rank runs both phase bodies directly.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != args.world_size:
            raise SystemExit(f"--world-size {args.world_size} does not match torchrun WORLD_SIZE {world_size}")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        from sft_core import load_jsonl_records

        records = load_jsonl_records(args.data)
        try:
            samples = encode_samples(tokenizer, records, args.max_length)
            filtered_no_supervision = 0
        except ValueError:
            # 12GB calibration: at reduced max_length a few corpus samples lose
            # all supervised targets to truncation. Drop them on every rank so
            # the comparison stays on identical usable samples.
            kept = []
            filtered_no_supervision = 0
            for record in records:
                try:
                    kept.extend(encode_samples(tokenizer, [record], args.max_length))
                except ValueError:
                    filtered_no_supervision += 1
            samples = kept
        # Batch plans need BATCH_SIZE members per step.
        usable = len(samples) - len(samples) % BATCH_SIZE
        dropped_for_divisibility = len(samples) - usable
        samples = samples[:usable]
        if rank == 0:
            print(
                f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
                f"dropped_for_divisibility={dropped_for_divisibility}"
            )
        plan = make_batch_plan(len(samples), BATCH_SIZE, args.seed)
        reference = None
        if rank == 0:
            reference = run_reference(args, samples, plan, torch.device("cuda:0"))
            gc.collect()
            torch.cuda.empty_cache()
        _worker(rank, world_size, 0, args_ns, samples_obj=samples, plan_obj=plan)
        if rank == 0:
            report = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
            payload = assemble_report(args, report, reference)
            write_json(args.output, payload)
            Path(args.part_path).unlink()
            args.stash_path.unlink()
            for step_index in (1, args.steps - 1):
                args.baseline_state_path(step_index).unlink()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if not payload["pass"]:
                raise SystemExit("correctness checks failed")
        return

    # mp.spawn path: single node.
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    from sft_core import load_jsonl_records

    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them on BOTH paths so the
        # comparison stays on identical samples.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    # Batch plans need BATCH_SIZE members per step.
    usable = len(samples) - len(samples) % BATCH_SIZE
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    plan = make_batch_plan(len(samples), BATCH_SIZE, args.seed)

    reference = run_reference(args, samples, plan, torch.device("cuda:0"))
    gc.collect()
    torch.cuda.empty_cache()

    torch.save({"samples": samples, "plan": plan}, args.samples_path)
    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns), nprocs=args.world_size, join=True)
    args.samples_path.unlink()

    report = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
    payload = assemble_report(args, report, reference)
    write_json(args.output, payload)
    Path(args.part_path).unlink()
    args.stash_path.unlink()
    for step_index in (1, args.steps - 1):
        args.baseline_state_path(step_index).unlink()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit("correctness checks failed")


if __name__ == "__main__":
    main()
