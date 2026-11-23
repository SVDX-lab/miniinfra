#!/usr/bin/env python3
"""Chapter 41 correctness on real Qwen3-0.6B weights: interleaved vs single GPU.

Phase A (one process, cuda:0): the replicated FP32 reference — the FULL model
consumes the SAME deterministic micro-batches (same plan, same loss scales),
accumulating gradients micro by micro; step 0 stashes every named gradient,
the per-virtual-stage-boundary input gradients (hooks at each virtual stage's
first layer), the micro-summed loss, and — after exact-norm clipping and
AdamW — the final parameters.  Boundaries follow the INTERLEAVED split
(``total_stages = world_size * vpp``).

Phase B (one GPU per physical rank, ``vpp`` virtual stages each): the SAME
total split first runs ``interleaved_1f1b_step``.  Comparisons per rank: the
micro-summed loss; every chunk weight gradient against its reference slice
(global layer names — chunks hold NON-contiguous layers); every received
stage-boundary input gradient (per chunk, per micro) against the hooked
reference gradients; the tied Embedding/LM Head gradients after the P2P
exchange (both must equal the tied reference gradient); then the
non-interleaved 1F1B baseline (``vpp=1``, ``total_stages = world_size``) must
reproduce the reference gradients on its own split — transitively equal to
the interleaved result, since both consume identical micro-batches and only
placement/scheduling differ — and the post-step interleaved parameters must
match the reference update.

``--num-micro`` must be divisible by the physical pipeline size (the
interleaved schedule processes micro-batches in groups of ``P``).  Launch
with ``python compare_correctness.py`` (mp.spawn, single node) or under
``torchrun`` (``--world-size`` must match ``WORLD_SIZE``).  Bars are tight:
the reference consumes identical micro shapes, so differences come from FP32
reduction order only.
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

from experiment_utils import environment_record, write_json
from interleaved import (
    StageTimeline,
    exchange_tied_gradients,
    interleaved_1f1b_step,
    load_pipeline_model,
    onef1b_step,
    pp_train_step,
    split_virtual_stages,
)
from pp_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    build_micro_batches,
    cleanup_distributed,
    encode_samples,
    free_port,
    load_tokenizer,
    make_batch_plan,
    micro_dicts,
    setup_distributed,
)
from qwen3_model import Qwen3Config, load_handwritten_model, resolve_model_directory
from sft_core import seed_everything


LOSS_REL_BAR = 1e-5
GRAD_SCALE_REL_BAR = 5e-3
TRAJECTORY_BAR = 5e-4
PARAM_SCALE_REL_BAR = 1e-3
SCHEDULE_PARITY_ATOL = 1e-6


def tensor_report(actual: torch.Tensor, reference: torch.Tensor) -> dict:
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


def build_correctness_micros(args: SimpleNamespace, pad_token_id: int):
    """The deterministic micro-batch plan shared by reference and stages."""

    from sft_core import load_jsonl_records
    from pp_runtime import encode_samples as encode

    if args.num_micro % args.world_size != 0:
        raise SystemExit(
            f"--num-micro {args.num_micro} must be divisible by the physical pipeline size {args.world_size} "
            "(the interleaved schedule processes micro-batches in groups of the pipeline size)"
        )
    tokenizer = load_tokenizer(torch.device("cpu"))
    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them so reference and pipeline
        # stages stay on identical usable samples.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    # The correctness plan needs num_micro * samples_per_micro members.
    divisor = args.num_micro * args.samples_per_micro
    usable = len(samples) - len(samples) % divisor
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    plan = make_batch_plan(len(samples), args.num_micro * args.samples_per_micro, args.seed)
    micro_batches = build_micro_batches(samples, plan[0], args.num_micro, pad_token_id)
    supervised_global = float(sum(batch.supervised_tokens for batch in micro_batches))
    return micro_dicts(micro_batches, supervised_global)


def run_reference(args: SimpleNamespace, micros: list[dict], device) -> dict:
    """Phase A: full-model reference trajectory with micro accumulation."""

    from mini_batch import batch_loss
    from qwen3_model import load_handwritten_model as load_full

    config = Qwen3Config.from_model_directory(resolve_model_directory(MODEL_ID, MODEL_REVISION))
    total_stages = args.world_size * args.vpp
    boundary_layers = [begin for begin, _ in split_virtual_stages(config.num_hidden_layers, total_stages)[0][1:]]

    model = load_full(resolve_model_directory(MODEL_ID, MODEL_REVISION), device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)

    captured: list[dict] = []
    handles = []
    for layer_index in boundary_layers:
        def hook(module, grad_input, grad_output, _layer=layer_index):
            captured.append({"layer": _layer, "grad": grad_input[0].detach().to("cpu")})
        handles.append(model.model.layers[layer_index].register_full_backward_hook(hook))

    def micro_pass(accumulate: bool, capture_grads: bool = False) -> tuple[float, dict]:
        if accumulate:
            optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in micros:
            # micros arrive as host tensors; the reference model lives on device
            input_ids = micro["input_ids"].to(device)
            attention_mask = micro["attention_mask"].to(device)
            labels = micro["labels"].to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            loss = batch_loss(logits, labels, "token_mean") * micro["scale"]
            if accumulate:
                loss.backward()
            total += float(loss.detach())
        grads = {}
        if capture_grads:
            grads = {name: p.grad.detach().to("cpu") for name, p in model.named_parameters()}
        return total, grads

    step0_loss, reference_grads = micro_pass(accumulate=True, capture_grads=True)
    for handle in handles:
        handle.remove()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    steps = [{"step": 0, "loss": step0_loss, "gradient_norm": float(grad_norm.item())}]
    for step_index in range(1, args.steps):
        loss, _ = micro_pass(accumulate=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        steps.append({"step": step_index, "loss": loss, "gradient_norm": float(grad_norm.item())})

    final_params = {name: parameter.detach().to("cpu").clone() for name, parameter in model.named_parameters()}
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "steps": steps,
        "step0_loss": step0_loss,
        "reference_grads": reference_grads,
        "boundary_captures": captured,
        "boundary_layers": boundary_layers,
        "final_params": final_params,
        "split": split_virtual_stages(config.num_hidden_layers, total_stages)[1],
        "total_stages": total_stages,
    }


def compare_named_grads(parameter, name: str, reference_grads: dict, reference_name: str,
                        device, reports: list, worst_box: dict) -> None:
    if parameter.grad is None:
        return
    report = tensor_report(parameter.grad, reference_grads[reference_name].to(device))
    report["parameter"] = name
    reports.append(report)
    if worst_box["value"] is None or report["scale_relative_error"] > worst_box["value"]["scale_relative_error"]:
        worst_box["value"] = report


def _worker(rank: int, world_size: int, port: int, args_ns: dict, micros_meta) -> None:
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
    micros = [
        {
            **micro,
            "input_ids": micro["input_ids"].to(device),
            "attention_mask": micro["attention_mask"].to(device),
            "labels": micro["labels"].to(device),
        }
        for micro in micros_meta
    ]

    reference = torch.load(args.stash_path, map_location="cpu", weights_only=False)

    def reference_name_for(chunk, name: str) -> str:
        begin, _ = chunk.layer_range
        if name.startswith("layers."):
            local_layer, rest = name.split(".", 1)[1].split(".", 1)
            return f"model.layers.{begin + int(local_layer)}.{rest}"
        if name in ("embed_tokens.weight", "lm_head.weight"):
            return "model.embed_tokens.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        raise ValueError(f"unexpected chunk parameter {name}")

    def snapshot_key_for(chunk, name: str) -> str:
        """Global key for cross-schedule comparison; the tied pair keeps two
        distinct keys so both partial gradients survive the snapshot."""

        if name == "lm_head.weight":
            return "model.lm_head.weight"
        return reference_name_for(chunk, name)

    # ---- interleaved step-0 capture ----
    chunks = load_pipeline_model(MODEL_ID, device, rank, world_size, args.vpp, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        [parameter for chunk in chunks for parameter in chunk.parameters()],
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay,
        foreach=False, fused=False,
    )
    optimizer.zero_grad(set_to_none=True)
    timeline = StageTimeline(device)
    interleaved_result = interleaved_1f1b_step(chunks, micros, device, timeline)
    interleaved_events = timeline.end_step()
    exchange_tied_gradients(chunks, device)
    loss_shards = [None] * world_size
    dist.all_gather_object(loss_shards, interleaved_result["local_loss_shard"])
    step0_loss = float(sum(loss_shards))

    grads_report = []
    worst_box = {"value": None}
    for chunk in chunks:
        for name, parameter in chunk.named_parameters():
            compare_named_grads(parameter, f"chunk{chunk.local_chunk}.{name}",
                                reference["reference_grads"], reference_name_for(chunk, name),
                                device, grads_report, worst_box)
    worst = worst_box["value"]

    boundary_reports = []
    for chunk in chunks:
        if chunk.is_first:
            continue
        begin = chunk.layer_range[0]
        captures = [entry for entry in reference["boundary_captures"] if entry["layer"] == begin]
        assert len(captures) == len(micros), (len(captures), len(micros))
        for micro_index in range(len(micros)):
            grad = interleaved_result["input_grads"][f"chunk{chunk.local_chunk}_micro{micro_index}"]
            report = tensor_report(grad, captures[micro_index]["grad"].to(device))
            report["chunk"] = chunk.local_chunk
            report["micro"] = micro_index
            boundary_reports.append(report)
    boundary_worst = max(boundary_reports, key=lambda r: r["scale_relative_error"]) if boundary_reports else None

    # ---- non-interleaved 1F1B baseline (vpp=1) reproduces the reference ----
    # Free the interleaved optimizer first; the interleaved step-0 gradients
    # move to CPU under GLOBAL parameter names so the two schedules can be
    # compared without holding two optimizer states on one 12GB card (their
    # chunk-local names address different layers — only global names agree).
    interleaved_grads_global = {
        snapshot_key_for(chunk, name): parameter.grad.detach().to("cpu").clone()
        for chunk in chunks for name, parameter in chunk.named_parameters()
        if parameter.grad is not None
    }
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    # The two arms slice the model differently (vpp=2 vs vpp=1), so every
    # rank needs the FULL global-name snapshot: one object gather, CPU side.
    gathered_grads = [None] * world_size
    dist.all_gather_object(gathered_grads, interleaved_grads_global)
    merged_grads: dict = {}
    for partial in gathered_grads:
        merged_grads.update(partial)
    del gathered_grads
    interleaved_grads_global = merged_grads

    baseline_chunks = load_pipeline_model(MODEL_ID, device, rank, world_size, 1, dtype=torch.float32)
    baseline_timeline = StageTimeline(device)
    baseline_result = onef1b_step(baseline_chunks, micros, device, baseline_timeline)
    baseline_events = baseline_timeline.end_step()
    exchange_tied_gradients(baseline_chunks, device)

    baseline_shards = [None] * world_size
    dist.all_gather_object(baseline_shards, baseline_result["local_loss_shard"])
    baseline_loss = float(sum(baseline_shards))

    baseline_vs_reference_worst = None
    baseline_vs_interleaved_worst = None
    for chunk in baseline_chunks:
        for name, parameter in chunk.named_parameters():
            if parameter.grad is None:
                continue
            reference_report = tensor_report(parameter.grad, reference["reference_grads"][reference_name_for(chunk, name)].to(device))
            if baseline_vs_reference_worst is None or reference_report["scale_relative_error"] > baseline_vs_reference_worst["scale_relative_error"]:
                baseline_vs_reference_worst = reference_report
            interleaved_grad = interleaved_grads_global[snapshot_key_for(chunk, name)].to(device)
            difference = (parameter.grad - interleaved_grad).abs().max().item()
            scale = interleaved_grad.abs().max().item()
            relative = difference / max(scale, 1e-8)
            if baseline_vs_interleaved_worst is None or relative > baseline_vs_interleaved_worst["relative_error"]:
                baseline_vs_interleaved_worst = {"parameter": snapshot_key_for(chunk, name), "relative_error": relative}
    del baseline_chunks
    gc.collect()
    torch.cuda.empty_cache()

    # ---- trajectory through the real step function (interleaved) ----
    optimizer = torch.optim.AdamW(
        [parameter for chunk in chunks for parameter in chunk.parameters()],
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay,
        foreach=False, fused=False,
    )
    steps = []
    for step_index in range(args.steps):
        metrics = pp_train_step(chunks, optimizer, micros, device, "interleaved", args.max_grad_norm)
        steps.append({
            "step": step_index,
            "loss": metrics["global_loss"],
            "gradient_norm": metrics["gradient_norm_before_clip"],
            "step_time_ms": metrics["step_time_ms"],
            "bubble_ratio": metrics["bubble"]["bubble_ratio"],
            "p2p_time_s": metrics["p2p_time_s"],
            "p2p_count": metrics["p2p_count"],
            "in_flight_micros": metrics["in_flight_micros"],
        })

    # ---- final parameters vs the reference ----
    param_reports = []
    param_worst = None
    named_reference = reference["final_params"]
    for chunk in chunks:
        for name, parameter in chunk.named_parameters():
            # final-param comparison runs on the CPU: the stage still holds grads and
            # AdamW state, and the double() copies of the tied embedding need ~3.5 GiB
            report = tensor_report(parameter.detach().to("cpu"), named_reference[reference_name_for(chunk, name)])
            report["parameter"] = f"chunk{chunk.local_chunk}.{name}"
            param_reports.append(report)
            if param_worst is None or report["scale_relative_error"] > param_worst["scale_relative_error"]:
                param_worst = report

    worker_report = {
        "rank": rank,
        "chunk_layer_ranges": [f"{chunk.layer_range[0]}:{chunk.layer_range[1]}" for chunk in chunks],
        "step0_loss_rel_error": abs(step0_loss - reference["step0_loss"]) / abs(reference["step0_loss"]),
        "gradient_worst": worst,
        "boundary_worst": boundary_worst,
        "baseline_reference_worst": baseline_vs_reference_worst,
        "baseline_interleaved_parity_worst": baseline_vs_interleaved_worst,
        "baseline_step0_loss_rel_error": abs(baseline_loss - reference["step0_loss"]) / abs(reference["step0_loss"]),
        "trajectory": steps,
        "param_worst": param_worst,
        "interleaved_timeline_step0": interleaved_events,
        "onef1b_timeline_step0": baseline_events,
    }
    del chunks, optimizer, interleaved_grads_global, reference
    gc.collect()
    torch.cuda.empty_cache()

    reports = [None] * world_size
    dist.all_gather_object(reports, worker_report)
    cleanup_distributed()
    if rank == 0:
        write_json(Path(args.part_path), {"world_size": world_size, "vpp": args.vpp, "ranks": reports})


def assemble_report(args: SimpleNamespace, part: dict, reference: dict) -> dict:
    ranks = part["ranks"]
    rank0 = ranks[0]
    trajectory_errors = [
        abs(step["loss"] - ref["loss"]) / abs(ref["loss"])
        for ref, step in zip(reference["steps"], rank0["trajectory"])
    ]
    expected_ranges = []
    start = 0
    for count in reference["split"]["stage_layer_counts"]:
        expected_ranges.append(f"{start}:{start + count}")
        start += count
    checks = {
        "forward_loss_rel_ok": rank0["step0_loss_rel_error"] < LOSS_REL_BAR,
        "gradient_worst_ok": rank0["gradient_worst"]["scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "boundary_grad_ok": (rank0["boundary_worst"] or {"scale_relative_error": 0.0})["scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "baseline_reference_ok": rank0["baseline_reference_worst"]["scale_relative_error"] < GRAD_SCALE_REL_BAR,
        "baseline_interleaved_parity_ok": rank0["baseline_interleaved_parity_worst"]["relative_error"] < SCHEDULE_PARITY_ATOL,
        "baseline_loss_rel_ok": rank0["baseline_step0_loss_rel_error"] < LOSS_REL_BAR,
        "trajectory_ok": max(trajectory_errors) < TRAJECTORY_BAR,
        "final_params_ok": rank0["param_worst"]["scale_relative_error"] < PARAM_SCALE_REL_BAR,
        "chunk_ranges_ok": sorted(
            chunk_range for rank in ranks for chunk_range in rank["chunk_layer_ranges"]
        ) == sorted(expected_ranges),
    }
    return {
        "schema_note": {
            "equivalence": "Interleaved 1F1B reorders WHEN (virtual stage, micro-batch) items run and WHO owns them, not WHAT is computed; the reference consumes identical micro shapes so tolerance bars cover FP32 reduction order only",
            "bars": {
                "loss_rel": LOSS_REL_BAR,
                "grad_scale_rel": GRAD_SCALE_REL_BAR,
                "trajectory_rel": TRAJECTORY_BAR,
                "param_scale_rel": PARAM_SCALE_REL_BAR,
                "baseline_parity": SCHEDULE_PARITY_ATOL,
            },
        },
        "world_size": args.world_size,
        "pp_degree": args.world_size,
        "vpp": args.vpp,
        "total_virtual_stages": reference["total_stages"],
        "mode": "interleaved (non-interleaved onef1b checked for gradient parity)",
        "steps": args.steps,
        "num_micro": args.num_micro,
        "samples_per_micro": args.samples_per_micro,
        "max_length": args.max_length,
        "split": reference["split"],
        "nnodes": int(os.environ.get("NNODES", "1")),
        "node_rank": int(os.environ.get("NODE_RANK", "0")),
        "launcher": "torchrun" if "RANK" in os.environ else "mp.spawn",
        "per_rank": [
            {
                "rank": rank["rank"],
                "chunk_layer_ranges": rank["chunk_layer_ranges"],
                "step0_loss_rel_error": rank["step0_loss_rel_error"],
                "gradient_worst": rank["gradient_worst"],
                "boundary_worst": rank["boundary_worst"],
                "baseline_reference_worst": rank["baseline_reference_worst"],
                "baseline_interleaved_parity_worst": rank["baseline_interleaved_parity_worst"],
                "param_worst": rank["param_worst"],
            }
            for rank in ranks
        ],
        "stage0_interleaved_timeline_step0": rank0["interleaved_timeline_step0"],
        "stage0_onef1b_timeline_step0": rank0["onef1b_timeline_step0"],
        "trajectory_max_loss_rel_error": max(trajectory_errors),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _finalize(args: SimpleNamespace, args_ns: dict) -> None:
    part = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
    reference = torch.load(args.stash_path, map_location="cpu", weights_only=False)
    reference_meta = {"steps": reference["steps"], "step0_loss": reference["step0_loss"],
                      "boundary_layers": reference["boundary_layers"], "split": reference["split"],
                      "total_stages": reference["total_stages"]}
    payload = environment_record(world_size=args.world_size, dp_degree=1, pp_degree=args.world_size)
    payload.update(assemble_report(args, part, reference))
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "float32",
        "tf32": False,
        "attention_implementation": "eager",
        "data_file": str(args.data),
        "reference_meta": reference_meta,
    })
    write_json(args.output, payload)
    Path(args.part_path).unlink()
    args.stash_path.unlink()
    print(json.dumps({
        key: payload[key]
        for key in ("checks", "pass", "trajectory_max_loss_rel_error", "split", "total_virtual_stages")
    }, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit("correctness checks failed")


def _wait_for_file(path: Path) -> None:
    while not path.exists():
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter41_train.jsonl"))
    parser.add_argument("--world-size", type=int, default=2, choices=(2, 4),
                        help="the PHYSICAL pipeline size equals the world size in this chapter")
    parser.add_argument("--vpp", type=int, default=2, choices=(2, 4),
                        help="virtual stages per physical rank; total_stages = world_size * vpp")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-micro", type=int, default=4,
                        help="must be divisible by the physical pipeline size (interleaved groups)")
    parser.add_argument("--samples-per-micro", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    if args.num_micro % args.world_size != 0:
        raise SystemExit(f"--num-micro {args.num_micro} must be divisible by --world-size {args.world_size}")
    args.stash_path = Path("correctness-pp-reference.pt")
    args.part_path = Path("correctness-pp-part.json")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args_ns = {
        "seed": args.seed, "steps": args.steps, "num_micro": args.num_micro,
        "samples_per_micro": args.samples_per_micro, "max_length": args.max_length,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "data": str(args.data),
        "vpp": args.vpp,
        "world_size": args.world_size,
        "stash_path": str(args.stash_path), "part_path": str(args.part_path),
    }
    micros_meta = build_correctness_micros(args, pad_token_id=load_tokenizer(torch.device("cpu")).pad_token_id)

    def atomic_torch_save(obj, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, temporary)
        os.replace(temporary, path)

    if "RANK" in os.environ:
        # torchrun path: rank 0 alone builds the Phase A reference, then every
        # rank runs the pipeline worker body directly.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != args.world_size:
            raise SystemExit(f"--world-size {args.world_size} does not match torchrun WORLD_SIZE {world_size}")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        if rank == 0:
            stash = run_reference(args, micros_meta, torch.device("cuda:0"))
            atomic_torch_save(stash, args.stash_path)
            del stash
            gc.collect()
            torch.cuda.empty_cache()
        # No process group exists at launcher level under torchrun; the other
        # ranks wait for the atomically-renamed stash file instead.
        _wait_for_file(args.stash_path)
        _worker(rank, world_size, 0, args_ns, micros_meta)
        if rank == 0:
            _finalize(args, args_ns)
        return

    # mp.spawn path: single node.
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    stash = run_reference(args, micros_meta, torch.device("cuda:0"))
    atomic_torch_save(stash, args.stash_path)
    del stash
    gc.collect()
    torch.cuda.empty_cache()

    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns, micros_meta), nprocs=args.world_size, join=True)
    _finalize(args, args_ns)


if __name__ == "__main__":
    main()
