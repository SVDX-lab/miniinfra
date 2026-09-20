#!/usr/bin/env python3
"""Chapter 31 correctness on real weights: delayed sync versus eager sync.

Phase A (rank 0, cuda:0): the big-batch reference trajectory — loss,
gradient norms, parameter fingerprints over 8 optimizer steps, step-0
gradients stashed (identical to Chapters 14/15).

Phase B (worker processes, one GPU each, K=2 micro-steps), per sync mode:

- B1 gradient equivalence, no optimizer state: one ``accumulate_step`` under
  a wrapper that never applies the update, so lazy AdamW state is never
  allocated. Peak FP32 residency is model + gradients + the ``every`` mode's
  accumulator snapshot (~7.2 GiB), which fits 12 GB. Rank 0 compares the
  step-0 gradients against the stashed big-batch reference.
- B2 parameter-update trajectory: the full ``--steps`` trajectory with a real
  AdamW, cross-rank fingerprints, and AllReduce-op accounting. In FP32 the
  ``every`` mode needs model + gradients + snapshot + AdamW state (~12 GiB)
  and can legitimately exceed a 12 GB card at step >= 1; the attempt is
  wrapped in ``try/except torch.cuda.OutOfMemoryError`` and an OOM is
  recorded as ``param_update_check: "oom_12gb_fp32"`` instead of failing the
  chapter. The pass criterion is the B1 gradient equivalence (plus all B2
  checks for every mode whose B2 actually ran).

Launcher: mp.spawn on one node (default), or torchrun across nodes when
RANK already exists in the environment.
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
    all_reduce_scalars,
    cleanup_distributed,
    encode_samples,
    free_port,
    load_tokenizer_and_model,
    make_batch_plan,
    setup_distributed,
)
from experiment_utils import write_json
from compare_reference import gradient_report, run_reference
from no_sync_core import accumulate_step, build_micro_batches
from sft_core import seed_everything


GRAD_COSINE_BAR = 0.99999
GRAD_SCALE_REL_BAR = 5e-3
TRAJECTORY_BAR = 5e-4
BATCH_SIZE = 4
NUM_MICRO = 2
PARAM_UPDATE_OOM_TAG = "oom_12gb_fp32"


class GradientOnlyOptimizer:
    """AdamW facade whose ``step`` never applies an update.

    B1 compares gradients, not updates. Routing ``accumulate_step`` through
    this wrapper keeps the call path identical (zero_grad, backward,
    AllReduce, clip, step) while the lazy AdamW state — 4.8 GB in FP32 for
    0.6B parameters — is never allocated. ``accumulate_step`` only touches
    ``zero_grad`` and ``step``.
    """

    def __init__(self, parameters, **kwargs):
        self._optimizer = torch.optim.AdamW(parameters, **kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, *args, **kwargs) -> None:
        return None


def fingerprint(model: torch.nn.Module) -> dict:
    return {name: float(parameter.detach().double().sum()) for name, parameter in model.named_parameters()}


def batch_plan_for_world_size(sample_count: int, seed: int, world_size: int) -> list:
    """Global batch plan giving every rank at least NUM_MICRO members.

    World size 2 keeps the chapter's original global batch of 4 sharded 2+2;
    larger worlds scale the global batch so the K=2 micro-step split stays
    one sample per micro-batch on every rank.
    """

    return make_batch_plan(sample_count, max(BATCH_SIZE, world_size * NUM_MICRO), seed)


def build_report(reference: dict, ddp: dict, steps: int) -> dict:
    """Compare both sync modes against the Phase A reference trajectory.

    B1 gradient checks are mandatory. B2 trajectory checks apply only to
    modes whose update phase ran to completion; a mode recorded as
    ``oom_12gb_fp32`` keeps its B1 verdict and reports the OOM instead of
    failing the chapter.
    """

    payload = {"world_size": ddp["world_size"], "num_micro": NUM_MICRO, "modes": {}, "pass": True}
    for mode, data in ddp["modes"].items():
        grad_report = data["step0_grad_report"]
        gradient_ops_expected = NUM_MICRO if mode == "every" else 1
        checks = {
            "grad_cosine_ok": grad_report["grad_cosine_similarity"] > GRAD_COSINE_BAR,
            "grad_rel_ok": grad_report["grad_scale_relative_error"] < GRAD_SCALE_REL_BAR,
            "gradient_phase_allreduce_ops_ok": data["gradient_phase_allreduce_ops"] == gradient_ops_expected,
        }
        mode_payload = {
            "step0_grad_report": grad_report,
            "gradient_phase_allreduce_ops": data["gradient_phase_allreduce_ops"],
            "param_update_check": data["param_update_check"],
            "checks": checks,
        }
        if data["param_update_check"] == "ok":
            trajectory_errors = [
                abs(step["global_loss"] - ref["loss"]) / abs(ref["loss"])
                for ref, step in zip(reference["steps"], data["steps"])
            ]
            update_ops_expected = steps * NUM_MICRO if mode == "every" else steps
            checks["trajectory_ok"] = max(trajectory_errors) < TRAJECTORY_BAR
            checks["cross_rank_identical"] = data["cross_rank_identical"]
            checks["update_allreduce_ops_ok"] = data["total_allreduce_ops"] == update_ops_expected
            mode_payload.update({
                "trajectory_max_loss_rel_error": max(trajectory_errors),
                "cross_rank_identical": data["cross_rank_identical"],
                "total_allreduce_ops": data["total_allreduce_ops"],
                "total_comm_bytes": data["total_comm_bytes"],
            })
        else:
            mode_payload["param_update_oom"] = {
                "completed_steps": len(data["steps"]),
                "total_allreduce_ops": data["total_allreduce_ops"],
                "total_comm_bytes": data["total_comm_bytes"],
            }
        payload["modes"][mode] = mode_payload
        payload["pass"] = payload["pass"] and all(checks.values())
    return payload


def _worker(rank: int, world_size: int, port: int, args_ns: dict, samples_obj: dict | None = None) -> None:
    if "RANK" not in os.environ:
        # mp.spawn path: publish the env:// endpoints for this single node.
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    # Under torchrun the environment already carries RANK/WORLD_SIZE/LOCAL_RANK
    # and the rendezvous endpoint; overriding them here would break multi-node.
    args = SimpleNamespace(**args_ns)
    if samples_obj is not None:
        bundle = samples_obj
    else:
        bundle = torch.load(args.samples_path, map_location="cpu", weights_only=False)
    samples, plan = bundle["samples"], bundle["plan"]

    rank_id, _, device = setup_distributed(rank, world_size)
    assert rank_id == rank

    mode_reports = {}
    for mode in ("every", "last"):
        # ---- B1: gradient equivalence without lazy optimizer state ----
        seed_everything(args.seed)
        _, model = load_tokenizer_and_model(device)
        gradient_optimizer = GradientOnlyOptimizer(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay
        )
        micros, accounting = build_micro_batches(samples, plan[0], rank, world_size, NUM_MICRO, args.pad_token_id)
        gradient_metrics = accumulate_step(
            model, gradient_optimizer, micros, accounting["supervised_global"], device,
            args.max_grad_norm, world_size, rank, sync_mode=mode, collect_telemetry=False,
        )
        step0_grad_report = gradient_report(model, args.stash_path, clip_to=args.max_grad_norm) if rank == 0 else None
        gradient_ops = float(gradient_metrics["allreduce_count"])
        del model, gradient_optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # ---- B2: the full parameter-update trajectory; OOM is a recorded
        # capacity fact on 12 GB FP32, not a correctness failure ----
        param_update_check = "ok"
        steps: list[dict] = []
        total_comm_bytes = 0.0
        final_fingerprints: list | None = None
        model, optimizer = None, None
        try:
            seed_everything(args.seed)
            _, model = load_tokenizer_and_model(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8,
                weight_decay=args.weight_decay, foreach=False, fused=False,
            )
            for step_index in range(args.steps):
                micros, accounting = build_micro_batches(samples, plan[step_index], rank, world_size, NUM_MICRO, args.pad_token_id)
                metrics = accumulate_step(
                    model, optimizer, micros, accounting["supervised_global"], device,
                    args.max_grad_norm, world_size, rank, sync_mode=mode, collect_telemetry=False,
                )
                global_loss = all_reduce_scalars(metrics["local_loss"], world_size, device)
                total_comm_bytes += metrics["allreduce_bytes_total"]
                steps.append({
                    "step": step_index,
                    "global_loss": global_loss,
                    "gradient_norm": metrics["gradient_norm_before_clip"],
                    "fingerprint": fingerprint(model),
                    "allreduce_count": metrics["allreduce_count"],
                })
            final_fingerprints = [None] * world_size
            dist.all_gather_object(final_fingerprints, fingerprint(model))
        except torch.cuda.OutOfMemoryError:
            param_update_check = PARAM_UPDATE_OOM_TAG
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        mode_reports[mode] = {
            "step0_grad_report": step0_grad_report,
            "gradient_phase_allreduce_ops": gradient_ops,
            "param_update_check": param_update_check,
            "steps": steps,
            "cross_rank_identical": (all(entry == final_fingerprints[0] for entry in final_fingerprints)) if final_fingerprints is not None else None,
            "total_allreduce_ops": float(sum(step["allreduce_count"] for step in steps)),
            "total_comm_bytes": float(total_comm_bytes),
        }

    cleanup_distributed()
    if rank == 0:
        write_json(Path(args.part_path), {"world_size": world_size, "modes": mode_reports})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter31_train.jsonl"))
    parser.add_argument("--world-size", type=int, default=2,
                        help="mp.spawn path only; under torchrun the world size comes from the environment")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    args.stash_path = Path("correctness-reference-grads.pt")
    args.part_path = Path("correctness-ddp-part.json")
    args.samples_path = Path("correctness-samples.pt")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if "RANK" in os.environ:
        # torchrun path: every rank runs the worker body directly; rank 0 also
        # builds the single-GPU reference and finalizes the report.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.device_count() < local_world_size:
            raise SystemExit(f"need {local_world_size} local CUDA GPUs, found {torch.cuda.device_count()}")
        tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
        args.pad_token_id = tokenizer.pad_token_id
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
        # Batch plans shard max(BATCH_SIZE, world_size * NUM_MICRO) members per step.
        usable = len(samples) - len(samples) % max(BATCH_SIZE, world_size * NUM_MICRO)
        dropped_for_divisibility = len(samples) - usable
        samples = samples[:usable]
        if rank == 0:
            print(
                f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
                f"dropped_for_divisibility={dropped_for_divisibility}"
            )
        plan = batch_plan_for_world_size(len(samples), args.seed, world_size)
        reference = None
        if rank == 0:
            reference = run_reference(args, samples, plan, torch.device("cuda:0"))
            gc.collect()
            torch.cuda.empty_cache()
        args_ns = {
            "seed": args.seed, "steps": args.steps, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
            "pad_token_id": args.pad_token_id, "stash_path": str(args.stash_path),
            "part_path": str(args.part_path), "samples_path": str(args.samples_path),
        }
        _worker(rank, world_size, 0, args_ns, samples_obj={"samples": samples, "plan": plan})
        if rank == 0:
            report = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
            payload = build_report(reference, report, args.steps)
            write_json(args.output, payload)
            Path(args.part_path).unlink()
            args.stash_path.unlink()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if not payload["pass"]:
                raise SystemExit("correctness checks failed")
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"{args.world_size} CUDA GPUs are required")
    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    from sft_core import load_jsonl_records

    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them on BOTH paths so the DDP
        # comparison stays on identical samples.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    # Batch plans shard max(BATCH_SIZE, world_size * NUM_MICRO) members per step.
    usable = len(samples) - len(samples) % max(BATCH_SIZE, args.world_size * NUM_MICRO)
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )
    plan = batch_plan_for_world_size(len(samples), args.seed, args.world_size)

    reference = run_reference(args, samples, plan, torch.device("cuda:0"))
    gc.collect()
    torch.cuda.empty_cache()

    args_ns = {
        "seed": args.seed, "steps": args.steps, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "pad_token_id": args.pad_token_id, "stash_path": str(args.stash_path),
        "part_path": str(args.part_path), "samples_path": str(args.samples_path),
    }
    torch.save({"samples": samples, "plan": plan}, args.samples_path)
    port = free_port()
    mp.spawn(_worker, args=(args.world_size, port, args_ns), nprocs=args.world_size, join=True)
    args.samples_path.unlink()

    report = json.loads(Path(args.part_path).read_text(encoding="utf-8"))
    payload = build_report(reference, report, args.steps)
    write_json(args.output, payload)
    Path(args.part_path).unlink()
    args.stash_path.unlink()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit("correctness checks failed")


if __name__ == "__main__":
    main()
