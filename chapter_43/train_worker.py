"""Chapter 43 training worker: DDP training with checkpoint hooks (torchrun entry).

One worker covers every role of this chapter — the clean reference run, the
save/resume runs, and the fault-injection run — selected purely by flags:

- ``--save-dir`` + ``--save-modes centralized,sharded,dcp`` + ``--save-every``:
  writes checkpoints every K steps and at the final step.  Modes are the
  chapter's ONLY variable: ``centralized`` (rank 0 writes one file),
  ``sharded`` (every rank writes a shard + manifest), ``dcp`` (official
  ``torch.distributed.checkpoint`` comparison).
- ``--resume-dir`` (+ optional ``--resume-mode``): restores the full state
  (model, optimizer, scheduler, progress, sampler, per-rank RNG) and
  continues from the recorded global step.  Works under a different world
  size than the save (resharding by replica selection; see checkpointing.py).
- ``--fault-at-step`` + ``--fault-rank``: after the step's checkpoint is
  durable, that rank exits with code 70 WITHOUT any distributed cleanup —
  a hard kill the launcher must recover from.  The step must be aligned
  with a save, otherwise the worker refuses (a checkpoint that is one step
  older than the kill is the whole point of the exercise).

The training itself is the chapter-29 DDP semantics (supervised-token loss
weighting, post-backward gradient AllReduce) over a deterministic batch
plan, plus an LR scheduler so its state is part of the checkpoint.  Per-step
records (global loss, LR, RNG fingerprint, next-batch preview) are what the
continuity scripts compare.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from assistant_loss import EncodedConversation, IGNORE_INDEX
from checkpointing import (
    capture_rng,
    detect_mode,
    load_for_resume,
    model_checksum,
    restore_rng,
    rng_fingerprint,
    save_centralized,
    save_sharded,
    sync_scheduler_lr,
    dcp_available,
    dcp_load_checkpoint,
    dcp_save_checkpoint,
)
from ddp_core import (
    build_global_batch,
    cleanup_distributed,
    ddp_train_step,
    encode_samples,
    load_tokenizer_and_model,
    setup_distributed,
)
from experiment_utils import write_json
from mini_batch import pad_samples
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from sft_core import load_jsonl_records, seed_everything


FAULT_EXIT_CODE = 70


# ---------------------------------------------------------------------------
# data: deterministic plans over the synthetic or pinned corpus
# ---------------------------------------------------------------------------


TINY_CONFIG = Qwen3Config(
    vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
)
TINY_SAMPLES = 32
TINY_SEQ = 16


def synthetic_samples(count: int = TINY_SAMPLES, seq: int = TINY_SEQ) -> list[EncodedConversation]:
    """Deterministic synthetic conversations (identical on every rank)."""

    generator = torch.Generator().manual_seed(5000)
    tokens = torch.randint(4, TINY_CONFIG.vocab_size, (count, seq), generator=generator)
    samples = []
    for row in range(count):
        ids = tokens[row].unsqueeze(0)
        labels = ids.clone()
        labels[:, :2] = IGNORE_INDEX
        samples.append(EncodedConversation(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=labels,
            assistant_loss_mask=labels != IGNORE_INDEX,
            source_line=row,
            rendered_text="",
            token_spans=(),
            loss_mode="assistant_only",
        ))
    return samples


def build_batch_plan(sample_count: int, global_batch: int, seed: int) -> list[tuple[int, ...]]:
    """The exact Chapter 22 shuffle (members sorted inside each batch)."""

    import random

    rng = random.Random(f"{seed}|1|shuffle")
    order = list(range(sample_count))
    rng.shuffle(order)
    return [tuple(sorted(order[i : i + global_batch])) for i in range(0, len(order), global_batch)]


# ---------------------------------------------------------------------------
# the training loop with checkpoint hooks
# ---------------------------------------------------------------------------


def build_optimizer_and_scheduler(model: torch.nn.Module, learning_rate: float,
                                  warmup_steps: int, total_steps: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1, foreach=False, fused=False)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / float(decay_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def preview_next_batch(plan: list[tuple[int, ...]], step: int, samples, pad_token_id: int) -> list[int]:
    """First input ids of the NEXT batch's first sample (continuity probe)."""

    if step + 1 >= len(plan):
        return []
    members = plan[step + 1]
    batch = pad_samples([samples[index] for index in sorted(members)[:1]], pad_token_id)
    return [int(value) for value in batch.input_ids[0, :4]]


def run_training(args: SimpleNamespace, rank: int, world_size: int, device: torch.device,
                 backend: str) -> dict:
    seed_everything(args.seed)
    pad_token_id = 0
    if args.tiny:
        torch.manual_seed(args.seed)
        model = Qwen3ForCausalLM(TINY_CONFIG)
        model = model.to(device)
        samples = synthetic_samples()
    else:
        tokenizer, model = load_tokenizer_and_model(device)
        pad_token_id = tokenizer.pad_token_id
        records = load_jsonl_records(args.data)
        try:
            samples = encode_samples(tokenizer, records, args.max_length)
            filtered_no_supervision = 0
        except ValueError:
            # 12GB calibration: at reduced max_length a few corpus samples lose
            # all supervised targets to truncation. Drop them so every rank
            # trains on the same usable corpus.
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
    global_batch = args.global_batch
    plan = build_batch_plan(len(samples), global_batch, args.seed)
    if len(plan) < args.steps:
        raise SystemExit(f"need {args.steps} batches, the plan has {len(plan)}")

    optimizer, scheduler = build_optimizer_and_scheduler(model, args.learning_rate,
                                                         args.warmup_steps, args.steps)
    progress = {
        "global_step": 0,
        "consumed_supervised_tokens": 0.0,
        "consumed_samples": 0,
    }
    sampler = {
        "plan_seed": args.seed,
        "global_batch_sequences": float(global_batch),
        "plan_batches": float(len(plan)),
        "cursor": 0,
    }
    config = {
        "model_id": "tiny-synthetic" if args.tiny else "Qwen/Qwen3-0.6B-Base",
        "tiny": bool(args.tiny),
        "global_batch_sequences": float(global_batch),
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "total_steps": args.steps,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "data": str(args.data),
    }
    topology = {
        "world_size": world_size,
        "dp_size": world_size,
        "tp_size": 1,
        "pp_size": 1,
        "backend": backend,
        "rank": rank,
    }

    resume_info = None
    if args.resume_dir:
        resume_dir = Path(args.resume_dir)
        mode = args.resume_mode
        if mode in (None, "auto"):
            try:
                mode = detect_mode(resume_dir)
            except FileNotFoundError:
                mode = "dcp" if (resume_dir / ".metadata").is_file() else None
            if mode is None:
                raise SystemExit(f"no checkpoint found in {resume_dir}")
        if mode == "dcp":
            dcp_load_checkpoint(resume_dir, model, optimizer)
            payload = torch.load(resume_dir / "worker-meta.pt", map_location="cpu", weights_only=False)
            progress = payload["progress"]
            sampler = payload["sampler"]
            scheduler.load_state_dict(payload["scheduler"])
            sync_scheduler_lr(optimizer, scheduler)
            restore_rng(payload["rng_per_rank"][rank % len(payload["rng_per_rank"])])
            resume_info = {"mode": "dcp", "loaded_shard": None, "resharded": world_size != payload["topology"]["world_size"]}
        else:
            payload = load_for_resume(resume_dir, model, optimizer, scheduler, device, rank, world_size, mode)
            progress = dict(payload["progress"])
            sampler = dict(payload["sampler"])
            resume_info = {
                "mode": "sharded" if payload.get("loaded_shard") else "centralized",
                "loaded_shard": payload.get("loaded_shard"),
                "resharded": bool(payload.get("resharded")),
                "saved_world_size": payload.get("saved_world_size"),
            }

    save_modes = [mode for mode in args.save_modes.split(",") if mode] if args.save_modes else []
    unknown = set(save_modes) - {"centralized", "sharded", "dcp"}
    if unknown:
        raise SystemExit(f"unknown save modes {sorted(unknown)}")
    if "dcp" in save_modes and not dcp_available():
        raise SystemExit("torch.distributed.checkpoint is not available in this torch build")

    def save_checkpoints(step_done: int) -> list[dict]:
        metrics = []
        if not args.save_dir:
            return metrics
        target = Path(args.save_dir) / f"step-{step_done:05d}"
        common = (model, optimizer, scheduler, progress, sampler, config, topology)
        if "centralized" in save_modes:
            entry = save_centralized(target / "centralized", *common, rank=rank, world_size=world_size)
            entry["step"] = step_done
            metrics.append(entry)
        if "sharded" in save_modes:
            entry = save_sharded(target / "sharded", *common, rank=rank, world_size=world_size)
            entry["step"] = step_done
            metrics.append(entry)
        if "dcp" in save_modes:
            entry = dcp_save_checkpoint(target / "dcp", model, optimizer)
            entry["step"] = step_done
            # DCP stores only model/optimizer tensors; the loop state (progress,
            # sampler, scheduler, per-rank RNG) travels in a worker-meta sidecar.
            gathered_rng = [None] * world_size
            dist.all_gather_object(gathered_rng, capture_rng())
            if rank == 0:
                payload = {
                    "progress": dict(progress),
                    "sampler": dict(sampler),
                    "scheduler": scheduler.state_dict(),
                    "rng_per_rank": gathered_rng,
                    "topology": dict(topology),
                }
                path = target / "dcp" / "worker-meta.pt"
                temporary = path.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, path)
            metrics.append(entry)
        return metrics

    step_records: list[dict] = []
    save_records: list[dict] = []
    for step in range(progress["global_step"], args.steps):
        members = plan[step]
        batch, accounting = build_global_batch(samples, members, rank, world_size, pad_token_id)
        metrics = ddp_train_step(model, optimizer, batch, accounting["loss_scale"], device,
                                 args.max_grad_norm, world_size, rank, collect_telemetry=False)
        scheduler.step()
        progress["global_step"] = step + 1
        # GLOBAL counters only: per-rank values would diverge across ranks and
        # break the checkpoint's cross-rank consistency validation.
        progress["consumed_supervised_tokens"] += accounting["supervised_global"]
        progress["consumed_samples"] += len(members)
        sampler["cursor"] = step + 1
        step_records.append({
            "step": step,
            "loss": metrics["global_loss"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "rng_fingerprint": rng_fingerprint(),
            "next_batch_preview": preview_next_batch(plan, step, samples, pad_token_id),
            "supervised_tokens_local": accounting["supervised_local"],
        })
        save_due = args.save_every > 0 and (step + 1) % args.save_every == 0
        final_step = step + 1 == args.steps
        if args.save_dir and (save_due or final_step):
            save_records.extend(save_checkpoints(step + 1))
        if args.fault_at_step is not None and step == args.fault_at_step and rank == args.fault_rank:
            if not save_due:
                raise SystemExit(
                    f"--fault-at-step {step} must align with a checkpoint save (--save-every)"
                )
            print(f"[fault-inject] rank {rank} hard-exiting after step {step} checkpoint", flush=True)
            sys.stdout.flush()
            os._exit(FAULT_EXIT_CODE)

    records = [None] * world_size
    dist.all_gather_object(records, {
        "rank": rank,
        "steps": step_records,
        "final_model_checksum": model_checksum(model),
        "progress": dict(progress),
        "rng_fingerprint": rng_fingerprint(),
    })
    result = {
        "backend": backend,
        "world_size": world_size,
        "resume_info": resume_info,
        "save_metrics": save_records,
        "final_model_checksum": [item["final_model_checksum"] for item in records],
        "final_progress": records[0]["progress"],
        "per_rank": records,
    }
    if rank == 0 and args.output:
        write_json(Path(args.output), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true", help="tiny synthetic model instead of Qwen3-0.6B")
    parser.add_argument("--data", type=Path, default=Path("data/chapter43_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--global-batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=0, help="save every K steps (0 = only the final step)")
    parser.add_argument("--save-modes", default="", help="comma list: centralized,sharded,dcp")
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--resume-mode", default="auto", choices=("auto", "centralized", "sharded", "dcp"))
    parser.add_argument("--fault-at-step", type=int, default=None)
    parser.add_argument("--fault-rank", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    use_cuda = torch.cuda.is_available() and not args.tiny
    backend = "nccl" if use_cuda else "gloo"
    rank, world_size, device = setup_distributed(rank, world_size, backend=backend)
    try:
        run_training(args, rank, world_size, device, backend)
        cleanup_distributed()
    except Exception:
        # Re-raise after teardown so torchrun reports the failure non-zero.
        if dist.is_initialized():
            dist.destroy_process_group()
        raise


if __name__ == "__main__":
    main()
