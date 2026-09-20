#!/usr/bin/env python3
"""Chapter 22 correctness: loader parity and bitwise-identical trajectories.

Layers:

1. Shard build (one-time) and device-batch parity: the synchronous loader and
   the prefetch loader must return bitwise-identical device tensors for every
   batch of the epoch.
2. Trajectory: a full training epoch driven by each loader, from fresh models
   and the same seed, must produce bitwise-identical per-step losses and
   gradient norms.  Overlapping data supply must not change training.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from assistant_loss import EncodedConversation
from data_paths import OnlineCorpus, ShardCorpus
from experiment_utils import environment_record, write_json
from prefetch_pipeline import PrefetchBatchLoader, SyncBatchLoader
from qwen3_model import resolve_model_directory
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything
from token_shard import build_shard
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_optimizer,
    load_tokenizer_and_model,
    train_device_batch_step,
)


def run_trajectory(args, loader, device: torch.device) -> list[dict[str, float]]:
    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    loader.start(device)
    steps: list[dict[str, float]] = []
    try:
        while True:
            try:
                gpu_batch, host, exposed_ms, supply_ms = loader.next_batch()
            except StopIteration:
                break
            metrics = train_device_batch_step(model, optimizer, gpu_batch, device, args.max_grad_norm, collect_telemetry=False)
            metrics["exposed_copy_ms"] = exposed_ms
            metrics["supply_ms"] = supply_ms
            metrics["supervised_tokens"] = host.supervised_tokens
            steps.append(metrics)
    finally:
        loader.shutdown()
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter22_train.jsonl"))
    parser.add_argument("--shard", type=Path, default=Path("data/chapter22_shard.bin"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    online = OnlineCorpus(args.data, tokenizer, args.max_length)
    samples = []
    for index in range(len(online)):
        source_line, ids, labels = online.materialize(index)
        samples.append(EncodedConversation(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=labels,
            assistant_loss_mask=labels != -100,
            source_line=source_line,
            rendered_text="",
            token_spans=(),
            loss_mode="assistant_only",
        ))
    shard_stats = build_shard(args.shard, samples, MODEL_REVISION, DATASET_REVISION, args.max_length)
    del samples
    gc.collect()

    import random

    rng = random.Random(f"{args.seed}|1|shuffle")
    order = list(range(len(online)))
    rng.shuffle(order)
    batches = [tuple(sorted(order[i : i + args.batch_size])) for i in range(0, len(order), args.batch_size)]
    pad_token_id = tokenizer.pad_token_id

    # ---- 1. device batch parity ----
    sync = SyncBatchLoader(ShardCorpus(args.shard, MODEL_REVISION, DATASET_REVISION), batches, pad_token_id, pin=False)
    prefetch = PrefetchBatchLoader(ShardCorpus(args.shard, MODEL_REVISION, DATASET_REVISION), batches, pad_token_id, queue_depth=args.queue_depth, pin=True)
    sync.start(device)
    prefetch.start(device)
    mismatched = 0
    compared = 0
    while True:
        try:
            sync_gpu, _, _, _ = sync.next_batch()
        except StopIteration:
            break
        prefetch_gpu, _, _, _ = prefetch.next_batch()
        same = all(torch.equal(sync_gpu[name], prefetch_gpu[name]) for name in sync_gpu)
        compared += 1
        if not same:
            mismatched += 1
    sync.shutdown()
    prefetch.shutdown()
    if mismatched:
        raise AssertionError(f"{mismatched} of {compared} device batches differ between loaders")

    # ---- 2. bitwise-identical trajectories ----
    trajectories: dict[str, list[dict[str, float]]] = {}
    for name, loader_factory in (
        ("sync", lambda: SyncBatchLoader(ShardCorpus(args.shard, MODEL_REVISION, DATASET_REVISION), batches, pad_token_id, pin=False)),
        ("prefetch", lambda: PrefetchBatchLoader(ShardCorpus(args.shard, MODEL_REVISION, DATASET_REVISION), batches, pad_token_id, queue_depth=args.queue_depth, pin=True)),
    ):
        trajectories[name] = run_trajectory(args, loader_factory(), device)
    if len(trajectories["sync"]) != len(trajectories["prefetch"]):
        raise AssertionError("trajectory lengths differ")
    loss_mismatches = 0
    for sync_step, prefetch_step in zip(trajectories["sync"], trajectories["prefetch"]):
        if sync_step["loss"] != prefetch_step["loss"]:
            loss_mismatches += 1
        if sync_step["gradient_norm_before_clip"] != prefetch_step["gradient_norm_before_clip"]:
            raise AssertionError("gradient norm diverged between loaders")
    if loss_mismatches:
        raise AssertionError(f"{loss_mismatches} step losses differ between loaders")

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "seed": args.seed, "num_samples": len(online), "batch_size": args.batch_size,
        "max_length": args.max_length, "queue_depth": args.queue_depth,
        "shard_bytes": shard_stats["file_bytes"],
        "device_batch_parity": {"batches_compared": compared, "mismatched": mismatched},
        "trajectory": {
            "steps": len(trajectories["sync"]),
            "loss_mismatches": loss_mismatches,
            "first_loss": trajectories["sync"][0]["loss"],
            "last_loss": trajectories["sync"][-1]["loss"],
        },
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_22 correctness checks passed")


if __name__ == "__main__":
    main()
