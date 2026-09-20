#!/usr/bin/env python3
"""Chapter 21 benchmark: online tokenization versus the offline token shard.

Phases:

1. One-time offline build: encode all samples through the online pipeline and
   write the shard (both costs timed), record file sizes.
2. Data pipeline (no GPU): cold start to the first ready batch, per-epoch data
   time, and per-record random-access cost for both paths.
3. GPU data wait: one real training epoch per path with the shared step body;
   the data phase plus synchronous H2D is exactly how long the GPU waits.
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path
from statistics import mean, median

import torch

from assistant_loss import EncodedConversation
from data_paths import OnlineCorpus, ShardCorpus, pad_pairs
from experiment_utils import environment_record, percentile, write_json
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything
from token_shard import build_shard
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_optimizer,
    load_tokenizer_and_model,
    memory_record,
    train_batch_step,
)


def random_batches(num_samples: int, batch_size: int, seed: int, epoch: int) -> list[tuple[int, ...]]:
    rng = random.Random(f"{seed}|{epoch}|shuffle")
    order = list(range(num_samples))
    rng.shuffle(order)
    return [tuple(sorted(order[start : start + batch_size])) for start in range(0, num_samples, batch_size)]


def data_epoch(corpus, batches, pad_token_id) -> tuple[float, int, int]:
    """Materialize and pad every batch; return (seconds, samples, supervised)."""

    start = time.perf_counter()
    samples = 0
    supervised = 0
    for members in batches:
        pairs = [corpus.materialize(index) for index in members]
        batch = pad_pairs(pairs, pad_token_id)
        samples += batch.batch_size
        supervised += batch.supervised_tokens
    return time.perf_counter() - start, samples, supervised


def gpu_epoch(args, corpus, batches, pad_token_id) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    try:
        for members in batches[: args.warmup]:
            pairs = [corpus.materialize(index) for index in members]
            batch = pad_pairs(pairs, pad_token_id)
            train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
        torch.cuda.reset_peak_memory_stats(device)
        steps = []
        for step_index, members in enumerate(batches[args.warmup :], start=args.warmup):
            data_start = time.perf_counter()
            pairs = [corpus.materialize(index) for index in members]
            batch = pad_pairs(pairs, pad_token_id)
            data_ms = (time.perf_counter() - data_start) * 1000
            metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm)
            metrics["data_time_ms"] = data_ms
            metrics["step_time_ms"] += data_ms
            metrics["step"] = float(step_index + 1)
            steps.append(metrics)
            del batch
        total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
        supervised = sum(step["supervised_tokens"] for step in steps)
        return {
            "status": "success",
            "warmup_steps": args.warmup,
            "measured_steps": len(steps),
            "steps": steps,
            "aggregate": {
                "step_time_mean_ms": mean(step["step_time_ms"] for step in steps),
                "step_time_p50_ms": percentile([step["step_time_ms"] for step in steps], 0.5),
                "step_time_p95_ms": percentile([step["step_time_ms"] for step in steps], 0.95),
                "step_time_max_ms": max(step["step_time_ms"] for step in steps),
            },
            "data_time_mean_ms": mean(step["data_time_ms"] for step in steps),
            "h2d_time_mean_ms": mean(step["h2d_time_ms"] for step in steps),
            "forward_time_mean_ms": mean(step["forward_time_ms"] for step in steps),
            "optimizer_time_mean_ms": mean(step["optimizer_time_ms"] for step in steps),
            "gpu_wait_share": mean((step["data_time_ms"] + step["h2d_time_ms"]) / step["step_time_ms"] for step in steps),
            "throughput": {
                "supervised_tokens_per_second": supervised / total_seconds,
            },
            "memory": memory_record(model, optimizer),
        }
    except torch.cuda.OutOfMemoryError:
        return {"status": "oom", "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter21_train.jsonl"))
    parser.add_argument("--shard", type=Path, default=Path("data/chapter21_shard.bin"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--random-access-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    records_probe = OnlineCorpus(args.data, Qwen3Tokenizer(resolve_model_directory(MODEL_ID, MODEL_REVISION)), args.max_length)
    num_samples = len(records_probe)
    pad_token_id = records_probe.tokenizer.pad_token_id
    batches = random_batches(num_samples, args.batch_size, args.seed, 1)

    # ---- Phase 1: one-time offline build ----
    build_start = time.perf_counter()
    samples = []
    for index in range(num_samples):
        source_line, ids, labels = records_probe.materialize(index)
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
    encode_seconds = time.perf_counter() - build_start
    shard_stats = build_shard(args.shard, samples, MODEL_REVISION, DATASET_REVISION, args.max_length)
    jsonl_bytes = Path(args.data).stat().st_size
    print(f"offline build: encode={encode_seconds:.1f}s shard={shard_stats['file_bytes']}B jsonl={jsonl_bytes}B", flush=True)

    # ---- Phase 2: data pipeline (no GPU) ----
    pipeline: dict[str, object] = {}
    for name, factory in (
        ("online", lambda: OnlineCorpus(args.data, records_probe.tokenizer, args.max_length)),
        ("shard", lambda: ShardCorpus(args.shard, MODEL_REVISION, DATASET_REVISION)),
    ):
        start = time.perf_counter()
        corpus = factory()
        init_ms = (time.perf_counter() - start) * 1000
        batch_start = time.perf_counter()
        pairs = [corpus.materialize(index) for index in batches[0]]
        first_batch = pad_pairs(pairs, pad_token_id)
        cold_start_ms = (time.perf_counter() - batch_start) * 1000 + init_ms
        epoch_times = []
        for _ in range(args.epochs):
            seconds, samples_count, supervised = data_epoch(corpus, batches, pad_token_id)
            epoch_times.append(seconds)
        rng = random.Random(args.seed)
        probe_indices = [rng.randrange(num_samples) for _ in range(args.random_access_count)]
        probe_times = []
        for index in probe_indices:
            start = time.perf_counter()
            corpus.materialize(index)
            probe_times.append((time.perf_counter() - start) * 1000)
        pipeline[name] = {
            "init_ms": init_ms,
            "cold_start_first_batch_ms": cold_start_ms,
            "epoch_data_seconds": epoch_times,
            "epoch_supervised_tokens": supervised,
            "data_samples_per_second_epoch2": samples_count / epoch_times[-1],
            "random_access_ms_p50": median(probe_times),
            "random_access_ms_mean": mean(probe_times),
        }
        print(f"{name}: cold={cold_start_ms:.1f}ms epochs={[round(t, 1) for t in epoch_times]} ra_p50={median(probe_times):.3f}ms", flush=True)

    # ---- Phase 3: GPU data wait ----
    gpu: dict[str, object] = {}
    for name, factory in (
        ("online", lambda: OnlineCorpus(args.data, records_probe.tokenizer, args.max_length)),
        ("shard", lambda: ShardCorpus(args.shard, MODEL_REVISION, DATASET_REVISION)),
    ):
        corpus = factory()
        result = gpu_epoch(args, corpus, batches, pad_token_id)
        gpu[name] = result
        if result["status"] == "success":
            print(f"{name}: gpu_epoch sup/s={result['throughput']['supervised_tokens_per_second']:.2f} "
                  f"data_wait_share={result['gpu_wait_share']:.3f}", flush=True)
        else:
            print(f"{name}: {result['status']}", flush=True)

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "world_size": 1, "batch_size": args.batch_size, "num_samples": num_samples,
        "max_length": args.max_length, "epochs": args.epochs, "warmup_steps": args.warmup,
        "seed": args.seed, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "offline_build": {
            "encode_seconds": encode_seconds,
            "write_seconds": shard_stats["write_seconds"],
            "shard_bytes": shard_stats["file_bytes"],
            "jsonl_bytes": jsonl_bytes,
        },
        "pipeline": pipeline,
        "gpu": gpu,
        "timing_scope": {
            "data_phase_includes": ["jsonl_record_parse_or_shard_seek", "chat_template_render", "bpe_tokenize_online_only", "checksum_verify_shard_only", "pad_pairs_collate"],
            "gpu_wait_definition": "data_time_ms + synchronous h2d_time_ms per step",
            "excluded": ["plan_construction", "checkpoint", "gpu_telemetry_query"],
        },
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
