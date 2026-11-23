#!/usr/bin/env python3
"""Chapter 22 benchmark: synchronous data path versus prefetch with async H2D.

Each repeat drives one full training epoch with the shared step body, first
through ``SyncBatchLoader`` (materialize + blocking H2D inside the step path),
then through ``PrefetchBatchLoader`` (worker thread, pinned host buffers,
side-stream double-buffered copies).  Per-step timings are decomposed into the
data-side cost each path exposes to the critical path and the compute cost.
"""

from __future__ import annotations

import argparse
import gc
import resource
import time
from pathlib import Path
from statistics import mean

import torch

from assistant_loss import EncodedConversation
from data_paths import OnlineCorpus, ShardCorpus
from experiment_utils import environment_record, percentile, write_json
from prefetch_pipeline import PrefetchBatchLoader, SyncBatchLoader
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything
from token_shard import build_shard
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_optimizer,
    load_tokenizer_and_model,
    train_device_batch_step,
)


def run_epoch(args, loader, device: torch.device) -> dict[str, object]:
    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    loader.start(device)
    steps = []
    try:
        for _ in range(args.warmup):
            gpu_batch, _, _, _ = loader.next_batch()
            train_device_batch_step(model, optimizer, gpu_batch, device, args.max_grad_norm, collect_telemetry=False)
        torch.cuda.reset_peak_memory_stats(device)
        for step_index in range(len(loader.batches) - args.warmup):
            data_start = time.perf_counter()
            gpu_batch, host, exposed_ms, supply_ms = loader.next_batch()
            data_wait_ms = (time.perf_counter() - data_start) * 1000
            metrics = train_device_batch_step(model, optimizer, gpu_batch, device, args.max_grad_norm)
            metrics.update({
                "step": float(step_index + 1),
                "data_wait_ms": data_wait_ms,
                "exposed_copy_ms": exposed_ms,
                "supply_ms": supply_ms,
                "supervised_tokens": float(host.supervised_tokens),
                "physical_tokens": float(gpu_batch["input_ids"].numel()),
                "non_padding_tokens": float(gpu_batch["attention_mask"].sum().item()),
            })
            steps.append(metrics)
        total_seconds = sum(step["step_time_ms"] for step in steps) / 1000
        supervised = sum(step["supervised_tokens"] for step in steps)
        step_times = [step["step_time_ms"] for step in steps]
        return {
            "status": "success",
            "warmup_steps": args.warmup,
            "measured_steps": len(steps),
            "steps": steps,
            "aggregate": {
                "step_time_mean_ms": mean(step_times),
                "step_time_p50_ms": percentile(step_times, 0.5),
                "step_time_p95_ms": percentile(step_times, 0.95),
                "step_time_max_ms": max(step_times),
            },
            "phase_means_ms": {
                "data_wait": mean(step["data_wait_ms"] for step in steps),
                "forward": mean(step["forward_time_ms"] for step in steps),
                "backward": mean(step["backward_time_ms"] for step in steps),
                "optimizer": mean(step["optimizer_time_ms"] for step in steps),
            },
            "gpu_wait_share": mean(step["data_wait_ms"] / step["step_time_ms"] for step in steps),
            "throughput": {
                "supervised_tokens_per_second": supervised / total_seconds,
            },
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            },
        }
    except torch.cuda.OutOfMemoryError:
        return {"status": "oom", "allocated_bytes": torch.cuda.memory_allocated()}
    finally:
        loader.shutdown()
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter22_train.jsonl"))
    parser.add_argument("--shard", type=Path, default=Path("data/chapter22_shard.bin"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--queue-depth", type=int, default=2)
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
    jsonl_bytes = Path(args.data).stat().st_size

    import random

    rng = random.Random(f"{args.seed}|1|shuffle")
    order = list(range(len(online)))
    rng.shuffle(order)
    batches = [tuple(sorted(order[i : i + args.batch_size])) for i in range(0, len(order), args.batch_size)]

    repeats: list[dict[str, object]] = []
    for repeat in range(1, args.repeats + 1):
        for name, factory in (
            ("sync", lambda: SyncBatchLoader(ShardCorpus(args.shard, MODEL_REVISION, DATASET_REVISION), batches, pad_token_id=tokenizer.pad_token_id, pin=False)),
            ("prefetch", lambda: PrefetchBatchLoader(ShardCorpus(args.shard, MODEL_REVISION, DATASET_REVISION), batches, pad_token_id=tokenizer.pad_token_id, queue_depth=args.queue_depth, pin=True)),
        ):
            result = run_epoch(args, factory(), torch.device("cuda:0"))
            entry = {"repeat": repeat, "path": name, "result": result}
            repeats.append(entry)
            if result["status"] == "success":
                print(
                    f"repeat={repeat} {name}: sup/s={result['throughput']['supervised_tokens_per_second']:.2f} "
                    f"data_wait={result['phase_means_ms']['data_wait']:.3f}ms step_p50={result['aggregate']['step_time_p50_ms']:.1f}ms",
                    flush=True,
                )
            else:
                print(f"repeat={repeat} {name}: {result['status']}", flush=True)

    def aggregate(path: str) -> dict[str, float]:
        successful = [entry["result"] for entry in repeats if entry["path"] == path and entry["result"]["status"] == "success"]
        return {
            "successful_repeats": len(successful),
            "step_time_p50_ms_mean": mean(r["aggregate"]["step_time_p50_ms"] for r in successful),
            "supervised_tokens_per_second_mean": mean(r["throughput"]["supervised_tokens_per_second"] for r in successful),
            "data_wait_ms_mean": mean(r["phase_means_ms"]["data_wait"] for r in successful),
        }

    summary = {path: aggregate(path) for path in ("sync", "prefetch")}
    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "world_size": 1, "batch_size": args.batch_size, "num_samples": len(online),
        "max_length": args.max_length, "warmup_steps": args.warmup, "repeats": args.repeats,
        "queue_depth": args.queue_depth, "seed": args.seed,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "shard_bytes": shard_stats["file_bytes"], "jsonl_bytes": jsonl_bytes,
        "repeats_detail": repeats,
        "summary": summary,
        "pinned_memory_note": (
            "prefetch path keeps one pinned host batch per queue slot plus the in-flight batch; "
            "bytes per batch are reported per repeat from HostBatch.tensor_bytes"
        ),
        "timing_scope": {
            "sync_data_wait_includes": ["shard_seek_read", "checksum_verify", "pad_pairs_collate", "blocking_h2d_plus_sync"],
            "prefetch_data_wait_includes": ["event_synchronize_residual_copy", "queue_wait_for_batch_after_next"],
            "excluded": ["checkpoint", "gpu_telemetry_query"],
        },
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
