#!/usr/bin/env python3
"""Chapter 20 benchmark: padded batches versus sequence packing.

Regime A uses the fixed 512-sample corpus with the encoder cap of 512, which is
the natural training distribution of this chapter.  Regime B is a mechanism
amplifier: synthetic shapes (uniform length distributions, random token ids)
whose step time is content independent under the eager implementation.  It
shows how the same two paths behave once lengths actually vary; it is not a
measurement of the chapter corpus.
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path
from statistics import mean

import torch

from assistant_loss import EncodedConversation, IGNORE_INDEX
from experiment_utils import aggregate_steps, environment_record, percentile, write_json
from qwen3_model import Qwen3Config, resolve_model_directory
from sequence_packing import build_pack, greedy_packing_plan
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
    memory_record,
    train_batch_step,
    train_pack_step,
)


TIMING_FIELDS_BASELINE = ("data_time_ms", "h2d_time_ms", "forward_time_ms", "backward_time_ms", "optimizer_time_ms", "step_time_ms")
TIMING_FIELDS_PACK = ("data_time_ms", "h2d_mask_time_ms", "forward_time_ms", "backward_time_ms", "optimizer_time_ms", "step_time_ms")


def random_batches(num_samples: int, batch_size: int, seed: int, epoch: int) -> list[tuple[int, ...]]:
    """The Chapter 18/04 random baseline sampler, inlined for this chapter."""

    rng = random.Random(f"{seed}|{epoch}|shuffle")
    order = list(range(num_samples))
    rng.shuffle(order)
    return [tuple(sorted(order[start : start + batch_size])) for start in range(0, num_samples, batch_size)]


def synthetic_samples(count: int, low: int, high: int, vocabulary: int, supervised_ratio: float, seed: int) -> list[EncodedConversation]:
    """Random token ids with tail supervision; shapes are the measured object."""

    rng = random.Random(f"{seed}|synthetic|{low}|{high}")
    torch_rng = torch.Generator().manual_seed(seed + 1)
    samples: list[EncodedConversation] = []
    for index in range(count):
        length = rng.randint(low, high)
        ids = torch.randint(1, vocabulary, (1, length), generator=torch_rng)
        labels = ids.clone()
        supervised_from = max(1, int(length * (1.0 - supervised_ratio)))
        labels[0, :supervised_from] = IGNORE_INDEX
        samples.append(EncodedConversation(
            input_ids=ids,
            attention_mask=torch.ones((1, length), dtype=torch.long),
            labels=labels,
            assistant_loss_mask=labels != IGNORE_INDEX,
            source_line=index + 1,
            rendered_text="",
            token_spans=(),
            loss_mode="assistant_only",
        ))
    return samples


def run_epoch(args, step_runner, plan_descriptions, warmup_steps: int, is_pack: bool) -> dict[str, object]:
    """Shared epoch runner; ``plan_descriptions`` yields per-step member tuples."""

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    steps: list[dict[str, float]] = []
    try:
        steps_list = list(plan_descriptions)
        for members in steps_list[:warmup_steps]:
            metrics = step_runner(model, optimizer, members, device, collect_telemetry=False)
        torch.cuda.reset_peak_memory_stats(device)
        for step_index, members in enumerate(steps_list[warmup_steps:], start=warmup_steps):
            metrics = step_runner(model, optimizer, members, device)
            metrics["step"] = float(step_index + 1)
            steps.append(metrics)
        measured_seconds = sum(step["step_time_ms"] for step in steps) / 1000
        totals = {
            field: sum(step[field] for step in steps)
            for field in ("physical_tokens", "non_padding_tokens", "supervised_tokens", "padding_tokens")
        }
        timing_fields = TIMING_FIELDS_PACK if is_pack else TIMING_FIELDS_BASELINE
        timing_fields = tuple(field for field in timing_fields if field in steps[0])
        aggregate = aggregate_steps([{"step_time_ms": step["step_time_ms"], "loss": step["loss"]} for step in steps])
        return {
            "status": "success",
            "warmup_steps": warmup_steps,
            "measured_steps": len(steps),
            "steps": steps,
            "aggregate": aggregate,
            "step_time_p99_ms": percentile([step["step_time_ms"] for step in steps], 0.99),
            "timing_mean_ms": {field: mean(step[field] for step in steps) for field in timing_fields},
            "measured_tokens": totals,
            "throughput": {
                "physical_tokens_per_second": totals["physical_tokens"] / measured_seconds,
                "non_padding_tokens_per_second": totals["non_padding_tokens"] / measured_seconds,
                "supervised_tokens_per_second": totals["supervised_tokens"] / measured_seconds,
            },
            "loss_weighted_by_epoch": sum(step["loss"] * step["supervised_tokens"] for step in steps) / totals["supervised_tokens"],
            "memory": memory_record(model, optimizer),
        }
    except torch.cuda.OutOfMemoryError:
        return {
            "status": "oom",
            "failure_stage": "warmup_or_measured_step",
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter20_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pack-capacity", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--synthetic-count", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    records = load_jsonl_records(args.data)
    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    samples = encode_samples(tokenizer, records, args.max_length)
    lengths = [sample.non_padding_tokens for sample in samples]
    pad_token_id = tokenizer.pad_token_id
    encoded_supervised = sum(sample.supervised_tokens for sample in samples)
    vocabulary = Qwen3Config.from_model_directory(
        resolve_model_directory(MODEL_ID, MODEL_REVISION)
    ).vocab_size

    configurations: list[dict[str, object]] = []

    def run_padded_epoch(tag: str, sample_pool, batches: list[tuple[int, ...]], supervised_total: int) -> dict[str, object]:
        def runner(model, optimizer, members, device, collect_telemetry: bool = True):
            data_start = time.perf_counter()
            host_batch = build_batch(sample_pool, members, pad_token_id)
            data_ms = (time.perf_counter() - data_start) * 1000
            metrics = train_batch_step(model, optimizer, host_batch, device, args.max_grad_norm, collect_telemetry)
            metrics["data_time_ms"] = data_ms
            metrics["step_time_ms"] += data_ms
            return metrics

        result = run_epoch(args, runner, batches, args.warmup, is_pack=False)
        if result["status"] == "success":
            measured = result["measured_tokens"]
            warmup_supervised = sum(
                sample_pool[index].supervised_tokens
                for batch in batches[: args.warmup] for index in batch
            )
            if int(measured["supervised_tokens"]) + warmup_supervised != supervised_total:
                raise AssertionError(f"{tag}: supervised accounting mismatch")
        return result

    def run_packed_epoch(tag: str, sample_pool, packs, supervised_total: int, capacity: int) -> dict[str, object]:
        pack_objects = {members: build_pack(sample_pool, members, capacity, pad_token_id) for members in packs}

        def runner(model, optimizer, members, device, collect_telemetry: bool = True):
            data_start = time.perf_counter()
            host_pack = pack_objects[members]
            data_ms = (time.perf_counter() - data_start) * 1000
            metrics = train_pack_step(model, optimizer, host_pack, device, args.max_grad_norm, collect_telemetry)
            metrics["data_time_ms"] = data_ms
            metrics["step_time_ms"] += data_ms
            return metrics

        result = run_epoch(args, runner, packs, args.warmup, is_pack=True)
        if result["status"] == "success":
            measured = result["measured_tokens"]
            warmup_supervised = sum(
                sample_pool[index].supervised_tokens
                for members in packs[: args.warmup] for index in members
            )
            if int(measured["supervised_tokens"]) + warmup_supervised != supervised_total:
                raise AssertionError(f"{tag}: supervised accounting mismatch")
            result["pack_utilization"] = sum(
                pack_objects[members].non_padding_tokens for members in packs
            ) / (len(packs) * capacity)
            result["packs_total"] = len(packs)
        return result

    # ---- Regime A: the natural corpus ----
    baseline_batches = random_batches(len(samples), args.batch_size, args.seed, 1)
    baseline_run = run_padded_epoch("natural-padded", samples, baseline_batches, encoded_supervised)
    configurations.append({
        "regime": "natural", "path": "padded", "batch_size": args.batch_size,
        "num_samples": len(samples), "num_steps": len(baseline_batches),
        "run": baseline_run,
    })
    if baseline_run["status"] == "success":
        print(f"natural padded supervised_tok_s={baseline_run['throughput']['supervised_tokens_per_second']:.2f}")
    else:
        print(f"natural padded status={baseline_run['status']} allocated={baseline_run.get('allocated_bytes', 0) / 2**30:.2f} GiB")

    pack_plan = greedy_packing_plan(lengths, args.pack_capacity, args.seed, 1)
    packing_run = run_packed_epoch("natural-packed", samples, pack_plan.packs, encoded_supervised, args.pack_capacity)
    configurations.append({
        "regime": "natural", "path": "packed", "pack_capacity": args.pack_capacity,
        "num_samples": len(samples), "num_steps": pack_plan.num_packs,
        "samples_per_pack_p50": sorted(pack_plan.samples_per_pack())[pack_plan.num_packs // 2],
        "max_samples_per_pack": max(pack_plan.samples_per_pack()),
        "run": packing_run,
    })
    if packing_run["status"] == "success":
        print(f"natural packed supervised_tok_s={packing_run['throughput']['supervised_tokens_per_second']:.2f} utilization={packing_run.get('pack_utilization', 0):.4f}")
    else:
        print(f"natural packed status={packing_run['status']} allocated={packing_run.get('allocated_bytes', 0) / 2**30:.2f} GiB")

    # ---- Regime B: synthetic shapes (mechanism amplifier) ----
    distributions = [
        {"name": "uniform_96_256", "low": 96, "high": 256, "capacity": 512},
        {"name": "uniform_96_512", "low": 96, "high": 512, "capacity": 512},
        {"name": "uniform_384_1024", "low": 384, "high": 1024, "capacity": 1024},
    ]
    for spec in distributions:
        synthetic = synthetic_samples(
            args.synthetic_count, spec["low"], spec["high"], vocabulary, 0.635, args.seed
        )
        synthetic_lengths = [sample.non_padding_tokens for sample in synthetic]
        synthetic_supervised = sum(sample.supervised_tokens for sample in synthetic)
        synthetic_batches = random_batches(len(synthetic), args.batch_size, args.seed, 1)
        padded = run_padded_epoch(f"{spec['name']}-padded", synthetic, synthetic_batches, synthetic_supervised)
        configurations.append({
            "regime": "synthetic", "path": "padded", "distribution": spec["name"],
            "length_range": [spec["low"], spec["high"]], "batch_size": args.batch_size,
            "num_samples": len(synthetic), "num_steps": len(synthetic_batches),
            "run": padded,
        })
        synthetic_plan = greedy_packing_plan(synthetic_lengths, spec["capacity"], args.seed, 1)
        packed = run_packed_epoch(
            f"{spec['name']}-packed", synthetic, synthetic_plan.packs, synthetic_supervised, spec["capacity"]
        )
        configurations.append({
            "regime": "synthetic", "path": "packed", "distribution": spec["name"],
            "length_range": [spec["low"], spec["high"]], "pack_capacity": spec["capacity"],
            "num_samples": len(synthetic), "num_steps": synthetic_plan.num_packs,
            "samples_per_pack_p50": sorted(synthetic_plan.samples_per_pack())[synthetic_plan.num_packs // 2],
            "max_samples_per_pack": max(synthetic_plan.samples_per_pack()),
            "run": packed,
        })
        def describe(result, label):
            if result["status"] != "success":
                return f"{label}=OOM"
            return f"{label}={result['throughput']['supervised_tokens_per_second']:.2f}"

        print(
            f"{spec['name']}: {describe(padded, 'padded')} "
            f"{describe(packed, 'packed')} "
            f"utilization={packed.get('pack_utilization', 0):.4f}"
        )

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "formal_dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "world_size": 1, "batch_size": args.batch_size, "pack_capacity": args.pack_capacity,
        "max_length": args.max_length, "reduction": "token_mean",
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "seed": args.seed,
        "warmup_steps": args.warmup, "epochs": 1, "repeats": 1,
        "sample_count": len(samples),
        "corpus_length_p50": sorted(lengths)[len(lengths) // 2],
        "configurations": configurations,
        "timing_scope": {
            "included": ["in_memory_batch_or_pack_construction", "synchronous_h2d", "block_diagonal_mask_build_packed_path", "forward_and_backward", "gradient_clipping", "optimizer"],
            "excluded": ["jsonl_read", "chat_template", "tokenization", "plan_construction", "checkpoint", "gpu_telemetry_query"],
            "communication_time_ms": 0.0, "checkpoint_time_ms": 0.0,
            "cuda_intervals": "CUDA events", "end_to_end": "synchronized wall clock",
        },
        "synthetic_shapes_note": (
            "Regime B uses random token ids and uniform length distributions; under the eager "
            "implementation step time depends on shapes, not token values. It is a mechanism "
            "amplifier, not a measurement of the chapter corpus."
        ),
    })
    write_json(args.output, payload)
    print(f"saved benchmark to {args.output}")


if __name__ == "__main__":
    main()
