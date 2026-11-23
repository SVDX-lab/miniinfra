#!/usr/bin/env python3
"""Run short matched trajectories for serial and fixed-mini-batch execution."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from assistant_loss import encode_conversation
from experiment_utils import environment_record, write_json
from mini_batch import pad_samples, trim_sample
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import MODEL_ID, MODEL_REVISION, build_optimizer, load_tokenizer_and_model, memory_record, train_group_step


def run(args, batch, execution: str) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    torch.cuda.reset_peak_memory_stats(device)
    steps = []
    for index in range(args.steps):
        metrics = train_group_step(model, optimizer, batch, device, execution, args.reduction, args.max_grad_norm)
        metrics["step"] = index + 1
        steps.append(metrics)
        print(f"execution={execution} step={index + 1} loss={metrics['loss']:.6f}")
    result = {"execution": execution, "steps": steps, "memory": memory_record(model, optimizer)}
    del optimizer, model
    gc.collect(); torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter18_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-cursor", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--reduction", choices=["token_mean", "sample_mean"], default="token_mean")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output", type=Path, default=Path("train-results.local.json"))
    args = parser.parse_args()
    if min(args.batch_size, args.steps) < 1:
        parser.error("batch size and steps must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = load_jsonl_records(args.data)[args.sample_cursor : args.sample_cursor + args.batch_size]
    if len(records) != args.batch_size:
        parser.error("data does not contain enough records")
    tokenizer = Qwen3Tokenizer(resolve_model_directory(MODEL_ID, MODEL_REVISION))
    samples = [trim_sample(encode_conversation(tokenizer, record, args.max_length, "assistant_only")) for record in records]
    batch = pad_samples(samples, tokenizer.pad_token_id)
    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "tf32": False, "batch_size": args.batch_size,
        "sample_cursor": args.sample_cursor, "max_length": args.max_length, "reduction": args.reduction,
        "source_lines": list(batch.source_lines), "sequence_lengths": list(batch.lengths),
        "trajectories": {execution: run(args, batch, execution) for execution in ("serial", "batched")},
    })
    write_json(args.output, payload)
    print(f"saved trajectories to {args.output}")


if __name__ == "__main__":
    main()
