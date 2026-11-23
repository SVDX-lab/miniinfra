#!/usr/bin/env python3
"""Run matched multi-epoch trajectories for selected Chapter 19 samplers."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from statistics import mean

import torch

from experiment_utils import aggregate_steps, environment_record, write_json
from length_bucket import bucketed_plan, random_plan
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
)


def run_trajectory(args, samples, lengths, pad_token_id, plans) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    steps: list[dict[str, float]] = []
    try:
        global_step = 0
        for plan in plans:
            epoch_steps: list[dict[str, float]] = []
            for batch_indices in plan.batches:
                host_batch = build_batch(samples, batch_indices, pad_token_id)
                metrics = train_batch_step(model, optimizer, host_batch, device, args.max_grad_norm, collect_telemetry=False)
                global_step += 1
                metrics["step"] = float(global_step)
                metrics["epoch"] = float(plan.epoch)
                steps.append(metrics)
                epoch_steps.append(metrics)
            epoch_supervised = sum(metrics["supervised_tokens"] for metrics in epoch_steps)
            print(
                f"strategy={plan.strategy} window={plan.window} epoch={plan.epoch} "
                f"loss_weighted={sum(m['loss'] * m['supervised_tokens'] for m in epoch_steps) / epoch_supervised:.6f}"
            )
        epoch_boundary = plans[0].num_batches
        result = {
            "status": "success",
            "steps": steps,
            "epoch_aggregates": [
                {
                    "epoch": plan.epoch,
                    "aggregate": aggregate_steps(
                        [step for step in steps if step["epoch"] == plan.epoch]
                    ),
                    "loss_weighted": sum(
                        step["loss"] * step["supervised_tokens"]
                        for step in steps if step["epoch"] == plan.epoch
                    ) / sum(step["supervised_tokens"] for step in steps if step["epoch"] == plan.epoch),
                    "padding_ratio": mean(step["padding_ratio"] for step in steps if step["epoch"] == plan.epoch),
                }
                for plan in plans
            ],
            "memory": memory_record(model, optimizer),
        }
        return result
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter19_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mid-window", type=int, default=32)
    parser.add_argument("--full-sort-window", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("train-results.local.json"))
    args = parser.parse_args()
    if min(args.batch_size, args.epochs) < 1:
        parser.error("batch size and epochs must be positive")
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

    trajectories = {}
    for name, plan_builder in (
        ("random", lambda epoch: random_plan(len(samples), args.batch_size, args.seed, epoch)),
        ("bucketed_mid", lambda epoch: bucketed_plan(lengths, args.batch_size, args.mid_window, args.seed, epoch)),
        ("bucketed_full_sort", lambda epoch: bucketed_plan(lengths, args.batch_size, args.full_sort_window, args.seed, epoch)),
    ):
        plans = [plan_builder(epoch) for epoch in range(1, args.epochs + 1)]
        print(f"running trajectory: {name}")
        trajectories[name] = {
            "window": plans[0].window,
            "epochs": args.epochs,
            **run_trajectory(args, samples, lengths, pad_token_id, plans),
        }

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "world_size": 1, "batch_size": args.batch_size, "num_samples": len(samples),
        "reduction": "token_mean", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "seed": args.seed, "max_length": args.max_length, "epochs": args.epochs,
        "trajectories": trajectories,
    })
    write_json(args.output, payload)
    print(f"saved trajectories to {args.output}")


if __name__ == "__main__":
    main()
