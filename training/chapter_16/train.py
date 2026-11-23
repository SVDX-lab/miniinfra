#!/usr/bin/env python3
"""Minimal full-parameter SFT loop for Qwen3-0.6B-Base."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiment_utils import aggregate_steps, environment_record, write_json
from sft_core import (
    DATASET_FILE,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SHA256,
    encode_conversation,
    load_jsonl_records,
    seed_everything,
    select_record,
    trainable_parameter_count,
)
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_optimizer_and_scheduler,
    generate_text,
    load_tokenizer_and_model,
    memory_ledger,
    save_checkpoint,
    train_step,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter16_train.jsonl"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-cursor", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/chapter16.pt"))
    parser.add_argument("--output", type=Path, default=Path("train-results.local.json"))
    parser.add_argument("--prompt", default="用一句话解释什么是梯度下降。")
    parser.add_argument("--generation-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("Chapter 16 real-model training requires one CUDA GPU")

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tokenizer, model = load_tokenizer_and_model(device)
    records = load_jsonl_records(args.data)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        args.learning_rate,
        args.weight_decay,
        args.warmup_steps,
        args.steps,
    )

    before = generate_text(model, tokenizer, args.prompt, device, args.generation_tokens)
    torch.cuda.reset_peak_memory_stats(device)
    step_records: list[dict[str, float]] = []
    consumed_supervised_tokens = 0
    for step in range(args.steps):
        record = select_record(records, args.sample_cursor)
        sample = encode_conversation(tokenizer, record, args.max_length)
        metrics = train_step(model, optimizer, scheduler, sample, device, args.max_grad_norm)
        metrics["step"] = float(step + 1)
        metrics["source_line"] = float(sample.source_line)
        step_records.append(metrics)
        consumed_supervised_tokens += sample.supervised_tokens
        print(
            f"step={step + 1} loss={metrics['loss']:.6f} "
            f"supervised_tok_s={metrics['supervised_tokens_per_second']:.2f}"
        )

    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        scheduler,
        args.steps,
        consumed_supervised_tokens,
        args.sample_cursor,
    )
    after = generate_text(model, tokenizer, args.prompt, device, args.generation_tokens)

    payload = environment_record()
    payload.update(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "dataset": {
                "repo": DATASET_REPO,
                "revision": DATASET_REVISION,
                "file": DATASET_FILE,
                "sha256": DATASET_SHA256,
                "sample_source_lines": sorted({int(step["source_line"]) for step in step_records}),
            },
            "dtype": "float32",
            "attention_implementation": "eager",
            "tf32": False,
            "seed": args.seed,
            "data_path": str(args.data),
            "sample_count": len(records),
            "sample_cursor": args.sample_cursor,
            "max_length": args.max_length,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "global_batch_sequences": 1,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_gradient_norm": args.max_grad_norm,
            "warmup_steps": args.warmup_steps,
            "steps": step_records,
            "aggregate": aggregate_steps(step_records),
            "memory": memory_ledger(model, optimizer),
            "trainable_parameters": trainable_parameter_count(model),
            "checkpoint": str(args.checkpoint),
            "consumed_supervised_tokens": consumed_supervised_tokens,
            "generation": {"prompt": args.prompt, "before": before, "after": after},
            "timing_scope": {
                "included": ["synchronous_h2d", "forward", "backward", "gradient_clipping", "optimizer", "scheduler"],
                "excluded": ["jsonl_read", "chat_template", "tokenization", "generation", "checkpoint", "gpu_telemetry_query"],
            },
        }
    )
    write_json(args.output, payload)
    print(f"saved checkpoint={args.checkpoint} results={args.output}")


if __name__ == "__main__":
    main()
