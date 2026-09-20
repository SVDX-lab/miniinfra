#!/usr/bin/env python3
"""Verify that a saved checkpoint reproduces the next optimizer step."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from experiment_utils import environment_record, write_json
from sft_core import (
    DATASET_FILE,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SHA256,
    encode_conversation,
    load_jsonl_records,
    seed_everything,
)
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_optimizer_and_scheduler,
    load_checkpoint,
    load_tokenizer_and_model,
    save_checkpoint,
    train_step,
)


def probe_parameter(model: torch.nn.Module) -> tuple[str, torch.Tensor]:
    name, parameter = min(model.named_parameters(), key=lambda item: item[1].numel())
    return name, parameter.detach().float().cpu().clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter16_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/resume-test.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoint-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required for the checkpoint test")

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    tokenizer, model = load_tokenizer_and_model(device)
    records = load_jsonl_records(args.data)
    first = encode_conversation(tokenizer, records[0], args.max_length)
    second = encode_conversation(tokenizer, records[1], args.max_length)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, args.learning_rate, args.weight_decay, 1, 3
    )

    first_metrics = train_step(model, optimizer, scheduler, first, device, args.max_grad_norm)
    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        scheduler,
        step=1,
        consumed_supervised_tokens=first.supervised_tokens,
        sample_cursor=1,
    )
    reference_metrics = train_step(model, optimizer, scheduler, second, device, args.max_grad_norm)
    probe_name, reference_parameter = probe_parameter(model)

    del optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, args.learning_rate, args.weight_decay, 1, 3
    )
    restored = load_checkpoint(args.checkpoint, model, optimizer, scheduler, device)
    resumed_metrics = train_step(model, optimizer, scheduler, second, device, args.max_grad_norm)
    _, resumed_parameter = probe_parameter(model)

    loss_error = abs(reference_metrics["loss"] - resumed_metrics["loss"])
    parameter_error = float((reference_parameter - resumed_parameter).abs().max().item())
    if restored != {"step": 1, "consumed_supervised_tokens": first.supervised_tokens, "sample_cursor": 1}:
        raise AssertionError(f"restored metadata mismatch: {restored}")
    if loss_error > 1e-7 or parameter_error > 1e-7:
        raise AssertionError(
            f"resume mismatch: loss_error={loss_error}, parameter_error={parameter_error}"
        )

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
                "sample_source_lines": [first.source_line, second.source_line],
            },
            "dtype": "float32",
            "seed": args.seed,
            "max_length": args.max_length,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "global_batch_sequences": 1,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_gradient_norm": args.max_grad_norm,
            "checkpoint": str(args.checkpoint),
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "first_step_loss": first_metrics["loss"],
            "reference_next_step_loss": reference_metrics["loss"],
            "resumed_next_step_loss": resumed_metrics["loss"],
            "next_step_loss_absolute_error": loss_error,
            "probe_parameter": probe_name,
            "probe_parameter_max_absolute_error": parameter_error,
            "restored_metadata": restored,
            "status": "pass",
        }
    )
    write_json(args.output, payload)
    print(
        f"checkpoint continuation passed: loss_error={loss_error:.3e}, "
        f"parameter_error={parameter_error:.3e}"
    )


if __name__ == "__main__":
    main()
