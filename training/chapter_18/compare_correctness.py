#!/usr/bin/env python3
"""Compare serial accumulation with one padded FP32 mini-batch on a real Qwen3 model."""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from assistant_loss import encode_conversation
from experiment_utils import environment_record, write_json
from mini_batch import pad_samples, trim_sample
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import MODEL_ID, MODEL_REVISION, build_optimizer, load_tokenizer_and_model, move_batch
from mini_batch import batched_backward, serial_backward


PROBE_PARAMETER = "model.layers.0.self_attn.q_norm.weight"


def _norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(float(parameter.grad.double().square().sum()) for parameter in model.parameters() if parameter.grad is not None))


def _run(
    execution: str,
    host_batch,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    reduction: str,
    max_grad_norm: float,
) -> dict[str, object]:
    seed_everything(seed)
    device = torch.device("cuda:0")
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    batch = move_batch(host_batch, device)
    optimizer.zero_grad(set_to_none=True)
    if execution == "serial":
        loss, logits = serial_backward(model, batch, reduction)
        logits_probe = [tensor[0, : host_batch.lengths[index], :32].cpu() for index, tensor in enumerate(logits)]
    else:
        loss, batch_logits = batched_backward(model, batch, reduction)
        logits_probe = [batch_logits[index, :length, :32].cpu() for index, length in enumerate(host_batch.lengths)]
    parameter = dict(model.named_parameters())[PROBE_PARAMETER]
    gradient = parameter.grad.detach().float().cpu().clone()
    gradient_norm = _norm(model)
    before = parameter.detach().float().cpu().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    update = parameter.detach().float().cpu() - before
    probe_sum = float(parameter.detach().double().sum().cpu())
    result = {
        "execution": execution,
        "loss": float(loss.item()),
        "gradient_norm": gradient_norm,
        "probe_gradient": gradient.tolist(),
        "probe_update": update.tolist(),
        "probe_parameter_sum_after_step": probe_sum,
        "logits_probe": [tensor.tolist() for tensor in logits_probe],
    }
    del optimizer, model, batch
    gc.collect(); torch.cuda.empty_cache()
    return result


def _flatten(records: list) -> torch.Tensor:
    return torch.tensor(records, dtype=torch.float64).flatten()


def _flatten_logits(records: list) -> torch.Tensor:
    return torch.cat([torch.tensor(record, dtype=torch.float64).flatten() for record in records])


def comparison(serial: dict[str, object], batched: dict[str, object]) -> dict[str, float]:
    sg = _flatten(serial["probe_gradient"]); bg = _flatten(batched["probe_gradient"])
    su = _flatten(serial["probe_update"]); bu = _flatten(batched["probe_update"])
    sl = _flatten_logits(serial["logits_probe"]); bl = _flatten_logits(batched["logits_probe"])
    return {
        "loss_absolute_error": abs(float(serial["loss"]) - float(batched["loss"])),
        "loss_relative_error": abs(float(serial["loss"]) - float(batched["loss"])) / max(abs(float(serial["loss"])), 1e-12),
        "logits_max_absolute_error": float((sl - bl).abs().max()),
        "logits_mean_absolute_error": float((sl - bl).abs().mean()),
        "probe_gradient_max_absolute_error": float((sg - bg).abs().max()),
        "probe_gradient_relative_l2_error": float(torch.linalg.vector_norm(sg - bg) / torch.linalg.vector_norm(sg)),
        "probe_gradient_cosine_similarity": float(F.cosine_similarity(sg, bg, dim=0)),
        "gradient_norm_absolute_error": abs(float(serial["gradient_norm"]) - float(batched["gradient_norm"])),
        "probe_update_max_absolute_error": float((su - bu).abs().max()),
        "probe_parameter_sum_absolute_error": abs(float(serial["probe_parameter_sum_after_step"]) - float(batched["probe_parameter_sum_after_step"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter18_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sample-cursor", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--reduction", choices=["token_mean", "sample_mean"], default="token_mean")
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    parser.add_argument("--skip-real-model", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    records = load_jsonl_records(args.data)
    selected_records = records[args.sample_cursor : args.sample_cursor + args.batch_size]
    if len(selected_records) < args.batch_size:
        parser.error("data does not contain enough records")
    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "seed": args.seed, "batch_size": args.batch_size, "max_length": args.max_length, "sample_cursor": args.sample_cursor,
        "max_grad_norm": args.max_grad_norm,
        "reduction": args.reduction, "status": "not_run" if args.skip_real_model else "pass",
    })
    if not args.skip_real_model:
        if not torch.cuda.is_available():
            raise SystemExit("one CUDA GPU is required unless --skip-real-model is used")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer = Qwen3Tokenizer(resolve_model_directory(MODEL_ID, MODEL_REVISION))
        samples = [trim_sample(encode_conversation(tokenizer, record, args.max_length, "assistant_only")) for record in selected_records]
        group = pad_samples(samples, tokenizer.pad_token_id)
        serial = _run("serial", group, args.seed, args.learning_rate, args.weight_decay, args.reduction, args.max_grad_norm)
        batched = _run("batched", group, args.seed, args.learning_rate, args.weight_decay, args.reduction, args.max_grad_norm)
        metrics = comparison(serial, batched)
        if metrics["loss_absolute_error"] > 2e-4 or metrics["probe_gradient_relative_l2_error"] > 2e-3 or metrics["probe_update_max_absolute_error"] > 2e-6:
            raise AssertionError(f"serial and batched paths exceed FP32 tolerance: {metrics}")
        payload["sample_group"] = {
            "source_lines": list(group.source_lines), "lengths": list(group.lengths),
            "physical_tokens_serial": sum(group.lengths), "physical_tokens_batched": group.physical_tokens,
            "non_padding_tokens": group.non_padding_tokens, "supervised_tokens": group.supervised_tokens,
            "padding_tokens_batched": group.padding_tokens, "padding_ratio_batched": group.padding_ratio,
        }
        payload["paths"] = {"serial": serial, "batched": batched}
        payload["comparison"] = metrics
    write_json(args.output, payload)
    print("chapter_18 correctness checks passed")


if __name__ == "__main__":
    main()
