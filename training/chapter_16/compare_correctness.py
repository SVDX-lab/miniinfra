#!/usr/bin/env python3
"""Check causal loss, gradients, masks, and the real handwritten Qwen3 model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from experiment_utils import environment_record, write_json
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from sft_core import (
    DATASET_FILE,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SHA256,
    IGNORE_INDEX,
    encode_conversation,
    load_jsonl_records,
    seed_everything,
)
from sft_core import shifted_cross_entropy
from training import MODEL_ID, MODEL_REVISION, load_tokenizer_and_model, move_batch


def explicit_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for position in range(labels.shape[1] - 1):
        target = int(labels[0, position + 1].item())
        if target == IGNORE_INDEX:
            continue
        terms.append(-torch.log_softmax(logits[0, position].float(), dim=-1)[target])
    if not terms:
        raise ValueError("no supervised next-token targets")
    return torch.stack(terms).mean()


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )


def tiny_model_checks() -> dict[str, float]:
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(tiny_config())
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    labels = torch.tensor([[1, 2, 3, IGNORE_INDEX]])

    logits = model(input_ids, attention_mask=attention_mask)
    vectorized = shifted_cross_entropy(logits, labels)
    explicit = explicit_token_loss(logits, labels)
    loss_error = float((vectorized - explicit).abs().item())
    torch.testing.assert_close(vectorized, explicit, atol=1e-6, rtol=1e-6)

    vectorized.backward()
    parameter = model.model.layers[0].mlp.down_proj.weight
    row, column = 0, 0
    analytical = float(parameter.grad[row, column].item())
    original = float(parameter[row, column].item())
    epsilon = 1e-3
    with torch.no_grad():
        parameter[row, column] = original + epsilon
    plus = float(
        shifted_cross_entropy(model(input_ids, attention_mask=attention_mask), labels).item()
    )
    with torch.no_grad():
        parameter[row, column] = original - epsilon
    minus = float(
        shifted_cross_entropy(model(input_ids, attention_mask=attention_mask), labels).item()
    )
    with torch.no_grad():
        parameter[row, column] = original
    numerical = (plus - minus) / (2 * epsilon)
    finite_difference_error = abs(analytical - numerical)
    tolerance = 2e-3 + 0.03 * abs(numerical)
    if finite_difference_error > tolerance:
        raise AssertionError(
            f"finite difference mismatch: analytical={analytical}, numerical={numerical}"
        )

    changed_padding = input_ids.clone()
    changed_padding[0, -1] = 17
    with torch.no_grad():
        original_logits = model(input_ids, attention_mask=attention_mask)
        changed_logits = model(changed_padding, attention_mask=attention_mask)
    padding_influence = float(
        (original_logits[:, :3] - changed_logits[:, :3]).abs().max().item()
    )
    if padding_influence != 0.0:
        raise AssertionError(f"right padding changed supervised logits by {padding_influence}")

    return {
        "explicit_loss": float(explicit.item()),
        "vectorized_loss": float(vectorized.item()),
        "loss_absolute_error": loss_error,
        "analytical_gradient": analytical,
        "numerical_gradient": numerical,
        "finite_difference_absolute_error": finite_difference_error,
        "padding_influence_max_absolute_error": padding_influence,
    }


def real_model_checks(data_path: Path, max_length: int) -> dict[str, float | int]:
    device = torch.device("cuda:0")
    tokenizer, model = load_tokenizer_and_model(device)
    record = load_jsonl_records(data_path)[0]
    sample = encode_conversation(tokenizer, record, max_length)
    batch = move_batch(sample, device)
    if sample.input_ids[0, sample.non_padding_tokens - 1].item() != tokenizer.eos_token_id:
        raise AssertionError("encoded conversation does not end with EOS")

    model.train()
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    loss = shifted_cross_entropy(logits, batch["labels"])
    if not torch.isfinite(loss):
        raise AssertionError("real-model loss is not finite")
    loss.backward()
    squared_norm = 0.0
    parameters_with_gradient = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameters_with_gradient += 1
        squared_norm += float(parameter.grad.detach().float().square().sum().item())
    gradient_norm = math.sqrt(squared_norm)
    if not math.isfinite(gradient_norm) or gradient_norm == 0.0:
        raise AssertionError(f"invalid gradient norm: {gradient_norm}")
    return {
        "loss": float(loss.item()),
        "gradient_norm": gradient_norm,
        "parameters_with_gradient": parameters_with_gradient,
        "source_line": sample.source_line,
        "physical_tokens": sample.physical_tokens,
        "non_padding_tokens": sample.non_padding_tokens,
        "supervised_tokens": sample.supervised_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter16_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--skip-real-model", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()

    seed_everything(args.seed)
    result: dict[str, object] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "file": DATASET_FILE,
            "sha256": DATASET_SHA256,
        },
        "seed": args.seed,
        "max_length": args.max_length,
        "dtype": "float32",
        "tiny_model": tiny_model_checks(),
        "status": "pass",
    }
    if not args.skip_real_model:
        if not torch.cuda.is_available():
            raise SystemExit("use --skip-real-model for a CPU-only check")
        result["real_model"] = real_model_checks(args.data, args.max_length)
    payload = environment_record()
    payload.update(result)
    write_json(args.output, payload)
    print("chapter_16 correctness checks passed")


if __name__ == "__main__":
    main()
