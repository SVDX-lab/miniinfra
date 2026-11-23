#!/usr/bin/env python3
"""Check Qwen3 template boundaries and compare the two FP32 objectives."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from jinja2 import Environment

from assistant_loss import (
    IGNORE_INDEX,
    assert_only_labels_differ,
    encode_conversation,
    supervised_token_indices,
)
from boundary_fixtures import BOUNDARY_CASES
from experiment_utils import environment_record, write_json
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything, shifted_cross_entropy
from training import MODEL_ID, MODEL_REVISION, load_tokenizer_and_model, move_batch


PROBE_PARAMETER = "model.layers.0.self_attn.q_norm.weight"


def _gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(squared)


def _optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )


def boundary_checks(tokenizer: Any) -> dict[str, object]:
    results: dict[str, object] = {}
    for name, record in BOUNDARY_CASES.items():
        baseline = encode_conversation(tokenizer, record, 256, "full_conversation")
        variant = encode_conversation(tokenizer, record, 256, "assistant_only")
        assert_only_labels_differ(baseline, variant)

        spans = []
        for span in variant.token_spans:
            active = [
                index
                for index in range(span.start, span.end)
                if index < variant.labels.shape[1] and variant.labels[0, index].item() != IGNORE_INDEX
            ]
            expected_count = span.end - span.start if span.supervised else 0
            if len(active) != expected_count:
                raise AssertionError(f"{name}: off-by-one in {span.role}/{span.kind}")
            spans.append(
                {
                    "role": span.role,
                    "kind": span.kind,
                    "start": span.start,
                    "end_exclusive": span.end,
                    "token_count": span.end - span.start,
                    "expected_supervised": span.supervised,
                    "active_label_indices": active,
                }
            )

        assistant_ends = [span for span in variant.token_spans if span.role == "assistant" and span.kind == "end"]
        if any(span.end - span.start != 1 for span in assistant_ends):
            raise AssertionError(f"{name}: <|im_end|> was not one atomic token")
        if name == "empty_assistant" and variant.supervised_tokens != 1:
            raise AssertionError("empty assistant must supervise exactly its real <|im_end|>")
        if name == "tool_call_and_response":
            tool_spans = [span for span in variant.token_spans if span.role == "tool"]
            if not tool_spans or any(span.supervised for span in tool_spans):
                raise AssertionError("tool responses must stay outside Assistant-only loss")

        results[name] = {
            "status": "pass",
            "input_ids_equal": torch.equal(baseline.input_ids, variant.input_ids),
            "attention_mask_equal": torch.equal(baseline.attention_mask, variant.attention_mask),
            "non_padding_tokens": variant.non_padding_tokens,
            "baseline_supervised_tokens": baseline.supervised_tokens,
            "variant_supervised_tokens": variant.supervised_tokens,
            "variant_supervised_indices": supervised_token_indices(variant),
            "spans": spans,
            "off_by_one_errors": 0,
        }
    return results


def official_tool_template_check(model_directory: str) -> dict[str, object]:
    """Compare one tool-call round against the pinned upstream Jinja template."""

    tokenizer_config = Path(model_directory) / "tokenizer_config.json"
    raw = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    template_text = raw["chat_template"]
    messages = BOUNDARY_CASES["tool_call_and_response"]["conversations"][:3]
    official = Environment().from_string(template_text).render(
        messages=messages,
        tools=None,
        add_generation_prompt=False,
    )
    from assistant_loss import render_qwen3_segments

    handwritten = "".join(segment.text for segment in render_qwen3_segments(messages))
    if handwritten != official:
        raise AssertionError("handwritten tool-call boundary differs from pinned Qwen3 chat template")
    return {
        "status": "pass",
        "exact_render_match": True,
        "chat_template_sha256": hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
        "messages_checked": len(messages),
    }


def dataset_mask_statistics(tokenizer: Any, data_path: Path, max_length: int) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    totals = {"physical": 0, "non_padding": 0, "baseline_supervised": 0, "variant_supervised": 0}
    for record in load_jsonl_records(data_path):
        baseline = encode_conversation(tokenizer, record, max_length, "full_conversation")
        variant = encode_conversation(tokenizer, record, max_length, "assistant_only")
        assert_only_labels_differ(baseline, variant)
        row = {
            "source_line": int(record["source_line"]),
            "physical_tokens": baseline.physical_tokens,
            "non_padding_tokens": baseline.non_padding_tokens,
            "baseline_supervised_tokens": baseline.supervised_tokens,
            "variant_supervised_tokens": variant.supervised_tokens,
            "baseline_supervision_ratio": baseline.supervision_ratio,
            "variant_supervision_ratio": variant.supervision_ratio,
        }
        samples.append(row)
        totals["physical"] += baseline.physical_tokens
        totals["non_padding"] += baseline.non_padding_tokens
        totals["baseline_supervised"] += baseline.supervised_tokens
        totals["variant_supervised"] += variant.supervised_tokens
    target_slots = sum(max(0, int(sample["non_padding_tokens"]) - 1) for sample in samples)
    return {
        "max_length": max_length,
        "sample_count": len(samples),
        "samples": samples,
        "totals": totals,
        "target_slots_after_shift": target_slots,
        "baseline_supervision_ratio": totals["baseline_supervised"] / target_slots,
        "variant_supervision_ratio": totals["variant_supervised"] / target_slots,
    }


def run_objective(
    model: torch.nn.Module,
    sample: Any,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    model.train()
    optimizer = _optimizer(model, learning_rate, weight_decay)
    batch = move_batch(sample, device)
    initial_probe = dict(model.named_parameters())[PROBE_PARAMETER].detach().float().cpu().clone()
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    loss = shifted_cross_entropy(logits, batch["labels"])
    loss.backward()
    probe_parameter = dict(model.named_parameters())[PROBE_PARAMETER]
    probe_gradient = probe_parameter.grad.detach().float().cpu().clone()
    gradient_norm = _gradient_norm(model)
    clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    delta = probe_parameter.detach().float().cpu() - initial_probe
    return {
        "loss": float(loss.detach().item()),
        "gradient_norm_before_clip": gradient_norm,
        "clip_returned_norm": float(clipped_norm.detach().item()),
        "probe_gradient": probe_gradient,
        "probe_delta": delta,
        "logits_probe": logits.detach()[0, :4, :128].float().cpu(),
        "supervised_tokens": sample.supervised_tokens,
        "non_padding_tokens": sample.non_padding_tokens,
        "supervision_ratio": sample.supervision_ratio,
    }


def real_model_checks(
    data_path: Path,
    max_length: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, object]:
    device = torch.device("cuda:0")
    tokenizer, baseline_model = load_tokenizer_and_model(device)
    record = load_jsonl_records(data_path)[0]
    baseline_sample = encode_conversation(tokenizer, record, max_length, "full_conversation")
    variant_sample = encode_conversation(tokenizer, record, max_length, "assistant_only")
    assert_only_labels_differ(baseline_sample, variant_sample)

    seed_everything(seed)
    baseline = run_objective(baseline_model, baseline_sample, device, learning_rate, weight_decay)
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    seed_everything(seed)
    _, variant_model = load_tokenizer_and_model(device)
    variant = run_objective(variant_model, variant_sample, device, learning_rate, weight_decay)

    logits_error = float((baseline["logits_probe"] - variant["logits_probe"]).abs().max().item())
    if logits_error != 0.0:
        raise AssertionError(f"labels changed forward logits by {logits_error}")
    gradient_cosine = float(F.cosine_similarity(baseline["probe_gradient"].flatten(), variant["probe_gradient"].flatten(), dim=0).item())
    update_cosine = float(F.cosine_similarity(baseline["probe_delta"].flatten(), variant["probe_delta"].flatten(), dim=0).item())
    update_max_error = float((baseline["probe_delta"] - variant["probe_delta"]).abs().max().item())
    if not all(math.isfinite(value) for value in (gradient_cosine, update_cosine, update_max_error)):
        raise AssertionError("non-finite gradient or update comparison")

    return {
        "source_line": int(record["source_line"]),
        "input_ids_equal": torch.equal(baseline_sample.input_ids, variant_sample.input_ids),
        "attention_mask_equal": torch.equal(baseline_sample.attention_mask, variant_sample.attention_mask),
        "forward_logits_probe_max_absolute_error": logits_error,
        "probe_parameter": PROBE_PARAMETER,
        "gradient_cosine_similarity": gradient_cosine,
        "one_step_update_cosine_similarity": update_cosine,
        "one_step_update_max_absolute_difference": update_max_error,
        "baseline": {key: value for key, value in baseline.items() if not torch.is_tensor(value)},
        "variant": {key: value for key, value in variant.items() if not torch.is_tensor(value)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter17_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--stats-max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--skip-real-model", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.skip_real_model:
        from smoke_test import BoundaryTokenizer

        tokenizer: Any = BoundaryTokenizer()
    else:
        if not torch.cuda.is_available():
            raise SystemExit("use --skip-real-model for a CPU-only boundary check")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        model_directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
        tokenizer = Qwen3Tokenizer(model_directory)

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
            },
            "dtype": "float32",
            "formal_dtype": "float32",
            "seed": args.seed,
            "max_length": args.max_length,
            "boundary_cases": boundary_checks(tokenizer),
            "official_tool_template_check": official_tool_template_check(model_directory)
            if not args.skip_real_model
            else {"status": "not_run", "reason": "real tokenizer config not loaded"},
            "dataset_mask_statistics": dataset_mask_statistics(tokenizer, args.data, args.stats_max_length)
            if args.data.is_file()
            else {"status": "not_run", "reason": f"missing {args.data}"},
            "status": "pass",
        }
    )
    if not args.skip_real_model:
        payload["real_model"] = real_model_checks(
            args.data,
            args.max_length,
            args.seed,
            args.learning_rate,
            args.weight_decay,
        )
    write_json(args.output, payload)
    print("chapter_17 correctness checks passed")


if __name__ == "__main__":
    main()
