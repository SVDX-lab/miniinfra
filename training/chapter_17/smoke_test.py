#!/usr/bin/env python3
"""CPU-only boundary, label, and tiny-model checks for Chapter 17."""

from __future__ import annotations

import re

import torch

from assistant_loss import IM_END, IM_START, assert_only_labels_differ, encode_conversation
from boundary_fixtures import BOUNDARY_CASES
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from sft_core import IGNORE_INDEX, shifted_cross_entropy


class BoundaryTokenizer:
    """Deterministic tokenizer that keeps Qwen special tokens atomic."""

    def __init__(self) -> None:
        self.eos_token_id = 1
        self.pad_token_id = 0
        self._ordinary: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in re.split(f"({re.escape(IM_START)}|{re.escape(IM_END)})", text):
            if not part:
                continue
            if part == IM_START:
                ids.append(2)
            elif part == IM_END:
                ids.append(self.eos_token_id)
            else:
                for character in part:
                    if character not in self._ordinary:
                        self._ordinary[character] = len(self._ordinary) + 3
                    ids.append(self._ordinary[character])
        return ids


def check_case(name: str, record: dict[str, object]) -> dict[str, int]:
    tokenizer = BoundaryTokenizer()
    baseline = encode_conversation(tokenizer, record, 512, "full_conversation")
    variant = encode_conversation(tokenizer, record, 512, "assistant_only")
    assert_only_labels_differ(baseline, variant)

    expected = {
        index
        for span in variant.token_spans
        if span.supervised
        for index in range(span.start, span.end)
        if index > 0
    }
    actual = {
        index
        for index, label in enumerate(variant.labels[0].tolist())
        if label != IGNORE_INDEX and index > 0
    }
    assert actual == expected, (name, sorted(actual ^ expected))

    assistant_headers = [span for span in variant.token_spans if span.role == "assistant" and span.kind == "header"]
    assistant_ends = [span for span in variant.token_spans if span.role == "assistant" and span.kind == "end"]
    assert all(not span.supervised for span in assistant_headers)
    assert all(span.supervised and span.end - span.start == 1 for span in assistant_ends)
    for header in assistant_headers:
        assert variant.labels[0, header.end - 1].item() == IGNORE_INDEX

    if name == "multi_turn":
        assert len(assistant_ends) == 2
    elif name == "system_prompt":
        assert all(
            not span.supervised for span in variant.token_spans if span.role == "system"
        )
    elif name == "tool_call_and_response":
        assert "<tool_call>" in variant.rendered_text
        assert "<tool_response>" in variant.rendered_text
        assert all(not span.supervised for span in variant.token_spans if span.role == "tool")
        tool_call_content = [
            span
            for span in variant.token_spans
            if span.role == "assistant" and span.kind == "content" and span.end > span.start
        ][0]
        assert tool_call_content.supervised
    elif name == "empty_assistant":
        assert variant.supervised_tokens == 1

    return {
        "non_padding_tokens": variant.non_padding_tokens,
        "baseline_supervised_tokens": baseline.supervised_tokens,
        "variant_supervised_tokens": variant.supervised_tokens,
        "assistant_messages": len(assistant_ends),
    }


def tiny_model_check() -> None:
    torch.manual_seed(7)
    config = Qwen3Config(
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
    model = Qwen3ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    baseline_labels = torch.tensor([[1, 2, 3, 4, 5, IGNORE_INDEX]])
    variant_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, IGNORE_INDEX]])
    logits = model(input_ids, attention_mask=attention_mask)
    baseline_loss = shifted_cross_entropy(logits, baseline_labels)
    variant_loss = shifted_cross_entropy(logits, variant_labels)
    baseline_loss.backward(retain_graph=True)
    baseline_gradient = model.model.layers[0].mlp.down_proj.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    variant_loss.backward()
    variant_gradient = model.model.layers[0].mlp.down_proj.weight.grad
    assert torch.isfinite(baseline_loss) and torch.isfinite(variant_loss)
    assert not torch.equal(baseline_gradient, variant_gradient)


def main() -> None:
    summaries = {name: check_case(name, record) for name, record in BOUNDARY_CASES.items()}
    tiny_model_check()
    print("chapter_17 smoke tests passed")
    for name, summary in summaries.items():
        print(name, summary)


if __name__ == "__main__":
    main()
