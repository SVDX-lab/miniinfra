#!/usr/bin/env python3
"""CPU-only checks for Chapter 16 data, causal shift, optimizer, and state."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from sft_core import (
    IGNORE_INDEX,
    is_plain_conversation,
    load_jsonl_records,
    shifted_cross_entropy,
)
from qwen3_model import Qwen3Config, Qwen3ForCausalLM


def test_shifted_cross_entropy() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [1.0, 0.0, 4.0],
                [2.0, 1.0, 0.0],
            ]
        ]
    )
    labels = torch.tensor([[0, 1, 2, IGNORE_INDEX]])
    actual = shifted_cross_entropy(logits, labels)
    expected = (
        F.cross_entropy(logits[:, 0, :], labels[:, 1], reduction="sum")
        + F.cross_entropy(logits[:, 1, :], labels[:, 2], reduction="sum")
    ) / 2
    torch.testing.assert_close(actual, expected)


def test_plain_record_filter() -> None:
    plain = {"conversations": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
    tool = {"conversations": [{"role": "assistant", "content": "", "tool_calls": "[]"}]}
    assert is_plain_conversation(plain)
    assert not is_plain_conversation(tool)


def test_jsonl_and_checkpoint_state() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.jsonl"
        path.write_text(
            json.dumps(
                {"conversations": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
            )
            + "\n",
            encoding="utf-8",
        )
        records = load_jsonl_records(path)
        assert records[0]["source_line"] == 1

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = model(torch.ones(1, 2)).square().mean()
        loss.backward()
        optimizer.step()
        assert optimizer.state

        checkpoint = Path(directory) / "state.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 1}, checkpoint)
        restored = torch.nn.Linear(2, 2)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restored.load_state_dict(state["model"])
        restored_optimizer.load_state_dict(state["optimizer"])
        assert state["step"] == 1
        for left, right in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(left, right)
        assert restored_optimizer.state


def test_tiny_qwen_forward_and_backward() -> None:
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
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    labels = torch.tensor([[1, 2, 3, IGNORE_INDEX]])
    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (1, 4, 32)
    loss = shifted_cross_entropy(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def main() -> None:
    test_shifted_cross_entropy()
    test_plain_record_filter()
    test_jsonl_and_checkpoint_state()
    test_tiny_qwen_forward_and_backward()
    print("chapter_16 smoke tests passed")


if __name__ == "__main__":
    main()
