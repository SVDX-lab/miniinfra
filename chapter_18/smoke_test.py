#!/usr/bin/env python3
"""CPU-only fixed-mini-batch shape, reduction, gradient, and update checks."""

from __future__ import annotations

import copy

import torch

from assistant_loss import EncodedConversation, IGNORE_INDEX
from mini_batch import batched_backward, pad_samples, serial_backward
from qwen3_model import Qwen3Config, Qwen3ForCausalLM


def sample(ids: list[int], active_from: int, source_line: int) -> EncodedConversation:
    labels = [IGNORE_INDEX if index < active_from else token for index, token in enumerate(ids)]
    ones = [1] * len(ids)
    return EncodedConversation(
        input_ids=torch.tensor([ids]), attention_mask=torch.tensor([ones]), labels=torch.tensor([labels]),
        assistant_loss_mask=torch.tensor([[label != IGNORE_INDEX for label in labels]]),
        source_line=source_line, rendered_text="", token_spans=(), loss_mode="assistant_only",
    )


def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(17)
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        rms_norm_eps=1e-6, rope_theta=10_000.0, torch_dtype="float32", tie_word_embeddings=True,
    ))


def compare(reduction: str) -> None:
    group = pad_samples([sample([1, 2, 3, 4, 5], 2, 11), sample([6, 7, 8, 9, 10, 11, 12], 4, 12)], 0)
    assert group.input_ids.shape == (2, 7)
    assert group.lengths == (5, 7)
    assert group.padding_tokens == 2 and abs(group.padding_ratio - 1 / 7) < 1e-12
    serial = tiny_model()
    batched = copy.deepcopy(serial)
    serial_loss, _ = serial_backward(serial, group, reduction)
    batched_loss, _ = batched_backward(batched, group, reduction)
    assert torch.allclose(serial_loss, batched_loss, atol=2e-6, rtol=2e-6)
    for left, right in zip(serial.parameters(), batched.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=3e-6, rtol=3e-5)
    left_optimizer = torch.optim.SGD(serial.parameters(), lr=1e-3)
    right_optimizer = torch.optim.SGD(batched.parameters(), lr=1e-3)
    left_optimizer.step(); right_optimizer.step()
    for left, right in zip(serial.parameters(), batched.parameters()):
        assert torch.allclose(left, right, atol=3e-8, rtol=3e-6)


def main() -> None:
    compare("token_mean")
    compare("sample_mean")
    print("chapter_18 smoke tests passed")


if __name__ == "__main__":
    main()
