#!/usr/bin/env python3
"""CPU-only Chapter 28 checks: compile parity, stats plumbing, pad helper."""

from __future__ import annotations

import torch

from compiled_step import dynamo_stats, reset_dynamo_stats
from mini_batch import pad_samples
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from training import build_optimizer, pad_batch_to, train_batch_step


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )


def synthetic_batch(sequence_length: int, generator: torch.Generator):
    tokens = torch.randint(4, 64, (2, sequence_length), generator=generator)
    labels = tokens.clone()
    labels[:, :2] = -100
    return tokens, labels


def check_padded_batch_helper() -> None:
    generator = torch.Generator().manual_seed(2)
    tokens, labels = synthetic_batch(10, generator)
    from assistant_loss import EncodedConversation, IGNORE_INDEX

    samples = [
        EncodedConversation(
            input_ids=tokens[row].unsqueeze(0),
            attention_mask=torch.ones(1, 10, dtype=torch.long),
            labels=labels[row].unsqueeze(0),
            assistant_loss_mask=labels[row].unsqueeze(0) != IGNORE_INDEX,
            source_line=row + 1,
            rendered_text="",
            token_spans=(),
            loss_mode="assistant_only",
        )
        for row in range(2)
    ]
    natural = pad_samples(samples, pad_token_id=0)
    fixed = pad_batch_to(natural, target_length=16, pad_token_id=0)
    assert fixed.input_ids.shape == (2, 16)
    assert fixed.supervised_tokens == natural.supervised_tokens
    assert int(fixed.attention_mask[:, 10:].sum()) == 0
    assert int((fixed.labels[:, 10:] == IGNORE_INDEX).all())
    assert torch.equal(fixed.input_ids[:, :10], natural.input_ids)
    try:
        pad_batch_to(natural, target_length=8, pad_token_id=0)
    except ValueError:
        return
    raise AssertionError("padding to a shorter length must raise")


def check_compiled_parity_on_cpu() -> None:
    """Compiled forward/backward must track eager within FP32 tolerance."""

    generator = torch.Generator().manual_seed(3)
    tokens, labels = synthetic_batch(12, generator)
    mask = torch.ones_like(tokens)

    eager = Qwen3ForCausalLM(tiny_config())
    compiled_inner = Qwen3ForCausalLM(tiny_config())
    compiled_inner.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_inner)

    reset_dynamo_stats()
    eager_optimizer = build_optimizer(eager, 2e-5, 0.1)
    compiled_optimizer = build_optimizer(compiled_inner, 2e-5, 0.1)

    eager_optimizer.zero_grad(set_to_none=True)
    logits_e = eager(tokens)
    loss_e = torch.nn.functional.cross_entropy(
        logits_e[:, :-1, :].reshape(-1, logits_e.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    loss_e.backward()

    compiled_optimizer.zero_grad(set_to_none=True)
    logits_c = compiled(tokens)
    loss_c = torch.nn.functional.cross_entropy(
        logits_c[:, :-1, :].reshape(-1, logits_c.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    loss_c.backward()

    rel = float(abs(loss_e.item() - loss_c.item()) / abs(loss_e.item()))
    assert rel < 1e-5, f"compiled loss rel error {rel}"
    for (name, parameter_e), (_other, parameter_c) in zip(
        eager.named_parameters(), compiled_inner.named_parameters()
    ):
        if parameter_e.grad is None or parameter_c.grad is None:
            raise AssertionError(f"missing grad at {name}")
        error = (parameter_e.grad - parameter_c.grad).abs().max()
        assert float(error) < 1e-5, f"grad mismatch at {name}: {float(error)}"
    stats = dynamo_stats()
    assert stats["unique_graphs"] >= 1, stats
    assert not stats["graph_breaks"], stats


def check_compiled_step_parity() -> None:
    """One full training step through the shared body stays within tolerance."""

    generator = torch.Generator().manual_seed(4)
    tokens, labels = synthetic_batch(12, generator)
    from assistant_loss import EncodedConversation, IGNORE_INDEX

    samples = [
        EncodedConversation(
            input_ids=tokens[row].unsqueeze(0),
            attention_mask=torch.ones(1, 12, dtype=torch.long),
            labels=labels[row].unsqueeze(0),
            assistant_loss_mask=labels[row].unsqueeze(0) != IGNORE_INDEX,
            source_line=row + 1,
            rendered_text="",
            token_spans=(),
            loss_mode="assistant_only",
        )
        for row in range(2)
    ]
    from mini_batch import pad_samples as pad

    batch = pad(samples, pad_token_id=0)
    device = torch.device("cpu")
    eager = Qwen3ForCausalLM(tiny_config())
    compiled_inner = Qwen3ForCausalLM(tiny_config())
    compiled_inner.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_inner)

    metrics_e = train_batch_step(eager, build_optimizer(eager, 2e-5, 0.1), batch, device, 1.0, collect_telemetry=False)
    metrics_c = train_batch_step(compiled, build_optimizer(compiled_inner, 2e-5, 0.1), batch, device, 1.0, collect_telemetry=False)
    rel = abs(metrics_e["loss"] - metrics_c["loss"]) / abs(metrics_e["loss"])
    assert rel < 1e-5, (metrics_e["loss"], metrics_c["loss"])
    norm_rel = abs(metrics_e["gradient_norm_before_clip"] - metrics_c["gradient_norm_before_clip"]) / metrics_e["gradient_norm_before_clip"]
    assert norm_rel < 1e-4, (metrics_e["gradient_norm_before_clip"], metrics_c["gradient_norm_before_clip"])


def main() -> None:
    check_padded_batch_helper()
    check_compiled_parity_on_cpu()
    check_compiled_step_parity()
    print("chapter_28 smoke tests passed")


if __name__ == "__main__":
    main()
