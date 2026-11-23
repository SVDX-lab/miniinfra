#!/usr/bin/env python3
"""CPU-only Chapter 25 checks: RNG preservation, bitwise parity, granularity."""

from __future__ import annotations

import torch
from torch import nn

from activation_checkpoint import (
    CheckpointedQwen3ForCausalLM,
    checkpoint_segment,
)
from mini_batch import batch_loss, pad_samples
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from training import build_optimizer, train_batch_step


def tiny_config(layers: int = 2) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )


def synthetic_samples(count: int) -> list:
    from assistant_loss import EncodedConversation, IGNORE_INDEX

    generator = torch.Generator().manual_seed(8)
    samples = []
    for index in range(count):
        length = 8 + index
        tokens = torch.randint(4, 64, (1, length), generator=generator)
        labels = tokens.clone()
        labels[0, :2] = IGNORE_INDEX
        samples.append(
            EncodedConversation(
                input_ids=tokens,
                attention_mask=torch.ones(1, length, dtype=torch.long),
                labels=labels,
                assistant_loss_mask=labels != IGNORE_INDEX,
                source_line=index + 1,
                rendered_text="",
                token_spans=(),
                loss_mode="assistant_only",
            )
        )
    return samples


def grads_and_loss(model, batch, seed: int):
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    logits = model(batch.input_ids, attention_mask=batch.attention_mask)
    loss = batch_loss(logits, batch.labels, "token_mean")
    loss.backward()
    selected = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return float(loss.item()), selected


def check_dropout_rng_preservation() -> None:
    """With dropout>0 the checkpoint must reproduce un-checkpointed grads."""

    # 先创建全部模块（初始化会消耗随机流），再在每次前向前重新播种，
    # 保证参照前向与重算拿到同一批 dropout 掩码。
    module = nn.Sequential(nn.Linear(16, 32), nn.Dropout(p=0.5), nn.Linear(32, 16), nn.Dropout(p=0.5))
    checkpointed_module = nn.Sequential(nn.Linear(16, 32), nn.Dropout(p=0.5), nn.Linear(32, 16), nn.Dropout(p=0.5))
    checkpointed_module.load_state_dict(module.state_dict())
    leaky = nn.Sequential(nn.Linear(16, 32), nn.Dropout(p=0.5), nn.Linear(32, 16), nn.Dropout(p=0.5))
    leaky.load_state_dict(module.state_dict())
    x = torch.randn(4, 16, requires_grad=True)

    torch.manual_seed(3)
    reference = module(x).pow(2).sum()
    reference.backward()
    reference_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }

    def segment(hidden):
        return checkpointed_module(hidden).pow(2).sum()

    torch.manual_seed(3)
    output = checkpoint_segment(segment, True, x)
    output.backward()

    for name, parameter in checkpointed_module.named_parameters():
        assert torch.equal(parameter.grad, reference_grads[name]), f"RNG divergence at {name}"

    # 反例：不保留 RNG 状态时，重算消费不同的随机流，梯度必须不同——
    # 这正是基础设施不能依赖“Qwen3 dropout 为零”的原因。
    class LeakyCheckpoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, function, *inputs):
            ctx.fn = function
            ctx.save_for_backward(*inputs)
            with torch.no_grad():
                return function(*inputs)

        @staticmethod
        def backward(ctx, grad_output):
            (tensor_input,) = ctx.saved_tensors
            detached = tensor_input.detach().requires_grad_(True)
            with torch.enable_grad():
                recomputed = ctx.fn(detached)
                torch.autograd.backward(recomputed, grad_output)
            return None, detached.grad

    torch.manual_seed(3)
    output = LeakyCheckpoint.apply(leaky, x)
    output.pow(2).sum().backward()
    diverged = any(
        not torch.equal(parameter.grad, reference_grads[name])
        for name, parameter in leaky.named_parameters()
    )
    assert diverged, "RNG-forgetting checkpoint should change dropout draws"


def check_bitwise_parity_on_tiny_qwen3() -> None:
    samples = synthetic_samples(4)
    batch = pad_samples(samples, pad_token_id=0)
    config = tiny_config()

    plain = Qwen3ForCausalLM(config)
    for segments in (1, 2):
        checkpointed = CheckpointedQwen3ForCausalLM(config, layers_per_segment=segments)
        checkpointed.load_state_dict(plain.state_dict())
        checkpointed.lm_head.weight = checkpointed.model.embed_tokens.weight

        plain_loss, plain_grads = grads_and_loss(plain, batch, seed=11)
        ckpt_loss, ckpt_grads = grads_and_loss(checkpointed, batch, seed=11)
        assert plain_loss == ckpt_loss, (plain_loss, ckpt_loss)
        for name, grad in plain_grads.items():
            assert torch.equal(grad, ckpt_grads[name]), f"grad mismatch at {name} (segments={segments})"

    # 两层模型上 layers_per_segment=3 必须被拒绝。
    try:
        CheckpointedQwen3ForCausalLM(config, layers_per_segment=3)
    except ValueError:
        pass
    else:
        raise AssertionError("indivisible layers_per_segment must raise")


def check_state_dict_compatibility() -> None:
    plain = Qwen3ForCausalLM(tiny_config())
    checkpointed = CheckpointedQwen3ForCausalLM(tiny_config(), layers_per_segment=1)
    result = checkpointed.load_state_dict(plain.state_dict(), strict=False)
    assert set(result.missing_keys) <= {"lm_head.weight"}, result.missing_keys
    assert not result.unexpected_keys, result.unexpected_keys


def check_full_step_parity() -> None:
    """One complete training step must be bitwise identical."""

    samples = synthetic_samples(4)
    batch = pad_samples(samples, pad_token_id=0)
    device = torch.device("cpu")
    plain = Qwen3ForCausalLM(tiny_config())
    checkpointed = CheckpointedQwen3ForCausalLM(tiny_config(), layers_per_segment=1)
    checkpointed.load_state_dict(plain.state_dict())
    checkpointed.lm_head.weight = checkpointed.model.embed_tokens.weight

    plain_metrics = train_batch_step(
        plain, build_optimizer(plain, 2e-5, 0.1), batch, device, 1.0, collect_telemetry=False
    )
    ckpt_metrics = train_batch_step(
        checkpointed, build_optimizer(checkpointed, 2e-5, 0.1), batch, device, 1.0, collect_telemetry=False
    )
    assert plain_metrics["loss"] == ckpt_metrics["loss"]
    assert plain_metrics["gradient_norm_before_clip"] == ckpt_metrics["gradient_norm_before_clip"]
    for (name, parameter), (_other, reference_parameter) in zip(
        plain.named_parameters(), checkpointed.named_parameters()
    ):
        assert torch.equal(parameter, reference_parameter), f"post-step mismatch at {name}"


def main() -> None:
    check_dropout_rng_preservation()
    check_bitwise_parity_on_tiny_qwen3()
    check_state_dict_compatibility()
    check_full_step_parity()
    print("chapter_25 smoke tests passed")


if __name__ == "__main__":
    main()
