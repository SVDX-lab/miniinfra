#!/usr/bin/env python3
"""CPU-only Chapter 23 checks: dtype policy, loss masking, state dtypes."""

from __future__ import annotations

import torch

from assistant_loss import EncodedConversation, IGNORE_INDEX
from mini_batch import batch_loss, pad_samples
from mixed_precision import (
    assert_no_nan_or_inf,
    autocast_policy_census,
    compare_trajectories,
    probe_config,
    state_dtype_census,
)
from qwen3_model import Qwen3ForCausalLM
from training import (
    PRECISION_MODES,
    build_optimizer,
    make_batch_plan,
    train_batch_step,
)


def synthetic_encoded_samples(config, count: int) -> list[EncodedConversation]:
    """Short hand-built samples with partial supervision and no padding."""

    generator = torch.Generator().manual_seed(8)
    samples = []
    for index in range(count):
        length = 8 + index
        tokens = torch.randint(4, config.vocab_size, (1, length), generator=generator)
        mask = torch.ones(1, length, dtype=torch.long)
        labels = tokens.clone()
        # 前两个非监督位清成 ignore_index，模拟 Assistant-only 前缀。
        labels[0, :2] = IGNORE_INDEX
        samples.append(
            EncodedConversation(
                input_ids=tokens,
                attention_mask=mask,
                labels=labels,
                assistant_loss_mask=labels != IGNORE_INDEX,
                source_line=index + 1,
                rendered_text="",
                token_spans=(),
                loss_mode="assistant_only",
            )
        )
    return samples


def tiny_inputs(config, batch_size: int = 2, sequence_length: int = 12):
    generator = torch.Generator().manual_seed(8)
    input_ids = torch.randint(4, config.vocab_size, (batch_size, sequence_length), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    # 一个样本全监督，另一个样本只保留开头两个监督位。
    labels[1, 2:] = IGNORE_INDEX
    return input_ids, attention_mask, labels


def check_autocast_census_on_cpu() -> None:
    config = probe_config()
    model = Qwen3ForCausalLM(config)
    input_ids, attention_mask, labels = tiny_inputs(config)

    def forward_and_loss():
        logits = model(input_ids, attention_mask=attention_mask)
        return batch_loss(logits, labels, "token_mean")

    plain = autocast_policy_census(model, forward_and_loss, torch.autocast("cpu", enabled=False))
    census = autocast_policy_census(model, forward_and_loss, torch.autocast("cpu", dtype=torch.bfloat16))

    assert plain["<loss>"] == "torch.float32", plain["<loss>"]
    assert census["<loss>"] == "torch.float32", f"loss must stay FP32 under autocast, got {census['<loss>']}"
    assert census["model.embed_tokens"] == "torch.float32", "embedding lookup is not an autocast op"
    linear_outputs = [dtype for name, dtype in census.items() if name.endswith(("proj", "lm_head"))]
    assert linear_outputs and all(dtype == "torch.bfloat16" for dtype in linear_outputs), census
    assert census["model.layers.0.self_attn"] == "torch.bfloat16", census
    # 参数本身不因 autocast 改变 dtype。
    states = state_dtype_census(model, None)
    assert states["parameters"] == {"torch.float32": sum(1 for _ in model.parameters())}


def check_fp32_mode_is_plain_body() -> None:
    config = probe_config()
    samples = synthetic_encoded_samples(config, count=4)
    batches = pad_samples(samples, pad_token_id=0)
    model = Qwen3ForCausalLM(config)
    reference_model = Qwen3ForCausalLM(config)
    reference_model.load_state_dict(model.state_dict())
    optimizer = build_optimizer(model, 2e-5, 0.1)
    reference_optimizer = build_optimizer(reference_model, 2e-5, 0.1)

    through_context = train_batch_step(
        model, optimizer, batches, torch.device("cpu"), 1.0, mode="fp32", collect_telemetry=False
    )
    # 参照体直接写一遍第 22 期步体，不经过任何 autocast 上下文。
    reference_optimizer.zero_grad(set_to_none=True)
    logits = reference_model(batches.input_ids, attention_mask=batches.attention_mask)
    loss = batch_loss(logits, batches.labels, "token_mean")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0)
    reference_optimizer.step()

    assert through_context["loss"] == float(loss.item())
    assert through_context["gradient_norm_before_clip"] == float(grad_norm.item())
    for (name, parameter), (_other, reference_parameter) in zip(
        model.named_parameters(), reference_model.named_parameters()
    ):
        assert torch.equal(parameter, reference_parameter), f"FP32 mode diverged at {name}"


def check_mask_accounting_under_both_modes() -> None:
    config = probe_config()
    samples = synthetic_encoded_samples(config, count=4)
    batch = pad_samples(samples, pad_token_id=0)
    supervised = batch.supervised_tokens
    assert supervised == sum(batch.supervised_tokens_per_sample)
    model = Qwen3ForCausalLM(config)
    losses = {}
    for mode in PRECISION_MODES:
        fresh = Qwen3ForCausalLM(config)
        fresh.load_state_dict(model.state_dict())
        result = train_batch_step(
            fresh, build_optimizer(fresh, 2e-5, 0.1), batch, torch.device("cpu"),
            1.0, mode=mode, collect_telemetry=False,
        )
        assert result["supervised_tokens"] == supervised
        losses[mode] = result["loss"]
        assert_no_nan_or_inf(fresh)
    # 同一 FP32 权重下两种模式都基于同一批监督位；差异只允许来自 dtype。
    assert abs(losses["fp32"] - losses["bf16_autocast"]) < 0.25, losses


def check_batch_plan_covers_corpus_once() -> None:
    plan = make_batch_plan(num_samples=32, batch_size=4, seed=20260912)
    assert len(plan) == 8
    flat = [index for batch in plan for index in batch]
    assert sorted(flat) == list(range(32))
    assert plan == make_batch_plan(num_samples=32, batch_size=4, seed=20260912)


def check_trajectory_helper() -> None:
    summary = compare_trajectories([1.0, 2.0, 3.0], [1.0, 2.5, 3.0])
    assert summary["max_abs_error"] == 0.5
    assert summary["steps"] == 3
    try:
        compare_trajectories([1.0], [1.0, 2.0])
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise")


def main() -> None:
    check_autocast_census_on_cpu()
    check_fp32_mode_is_plain_body()
    check_mask_accounting_under_both_modes()
    check_batch_plan_covers_corpus_once()
    check_trajectory_helper()
    print("chapter_23 smoke tests passed")


if __name__ == "__main__":
    main()
