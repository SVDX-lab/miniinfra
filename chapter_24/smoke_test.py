#!/usr/bin/env python3
"""CPU-only Chapter 24 checks: weighting math, plans, padding accounting."""

from __future__ import annotations

import torch

from assistant_loss import EncodedConversation, IGNORE_INDEX
from gradient_accumulation import accumulated_step, supervised_token_count, token_sum_loss
from mini_batch import batch_loss, pad_samples
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from training import (
    build_optimizer,
    make_batch_plan,
    make_step_plan,
    split_effective_batch,
    to_device_batch,
    train_batch_step,
)


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


def synthetic_sample(length: int, supervised_from: int, source_line: int) -> EncodedConversation:
    generator = torch.Generator().manual_seed(source_line)
    tokens = torch.randint(4, 64, (1, length), generator=generator)
    labels = tokens.clone()
    labels[0, :supervised_from] = IGNORE_INDEX
    return EncodedConversation(
        input_ids=tokens,
        attention_mask=torch.ones(1, length, dtype=torch.long),
        labels=labels,
        assistant_loss_mask=labels != IGNORE_INDEX,
        source_line=source_line,
        rendered_text="",
        token_spans=(),
        loss_mode="assistant_only",
    )


def tiny_model() -> Qwen3ForCausalLM:
    generator = torch.Generator().manual_seed(7)
    model = Qwen3ForCausalLM(tiny_config())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.5, 0.5, generator=generator)
    return model


def check_weighting_math_direct() -> None:
    """token_sum_loss divided by total must equal the union token_mean."""

    config = tiny_config()
    samples = [synthetic_sample(8 + index, 2 + index, index + 1) for index in range(4)]
    batch = pad_samples(samples, pad_token_id=0)
    model = tiny_model()
    logits = model(batch.input_ids, attention_mask=batch.attention_mask)
    union_mean = batch_loss(logits, batch.labels, "token_mean")
    total = supervised_token_count(batch.labels)
    direct_sum = token_sum_loss(logits, batch.labels) / total
    assert torch.allclose(union_mean, direct_sum, rtol=1e-6), (union_mean, direct_sum)


def check_accumulated_gradients_match_big_batch() -> None:
    """2+2 accumulation with *uneven* supervised counts still matches."""

    config = tiny_config()
    # 监督 Token 数故意不均匀：微批之间 5:13 的比例会放大加权错误。
    samples = [
        synthetic_sample(8, 4, 1),
        synthetic_sample(9, 7, 2),
        synthetic_sample(12, 2, 3),
        synthetic_sample(16, 2, 4),
    ]
    members = tuple(range(4))
    big = pad_samples([samples[i] for i in members], pad_token_id=0)

    model = tiny_model()
    reference_model = tiny_model()
    reference_model.load_state_dict(model.state_dict())
    optimizer = build_optimizer(model, 2e-5, 0.1)
    reference_optimizer = build_optimizer(reference_model, 2e-5, 0.1)
    device = torch.device("cpu")

    accumulated_step(
        model, optimizer, split_effective_batch(samples, members, 2, 0), device, 1.0
    )
    train_batch_step(reference_model, reference_optimizer, big, device, 1.0, collect_telemetry=False)

    worst_rel = 0.0
    for (name, parameter), (_other, reference_parameter) in zip(
        model.named_parameters(), reference_model.named_parameters()
    ):
        if parameter.grad is None or reference_parameter.grad is None:
            raise AssertionError(f"missing gradient at {name}")
        rel = (parameter.grad - reference_parameter.grad).abs().max() / reference_parameter.grad.abs().max().clamp_min(1e-12)
        worst_rel = max(worst_rel, float(rel))
        assert float(rel) < 1e-5, f"gradient mismatch at {name}: rel={float(rel)}"
    print(f"  accumulated-vs-big worst grad rel error: {worst_rel:.2e}")


def check_mean_of_means_diverges() -> None:
    """The naive objective (average of micro means) must NOT match."""

    config = tiny_config()
    samples = [synthetic_sample(8, 6, 1), synthetic_sample(16, 2, 2)]
    big = pad_samples(samples, pad_token_id=0)
    micro_batches = split_effective_batch(samples, (0, 1), 1, 0)
    device = torch.device("cpu")

    model = tiny_model()
    counts = [supervised_token_count(batch.labels) for batch in micro_batches]
    with torch.no_grad():
        naive = sum(
            float(token_sum_loss(model(batch.input_ids, attention_mask=batch.attention_mask), batch.labels))
            / count
            for batch, count in zip(micro_batches, counts)
        ) / len(micro_batches)
        honest = float(batch_loss(model(big.input_ids, attention_mask=big.attention_mask), big.labels, "token_mean"))
    assert abs(naive - honest) / honest > 1e-3, (naive, honest)


def check_plans_cover_and_split() -> None:
    step_plan = make_step_plan(64, 16, 20260912)
    assert len(step_plan) == 4
    flat = [index for group in step_plan for index in group]
    assert sorted(flat) == list(range(64))
    # 同一 shuffle 序列：step 16 计划的一组 = batch 4 计划连续 4 个批次的
    # 成员并集。组内排序不同（前者按 16 个成员排序，后者每 4 个排序），
    # 因此只断言集合相等。
    batch_plan = make_batch_plan(64, 4, 20260912)
    assert sorted(index for batch in batch_plan[:4] for index in batch) == sorted(step_plan[0])
    samples = [synthetic_sample(8 + (index % 5), 2, index + 1) for index in range(64)]
    micros = split_effective_batch(samples, step_plan[0], 4, 0)
    assert sum(batch.batch_size for batch in micros) == 16
    # 合成样本的 source_line = 成员下标 + 1，可据此检查每个微批的成员切片。
    for position, micro in enumerate(micros):
        assert micro.source_lines == tuple(index + 1 for index in step_plan[0][position * 4 : (position + 1) * 4])
    try:
        split_effective_batch(samples, step_plan[0], 3, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("indivisible split must raise")
    try:
        make_step_plan(63, 16, 20260912)
    except ValueError:
        pass
    else:
        raise AssertionError("indivisible step plan must raise")


def check_empty_supervision_raises() -> None:
    config = tiny_config()
    model = tiny_model()
    optimizer = build_optimizer(model, 2e-5, 0.1)
    unsupervised = EncodedConversation(
        input_ids=torch.ones(1, 6, dtype=torch.long),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        labels=torch.full((1, 6), IGNORE_INDEX, dtype=torch.long),
        assistant_loss_mask=torch.zeros(1, 6, dtype=torch.bool),
        source_line=1,
        rendered_text="",
        token_spans=(),
        loss_mode="assistant_only",
    )
    batch = pad_samples([unsupervised], pad_token_id=0)
    try:
        accumulated_step(model, optimizer, [batch], torch.device("cpu"), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupervised effective batch must raise")


def main() -> None:
    check_weighting_math_direct()
    check_accumulated_gradients_match_big_batch()
    check_mean_of_means_diverges()
    check_plans_cover_and_split()
    check_empty_supervision_raises()
    print("chapter_24 smoke tests passed")


if __name__ == "__main__":
    main()
