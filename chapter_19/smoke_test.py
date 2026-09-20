#!/usr/bin/env python3
"""CPU-only Chapter 19 sampler checks: partition, degeneration, determinism, edges."""

from __future__ import annotations

import torch

from assistant_loss import EncodedConversation, IGNORE_INDEX
from length_bucket import (
    BatchPlan,
    bucketed_plan,
    companion_repeat_ratio,
    membership_multiset,
    project_plan_tokens,
    random_plan,
)
from mini_batch import pad_samples


def synthetic_sample(ids: list[int], supervised_from: int, source_line: int) -> EncodedConversation:
    labels = [IGNORE_INDEX if index < supervised_from else token for index, token in enumerate(ids)]
    return EncodedConversation(
        input_ids=torch.tensor([ids]),
        attention_mask=torch.tensor([[1] * len(ids)]),
        labels=torch.tensor([labels]),
        assistant_loss_mask=torch.tensor([[label != IGNORE_INDEX for label in labels]]),
        source_line=source_line,
        rendered_text="",
        token_spans=(),
        loss_mode="assistant_only",
    )


def check_partition_and_degeneration() -> None:
    for plan in (random_plan(23, 4, 20260912, 1), bucketed_plan(list(range(2, 25)), 4, 8, 20260912, 1)):
        covered = sorted(index for batch in plan.batches for index in batch)
        assert covered == list(range(23)), "a plan must be a partition of the sample set"
        assert plan.num_batches == 6 and plan.partial_batches == 1
    # window == batch_size degenerates to the random baseline member sets
    for seed in (1, 20260912):
        baseline = random_plan(37, 4, seed, 3)
        degenerate = bucketed_plan([3] * 37, 4, 4, seed, 3)
        assert membership_multiset(degenerate) == membership_multiset(baseline)


def check_determinism() -> None:
    lengths = [(index * 37) % 61 + 2 for index in range(48)]
    left = bucketed_plan(lengths, 4, 16, 5, 1)
    right = bucketed_plan(lengths, 4, 16, 5, 1)
    assert left.batches == right.batches, "same seed and epoch must replay exactly"
    assert left.batches != bucketed_plan(lengths, 4, 16, 6, 1).batches, "seeds must matter"
    assert left.batches != bucketed_plan(lengths, 4, 16, 5, 2).batches, "epochs must reshuffle"
    assert left.batches != random_plan(48, 4, 5, 1).batches


def check_projection() -> None:
    lengths = [4, 6, 3, 9, 2, 7]
    plan = BatchPlan(
        batches=((0, 2, 1), (3, 4, 5)), strategy="bucketed", batch_size=3, window=3, seed=1, epoch=1,
    )
    projection = project_plan_tokens(lengths, plan)
    # batch 0: members {4,6,3} tokens wide=6 -> 18 physical; batch 1: {9,2,7} wide=9 -> 27
    assert projection.physical_tokens == 45
    assert projection.non_padding_tokens == 31
    assert projection.padding_tokens == 14
    assert abs(projection.padding_ratio - 14 / 45) < 1e-12
    assert projection.batch_physical_tokens == (18, 27)
    # full sort of pure length groups leaves no padding at all
    pure = [4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8]
    sorted_plan = bucketed_plan(pure, 4, 12, 7, 1, shuffle_batches=False)
    assert project_plan_tokens(pure, sorted_plan).padding_tokens == 0
    random_projection = project_plan_tokens(pure, random_plan(12, 4, 7, 1))
    assert random_projection.padding_tokens > 0


def check_window_boundaries() -> None:
    # length blocks of eight and four (both multiples of the batch size); every
    # window seam can create at most one mixed batch
    lengths = [2] * 8 + [6] * 4
    plan = bucketed_plan(lengths, 4, 8, 11, 1)
    mixed = sum(
        1 for batch in plan.batches if len({lengths[index] for index in batch}) > 1
    )
    assert mixed <= 2, f"only window seams may mix lengths, saw {mixed}"
    full = bucketed_plan(lengths, 4, 12, 11, 1, shuffle_batches=False)
    assert all(len({lengths[index] for index in batch}) == 1 for batch in full.batches)
    scattered = bucketed_plan(lengths, 4, 4, 11, 1)
    assert any(len({lengths[index] for index in batch}) > 1 for batch in scattered.batches)


def check_randomness_metrics() -> None:
    lengths = list(range(2, 26))  # 24 strictly increasing lengths: full sort is deterministic
    frozen = [bucketed_plan(lengths, 4, 24, 9, epoch) for epoch in (1, 2)]
    assert companion_repeat_ratio(frozen) == 1.0
    shuffled = [random_plan(24, 4, 9, epoch) for epoch in (1, 2)]
    assert companion_repeat_ratio(shuffled) < 1.0
    mid = [bucketed_plan(lengths, 4, 8, 9, epoch) for epoch in (1, 2)]
    assert 0.0 <= companion_repeat_ratio(mid) <= 1.0


def check_membership_does_not_reencode() -> None:
    samples = [
        synthetic_sample([1, 2, 3, 4, 5], 2, 101),
        synthetic_sample([6, 7, 8, 9], 1, 102),
        synthetic_sample([10, 11, 12, 13, 14, 15], 3, 103),
        synthetic_sample([16, 17], 1, 104),
    ]
    lengths = [sample.non_padding_tokens for sample in samples]
    plan = bucketed_plan(lengths, 2, 4, 3, 1, shuffle_batches=False)
    assert plan.batches == ((1, 3), (0, 2))
    for batch in plan.batches:
        group = pad_samples([samples[index] for index in batch], 0)
        for row, index in enumerate(batch):
            expected = samples[index]
            length = lengths[index]
            assert torch.equal(group.input_ids[row, :length], expected.input_ids[0])
            assert torch.equal(group.labels[row, :length], expected.labels[0])
            assert group.supervised_tokens_per_sample[row] == expected.supervised_tokens


def check_invalid_configuration() -> None:
    for callback in (
        lambda: bucketed_plan([2] * 12, 4, 3, 1, 1),      # window < batch_size
        lambda: bucketed_plan([2] * 12, 4, 6, 1, 1),      # window not a multiple of batch_size
        lambda: random_plan(3, 4, 1, 1),                  # fewer samples than one batch
        lambda: bucketed_plan([2] * 3, 4, 4, 1, 1),       # fewer samples than one batch, bucketed
    ):
        try:
            callback()
        except ValueError:
            continue
        raise AssertionError("expected a ValueError for invalid sampler configuration")


def main() -> None:
    check_partition_and_degeneration()
    check_determinism()
    check_projection()
    check_window_boundaries()
    check_randomness_metrics()
    check_membership_does_not_reencode()
    check_invalid_configuration()
    print("chapter_19 smoke tests passed")


if __name__ == "__main__":
    main()
