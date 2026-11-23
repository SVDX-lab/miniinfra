"""Chapter 19 core: length-bucketed batch plans and their measurable side effects.

A ``BatchPlan`` is a full-epoch partition of the sample set into batches.  Both
samplers consume the same encoded samples and only decide which samples share a
batch:

- ``random_plan`` shuffles all sample indices with a fixed seed and chunks the
  shuffled order into fixed-size batches.  This is the Chapter 18 style fixed
  random order baseline.
- ``bucketed_plan`` uses the same seeded shuffle, cuts it into shuffle windows
  of ``window`` samples, sorts every window by encoded length, chunks the sorted
  windows into batches, and finally shuffles the batch order.  Only batch
  membership changes; the sample set, the batch size and the number of steps do
  not.

Plans are pure functions of ``(lengths, seed, epoch, batch_size, window)``, so a
run can be replayed exactly from the result JSON.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class BatchPlan:
    """One epoch of batches, each a tuple of dataset sample indices."""

    batches: tuple[tuple[int, ...], ...]
    strategy: str
    batch_size: int
    window: int
    seed: int
    epoch: int

    def __post_init__(self) -> None:
        seen: set[int] = set()
        for batch in self.batches:
            if not batch:
                raise ValueError("every batch must contain at least one sample")
            if len(batch) != len(set(batch)):
                raise ValueError(f"batch repeats a sample: {sorted(batch)}")
            for index in batch:
                if index in seen:
                    raise ValueError(f"sample {index} appears in more than one batch")
                seen.add(index)
        if len(self.batches) > 1 and sum(1 for size in self.batch_sizes if size != self.batch_size) > 1:
            raise ValueError("at most one batch may be smaller than the batch size")

    @property
    def num_samples(self) -> int:
        return sum(len(batch) for batch in self.batches)

    @property
    def num_batches(self) -> int:
        return len(self.batches)

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(len(batch) for batch in self.batches)

    @property
    def partial_batches(self) -> int:
        target = self.batch_size
        return sum(1 for size in self.batch_sizes if size != target)

    def sequence(self) -> tuple[int, ...]:
        """Training order: batches in execution order, members sorted by index."""

        return tuple(index for batch in self.batches for index in sorted(batch))


def _resolve_window(window: int, batch_size: int, num_samples: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_samples < batch_size:
        raise ValueError(f"num_samples={num_samples} cannot fill one batch of {batch_size}")
    if window < batch_size:
        raise ValueError(
            f"window={window} must be at least batch_size={batch_size}, "
            "otherwise windows cannot be split into full batches"
        )
    if window % batch_size != 0:
        raise ValueError(
            f"window={window} must be a multiple of batch_size={batch_size} "
            "so every window splits into full batches"
        )
    return min(window, num_samples)


def random_plan(num_samples: int, batch_size: int, seed: int, epoch: int) -> BatchPlan:
    """Seeded global shuffle followed by fixed-size chunking."""

    if num_samples < batch_size:
        raise ValueError(f"num_samples={num_samples} cannot fill one batch of {batch_size}")
    rng = random.Random(f"{seed}|{epoch}|shuffle")
    order = list(range(num_samples))
    rng.shuffle(order)
    batches = tuple(sorted(order[start : start + batch_size]) for start in range(0, num_samples, batch_size))
    return BatchPlan(
        batches=batches, strategy="random", batch_size=batch_size,
        window=batch_size, seed=seed, epoch=epoch,
    )


def bucketed_plan(
    lengths: Sequence[int],
    batch_size: int,
    window: int,
    seed: int,
    epoch: int,
    shuffle_batches: bool = True,
) -> BatchPlan:
    """Shuffle, sort inside windows of ``window`` samples, chunk, shuffle batches.

    ``window == batch_size`` degenerates to the random baseline: every window is
    one shuffle chunk, sorting cannot change membership, so the batch member sets
    equal ``random_plan`` with the same seed and epoch.
    """

    num_samples = len(lengths)
    effective_window = _resolve_window(window, batch_size, num_samples)
    if any(length < 2 for length in lengths):
        raise ValueError("every sample needs at least two tokens after trimming")
    rng = random.Random(f"{seed}|{epoch}|shuffle")
    order = list(range(num_samples))
    rng.shuffle(order)
    batches: list[tuple[int, ...]] = []
    for start in range(0, num_samples, effective_window):
        chunk = order[start : start + effective_window]
        chunk.sort(key=lambda index: lengths[index])
        batches.extend(
            tuple(sorted(chunk[batch_start : batch_start + batch_size]))
            for batch_start in range(0, len(chunk), batch_size)
        )
    if shuffle_batches:
        rng.shuffle(batches)
    return BatchPlan(
        batches=tuple(batches), strategy="bucketed", batch_size=batch_size,
        window=effective_window, seed=seed, epoch=epoch,
    )


@dataclass(frozen=True)
class PlanTokenProjection:
    """Batch padding predicted from lengths alone, without running the model."""

    physical_tokens: int
    non_padding_tokens: int
    padding_tokens: int
    padding_ratio: float
    batch_physical_tokens: tuple[int, ...]

    @property
    def work_cv(self) -> float:
        """Coefficient of variation of per-batch physical work (DDP lookahead)."""

        return pstdev(self.batch_physical_tokens) / mean(self.batch_physical_tokens)

    @property
    def work_max_mean_ratio(self) -> float:
        return max(self.batch_physical_tokens) / mean(self.batch_physical_tokens)


def project_plan_tokens(lengths: Sequence[int], plan: BatchPlan) -> PlanTokenProjection:
    """Reproduce ``pad_samples`` accounting from lengths: pad each batch to its max."""

    batch_physical: list[int] = []
    non_padding = 0
    for batch in plan.batches:
        batch_lengths = [lengths[index] for index in batch]
        widest = max(batch_lengths)
        batch_physical.append(len(batch) * widest)
        non_padding += sum(batch_lengths)
    physical = sum(batch_physical)
    padding = physical - non_padding
    return PlanTokenProjection(
        physical_tokens=physical,
        non_padding_tokens=non_padding,
        padding_tokens=padding,
        padding_ratio=padding / physical,
        batch_physical_tokens=tuple(batch_physical),
    )


def batch_length_spreads(lengths: Sequence[int], plan: BatchPlan) -> tuple[float, ...]:
    """Per batch ``max(length) - min(length)``: the direct cause of padding."""

    return tuple(
        max(lengths[index] for index in batch) - min(lengths[index] for index in batch)
        for batch in plan.batches
    )


def companion_pair_counts(plans: Sequence[BatchPlan]) -> Counter[tuple[int, int]]:
    """Counts of within-batch sample pairs across the given epochs."""

    counts: Counter[tuple[int, int]] = Counter()
    for plan in plans:
        for batch in plan.batches:
            counts.update(combinations(sorted(batch), 2))
    return counts


def companion_repeat_ratio(plans: Sequence[BatchPlan]) -> float:
    """Fraction of distinct co-occurring pairs that co-occur in more than one epoch.

    Random batching co-occurs pairs essentially by chance, so the ratio stays
    near zero.  Full sorting freezes each sample's neighbours to its
    length-neighbours, pushing the ratio towards one.
    """

    counts = companion_pair_counts(plans)
    if not counts:
        return 0.0
    repeated = sum(1 for count in counts.values() if count >= 2)
    return repeated / len(counts)


def plans_are_identical(left: BatchPlan, right: BatchPlan) -> bool:
    return left.batches == right.batches


def membership_multiset(plan: BatchPlan) -> list[tuple[int, ...]]:
    """Batch member sets, order-insensitive inside and across batches."""

    return sorted(tuple(sorted(batch)) for batch in plan.batches)
