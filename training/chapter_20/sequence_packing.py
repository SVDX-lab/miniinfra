"""Chapter 20 core: sequence packing with strict per-sample attention isolation.

Packing places several unsplit samples into one fixed-capacity physical
sequence.  Three pieces of bookkeeping make a pack equivalent to running each
sample on its own:

- every sample's ``position_ids`` restart at zero (RoPE sees the same relative
  positions as an independent run);
- an additive block-diagonal causal mask forbids attention between different
  samples and into right padding;
- the loss mask is cut at sample boundaries so no target ever crosses from one
  sample into the next.

The greedy packer is a pure function of ``(lengths, capacity, seed, epoch)``
and never splits a sample.  The naive EOS-style concatenation that keeps
continuous positions and plain causal attention is provided only as a negative
control; it is not an equivalent packing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch

from assistant_loss import EncodedConversation, IGNORE_INDEX


@dataclass(frozen=True)
class PackPlan:
    """One epoch of packs, each a tuple of dataset sample indices."""

    packs: tuple[tuple[int, ...], ...]
    capacity: int
    seed: int
    epoch: int

    def __post_init__(self) -> None:
        seen: set[int] = set()
        for pack in self.packs:
            if not pack:
                raise ValueError("every pack must contain at least one sample")
            for index in pack:
                if index in seen:
                    raise ValueError(f"sample {index} appears in more than one pack")
                seen.add(index)
        if len(seen) != sum(len(pack) for pack in self.packs):
            raise AssertionError("pack coverage accounting failed")

    @property
    def num_packs(self) -> int:
        return len(self.packs)

    @property
    def num_samples(self) -> int:
        return sum(len(pack) for pack in self.packs)

    def utilization(self, lengths: Sequence[int]) -> float:
        """Non-padding tokens divided by physical pack slots."""

        physical = self.num_packs * self.capacity
        non_padding = sum(lengths[index] for pack in self.packs for index in pack)
        return non_padding / physical

    def samples_per_pack(self) -> tuple[int, ...]:
        return tuple(len(pack) for pack in self.packs)


def greedy_packing_plan(
    lengths: Sequence[int],
    capacity: int,
    seed: int,
    epoch: int,
) -> PackPlan:
    """Shuffle with a fixed seed, then first-fit packs without splitting samples.

    A sample longer than ``capacity`` is an error: this chapter never truncates
    inside the packer, truncation stays in the encoder where it is recorded.
    """

    if capacity < 2:
        raise ValueError("pack capacity must be at least two tokens")
    if any(length > capacity for length in lengths):
        offender = max(range(len(lengths)), key=lambda index: lengths[index])
        raise ValueError(
            f"sample {offender} has length {lengths[offender]} beyond pack capacity {capacity}; "
            "lower the encoder max_length instead of splitting samples"
        )
    rng = random.Random(f"{seed}|{epoch}|pack")
    order = list(range(len(lengths)))
    rng.shuffle(order)
    packs: list[tuple[int, ...]] = []
    current: list[int] = []
    current_length = 0
    for index in order:
        length = lengths[index]
        if current and current_length + length > capacity:
            packs.append(tuple(current))
            current = []
            current_length = 0
        current.append(index)
        current_length += length
    if current:
        packs.append(tuple(current))
    return PackPlan(packs=tuple(packs), capacity=capacity, seed=seed, epoch=epoch)


@dataclass(frozen=True)
class PackedSequence:
    """One pack as a [1, capacity] sequence, built on CPU."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    sample_spans: tuple[tuple[int, int], ...]
    source_lines: tuple[int, ...]
    lengths: tuple[int, ...]
    supervised_tokens_per_sample: tuple[int, ...]

    @property
    def capacity(self) -> int:
        return self.input_ids.shape[1]

    @property
    def non_padding_tokens(self) -> int:
        return int(self.attention_mask.sum().item())

    @property
    def supervised_tokens(self) -> int:
        return int((self.labels[..., 1:] != IGNORE_INDEX).sum().item())

    @property
    def physical_tokens(self) -> int:
        return self.input_ids.numel()


def build_pack(
    samples: Sequence[EncodedConversation],
    indices: Sequence[int],
    capacity: int,
    pad_token_id: int,
) -> PackedSequence:
    """Concatenate trimmed samples into one pack with isolation bookkeeping.

    The label of each sample's first position is forced to ``ignore_index``:
    after the causal shift it would otherwise become the target of the previous
    sample's last token.  Inside a sample, labels are the trimmed sample's own
    labels, so per-sample supervised token counts match an independent run.
    """

    if not indices:
        raise ValueError("a pack needs at least one sample")
    total_length = 0
    trimmed: list[EncodedConversation] = []
    for index in indices:
        sample = samples[index]
        if sample.input_ids.shape[0] != 1:
            raise ValueError("each encoded sample must have shape [1, sequence]")
        length = sample.non_padding_tokens
        if length < 2:
            raise ValueError("a sample needs at least two non-padding tokens")
        if total_length + length > capacity:
            raise ValueError(
                f"pack overflow: sample {index} needs {length} tokens, "
                f"{capacity - total_length} remain"
            )
        trimmed.append(sample)
        total_length += length

    input_ids = torch.full((1, capacity), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((1, capacity), dtype=torch.long)
    labels = torch.full((1, capacity), IGNORE_INDEX, dtype=torch.long)
    position_ids = torch.zeros((1, capacity), dtype=torch.long)

    spans: list[tuple[int, int]] = []
    lengths: list[int] = []
    supervised: list[int] = []
    source_lines: list[int] = []
    cursor = 0
    for sample in trimmed:
        length = sample.non_padding_tokens
        input_ids[0, cursor : cursor + length] = sample.input_ids[0, :length]
        attention_mask[0, cursor : cursor + length] = 1
        # 样本自己的 labels 已含 Assistant-only Mask；首个位置在 Shift 后不是
        # 任何位置的目标，这里显式置为 ignore_index，保证边界不跨样本。
        labels[0, cursor : cursor + length] = sample.labels[0, :length]
        labels[0, cursor] = IGNORE_INDEX
        position_ids[0, cursor : cursor + length] = torch.arange(length)
        spans.append((cursor, cursor + length))
        lengths.append(length)
        supervised.append(sample.supervised_tokens)
        source_lines.append(sample.source_line)
        cursor += length

    pack_supervised = int((labels[0, 1:] != IGNORE_INDEX).sum().item())
    if pack_supervised != sum(supervised):
        raise AssertionError("pack supervised tokens diverge from per-sample accounting")
    return PackedSequence(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
        sample_spans=tuple(spans),
        source_lines=tuple(source_lines),
        lengths=tuple(lengths),
        supervised_tokens_per_sample=tuple(supervised),
    )


def build_block_diagonal_mask(
    spans: Sequence[tuple[int, int]],
    capacity: int,
    dtype: torch.dtype,
    device: torch.device,
    total_length: int | None = None,
) -> torch.Tensor:
    """Additive [1, 1, capacity, capacity] mask: causal inside a sample only.

    Masked entries are ``torch.finfo(dtype).min`` so softmax gives them exactly
    zero weight; pad keys beyond ``total_length`` are masked as well.
    """

    if total_length is None:
        total_length = max((end for _, end in spans), default=0)
    same_sample = torch.zeros((capacity, capacity), dtype=torch.bool, device=device)
    for start, end in spans:
        same_sample[start:end, start:end] = True
    causal = torch.tril(torch.ones((capacity, capacity), dtype=torch.bool, device=device))
    key_is_real = torch.zeros((capacity,), dtype=torch.bool, device=device)
    key_is_real[:total_length] = True
    allowed = same_sample & causal & key_is_real.unsqueeze(0)

    mask = torch.zeros((capacity, capacity), dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


def pack_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-mean shifted cross entropy over the whole pack."""

    from mini_batch import batch_loss

    if logits.shape[0] != 1:
        raise ValueError("pack forward expects a single [1, capacity] sequence")
    return batch_loss(logits, labels, "token_mean")


def independent_supervised_losses(
    logits_by_sample: Sequence[torch.Tensor],
    samples: Sequence[EncodedConversation],
    indices: Sequence[int],
) -> tuple[list[torch.Tensor], list[float]]:
    """Per-sample unreduced losses from independent runs (for comparisons)."""

    from mini_batch import unreduced_shifted_cross_entropy

    per_token: list[torch.Tensor] = []
    means: list[float] = []
    for logits, index in zip(logits_by_sample, indices):
        labels = samples[index].labels[:, : logits.shape[1]]
        losses = unreduced_shifted_cross_entropy(logits, labels)
        active = labels[:, 1:] != IGNORE_INDEX
        per_token.append(losses.detach())
        means.append(float((losses * active).sum().item() / active.sum().item()))
    return per_token, means
