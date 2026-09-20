"""Padded mini-batch construction and matched loss reduction for Chapter 19.

The padded-batch construction and the ``token_mean`` reduction are inherited
unchanged from the stable Chapter 18 implementation: right padding to the batch
maximum, causal shift, and an average over effective supervised tokens.  The
serial per-sample backward proved equivalent in Chapter 18 and is not repeated
here; this chapter only changes which samples share a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from assistant_loss import EncodedConversation, IGNORE_INDEX


Reduction = Literal["token_mean", "sample_mean"]


@dataclass(frozen=True)
class PaddedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    source_lines: tuple[int, ...]
    lengths: tuple[int, ...]
    supervised_tokens_per_sample: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def physical_tokens(self) -> int:
        return self.input_ids.numel()

    @property
    def non_padding_tokens(self) -> int:
        return int(self.attention_mask.sum().item())

    @property
    def supervised_tokens(self) -> int:
        return sum(self.supervised_tokens_per_sample)

    @property
    def padding_tokens(self) -> int:
        return self.physical_tokens - self.non_padding_tokens

    @property
    def padding_ratio(self) -> float:
        return self.padding_tokens / self.physical_tokens


def trim_sample(sample: EncodedConversation) -> EncodedConversation:
    """Remove right padding without changing any non-padding token or label."""

    length = sample.non_padding_tokens
    if length < 2:
        raise ValueError("a sample needs at least two non-padding tokens")
    return EncodedConversation(
        input_ids=sample.input_ids[:, :length].clone(),
        attention_mask=sample.attention_mask[:, :length].clone(),
        labels=sample.labels[:, :length].clone(),
        assistant_loss_mask=sample.assistant_loss_mask[:, :length].clone(),
        source_line=sample.source_line,
        rendered_text=sample.rendered_text,
        token_spans=sample.token_spans,
        loss_mode=sample.loss_mode,
    )


def pad_samples(samples: Sequence[EncodedConversation], pad_token_id: int) -> PaddedBatch:
    """Right-pad a fixed ordered sample group to its longest sequence."""

    if not samples:
        raise ValueError("samples must not be empty")
    trimmed = [trim_sample(sample) for sample in samples]
    if any(sample.input_ids.shape[0] != 1 for sample in trimmed):
        raise ValueError("each encoded sample must have shape [1, sequence]")
    maximum = max(sample.input_ids.shape[1] for sample in trimmed)
    batch_size = len(trimmed)
    input_ids = torch.full((batch_size, maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, maximum), dtype=torch.long)
    labels = torch.full((batch_size, maximum), IGNORE_INDEX, dtype=torch.long)
    lengths: list[int] = []
    supervised: list[int] = []
    for row, sample in enumerate(trimmed):
        length = sample.input_ids.shape[1]
        input_ids[row, :length] = sample.input_ids[0]
        attention_mask[row, :length] = sample.attention_mask[0]
        labels[row, :length] = sample.labels[0]
        lengths.append(length)
        supervised.append(sample.supervised_tokens)
    return PaddedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        source_lines=tuple(sample.source_line for sample in trimmed),
        lengths=tuple(lengths),
        supervised_tokens_per_sample=tuple(supervised),
    )


def unreduced_shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [batch, sequence, vocab] and matching labels")
    shifted_labels = labels[:, 1:].contiguous()
    if not bool((shifted_labels != IGNORE_INDEX).any()):
        raise ValueError("no supervised next-token targets")
    losses = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    return losses.view(labels.shape[0], labels.shape[1] - 1)


def reduce_losses(per_token_losses: torch.Tensor, labels: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    active = labels[:, 1:] != IGNORE_INDEX
    counts = active.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every sample must contain a supervised next-token target")
    sums = (per_token_losses * active).sum(dim=1)
    if reduction == "token_mean":
        return sums.sum() / counts.sum()
    if reduction == "sample_mean":
        return (sums / counts).mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def batch_loss(logits: torch.Tensor, labels: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    return reduce_losses(unreduced_shifted_cross_entropy(logits, labels), labels, reduction)
