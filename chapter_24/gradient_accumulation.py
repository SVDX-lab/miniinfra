"""Chapter 24 core: gradient accumulation with supervised-token weighting.

One optimizer step over an effective batch of ``B`` samples can either run one
padded Forward/Backward on all ``B`` samples (big batch) or split them into
``K`` micro-batches and accumulate gradients.  With ``token_mean`` reduction
the two are the same objective only if every micro-step's loss is scaled by
its share of supervised tokens:

    total_loss = sum_i token_sum_i / sum_i supervised_i
               = sum_i (supervised_i / total) * token_mean_i

so ``accumulated_step`` backward-propagates ``token_sum_i / total`` per
micro-batch.  Accumulating *averages* (or forgetting to rescale) silently
changes the objective whenever micro-batches hold different supervised-token
counts - the exact mistake this chapter's correctness checks exist to catch.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from assistant_loss import IGNORE_INDEX
from mini_batch import PaddedBatch


def supervised_token_count(labels: torch.Tensor) -> int:
    """Supervised next-token targets after the causal shift."""

    return int((labels[:, 1:] != IGNORE_INDEX).sum().item())


def token_sum_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Unreduced shifted cross entropy summed over supervised tokens."""

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if not bool((shift_labels != IGNORE_INDEX).any()):
        raise ValueError("micro-batch has no supervised next-token targets")
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def accumulated_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    micro_batches: Sequence[PaddedBatch],
    device: torch.device,
    max_grad_norm: float,
    collect_telemetry: bool = False,
    telemetry_fn=None,
) -> dict[str, Any]:
    """K micro Forwards/Backwards, then one clip and one optimizer step.

    Returns per-micro timings plus whole-step totals.  The loss of
    micro-batch i is ``token_sum_i / total_supervised``; the accumulated
    gradient is exactly the gradient of the union ``token_mean`` objective,
    so no rescaling happens at the optimizer.
    """

    if not micro_batches:
        raise ValueError("accumulated_step needs at least one micro-batch")

    counts = [supervised_token_count(batch.labels) for batch in micro_batches]
    total_supervised = sum(counts)
    if total_supervised <= 0:
        raise ValueError("effective batch has no supervised next-token targets")

    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize(device)
    step_start = torch.cuda.Event(enable_timing=True) if on_cuda else None
    step_end = torch.cuda.Event(enable_timing=True) if on_cuda else None
    if step_start is not None:
        step_start.record()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_records: list[dict[str, float]] = []
    loss_value = 0.0
    for index, (batch, count) in enumerate(zip(micro_batches, counts)):
        micro_start = torch.cuda.Event(enable_timing=True) if on_cuda else None
        micro_end = torch.cuda.Event(enable_timing=True) if on_cuda else None
        if micro_start is not None:
            micro_start.record()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = token_sum_loss(logits, batch.labels) / total_supervised
        loss.backward()
        if micro_end is not None:
            micro_end.record()
        loss_value += float(loss.item())
        micro_records.append({
            "micro_batch": index,
            "supervised_tokens": float(count),
            "physical_tokens": float(batch.physical_tokens),
            "non_padding_tokens": float(batch.non_padding_tokens),
            "padding_ratio": batch.padding_ratio,
            "scaled_loss": float(loss.item()),
            "micro_time_ms": float(micro_start.elapsed_time(micro_end)) if on_cuda else 0.0,
        })

    optimizer_start = torch.cuda.Event(enable_timing=True) if on_cuda else None
    optimizer_end = torch.cuda.Event(enable_timing=True) if on_cuda else None
    if optimizer_start is not None:
        optimizer_start.record()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if optimizer_end is not None:
        optimizer_end.record()
    if step_end is not None:
        step_end.record()
    if on_cuda:
        torch.cuda.synchronize(device)

    metrics: dict[str, Any] = {
        "loss": loss_value,
        "gradient_norm_before_clip": float(grad_norm.item()),
        "micro_batch_count": float(len(micro_batches)),
        "micro_batch_size": float(micro_batches[0].batch_size),
        "effective_batch_size": float(sum(batch.batch_size for batch in micro_batches)),
        "supervised_tokens": float(total_supervised),
        "physical_tokens": float(sum(batch.physical_tokens for batch in micro_batches)),
        "non_padding_tokens": float(sum(batch.non_padding_tokens for batch in micro_batches)),
        "padding_tokens": float(sum(batch.padding_tokens for batch in micro_batches)),
        "padding_ratio": sum(batch.padding_tokens for batch in micro_batches)
        / sum(batch.physical_tokens for batch in micro_batches),
        "batch_max_length": float(max(max(batch.lengths) for batch in micro_batches)),
        "micro_records": micro_records,
    }
    if on_cuda:
        step_ms = float(step_start.elapsed_time(step_end))
        optimizer_ms = float(optimizer_start.elapsed_time(optimizer_end))
        metrics.update({
            "step_time_ms": step_ms,
            "optimizer_time_ms": optimizer_ms,
            "micro_total_time_ms": sum(record["micro_time_ms"] for record in micro_records),
            "supervised_tokens_per_second": metrics["supervised_tokens"] / (step_ms / 1000),
            "physical_tokens_per_second": metrics["physical_tokens"] / (step_ms / 1000),
            "non_padding_tokens_per_second": metrics["non_padding_tokens"] / (step_ms / 1000),
        })
    if collect_telemetry and telemetry_fn is not None:
        metrics.update(telemetry_fn(device))
    return metrics
