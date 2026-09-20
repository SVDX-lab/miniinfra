"""Plan-driven FP32 training steps for Chapter 20: padded batches and packs.

Both paths share the optimizer, gradient clipping, loss reduction and timing
instrumentation.  The padded path is the Chapter 18/04 baseline; the pack path
adds position reset and a block-diagonal additive mask, which are the chapter's
isolation machinery.  Membership plans come from ``sequence_packing``.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

import torch

from assistant_loss import EncodedConversation, encode_conversation
from mini_batch import PaddedBatch, batch_loss, pad_samples, trim_sample
from qwen3_model import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, load_handwritten_model, resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sequence_packing import PackedSequence, build_block_diagonal_mask


MODEL_ID = DEFAULT_MODEL_ID
MODEL_REVISION = DEFAULT_MODEL_REVISION


def load_tokenizer_and_model(device: torch.device) -> tuple[Qwen3Tokenizer, torch.nn.Module]:
    directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    return Qwen3Tokenizer(directory), load_handwritten_model(directory, device, dtype=torch.float32)


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=weight_decay, foreach=False, fused=False,
    )


def encode_samples(
    tokenizer: Qwen3Tokenizer,
    records: list[dict[str, Any]],
    max_length: int,
) -> list[EncodedConversation]:
    """Encode the fixed corpus once; physical arrangement never re-encodes."""

    return [
        trim_sample(encode_conversation(tokenizer, record, max_length, "assistant_only"))
        for record in records
    ]


def build_batch(samples: list[EncodedConversation], batch_indices: tuple[int, ...], pad_token_id: int) -> PaddedBatch:
    return pad_samples([samples[index] for index in batch_indices], pad_token_id)


def move_batch(batch: PaddedBatch, device: torch.device) -> PaddedBatch:
    return PaddedBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
        source_lines=batch.source_lines,
        lengths=batch.lengths,
        supervised_tokens_per_sample=batch.supervised_tokens_per_sample,
    )


def move_pack(pack: PackedSequence, device: torch.device) -> PackedSequence:
    return PackedSequence(
        input_ids=pack.input_ids.to(device),
        attention_mask=pack.attention_mask.to(device),
        labels=pack.labels.to(device),
        position_ids=pack.position_ids.to(device),
        sample_spans=pack.sample_spans,
        source_lines=pack.source_lines,
        lengths=pack.lengths,
        supervised_tokens_per_sample=pack.supervised_tokens_per_sample,
    )


def _telemetry(device: torch.device) -> dict[str, float]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={index}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


def train_batch_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    host_batch: PaddedBatch,
    device: torch.device,
    max_grad_norm: float,
    collect_telemetry: bool = True,
) -> dict[str, float]:
    """Baseline path: one padded Forward/Backward, clip, and AdamW step."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type != "cuda":
        start = time.perf_counter()
        batch = move_batch(host_batch, device)
        h2d_ms = (time.perf_counter() - start) * 1000
        forward_start = time.perf_counter()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean")
        forward_ms = (time.perf_counter() - forward_start) * 1000
        backward_start = time.perf_counter()
        loss.backward()
        backward_ms = (time.perf_counter() - backward_start) * 1000
        optimizer_start = time.perf_counter()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer_ms = (time.perf_counter() - optimizer_start) * 1000
        step_ms = h2d_ms + forward_ms + backward_ms + optimizer_ms
        metrics = {"h2d_time_ms": h2d_ms, "forward_time_ms": forward_ms, "backward_time_ms": backward_ms, "optimizer_time_ms": optimizer_ms, "step_time_ms": step_ms}
    else:
        torch.cuda.synchronize(device)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        wall_start = time.perf_counter()
        events[0].record()
        batch = move_batch(host_batch, device)
        events[1].record()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean")
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_end.record()
        loss.backward()
        events[2].record()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        events[3].record()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - wall_start) * 1000
        metrics = {
            "h2d_time_ms": float(events[0].elapsed_time(events[1])),
            "forward_time_ms": float(events[1].elapsed_time(forward_end)),
            "backward_time_ms": float(forward_end.elapsed_time(events[2])),
            "optimizer_time_ms": float(events[2].elapsed_time(events[3])),
            "step_time_ms": step_ms,
        }
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "loss": float(loss.item()),
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(batch.physical_tokens),
        "non_padding_tokens": float(batch.non_padding_tokens),
        "supervised_tokens": float(batch.supervised_tokens),
        "padding_tokens": float(batch.padding_tokens),
        "padding_ratio": batch.padding_ratio,
        "physical_tokens_per_second": batch.physical_tokens / seconds,
        "non_padding_tokens_per_second": batch.non_padding_tokens / seconds,
        "supervised_tokens_per_second": batch.supervised_tokens / seconds,
        "samples": float(batch.batch_size),
    })
    if device.type == "cuda" and collect_telemetry:
        metrics.update(_telemetry(device))
    return metrics


def train_pack_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    host_pack: PackedSequence,
    device: torch.device,
    max_grad_norm: float,
    collect_telemetry: bool = True,
    isolation: bool = True,
) -> dict[str, float]:
    """Packing path: one packed Forward/Backward with block-diagonal isolation.

    ``isolation=False`` builds the negative control: continuous positions and a
    plain causal padding mask, i.e. cross-sample attention is allowed.  The mask
    construction happens inside the timed data phase because it is part of the
    pack path's per-step cost.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type != "cuda":
        raise ValueError("pack steps are measured on CUDA; CPU runs use the smoke test")
    torch.cuda.synchronize(device)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    wall_start = time.perf_counter()
    events[0].record()
    pack = move_pack(host_pack, device)
    if isolation:
        mask = build_block_diagonal_mask(
            pack.sample_spans, pack.capacity, torch.float32, device,
            total_length=pack.non_padding_tokens,
        )
    else:
        mask = None
    events[1].record()
    if isolation:
        logits = model(
            pack.input_ids,
            attention_mask=pack.attention_mask,
            position_ids=pack.position_ids,
            attention_mask_4d=mask,
        )
    else:
        logits = model(pack.input_ids, attention_mask=pack.attention_mask)
    loss = batch_loss(logits, pack.labels, "token_mean")
    forward_end = torch.cuda.Event(enable_timing=True)
    forward_end.record()
    loss.backward()
    events[2].record()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    events[3].record()
    torch.cuda.synchronize(device)
    step_ms = (time.perf_counter() - wall_start) * 1000
    metrics = {
        "h2d_mask_time_ms": float(events[0].elapsed_time(events[1])),
        "forward_time_ms": float(events[1].elapsed_time(forward_end)),
        "backward_time_ms": float(forward_end.elapsed_time(events[2])),
        "optimizer_time_ms": float(events[2].elapsed_time(events[3])),
        "step_time_ms": step_ms,
    }
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "loss": float(loss.item()),
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(pack.physical_tokens),
        "non_padding_tokens": float(pack.non_padding_tokens),
        "supervised_tokens": float(pack.supervised_tokens),
        "padding_tokens": float(pack.physical_tokens - pack.non_padding_tokens),
        "padding_ratio": (pack.physical_tokens - pack.non_padding_tokens) / pack.physical_tokens,
        "pack_utilization": pack.non_padding_tokens / pack.physical_tokens,
        "samples_in_pack": float(len(pack.sample_spans)),
        "physical_tokens_per_second": pack.physical_tokens / seconds,
        "non_padding_tokens_per_second": pack.non_padding_tokens / seconds,
        "supervised_tokens_per_second": pack.supervised_tokens / seconds,
        "packs": 1.0,
    })
    if collect_telemetry:
        metrics.update(_telemetry(device))
    return metrics


def memory_record(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, int]:
    result = {
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "gradient_bytes": sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None),
        "optimizer_state_bytes": sum(v.numel() * v.element_size() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)),
    }
    if torch.cuda.is_available():
        result.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    return result
