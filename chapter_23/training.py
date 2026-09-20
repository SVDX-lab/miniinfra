"""Chapter 23 training step: FP32 baseline versus BF16 autocast.

The step body is the Chapter 22 stable body (padded forward, ``token_mean``
loss, global grad clip, AdamW); the only change is the dtype strategy:

- ``fp32``: everything FP32, TF32 off.  Bit-for-bit the Chapter 22 path, used
  as both the baseline arm and the numerical reference.
- ``bf16_autocast``: forward and loss run under ``torch.autocast`` with
  ``bfloat16``; master parameters, gradients and AdamW states stay FP32.
  BF16 needs no loss scaler; the dtype census in ``mixed_precision.py``
  records which Tensor actually holds which dtype.

Both arms consume identical device batches from the same encoded corpus and
batch plan inherited from the data-pipeline chapters.
"""

from __future__ import annotations

import random
import subprocess
import time
from typing import Any, Literal

import torch

from assistant_loss import EncodedConversation, encode_conversation
from mini_batch import PaddedBatch, batch_loss, pad_samples, trim_sample
from qwen3_model import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, load_handwritten_model, resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer


MODEL_ID = DEFAULT_MODEL_ID
MODEL_REVISION = DEFAULT_MODEL_REVISION

PrecisionMode = Literal["fp32", "bf16_autocast"]
PRECISION_MODES = ("fp32", "bf16_autocast")


def load_tokenizer_and_model(device: torch.device) -> tuple[Qwen3Tokenizer, torch.nn.Module]:
    """Master weights are always FP32; BF16 enters only through autocast."""

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
    """Encode the fixed corpus once; batch membership never re-encodes a sample."""

    return [
        trim_sample(encode_conversation(tokenizer, record, max_length, "assistant_only"))
        for record in records
    ]


def make_batch_plan(num_samples: int, batch_size: int, seed: int) -> list[tuple[int, ...]]:
    """The exact Chapter 22 shuffle: members sorted inside each batch."""

    rng = random.Random(f"{seed}|1|shuffle")
    order = list(range(num_samples))
    rng.shuffle(order)
    return [tuple(sorted(order[i : i + batch_size])) for i in range(0, len(order), batch_size)]


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


def _telemetry(device: torch.device) -> dict[str, float]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={index}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


def autocast_context(mode: PrecisionMode, device: torch.device):
    """Forward/loss precision context for a mode; FP32 keeps an empty context."""

    if mode == "bf16_autocast":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if mode == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    raise ValueError(f"unsupported precision mode: {mode}")


def train_batch_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    host_batch: PaddedBatch,
    device: torch.device,
    max_grad_norm: float,
    mode: PrecisionMode = "fp32",
    collect_telemetry: bool = True,
) -> dict[str, float]:
    """One padded Forward/Backward, global gradient clip, and AdamW step.

    CUDA timings use events around H2D, forward+loss, backward, and
    clip+optimizer; the wall-clock step time covers the synchronized whole.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type != "cuda":
        start = time.perf_counter()
        batch = move_batch(host_batch, device)
        h2d_ms = (time.perf_counter() - start) * 1000
        forward_start = time.perf_counter()
        with autocast_context(mode, device):
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
        with autocast_context(mode, device):
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
    loss_value = float(loss.item())
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "loss": loss_value,
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(batch.physical_tokens),
        "non_padding_tokens": float(batch.non_padding_tokens),
        "supervised_tokens": float(batch.supervised_tokens),
        "padding_tokens": float(batch.padding_tokens),
        "padding_ratio": batch.padding_ratio,
        "batch_max_length": float(max(batch.lengths)),
        "physical_tokens_per_second": batch.physical_tokens / seconds,
        "non_padding_tokens_per_second": batch.non_padding_tokens / seconds,
        "supervised_tokens_per_second": batch.supervised_tokens / seconds,
        "samples": float(batch.batch_size),
    })
    if device.type == "cuda" and collect_telemetry:
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
