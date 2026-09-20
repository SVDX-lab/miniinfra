"""Matched serial-sample and padded-mini-batch FP32 training steps."""

from __future__ import annotations

import subprocess
import time
from typing import Any, Literal

import torch

from mini_batch import PaddedBatch, Reduction, batch_loss
from qwen3_model import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, load_handwritten_model, resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer


Execution = Literal["serial", "batched"]
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


def train_group_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    host_batch: PaddedBatch,
    device: torch.device,
    execution: Execution,
    reduction: Reduction,
    max_grad_norm: float,
    collect_telemetry: bool = True,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    counts = torch.tensor(host_batch.supervised_tokens_per_sample, device=device)

    def forward_backward(batch: PaddedBatch) -> tuple[torch.Tensor, float, float]:
        sample_means: list[torch.Tensor] = []
        forward_ms = 0.0
        backward_ms = 0.0
        if execution == "serial":
            for row, length in enumerate(batch.lengths):
                if device.type == "cuda":
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    events[0].record()
                else:
                    forward_start = time.perf_counter()
                logits = model(batch.input_ids[row : row + 1, :length], attention_mask=batch.attention_mask[row : row + 1, :length])
                labels = batch.labels[row : row + 1, :length]
                sample_mean = batch_loss(logits, labels, "token_mean")
                if device.type == "cuda":
                    events[1].record(); events[2].record()
                else:
                    forward_ms += (time.perf_counter() - forward_start) * 1000
                    backward_start = time.perf_counter()
                scale = counts[row] / counts.sum() if reduction == "token_mean" else 1.0 / batch.batch_size
                (sample_mean * scale).backward()
                if device.type == "cuda":
                    events[3].record()
                    torch.cuda.synchronize(device)
                    forward_ms += float(events[0].elapsed_time(events[1]))
                    backward_ms += float(events[2].elapsed_time(events[3]))
                else:
                    backward_ms += (time.perf_counter() - backward_start) * 1000
                sample_means.append(sample_mean.detach())
            group_loss = torch.stack(sample_means)
            group_loss = (group_loss * counts).sum() / counts.sum() if reduction == "token_mean" else group_loss.mean()
            return group_loss, forward_ms, backward_ms
        if execution != "batched":
            raise ValueError(f"unsupported execution: {execution}")
        if device.type == "cuda":
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()
        else:
            forward_start = time.perf_counter()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, reduction)
        if device.type == "cuda":
            events[1].record(); events[2].record()
        else:
            forward_ms = (time.perf_counter() - forward_start) * 1000
            backward_start = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            events[3].record(); torch.cuda.synchronize(device)
            forward_ms = float(events[0].elapsed_time(events[1]))
            backward_ms = float(events[2].elapsed_time(events[3]))
        else:
            backward_ms = (time.perf_counter() - backward_start) * 1000
        return loss.detach(), forward_ms, backward_ms

    if device.type != "cuda":
        start = time.perf_counter()
        batch = move_batch(host_batch, device)
        h2d_ms = (time.perf_counter() - start) * 1000
        loss, forward_ms, backward_ms = forward_backward(batch)
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
        loss, forward_ms, backward_ms = forward_backward(batch)
        events[2].record()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        events[3].record()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - wall_start) * 1000
        metrics = {
            "h2d_time_ms": float(events[0].elapsed_time(events[1])),
            "forward_time_ms": forward_ms,
            "backward_time_ms": backward_ms,
            "optimizer_time_ms": float(events[2].elapsed_time(events[3])),
            "step_time_ms": step_ms,
        }
    seconds = metrics["step_time_ms"] / 1000
    physical = sum(host_batch.lengths) if execution == "serial" else host_batch.physical_tokens
    metrics.update({
        "loss": float(loss.item()),
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(physical),
        "non_padding_tokens": float(host_batch.non_padding_tokens),
        "supervised_tokens": float(host_batch.supervised_tokens),
        "padding_tokens": float(physical - host_batch.non_padding_tokens),
        "padding_ratio": (physical - host_batch.non_padding_tokens) / physical,
        "physical_tokens_per_second": physical / seconds,
        "non_padding_tokens_per_second": host_batch.non_padding_tokens / seconds,
        "supervised_tokens_per_second": host_batch.supervised_tokens / seconds,
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
