"""Single-GPU FP32 training functions shared by Chapter 17 commands."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import EncodedConversation, optimizer_state_bytes, parameter_bytes
from sft_core import shifted_cross_entropy


MODEL_ID = DEFAULT_MODEL_ID
MODEL_REVISION = DEFAULT_MODEL_REVISION


def load_tokenizer_and_model(device: torch.device) -> tuple[Any, torch.nn.Module]:
    model_directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    tokenizer = Qwen3Tokenizer(model_directory)
    model = load_handwritten_model(model_directory, device, dtype=torch.float32)
    return tokenizer, model


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / float(decay_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def move_batch(sample: EncodedConversation, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": sample.input_ids.to(device, non_blocking=False),
        "attention_mask": sample.attention_mask.to(device, non_blocking=False),
        "labels": sample.labels.to(device, non_blocking=False),
    }


def _cuda_elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def gpu_telemetry(device: torch.device) -> dict[str, float]:
    """Read power and temperature after a step without including CLI time in step time."""

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            f"--id={device_index}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    power, temperature = completed.stdout.strip().split(",")
    return {
        "power_watts": float(power.strip()),
        "temperature_celsius": float(temperature.strip()),
    }


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sample: EncodedConversation,
    device: torch.device,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        wall_start = time.perf_counter()
        events[0].record()
        batch = move_batch(sample, device)
        events[1].record()
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = shifted_cross_entropy(logits, batch["labels"])
        events[2].record()
        loss.backward()
        events[3].record()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        events[4].record()
        torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        h2d_ms = _cuda_elapsed(events[0], events[1])
        forward_ms = _cuda_elapsed(events[1], events[2])
        backward_ms = _cuda_elapsed(events[2], events[3])
        optimizer_ms = _cuda_elapsed(events[3], events[4])
    else:
        wall_start = time.perf_counter()
        h2d_start = time.perf_counter()
        batch = move_batch(sample, device)
        h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
        forward_start = time.perf_counter()
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = shifted_cross_entropy(logits, batch["labels"])
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
        backward_start = time.perf_counter()
        loss.backward()
        backward_ms = (time.perf_counter() - backward_start) * 1000.0
        optimizer_start = time.perf_counter()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer_ms = (time.perf_counter() - optimizer_start) * 1000.0
        wall_ms = (time.perf_counter() - wall_start) * 1000.0

    seconds = wall_ms / 1000.0
    metrics = {
        "loss": float(loss.detach().item()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "gradient_norm_before_clip": float(grad_norm.detach().item()),
        "h2d_time_ms": h2d_ms,
        "forward_time_ms": forward_ms,
        "backward_time_ms": backward_ms,
        "optimizer_time_ms": optimizer_ms,
        "step_time_ms": wall_ms,
        "physical_tokens": float(sample.physical_tokens),
        "non_padding_tokens": float(sample.non_padding_tokens),
        "supervised_tokens": float(sample.supervised_tokens),
        "physical_tokens_per_second": sample.physical_tokens / seconds,
        "non_padding_tokens_per_second": sample.non_padding_tokens / seconds,
        "supervised_tokens_per_second": sample.supervised_tokens / seconds,
    }
    if device.type == "cuda":
        metrics.update(gpu_telemetry(device))
    return metrics


def memory_ledger(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, int]:
    record = {
        "parameter_bytes": parameter_bytes(model),
        "gradient_bytes": sum(
            parameter.grad.numel() * parameter.grad.element_size()
            for parameter in model.parameters()
            if parameter.grad is not None
        ),
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
    }
    if torch.cuda.is_available():
        record.update(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    return record


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    consumed_supervised_tokens: int,
    sample_cursor: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": "chapter17.checkpoint.v1",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "consumed_supervised_tokens": consumed_supervised_tokens,
            "sample_cursor": sample_cursor,
            "rng": rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_revision") != MODEL_REVISION:
        raise ValueError("checkpoint model revision does not match Chapter 17")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint["rng"])
    return {
        "step": int(checkpoint["step"]),
        "consumed_supervised_tokens": int(checkpoint["consumed_supervised_tokens"]),
        "sample_cursor": int(checkpoint["sample_cursor"]),
    }


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    model.eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(text)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated_ids.append(token_id)
        input_ids = torch.cat(
            (input_ids, torch.tensor([[token_id]], dtype=torch.long, device=device)),
            dim=1,
        )
        if token_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
