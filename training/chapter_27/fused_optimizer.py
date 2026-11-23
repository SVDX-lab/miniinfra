"""Chapter 27 core: same AdamW math, three kernel organizations.

The optimizer's *math* is pinned: ``torch.optim.AdamW`` with
``betas=(0.9, 0.95)``, ``eps=1e-8``, a single parameter group, and the same
learning rate and weight decay everywhere.  Only the kernel organization of
the update changes:

- ``loop``     ``foreach=False, fused=False`` - one kernel launch per op per
  parameter tensor.  The Chapter 22 baseline.
- ``foreach``  ``foreach=True`` - the same elementwise ops batched across
  parameter tensors into multi-tensor kernels.
- ``fused``    ``fused=True`` - the whole update for every parameter runs in
  one fused CUDA kernel per step.

``count_step_kernels`` profiles one ``optimizer.step()`` and reports how many
CUDA kernels it launches plus their total device time - the mechanism number
this chapter exists to show.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.profiler import ProfilerActivity, profile


OptimizerArm = Literal["loop", "foreach", "fused"]


def build_optimizer(
    model,
    learning_rate: float,
    weight_decay: float,
    arm: OptimizerArm = "loop",
) -> torch.optim.AdamW:
    """``model`` is any ``nn.Module`` or an iterable of parameters."""
    if arm == "loop":
        foreach, fused = False, False
    elif arm == "foreach":
        foreach, fused = True, False
    elif arm == "fused":
        foreach, fused = False, True
    else:
        raise ValueError(f"unknown optimizer arm: {arm}")
    parameters = model.parameters() if hasattr(model, "parameters") else model
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        foreach=foreach,
        fused=fused,
    )


def reference_adamw_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> torch.Tensor:
    """Hand-written one-tensor AdamW update used as a math reference.

    Follows torch.optim.AdamW (decoupled weight decay, bias correction):

        p ← p - lr * weight_decay * p
        m ← β1 m + (1-β1) g
        v ← β2 v + (1-β2) g²
        m̂ = m / (1-β1^t); v̂ = v / (1-β2^t)
        p ← p - lr * m̂ / (√v̂ + eps)
    """

    parameter = parameter - learning_rate * weight_decay * parameter
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient * gradient
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    exp_avg_hat = exp_avg / bias_correction1
    exp_avg_sq_hat = exp_avg_sq / bias_correction2
    return parameter - learning_rate * exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)


def count_step_kernels(optimizer: torch.optim.Optimizer) -> dict:
    """Profile one ``optimizer.step()``: CUDA kernel count and device time."""

    optimizer.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        optimizer.step()
        torch.cuda.synchronize()
    kernel_events = [
        event for event in profiler.key_averages()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and event.self_device_time_total > 0
    ]
    kernel_count = sum(event.count for event in kernel_events)
    device_time_us = sum(event.self_device_time_total for event in kernel_events)
    top_kernels = sorted(
        (
            {"name": event.key[:80], "count": event.count, "device_time_us": event.self_device_time_total}
            for event in kernel_events
        ),
        key=lambda entry: entry["device_time_us"],
        reverse=True,
    )[:8]
    return {
        "cuda_kernel_count": int(kernel_count),
        "cuda_device_time_us": float(device_time_us),
        "distinct_kernels": len(kernel_events),
        "top_kernels": top_kernels,
    }


def step_peak_memory(optimizer: torch.nn.Module, model, max_grad_norm: float) -> dict:
    """Allocated memory before/during/after clip+step to expose temporaries."""

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    after = torch.cuda.memory_allocated()
    return {
        "allocated_before_step_bytes": before,
        "peak_during_step_bytes": peak,
        "allocated_after_step_bytes": after,
        "gradient_norm": float(grad_norm.item()),
    }
