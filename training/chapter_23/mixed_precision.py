"""Chapter 23 core: the BF16 autocast policy and its dtype census.

Mixed precision is a *policy over dtypes*, not one flag.  This module makes
the policy inspectable:

- ``autocast_policy_census`` runs a probe forward under a capture context and
  records the dtype of every named module output plus the loss, so the claim
  "matmuls are BF16, softmax/loss are FP32" is measured rather than assumed.
- ``state_dtype_census`` records the dtype of parameters, gradients and
  optimizer states, showing that the variant keeps FP32 master state.

The probe uses a tiny hand-built Qwen3 config so the census is cheap enough
for the CPU smoke test, and the same function re-runs on CUDA for the real
model.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from qwen3_model import Qwen3Config


def probe_config(vocab_size: int = 64, hidden_size: int = 32, layers: int = 2) -> Qwen3Config:
    """A tiny valid Qwen3 config for dtype census and CPU checks."""

    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )


def autocast_policy_census(
    model: torch.nn.Module,
    forward_and_loss: Callable[[], torch.Tensor],
    capture: Any,
) -> dict[str, str]:
    """Record the output dtype of every named module under ``capture``.

    ``capture`` is a context manager (autocast or an empty context).  The
    forward is executed once per module via forward hooks; hook recording and
    hook removal are explicit so a census can never leave hooks behind.
    """

    observed: dict[str, str] = {}

    def hook(name):
        def record(_module, _arguments, output):
            tensor = output
            while isinstance(tensor, (tuple, list)):
                tensor = tensor[0]
            if torch.is_tensor(tensor):
                observed[name] = str(tensor.dtype)

        return record

    handles = [
        (name, module.register_forward_hook(hook(name)))
        for name, module in model.named_modules()
        if name
    ]
    try:
        with capture:
            loss = forward_and_loss()
        observed["<loss>"] = str(loss.dtype)
    finally:
        for _name, handle in handles:
            handle.remove()
    return observed


def state_dtype_census(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> dict[str, Any]:
    """Dtype histogram over parameters, gradients, and optimizer states."""

    def histogram(tensors):
        counts: dict[str, int] = {}
        for tensor in tensors:
            name = str(tensor.dtype)
            counts[name] = counts.get(name, 0) + 1
        return counts

    result: dict[str, Any] = {
        "parameters": histogram(p for p in model.parameters()),
        "gradients": histogram(p.grad for p in model.parameters() if p.grad is not None),
    }
    if optimizer is not None:
        result["optimizer_states"] = histogram(
            value
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )
    return result


def assert_no_nan_or_inf(model: torch.nn.Module) -> None:
    # Chunked finiteness check: a full-shape bool temp for the tied embedding
    # grad (~155 MB) fragments the allocator at the 12GB FP32 margin.
    chunk = 8_000_000
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        flat = grad.view(-1)
        for begin in range(0, flat.numel(), chunk):
            if not bool(torch.isfinite(flat[begin : begin + chunk]).all()):
                raise FloatingPointError(f"non-finite gradient in {name}")


def compare_trajectories(
    reference: list[float], variant: list[float]
) -> dict[str, float]:
    """Absolute/relative error summary between two loss trajectories."""

    if len(reference) != len(variant):
        raise ValueError("trajectory lengths differ")
    absolute = [abs(a - b) for a, b in zip(reference, variant)]
    relative = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(reference, variant)]
    return {
        "steps": len(reference),
        "max_abs_error": max(absolute),
        "mean_abs_error": sum(absolute) / len(absolute),
        "max_rel_error": max(relative),
        "mean_rel_error": sum(relative) / len(relative),
    }
