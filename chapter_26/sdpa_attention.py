"""Chapter 26 core: swappable attention kernels for training.

``Qwen3Attention.attention_impl`` receives ``(query, key, value,
causal_mask, head_dim)`` where query is ``[batch, heads, seq, dim]`` and
key/value are still UN-repeated ``[batch, kv_heads, seq, dim]`` (GQA).  Every
implementation must produce the same ``[batch, heads, seq, dim]`` output and
work in both Forward and Backward.

- ``eager_attention`` materializes the full ``[batch, heads, seq, seq]``
  score and probability matrices - the Chapter 22 baseline, byte-identical to
  the inline body when ``attention_impl is None``.
- ``make_sdpa_attention(backend)`` wraps ``F.scaled_dot_product_attention``
  pinned to one backend via ``torch.nn.attention.sdpa_kernel``.  The MATH
  backend also materializes scores (mechanism control); the EFFICIENT backend
  tiles the computation and never forms the full matrix.
- ``flash_attention`` uses the FLASH backend with ``enable_gqa``: BF16/FP16
  CUDA only, and it cannot consume an arbitrary additive mask, so it takes
  ``is_causal=True`` and is only valid for padding-free batches.

``probe_backend_support`` measures which backends actually run on this
machine/dtype/mask combination instead of assuming.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from qwen3_model import Qwen3Attention


AttentionImpl = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]


def _repeat_to_query_heads(query, key, value):
    groups = query.shape[1] // key.shape[1]
    if groups == 1:
        return key, value
    return key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)


def eager_attention(query, key, value, causal_mask, head_dim):
    """Baseline kernel: materializes the full score/probability matrices."""

    key, value = _repeat_to_query_heads(query, key, value)
    attention_scores = torch.matmul(query, key.transpose(2, 3))
    attention_scores = attention_scores / math.sqrt(head_dim)
    attention_scores = attention_scores + causal_mask
    attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
    attention_probs = attention_probs.to(query.dtype)
    return torch.matmul(attention_probs, value)


def make_sdpa_attention(backend: SDPBackend) -> AttentionImpl:
    """SDPA pinned to one backend; keeps the explicit additive mask."""

    def implementation(query, key, value, causal_mask, head_dim):
        key, value = _repeat_to_query_heads(query, key, value)
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                scale=1.0 / math.sqrt(head_dim),
            )

    return implementation


def flash_attention(query, key, value, causal_mask, head_dim):
    """FLASH backend: requires BF16/FP16 CUDA, padding-free batches.

    The flash kernel cannot consume an additive mask; ``is_causal=True``
    expresses the causal part.  GQA is passed natively via ``enable_gqa``
    instead of materializing repeated keys/values.
    """

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=1.0 / math.sqrt(head_dim),
            enable_gqa=key.shape[1] != query.shape[1],
        )


ATTENTION_IMPLEMENTATIONS: dict[str, AttentionImpl] = {
    "eager": eager_attention,
    "sdpa_math": make_sdpa_attention(SDPBackend.MATH),
    "sdpa_mem_efficient": make_sdpa_attention(SDPBackend.EFFICIENT_ATTENTION),
    "flash": flash_attention,
}


def set_attention_implementation(model: torch.nn.Module, name: str) -> None:
    if name not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(f"unknown attention implementation: {name}")
    if name == "eager":
        # eager 走模块内置路径，保持与第 22 期逐位一致。
        implementation = None
    else:
        implementation = ATTENTION_IMPLEMENTATIONS[name]
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module.attention_impl = implementation


def probe_backend_support(device: torch.device) -> list[dict]:
    """Try every backend on this machine for FP32/BF16 with and without mask."""

    results = []
    generator = torch.Generator(device=device.type)
    generator.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        for mask_name, mask_kind in (("additive_causal", "mask"), ("is_causal_flag", "flag")):
            query = torch.randn(1, 4, 64, 8, generator=generator, device=device, dtype=dtype, requires_grad=True)
            key = torch.randn(1, 2, 64, 8, generator=generator, device=device, dtype=dtype, requires_grad=True)
            value = torch.randn(1, 2, 64, 8, generator=generator, device=device, dtype=dtype, requires_grad=True)
            if mask_kind == "mask":
                scores = torch.zeros(64, 64, device=device, dtype=dtype)
                scores = scores.masked_fill(
                    torch.triu(torch.ones(64, 64, device=device, dtype=torch.bool), diagonal=1),
                    torch.finfo(dtype).min,
                )
                mask_argument = {"attn_mask": scores.unsqueeze(0).unsqueeze(0)}
            else:
                mask_argument = {"is_causal": True}
            for name in ("MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION"):
                backend = getattr(SDPBackend, name)
                try:
                    with sdpa_kernel(backend):
                        output = F.scaled_dot_product_attention(
                            query, key, value, scale=1.0 / math.sqrt(8), enable_gqa=True, **mask_argument
                        )
                    loss = output.float().pow(2).sum()
                    loss.backward()
                    # 清掉探测用的梯度图
                    query.grad = key.grad = value.grad = None
                    status = "ok"
                    message = ""
                except Exception as error:  # noqa: BLE001 - census records failures
                    status = "unavailable"
                    message = str(error).splitlines()[0][:160]
                results.append({
                    "backend": name,
                    "dtype": str(dtype).replace("torch.", ""),
                    "mask": mask_name,
                    "status": status,
                    "message": message,
                })
    return results
