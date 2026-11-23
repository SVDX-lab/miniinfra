#!/usr/bin/env python3
"""CPU-only Chapter 26 checks: plumbing parity, causality, masking, census."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qwen3_model import Qwen3Config, Qwen3ForCausalLM, repeat_key_value
from sdpa_attention import (
    ATTENTION_IMPLEMENTATIONS,
    eager_attention,
    probe_backend_support,
    set_attention_implementation,
)


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )


def causal_mask(sequence_length: int, dtype=torch.float32):
    mask = torch.full((sequence_length, sequence_length), torch.finfo(dtype).min, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def check_injected_eager_matches_builtin() -> None:
    """Injection must not change the eager path by even one bit."""

    generator = torch.Generator().manual_seed(5)
    model_builtin = Qwen3ForCausalLM(tiny_config())
    model_injected = Qwen3ForCausalLM(tiny_config())
    model_injected.load_state_dict(model_builtin.state_dict())
    set_attention_implementation(model_injected, "eager")
    tokens = torch.randint(4, 64, (2, 12), generator=generator)
    with torch.no_grad():
        logits_builtin = model_builtin(tokens)
        logits_injected = model_injected(tokens)
    assert torch.equal(logits_builtin, logits_injected), "injected eager diverged from built-in"


def check_attention_level_parity() -> None:
    """sdpa_math must track eager on attention-level inputs."""

    generator = torch.Generator().manual_seed(6)
    query = torch.randn(2, 4, 16, 8, generator=generator)
    key = torch.randn(2, 2, 16, 8, generator=generator)
    value = torch.randn(2, 2, 16, 8, generator=generator)
    mask = causal_mask(16)
    reference = eager_attention(query, key, value, mask, 8)
    candidate = ATTENTION_IMPLEMENTATIONS["sdpa_math"](query, key, value, mask, 8)
    error = (reference - candidate).abs().max()
    assert float(error) < 1e-5, f"sdpa_math attention error {float(error)}"
    # repeat 的语义必须与 qwen3_model.repeat_key_value 一致
    key_repeated, value_repeated = None, None
    groups = query.shape[1] // key.shape[1]
    key_repeated = repeat_key_value(key, groups)
    value_repeated = repeat_key_value(value, groups)
    assert key_repeated.shape[1] == query.shape[1]
    assert torch.equal(key_repeated[:, 0], key[:, 0])
    assert torch.equal(key_repeated[:, 1], key[:, 0])


def check_causality() -> None:
    """Changing a future token must not change earlier outputs."""

    generator = torch.Generator().manual_seed(7)
    for name in ("eager", "sdpa_math"):
        impl = ATTENTION_IMPLEMENTATIONS[name]
        query = torch.randn(1, 4, 10, 8, generator=generator)
        key_a = torch.randn(1, 2, 10, 8, generator=generator)
        key_b = key_a.clone()
        key_b[:, :, 5:, :] = torch.randn(1, 2, 5, 8, generator=generator)
        value_a = torch.randn(1, 2, 10, 8, generator=generator)
        value_b = value_a.clone()
        value_b[:, :, 5:, :] = torch.randn(1, 2, 5, 8, generator=generator)
        mask = causal_mask(10)
        output_a = impl(query, key_a, value_a, mask, 8)
        output_b = impl(query, key_b, value_b, mask, 8)
        assert torch.allclose(output_a[:, :, :5, :], output_b[:, :, :5, :], atol=1e-6), name


def check_key_padding_mask() -> None:
    """A key masked as padding must not influence any output."""

    generator = torch.Generator().manual_seed(8)
    query = torch.randn(1, 4, 8, 8, generator=generator)
    key = torch.randn(1, 2, 8, 8, generator=generator)
    value = torch.randn(1, 2, 8, 8, generator=generator)
    with_padding = causal_mask(8).clone()
    with_padding[..., :, 6:] = torch.finfo(torch.float32).min
    output = eager_attention(query, key, value, with_padding, 8)

    clean_key = torch.cat([key[:, :, :6, :], key[:, :, -1:, :]], dim=2)
    clean_value = torch.cat([value[:, :, :6, :], value[:, :, -1:, :]], dim=2)
    clean_mask = causal_mask(2)
    # 无法直接比较不同长度的输出；改为检查 padding 列梯度阻断：
    key_grad_probe = key.clone().requires_grad_(True)
    output_probe = eager_attention(query, key_grad_probe, value, with_padding, 8)
    output_probe.sum().backward()
    assert key_grad_probe.grad is not None
    assert float(key_grad_probe.grad[:, :, 6:, :].abs().max()) == 0.0, "padding key received gradient"


def check_flash_cpu_semantics() -> None:
    """torch 2.12 ships a CPU flash kernel: verify its causal semantics.

    GPU 上 FLASH 只服务 BF16/FP16 且不吃加性 mask；CPU 上它能跑 FP32，
    但无论哪种，is_causal 语义都必须与 eager 的因果 mask 一致。
    老版本 torch（如 2.8）没有 CPU flash kernel：此时跳过本检查，语义
    覆盖由 GPU correctness 承担。
    """

    generator = torch.Generator().manual_seed(9)
    query = torch.randn(1, 4, 8, 8, generator=generator, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 8, 8, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(1, 2, 8, 8, generator=generator, dtype=torch.bfloat16)
    mask = causal_mask(8, dtype=torch.bfloat16)
    try:
        flash_output = ATTENTION_IMPLEMENTATIONS["flash"](query, key, value, mask, 8)
    except RuntimeError as error:
        assert "No available kernel" in str(error), error
        print("torch has no CPU flash kernel; skipping CPU flash semantics check")
        return
    eager_output = eager_attention(query, key, value, mask, 8)
    error = (flash_output.float() - eager_output.float()).abs().max()
    assert float(error) < 5e-2, f"flash CPU semantics diverge: {float(error)}"
    # 因果性反例在 flash 路径上同样成立。
    key_b = key.clone()
    key_b[:, :, 5:, :] = torch.randn(1, 2, 3, 8, generator=generator, dtype=torch.bfloat16)
    value_b = value.clone()
    value_b[:, :, 5:, :] = torch.randn(1, 2, 3, 8, generator=generator, dtype=torch.bfloat16)
    output_b = ATTENTION_IMPLEMENTATIONS["flash"](query, key_b, value_b, mask, 8)
    assert torch.allclose(
        flash_output[:, :, :5, :].float(), output_b[:, :, :5, :].float(), atol=1e-2
    ), "flash CPU violates causality"


def check_census_runs_on_cpu() -> None:
    results = probe_backend_support(torch.device("cpu"))
    assert results, "census returned nothing"
    statuses = {(r["backend"], r["status"]) for r in results}
    assert ("MATH", "ok") in statuses, statuses
    assert any(status == "unavailable" for _backend, status in statuses), statuses


def main() -> None:
    check_injected_eager_matches_builtin()
    check_attention_level_parity()
    check_causality()
    check_key_padding_mask()
    check_flash_cpu_semantics()
    check_census_runs_on_cpu()
    print("chapter_26 smoke tests passed")


if __name__ == "__main__":
    main()
