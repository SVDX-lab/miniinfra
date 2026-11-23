#!/usr/bin/env python3
"""第 01 期 Smoke Test：不下载模型、不占 GPU 的最低成本检查。

覆盖合同中的边界行为条目：
1. 手算 2×4 极小例（硬编码期望值）vs NumPy 朴素实现 vs torch 组合表达式；
2. eps 位置：全零输入输出全零（eps 在根号内），不产生 NaN；
3. NaN 传播、FP16 平方溢出防护（float32 上转换）；
4. 错误 Shape 显式报错；非连续 Tensor 的行为记录；
5. 手写 Qwen3 模型的 CPU 合成小模型前向与 Greedy 解码（不读真实权重）。
"""

from __future__ import annotations

import numpy as np
import torch

from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from rmsnorm_reference import (
    RMS_NORM_EPS,
    composite_rmsnorm_torch,
    hand_computed_example,
    naive_rmsnorm_numpy,
)

PASS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise SystemExit(f"Smoke Test 失败于: {name}")


def test_hand_computed_example() -> None:
    x, weight, expected = hand_computed_example()
    naive = naive_rmsnorm_numpy(x, weight, RMS_NORM_EPS)
    check("手算例 vs NumPy 朴素实现", np.allclose(naive, expected, atol=1e-5))
    torch_result = composite_rmsnorm_torch(
        torch.from_numpy(x), torch.from_numpy(weight), RMS_NORM_EPS
    ).numpy()
    check("手算例 vs torch 组合表达式", np.allclose(torch_result, expected, atol=1e-5))


def test_eps_and_zero_input() -> None:
    x = np.zeros((2, 8), dtype=np.float32)
    weight = np.ones(8, dtype=np.float32)
    result = naive_rmsnorm_numpy(x, weight, RMS_NORM_EPS)
    check("全零输入输出全零（eps 在根号内，无 NaN）", np.all(result == 0.0))


def test_nan_propagation() -> None:
    x = torch.ones(2, 8)
    x[0, 0] = float("nan")
    result = composite_rmsnorm_torch(x, torch.ones(8), RMS_NORM_EPS)
    check(
        "NaN 输入按逐行传播（NaN 行整行 NaN，另一行不受影响）",
        bool(torch.isnan(result[0]).all()) and bool(torch.isfinite(result[1]).all()),
    )


def test_fp16_overflow_guard() -> None:
    # FP16 最大值 65504：平方 = 4.29e9，超出 FP16 范围；若不升 FP32 会产生 Inf。
    x = torch.full((1, 8), 65504.0, dtype=torch.float16)
    weight = torch.ones(8, dtype=torch.float16)
    result = composite_rmsnorm_torch(x, weight, RMS_NORM_EPS)
    check("FP16 大数平方不溢出（FP32 上转换生效）", bool(torch.isfinite(result).all()))


def test_invalid_shape() -> None:
    try:
        composite_rmsnorm_torch(torch.ones(8), torch.ones(8), RMS_NORM_EPS)
    except ValueError:
        check("一维输入显式报错", True)
        return
    check("一维输入显式报错", False)


def test_non_contiguous_behavior() -> None:
    base = torch.randn(8, 4)
    transposed = base.t()  # [4, 8]，非连续
    result = composite_rmsnorm_torch(transposed, torch.ones(8), RMS_NORM_EPS)
    contiguous = composite_rmsnorm_torch(transposed.contiguous(), torch.ones(8), RMS_NORM_EPS)
    check(
        "非连续输入结果与连续一致（本期合同：原生表达式接受任意 stride）",
        torch.equal(result, contiguous),
    )


def test_tiny_model_greedy() -> None:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    generated, finished = model.greedy_generate(input_ids, max_new_tokens=5, eos_token_id=None)
    check(
        "合成小模型 Greedy 解码 Shape 正确",
        generated.shape == (1, 8) and not finished and bool((generated[:, :3] == input_ids[0]).all()),
    )
    logits, probes = model(input_ids, probe=True)
    check(
        "探针数量 = 2 层 + embedding + final_norm",
        set(probes) == {"embedding", "layer_0", "layer_1", "final_norm"},
    )
    check("探针 dtype 为 FP32", all(tensor.dtype == torch.float32 for tensor in probes.values()))
    check("logits Shape = [batch, seq, vocab]", tuple(logits.shape) == (1, 3, 32))


def main() -> int:
    test_hand_computed_example()
    test_eps_and_zero_input()
    test_nan_propagation()
    test_fp16_overflow_guard()
    test_invalid_shape()
    test_non_contiguous_behavior()
    test_tiny_model_greedy()
    print("[done] Smoke Test 全部通过（CPU-only，未下载模型）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
