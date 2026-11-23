#!/usr/bin/env python3
"""第 12 期 fixtures（固定 seed，参照第 01 期口径）。

本期 fixtures 是"能力探测与 dispatch 测试"的输入锚点（模型级复核直接消费
第 01 期 truth/ 共享锚点，不复制权重级真值）：
  - op_probe_{fp32,fp16,bf16}：融合 op / 组合表达式的 seeded 输入三元组
    （input/residual/weight，shape 8×1024）与 FP32 组合表达式期望输出；
  - rmsnorm_probe_fp32：独立 RMSNorm dispatch 测试输入与期望输出；
  - dispatch_prompt：第 01 期 truth prompt token ids（跨期共享锚点的引用）。

产物：fixtures/probes.npz + fixtures/manifest.json（逐张量 SHA256）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE / "implementations" / "runtime")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import sha256_of_tensor, write_results_json  # noqa: E402

SEED = 20261113
PROBE_ROWS, PROBE_HIDDEN = 8, 1024


def composite_reference(r: torch.Tensor, h: torch.Tensor, w: torch.Tensor,
                        eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """第 04 期 CPU composite 语义（期望输出生成口径，FP32）。"""
    t = r + h
    float_states = t.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    y = w * (float_states * torch.rsqrt(variance + eps)).to(t.dtype)
    return t, y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(HERE / "fixtures"))
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(SEED)

    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, object] = {
        "schema_version": "kernel-chapter13.fixtures.v1",
        "seed": SEED,
        "probe_shape": [PROBE_ROWS, PROBE_HIDDEN],
        "tensors": {},
        "shared_anchor": {
            "truth": "chapter_01/truth/truth-manifest.json（模型级复核锚点，本期不复制）",
        },
    }

    for dtype_name in ("float32", "float16", "bfloat16"):
        dtype = getattr(torch, dtype_name)
        r = torch.randn(PROBE_ROWS, PROBE_HIDDEN, generator=generator).to(dtype)
        h = torch.randn(PROBE_ROWS, PROBE_HIDDEN, generator=generator).to(dtype)
        w = torch.randn(PROBE_HIDDEN, generator=generator).to(dtype)

        def _store(name: str, tensor: torch.Tensor) -> None:
            # numpy 无 BFloat16 标量类型——bf16 以 uint8 原始字节视图存档
            # （storage_format 字段显式声明；消费方重建用 torch.frombuffer）。
            if tensor.dtype == torch.bfloat16:
                arrays[name] = tensor.contiguous().view(torch.uint8).numpy()
            else:
                arrays[name] = tensor.numpy()

        _store(f"op_probe_{dtype_name}_input", h)
        _store(f"op_probe_{dtype_name}_residual", r)
        _store(f"op_probe_{dtype_name}_weight", w)
        # 期望输出只在 FP32 生成（FP32 真值锚点口径；fp16/bf16 期望随 dtype
        # 语义在后端上即时计算——fixtures 不预置低精度"真值"）。
        if dtype_name == "float32":
            t, y = composite_reference(r, h, w)
            arrays["op_probe_fp32_expected_residual"] = t.numpy()
            arrays["op_probe_fp32_expected_output"] = y.numpy()
        for key, tensor in (
            (f"op_probe_{dtype_name}_input", h),
            (f"op_probe_{dtype_name}_residual", r),
            (f"op_probe_{dtype_name}_weight", w),
        ):
            manifest["tensors"][key] = {
                "sha256": sha256_of_tensor(tensor),
                "dtype": dtype_name,
                "shape": list(tensor.shape),
                "storage_format": "uint8-raw-bytes" if dtype == torch.bfloat16
                else "numpy-native",
            }

    x = torch.randn(PROBE_ROWS, PROBE_HIDDEN, generator=generator)
    w = torch.randn(PROBE_HIDDEN, generator=generator)
    float_states = x.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    y = w * (float_states * torch.rsqrt(variance + 1e-6)).to(x.dtype)
    arrays["rmsnorm_probe_fp32_input"] = x.numpy()
    arrays["rmsnorm_probe_fp32_weight"] = w.numpy()
    arrays["rmsnorm_probe_fp32_expected"] = y.numpy()
    manifest["tensors"]["rmsnorm_probe_fp32_input"] = {
        "sha256": sha256_of_tensor(x), "dtype": "float32", "shape": list(x.shape)}
    manifest["tensors"]["rmsnorm_probe_fp32_weight"] = {
        "sha256": sha256_of_tensor(w), "dtype": "float32", "shape": list(w.shape)}
    manifest["tensors"]["rmsnorm_probe_fp32_expected"] = {
        "sha256": sha256_of_tensor(y), "dtype": "float32", "shape": list(y.shape)}

    np.savez_compressed(out_dir / "probes.npz", **arrays)
    manifest["arrays"] = sorted(arrays)
    write_results_json(out_dir / "manifest.json", manifest)
    print(f"[fixtures] {len(arrays)} tensors, seed={SEED} → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
