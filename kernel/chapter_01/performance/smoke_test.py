#!/usr/bin/env python3
"""第 01 期性能诊断 smoke test（CPU-only）：roofline 方法论自检 + 被测对象 sanity。

检查项：
1. 理论字节/FLOPs/AI/延迟下限公式 vs 手算数（硬编码期望值）；
2. 未融合链路流量估算 vs 融合口径的倍率；
3. bound 判定逻辑的单元检查（合成数字，不依赖 GPU）；
4. 被测对象 sanity：composite_rmsnorm_torch（CPU FP32/FP16）与 NumPy 真值对照
   （第 01 期比较口径，验证本期跑的确实是同一个语义）；
5. fixtures 内联生成路径与落盘 fixtures 位级一致（同 seed 双路）。
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import torch

from roofline import (
    classify_eager_mode,
    classify_kernel_bound,
    latency_lower_bound_ms,
    per_size_launch_bound,
    theoretical_bytes,
    theoretical_flops,
    unfused_chain_bytes,
    unfused_chain_bytes_per_row,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmsnorm_reference import naive_rmsnorm_numpy, composite_rmsnorm_torch

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


def test_formulas() -> None:
    # 手算：rows=1, H=1024, fp16 → (1*1024*2 + 1024)*2 = 6144
    check("theoretical_bytes(1,1024,2)=6144", theoretical_bytes(1, 1024, 2) == 6144)
    check("theoretical_bytes(2048,1024,2)=8390656", theoretical_bytes(2048, 1024, 2) == 8_390_656)
    # FLOPs：2048*(4*1024+3) = 2048*4099 = 8390656*... 手算 = 8_394_752
    check("theoretical_flops(2048,1024)=8394752", theoretical_flops(2048, 1024) == 2048 * 4099)
    # AI = 8394752/8390656 ≈ 1.00049 FLOP/B
    ai = theoretical_flops(2048, 1024) / theoretical_bytes(2048, 1024, 2)
    check("arithmetic_intensity ≈ 1 FLOP/B", abs(ai - 1.0) < 0.01, f"AI={ai:.4f}")
    # 下限：8390656 B / 1008 GB/s = 8.324 µs
    lb = latency_lower_bound_ms(8_390_656, 1008e9)
    check("latency_lower_bound ≈ 8.32 µs @1008GB/s", abs(lb * 1000 - 8.324) < 0.01, f"{lb*1000:.3f} µs")
    lb_musa = latency_lower_bound_ms(8_390_656, 768e9)
    check("latency_lower_bound ≈ 10.925 µs @768GB/s", abs(lb_musa * 1000 - 10.925) < 0.01, f"{lb_musa*1000:.3f} µs")
    # 未融合链路：per-row = 38H+24 = 38936（reads 22H+12 + writes 16H+12）
    per_row = unfused_chain_bytes_per_row(1024)
    check("unfused per-row = 38H+24 = 38936", per_row["total"] == 38 * 1024 + 24 == 38936,
          f"total={per_row['total']}")
    ratio = unfused_chain_bytes(2048, 1024, 2) / theoretical_bytes(2048, 1024, 2)
    check("unfused/fused ≈ 9.5x", 9.0 < ratio < 10.0, f"ratio={ratio:.2f}x")


def test_classification_logic() -> None:
    # 内存受限判定：bytes=1GB, flops=1M → t_mem >> t_compute
    verdict = classify_kernel_bound(1 << 30, 1_000_000, 1008e9, 165.2e12)
    check("deep memory-bound 判定", verdict["kernel_bound_class"] == "memory-bound (deep)",
          f"{verdict}")
    # 计算受限：bytes 小、flops 巨大
    verdict = classify_kernel_bound(1_000_000, 500e12, 1008e9, 165.2e12)
    check("compute-bound 判定", "compute-bound" in verdict["kernel_bound_class"], f"{verdict}")
    # launch-dominated：eager 0.12ms vs gpu 0.033ms
    verdict = classify_eager_mode(0.1208, 0.0328)
    check("launch-dominated 判定", verdict["eager_mode_class"] == "launch-dominated (eager 被 CPU 发射主导)",
          f"ratio={verdict['eager_over_gpu_ratio']}")
    verdict = classify_eager_mode(0.033, 0.0328)
    check("gpu-dominated 判定", verdict["eager_mode_class"] == "gpu-dominated", f"{verdict}")
    # 逐尺寸 launch-bound：tight 5µs, gpu 1µs → bound；tight 10µs, gpu 9µs → not
    check("per-size launch-bound True", per_size_launch_bound(5.0, 1.0)["launch_bound"])
    check("per-size launch-bound False", not per_size_launch_bound(10.0, 9.0)["launch_bound"])


def test_truth_sanity() -> None:
    generator = torch.Generator().manual_seed(20260101)
    x = torch.randn(4, 128, generator=generator)
    weight = torch.randn(128, generator=generator)
    expected = naive_rmsnorm_numpy(x.numpy(), weight.numpy())
    actual = composite_rmsnorm_torch(x, weight, 1e-6).numpy()
    max_err = np.abs(actual - expected).max()
    check("composite vs numpy truth (fp32 CPU) max_err < 1e-5", max_err < 1e-5, f"max_err={max_err:.2e}")
    # fp16 CPU 路径（GPU fp16 的数量级参照）
    actual16 = composite_rmsnorm_torch(x.half(), weight.half(), 1e-6).float().numpy()
    max_err16 = np.abs(actual16 - expected).max()
    check("composite fp16 vs fp32 truth max_err < 5e-3", max_err16 < 5e-3, f"max_err={max_err16:.2e}")


def test_fixture_inline_consistency() -> None:
    fixture = Path("fixtures") / "x-4x1024-fp32.npy"
    if not fixture.exists():
        print("[SKIP] fixtures 不存在（先跑 generate_fixtures.py）")
        return
    on_disk = np.load(fixture)
    generator = torch.Generator().manual_seed(20260101 + 4 * 100000 + 1024)
    inline = torch.randn(4, 1024, generator=generator).numpy()
    check("fixture 与内联生成位级一致", np.array_equal(on_disk, inline),
          f"max_abs_diff={np.abs(on_disk - inline).max():.2e}")


def main() -> int:
    print("=== 第 01 期性能诊断 smoke test（CPU-only）===")
    test_formulas()
    test_classification_logic()
    test_truth_sanity()
    test_fixture_inline_consistency()
    if FAILURES:
        print(f"=== {len(FAILURES)} 项失败: {FAILURES} ===")
        return 1
    print("=== 全部通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
