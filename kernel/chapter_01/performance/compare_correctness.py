#!/usr/bin/env python3
"""第 01 期性能诊断正确性比较（轻量版，复用第 01 期比较口径）。

第 01 期性能诊断没有新的数值合同：被测对象与第 01 期是同一个 PyTorch 组合表达式
（rmsnorm_reference.py 由第一期共用），完整的正确性四件套（18 组合同
Shape fixtures、手算例、模型真值、确定性）已在第 01 期冻结，本章不重复。

本期 compare 的角色是"测量前的 sanity 锚"：对每个 benchmark shape 确认
composite（各后端）与 NumPy FP32 真值的误差在合同容差内——确保后续性能
数字测的确实是那个正确的算子。比较口径直接复用 ch01 experiment_utils
的 comparison_metrics（max/mean 绝对误差、相对误差、余弦相似度、NaN/Inf）。

用法：
    python compare_correctness.py --backend cpu
    python compare_correctness.py --backend cuda
    python compare_correctness.py --backend musa
输出：results/correctness-results-{backend}.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_utils import (
    backend_sync,
    comparison_metrics,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from roofline import HIDDEN_SIZE, ROW_COUNTS, RMS_NORM_EPS
from rmsnorm_reference import naive_rmsnorm_numpy, composite_rmsnorm_torch

SCHEMA_VERSION = "kernel-chapter02.correctness.v1"
# 合同容差：FP32 应在 1e-5 量级；FP16 对照容差 1e-2 —— 依据 ch01 dtype 误差分解实测
# （fp16 composite vs fp32 composite max_abs=7.03e-3，见 ch01 correctness-results-cuda.json）
TOLERANCE = {"float32": 1e-5, "float16": 1e-2}


def load_or_build(rows: int, hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fixture = Path("fixtures") / f"x-{rows}x{hidden_size}-fp32.npy"
    if fixture.exists():
        return torch.from_numpy(np.load(fixture)).to(device=device, dtype=dtype)
    generator = torch.Generator().manual_seed(20260101 + rows * 100000 + hidden_size)
    return torch.randn(rows, hidden_size, generator=generator).to(device=device, dtype=dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    weight_source = Path("fixtures") / "weight-fp32.npy"
    weight = (
        torch.from_numpy(np.load(weight_source)).to(device=device, dtype=dtype)
        if weight_source.exists()
        else torch.randn(HIDDEN_SIZE, generator=torch.Generator().manual_seed(20260101)).to(
            device=device, dtype=dtype
        )
    )

    entries = []
    worst = 0.0
    for rows in ROW_COUNTS:
        x = load_or_build(rows, HIDDEN_SIZE, device, dtype)
        actual = composite_rmsnorm_torch(x, weight, RMS_NORM_EPS)
        backend_sync(device)
        expected = torch.from_numpy(
            naive_rmsnorm_numpy(
                x.detach().cpu().float().numpy(), weight.detach().cpu().float().numpy(), RMS_NORM_EPS
            )
        )
        metrics = comparison_metrics(actual, expected)
        metrics["rows"] = rows
        metrics["hidden_size"] = HIDDEN_SIZE
        metrics["dtype"] = str(dtype).replace("torch.", "")
        metrics["within_tolerance"] = metrics["max_abs_error"] <= TOLERANCE[str(dtype).replace("torch.", "")]
        worst = max(worst, metrics["max_abs_error"])
        entries.append(metrics)
        print(
            f"[correctness/{arguments.backend}] rows={rows:<5} max_abs={metrics['max_abs_error']:.3e} "
            f"mean_abs={metrics['mean_abs_error']:.3e} cos={metrics['cosine_similarity']:.8f} "
            f"nan={metrics['nan_count']} inf={metrics['inf_count']} "
            f"容差内={metrics['within_tolerance']}"
        )

    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "tolerance": TOLERANCE,
        "worst_max_abs_error": worst,
        "all_within_tolerance": all(e["within_tolerance"] for e in entries),
        "note": "第 01 期性能诊断为测量前 sanity 锚；完整正确性合同（18 组 Shape/手算例/模型真值/确定性）沿用第 01 期四件套",
        "entries": entries,
        "environment": device_environment_record(device),
    }
    output = Path("results") / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    return 0 if results["all_within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
