#!/usr/bin/env python3
"""第 16 期正确性对照：各后端各变体输出 vs fixtures 中 CPU FP32 真值。

口径（照第 01 期 comparison_metrics）：max/mean 绝对误差、相对误差、余弦相似度、
NaN/Inf 计数、最大误差位置；另加：
- 确定性：同输入连续 3 次运行位级比对（第 07 期口径前置采样）；
- transpose 额外记录位级相等（纯重排应 bit-exact）；
- pass 判据：cosine ≥ 0.9999 且无 NaN/Inf（判据写进结果 JSON，不静默放宽）。

用法：python compare_correctness.py --backend {cpu,cuda,musa}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiment_utils import (
    backend_sync,
    comparison_metrics,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from ops_harness import (
    CPU_VARIANTS,
    load_binding,
    GPU_VARIANTS,
    SHAPES,
    cpu_truth,
    load_fixtures,
    run_cpu_variant,
    run_gpu_variant,
    run_torch_eager,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "implementations" / "cuda"))
sys.path.insert(0, str(ROOT / "implementations" / "musa"))

PASS_COSINE = 0.9999


def check_pass(metrics: dict) -> bool:
    return (
        metrics["nan_count"] == 0
        and metrics["inf_count"] == 0
        and metrics["cosine_similarity"] >= PASS_COSINE
    )


def compare_against_truth(actual: torch.Tensor, truth: torch.Tensor, op: str, variant: str,
                          rows: int, cols: int, backend: str) -> dict:
    metrics = comparison_metrics(actual, truth)
    metrics.update({
        "op": op,
        "variant": variant,
        "rows": rows,
        "cols": cols,
        "backend": backend,
        "pass": check_pass(metrics),
    })
    if op == "transpose":
        metrics["bitwise_equal_to_truth"] = bool(
            torch.equal(actual.detach().float().cpu(), truth.detach().float().cpu())
        )
    return metrics


def run_backend(backend: str, fixtures: dict, manifest: dict) -> dict:
    device = resolve_device(backend)
    results: dict = {
        "schema_version": "kernel-chapter17.correctness.v1",
        "backend": backend,
        "pass_judgement": f"cosine >= {PASS_COSINE} 且 NaN/Inf 计数为 0",
        "fixtures_npz_sha256": manifest.get("npz_sha256"),
        "entries": [],
    }

    if backend == "cpu":
        arrays = fixtures
        for rows, cols in SHAPES:
            for op, variants in CPU_VARIANTS.items():
                truth = cpu_truth(op, arrays, rows, cols)
                for variant in variants:
                    output = torch.from_numpy(np.ascontiguousarray(run_cpu_variant(op, arrays, rows, cols)))
                    entry = compare_against_truth(output, truth, op, f"{variant}", rows, cols, backend)
                    results["entries"].append(entry)
                    status = "PASS" if entry["pass"] else "FAIL"
                    print(f"[{status}] cpu {op}/{variant} {rows}x{cols} "
                          f"max_abs={entry['max_abs_error']:.3e} cos={entry['cosine_similarity']:.8f}")
        # torch 组合表达式交叉印证（CPU）：真值 vs torch
        for rows, cols in SHAPES:
            key = f"{rows}x{cols}"
            x = torch.from_numpy(arrays[f"x_{key}"])
            up = torch.from_numpy(arrays[f"up_{key}"])
            cross_checks = {
                "softmax": torch.softmax(x, dim=-1),
                "reduce": torch.cat([x.sum(dim=1), x.max(dim=1).values]),
                "transpose": x.t().contiguous(),
                "scan": torch.cumsum(x, dim=1),
                "silu_gate": torch.nn.functional.silu(x) * up,
            }
            for op, output in cross_checks.items():
                truth = cpu_truth(op, arrays, rows, cols)
                entry = compare_against_truth(output, truth, op, "torch_composite", rows, cols, backend)
                results["entries"].append(entry)
                status = "PASS" if entry["pass"] else "FAIL"
                print(f"[{status}] cpu {op}/torch_composite {rows}x{cols} "
                      f"max_abs={entry['max_abs_error']:.3e}")
        results["environment"] = device_environment_record(device)
        return results

    # GPU 后端：加载 C ABI Kernel 库
    lib_name = "libops_cuda.so" if backend == "cuda" else "libops_musa.so"
    binding = load_binding(backend, lib_name)

    for rows, cols in SHAPES:
        buffers = {
            "x": torch.empty(rows, cols, dtype=torch.float32, device=device),
            "up": torch.empty(rows, cols, dtype=torch.float32, device=device),
            "y_softmax": torch.empty(rows, cols, dtype=torch.float32, device=device),
            "y_scan": torch.empty(rows, cols, dtype=torch.float32, device=device),
            "y_transpose": torch.empty(cols, rows, dtype=torch.float32, device=device),
            "sum_out": torch.empty(rows, dtype=torch.float32, device=device),
            "max_out": torch.empty(rows, dtype=torch.float32, device=device),
            "tmp": torch.empty(rows, cols, dtype=torch.float32, device=device),
            "y_silu": torch.empty(rows, cols, dtype=torch.float32, device=device),
        }
        key = f"{rows}x{cols}"
        buffers["x"].copy_(torch.from_numpy(fixtures[f"x_{key}"]))
        buffers["up"].copy_(torch.from_numpy(fixtures[f"up_{key}"]))
        backend_sync(device)

        for op, variants in GPU_VARIANTS.items():
            truth = cpu_truth(op, fixtures, rows, cols).to(device)  # eager 对照用拼接真值
            for variant in variants:
                variant_truth = cpu_truth(op, fixtures, rows, cols, variant).to(device)
                started = time.perf_counter()
                output = run_gpu_variant(binding, op, variant, buffers, rows, cols)
                backend_sync(device)
                entry = compare_against_truth(output, variant_truth, op, variant, rows, cols, backend)

                # 确定性：连续 3 次位级比对（先再跑两次，分别与首次比较）
                deterministic = True
                for _ in range(2):
                    rerun = run_gpu_variant(binding, op, variant, buffers, rows, cols)
                    backend_sync(device)
                    if not torch.equal(rerun, output):
                        deterministic = False
                entry["deterministic_3x_bitwise"] = deterministic
                entry["first_run_wall_us"] = round((time.perf_counter() - started) * 1e6, 1)

                results["entries"].append(entry)
                status = "PASS" if entry["pass"] else "FAIL"
                extra = f" bitexact={entry['bitwise_equal_to_truth']}" if op == "transpose" else ""
                print(f"[{status}] {backend} {op}/{variant} {rows}x{cols} "
                      f"max_abs={entry['max_abs_error']:.3e} cos={entry['cosine_similarity']:.8f}"
                      f" det={deterministic}{extra}")

            # torch eager 对照组也做正确性对照（同真值）
            eager_variant = {
                "softmax": "torch_softmax", "reduce": "torch_sum_max",
                "transpose": "torch_t_contiguous", "scan": "torch_cumsum",
                "silu_gate": "torch_silu_mul",
            }[op]
            eager_output = run_torch_eager(op, buffers)
            backend_sync(device)
            entry = compare_against_truth(eager_output, truth, op, eager_variant, rows, cols, backend)
            results["entries"].append(entry)
            status = "PASS" if entry["pass"] else "FAIL"
            print(f"[{status}] {backend} {op}/{eager_variant} {rows}x{cols} "
                  f"max_abs={entry['max_abs_error']:.3e}")

        del buffers
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "musa":
            torch.musa.empty_cache()

    # softmax 数值稳定性压力例（x+100）：GPU kernel 输出 vs 稳定真值
    stress_x = torch.from_numpy(fixtures["x_stress"]).to(device)
    stress_truth = torch.from_numpy(fixtures["y_softmax_stress"]).to(device)
    stress_y = torch.empty_like(stress_x)
    binding.softmax_rows(stress_x, stress_y)
    backend_sync(device)
    entry = compare_against_truth(stress_y, stress_truth, "softmax", "kernel_stress_x_plus_100",
                                  int(stress_x.shape[0]), int(stress_x.shape[1]), backend)
    results["entries"].append(entry)
    status = "PASS" if entry["pass"] else "FAIL"
    print(f"[{status}] {backend} softmax/stress(x+100) 4x1024 max_abs={entry['max_abs_error']:.3e}")

    results["environment"] = device_environment_record(device)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    arguments = parser.parse_args()

    fixtures, manifest = load_fixtures()
    results = run_backend(arguments.backend, fixtures, manifest)

    failed = [entry for entry in results["entries"] if not entry["pass"]]
    results["failed_count"] = len(failed)
    output = ROOT / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    if failed:
        print(f"[summary] {len(failed)} 个条目未达判据——明细见 {output}")
        return 1
    print(f"[summary] 全部 {len(results['entries'])} 个条目通过判据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
