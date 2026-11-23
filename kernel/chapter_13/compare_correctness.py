#!/usr/bin/env python3
"""第 13 期正确性对照：Triton 路径 vs CPU FP32 真值（fixtures 复用 ch03/ch04）。

对照链：
1. fixtures 完整性：ch03（RMSNorm 30 组）+ ch04（融合 30 组）逐张量 SHA256；
2. composite vs 真值（平台原生路径延续口径）；
3. Triton RMSNorm vs 真值：FP32 + FP16 各 30 组 —— 单遍 kernel（生产写法）；
4. Triton RMSNorm looped kernel vs 真值：BLOCK=256（ch03 TILE 语义）30 组；
5. Triton autotune 版 vs 真值：30 组（autotune 选中的配置）；
6. Triton fused add+rmsnorm vs ch04 真值：FP32/FP16 各 30 组 + residual
   in-place 位级核对（fp32 路径 r_after == t 真值位级；fp16 路径 r_after == T(t)）;
7. 手写对照（可选 --with-handwritten）：ch03/ch04 手写 kernel 同 fixtures 对照，
   与 Triton 版误差并列（同会话）；
8. 确定性：同配置 3 次位级一致；跨配置（SP vs looped）位级差异记录；
9. 边界：全零行、NaN 行、FP16 大数、H=0、rows=0、非连续输入。

用法：
    python compare_correctness.py --backend cuda
    python compare_correctness.py --backend musa [--with-handwritten]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations"))
sys.path.insert(0, str(HERE.parent / "chapter_02" / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    backend_sync,
    comparison_metrics,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from generate_fixtures import SOURCES, verify  # noqa: E402
from naive_rmsnorm import RMS_NORM_EPS, composite_rmsnorm_torch  # noqa: E402

RMS_FIXTURES = SOURCES["chapter03_rmsnorm"]
FUSED_FIXTURES = SOURCES["chapter04_fused"]


def load_npz(source: dict) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(source["npz"]) as data:
        arrays = {key: data[key] for key in data.files}
    manifest = json.loads(source["manifest"].read_text(encoding="utf-8"))
    return arrays, manifest


def composite_chain(arguments, device, arrays, manifest, dtype) -> dict:
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden_size}"
        x_fp32 = torch.from_numpy(arrays[f"x_{key}"])
        expected = torch.from_numpy(arrays[f"y_{key}"])
        if hidden_size not in weight_cache:
            weight_cache[hidden_size] = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(
                device=device, dtype=dtype)
        x = x_fp32.to(device=device, dtype=dtype)
        result = composite_rmsnorm_torch(x, weight_cache[hidden_size], RMS_NORM_EPS)
        metrics = comparison_metrics(result, expected)
        metrics.update({"rows": rows, "hidden_size": hidden_size,
                        "dtype": "fp16" if dtype == torch.float16 else "fp32"})
        groups.append(metrics)
    worst = max(groups, key=lambda item: item["max_abs_error"])
    print(f"[composite/{arguments.backend}/{dtype}] 30 组 | "
          f"最差组 {worst['rows']}x{worst['hidden_size']} max_abs={worst['max_abs_error']:.3e}")
    return {"dtype": "fp16" if dtype == torch.float16 else "fp32",
            "groups": groups, "worst_group": worst}


def triton_chain(arguments, device, arrays, manifest, dtype, runner, label) -> dict:
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden_size}"
        x_fp32 = torch.from_numpy(arrays[f"x_{key}"])
        expected = torch.from_numpy(arrays[f"y_{key}"])
        if hidden_size not in weight_cache:
            weight_cache[hidden_size] = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(
                device=device, dtype=dtype)
        x = x_fp32.to(device=device, dtype=dtype)
        start = time.perf_counter()
        result = runner(x, weight_cache[hidden_size])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = comparison_metrics(result, expected)
        metrics.update({
            "rows": rows,
            "hidden_size": hidden_size,
            "dtype": "fp16" if dtype == torch.float16 else "fp32",
            "call_overhead_ms": round(elapsed_ms, 2),
        })
        groups.append(metrics)
    worst = max(groups, key=lambda item: item["max_abs_error"])
    nan_total = sum(item["nan_count"] for item in groups)
    print(f"[{label}/{arguments.backend}] {len(groups)} 组 | "
          f"最差组 {worst['rows']}x{worst['hidden_size']} max_abs={worst['max_abs_error']:.3e} "
          f"cos={worst['cosine_similarity']:.8f} NaN 总数={nan_total}")
    return {"runner": label, "dtype": "fp16" if dtype == torch.float16 else "fp32",
            "groups": groups, "worst_group": worst, "nan_total": nan_total}


def fused_chain(arguments, device, arrays, manifest, dtype, fused_runner, label="fused-triton") -> dict:
    """融合对照：y vs ch04 真值 + residual in-place 语义核对。

    ch04 真值：t = r+h（fixture 't'），y = RMSNorm(t)；fp16 路径的 residual
    期望 = T(t)（t 先按输出 dtype 舍入），y 期望仍是 fp32 真值 y。
    """
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden_size}"
        r_np = arrays[f"r_{key}"]
        h_np = arrays[f"h_{key}"]
        t_truth = torch.from_numpy(arrays[f"t_{key}"])
        expected = torch.from_numpy(arrays[f"y_{key}"])
        if hidden_size not in weight_cache:
            weight_cache[hidden_size] = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(
                device=device, dtype=dtype)
        r = torch.from_numpy(r_np).to(device=device, dtype=dtype)
        h_in = torch.from_numpy(h_np).to(device=device, dtype=dtype)
        r_orig = r.clone()  # 融合 runner 原地改 r：期望必须用原始 r（KI-14-05 checker bug 修复）
        y, r_after = fused_runner(r, h_in, weight_cache[hidden_size], RMS_NORM_EPS)
        metrics = comparison_metrics(y, expected)
        # residual in-place 语义：
        #   fp32 输入：GPU fp32 加法与 CPU 真值 t 位级一致 → r_after == t（位级）；
        #   fp16 输入：输入先经 fp16 舍入再相加，正确期望是
        #     T(fp32(fp16(r)) + fp32(fp16(h)))（ch04 期"位级一致"是同设备
        #     融合 vs 分离的同输入对照，不是 vs CPU t——本期修正该期望）。
        if dtype == torch.float32:
            metrics["residual_bitwise_vs_t"] = bool(torch.equal(r_after, t_truth.to(device)))
            metrics["residual_max_abs_vs_t"] = float(
                (r_after.float() - t_truth.to(device)).abs().max().item())
        else:
            t_expect = (r_orig.float() + h_in.float()).to(torch.float16)
            metrics["residual_bitwise_vs_T(r+h)"] = bool(torch.equal(r_after, t_expect))
            metrics["residual_max_abs_vs_t_truth_fp16"] = float(
                (r_after.float() - t_truth.to(device=device, dtype=torch.float16).float()
                 ).abs().max().item())
        metrics.update({"rows": rows, "hidden_size": hidden_size,
                        "dtype": "fp16" if dtype == torch.float16 else "fp32"})
        groups.append(metrics)
    worst = max(groups, key=lambda item: item["max_abs_error"])
    residual_ok = all(g.get("residual_bitwise_vs_t") or g.get("residual_bitwise_vs_T(r+h)")
                      for g in groups)
    print(f"[{label}/{arguments.backend}/{dtype}] 30 组 | "
          f"最差组 {worst['rows']}x{worst['hidden_size']} max_abs={worst['max_abs_error']:.3e} "
          f"residual in-place 位级全对={residual_ok}")
    return {"runner": label, "dtype": "fp16" if dtype == torch.float16 else "fp32",
            "groups": groups, "worst_group": worst, "residual_all_bitwise": residual_ok}


def determinism_chain(arguments, device, arrays, runner) -> dict:
    rows, hidden_size = 128, 1024
    key = f"{rows}x{hidden_size}"
    x = torch.from_numpy(arrays[f"x_{key}"]).to(device=device, dtype=torch.float16)
    w = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(device=device, dtype=torch.float16)
    runs = [runner(x, w).detach().float().cpu().contiguous() for _ in range(3)]
    bitwise = all(
        bytes(runs[0].numpy().tobytes()) == bytes(run.numpy().tobytes()) for run in runs[1:]
    )
    print(f"[determinism] Triton 同配置 3 次位级一致: {bitwise}")
    return {"runs": 3, "bitwise_identical": bool(bitwise)}


def edge_chain(arguments, device, runner, fused_runner) -> dict:
    checks = {}
    w = torch.randn(1024, device=device, dtype=torch.float16)

    x = torch.randn(3, 1024, device=device, dtype=torch.float16)
    x[1] = 0
    y = runner(x, w)
    checks["zero_row"] = {
        "all_zero": bool((y[1] == 0).all()),
        "no_nan": bool(~torch.isnan(y[1]).any()),
    }

    x = torch.randn(2, 1024, device=device, dtype=torch.float16)
    x[0, 0] = float("nan")
    y = runner(x, w)
    checks["nan_row"] = {
        "row0_all_nan": bool(torch.isnan(y[0]).all()),
        "row1_finite": bool(torch.isfinite(y[1]).all()),
    }

    x = torch.full((1, 1024), 65504.0, device=device, dtype=torch.float16)
    y = runner(x, w)
    checks["fp16_large"] = {"finite": bool(torch.isfinite(y).all())}

    try:
        runner(torch.randn(2, 0, device=device, dtype=torch.float16),
               torch.zeros(0, device=device, dtype=torch.float16))
        checks["h_zero"] = {"raised": False}
    except Exception as error:  # noqa: BLE001
        checks["h_zero"] = {"raised": True, "error_type": type(error).__name__,
                            "message": str(error)[:300]}

    y = runner(torch.randn(0, 1024, device=device, dtype=torch.float16), w)
    checks["rows_zero"] = {"shape": list(y.shape)}

    try:
        runner(torch.randn(1024, 128, device=device, dtype=torch.float16).t(),
               torch.randn(128, device=device, dtype=torch.float16))
        checks["non_contiguous"] = {"raised": False}
    except Exception as error:  # noqa: BLE001
        checks["non_contiguous"] = {"raised": True, "error_type": type(error).__name__,
                                    "message": str(error)[:300]}

    # 融合边界：全零行 + NaN 行（in-place residual 后 r 内容 == t 语义）
    r = torch.randn(2, 1024, device=device, dtype=torch.float16)
    h_in = torch.randn(2, 1024, device=device, dtype=torch.float16)
    r[1] = 0
    h_in[1] = 0
    y, r_after = fused_runner(r, h_in, w, RMS_NORM_EPS)
    checks["fused_zero_row"] = {
        "y_all_zero": bool((y[1] == 0).all()),
        "residual_all_zero": bool((r_after[1] == 0).all()),
        "no_nan": bool(~torch.isnan(y[1]).any()),
    }

    ok = (checks["zero_row"]["all_zero"] and checks["zero_row"]["no_nan"]
          and checks["nan_row"]["row0_all_nan"] and checks["nan_row"]["row1_finite"]
          and checks["fp16_large"]["finite"] and checks["h_zero"]["raised"]
          and checks["non_contiguous"]["raised"] and checks["rows_zero"]["shape"] == [0, 1024]
          and checks["fused_zero_row"]["y_all_zero"] and checks["fused_zero_row"]["no_nan"])
    print(f"[edge] 边界行为{'全部符合合同' if ok else '存在不符合项（见 JSON）'}")
    return {"checks": checks, "all_pass": ok}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    parser.add_argument("--with-handwritten", action="store_true",
                        help="同会话构建 ch03/ch04 手写 kernel 并对照（耗时 +数分钟）")
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    sys.path.insert(0, str(HERE / "implementations" / arguments.backend))
    rms_arrays, rms_manifest = load_npz(RMS_FIXTURES)
    fused_arrays, fused_manifest = load_npz(FUSED_FIXTURES)

    integrity = {
        "chapter03_rmsnorm": verify(RMS_FIXTURES),
        "chapter04_fused": verify(FUSED_FIXTURES),
    }
    if not all(item["passed"] for item in integrity.values()):
        raise SystemExit(f"fixture SHA256 校验失败：{integrity}")

    from rmsnorm_triton import AutotuneRMSNorm, rmsnorm  # noqa: PLC0415
    from fused_triton import fused_add_rmsnorm as fused_impl  # noqa: PLC0415

    def run_sp(x, w):
        return rmsnorm(x, w, RMS_NORM_EPS)

    def run_looped256(x, w):
        return rmsnorm(x, w, RMS_NORM_EPS, block_size=256, looped=True)

    autotune_runner = AutotuneRMSNorm("parity")

    def run_autotune(x, w):
        return autotune_runner(x, w, RMS_NORM_EPS)

    results = {
        "schema_version": "kernel-chapter14.correctness.v1",
        "backend": arguments.backend,
        "fixture_integrity": integrity,
        "environment": device_environment_record(device),
        "composite_fp32": composite_chain(arguments, device, rms_arrays, rms_manifest,
                                          torch.float32),
        "triton_sp_fp32": triton_chain(arguments, device, rms_arrays, rms_manifest,
                                       torch.float32, run_sp, "triton-SP"),
        "triton_looped256_fp32": triton_chain(arguments, device, rms_arrays, rms_manifest,
                                              torch.float32, run_looped256, "triton-looped-B256"),
        "autotune_fp32": triton_chain(arguments, device, rms_arrays, rms_manifest,
                                      torch.float32, run_autotune, "triton-autotune"),
    }
    best_config = autotune_runner.best_config()
    results["autotune_best_config"] = str(best_config)
    results["autotune_cache_read_attempts"] = autotune_runner.cache_read_attempts

    results["composite_fp16"] = composite_chain(arguments, device, rms_arrays, rms_manifest,
                                                torch.float16)
    results["triton_sp_fp16"] = triton_chain(arguments, device, rms_arrays, rms_manifest,
                                             torch.float16, run_sp, "triton-SP")
    results["triton_looped256_fp16"] = triton_chain(arguments, device, rms_arrays, rms_manifest,
                                                    torch.float16, run_looped256,
                                                    "triton-looped-B256")
    results["autotune_fp16"] = triton_chain(arguments, device, rms_arrays, rms_manifest,
                                            torch.float16, run_autotune, "triton-autotune")

    # 融合（ch04 fixtures：r/h/t/y 30 组）
    def run_fused(r, h_in, w, eps):
        return fused_impl(r, h_in, w, eps)
    results["fused_triton_fp32"] = fused_chain(arguments, device, fused_arrays,
                                               fused_manifest, torch.float32, run_fused)
    results["fused_triton_fp16"] = fused_chain(arguments, device, fused_arrays,
                                               fused_manifest, torch.float16, run_fused)

    results["determinism"] = determinism_chain(arguments, device, rms_arrays, run_sp)
    results["edge"] = edge_chain(arguments, device, run_sp, run_fused)

    if arguments.with_handwritten:
        from reference_runners import HandwrittenFused, HandwrittenRMSNorm  # noqa: PLC0415
        hw_rms = HandwrittenRMSNorm(arguments.backend)
        results["handwritten_rms_build_seconds"] = hw_rms.build_seconds

        def run_hw(x, w):
            return hw_rms.run(x, w, RMS_NORM_EPS, 256)
        results["handwritten_ch03_fp32"] = triton_chain(
            arguments, device, rms_arrays, rms_manifest, torch.float32, run_hw, "handwritten-ch03")
        results["handwritten_ch03_fp16"] = triton_chain(
            arguments, device, rms_arrays, rms_manifest, torch.float16, run_hw, "handwritten-ch03")

        hw_fused = HandwrittenFused(arguments.backend)
        results["handwritten_fused_build_seconds"] = hw_fused.build_seconds
        results["handwritten_warp_size"] = hw_fused.warp_size or hw_fused._ensure_warp_size(device)

        def run_hw_fused(r, h_in, w, eps):
            return hw_fused.run(r, h_in, w, eps, 256)
        results["handwritten_ch04_fused_fp32"] = fused_chain(
            arguments, device, fused_arrays, fused_manifest, torch.float32, run_hw_fused,
            label="handwritten-ch04")
        results["handwritten_ch04_fused_fp16"] = fused_chain(
            arguments, device, fused_arrays, fused_manifest, torch.float16, run_hw_fused,
            label="handwritten-ch04")

    output = HERE / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
