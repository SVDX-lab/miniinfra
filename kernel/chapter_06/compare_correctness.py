#!/usr/bin/env python3
"""第 06 期正确性比较（复用第 01/02 期比较口径）。

被测矩阵 = 30 组合同 Shape × 布局 × 策略：
  布局 L0 连续（行主序）/ L1 转置视图（stride=(1,rows)）/ L2 行切片视图
  （stride=(H+pad,1)，pad=8）；L1/L2 由 x 零拷贝派生（逻辑值不变 → 期望
  y 复用 fixtures 一份）。
  策略 A（reject）：L0 正常；L1/L2 必须显式报错（报错全文快照存 results/）。
  策略 B（auto-copy）/ C（strided）：L0/L1/L2 全部数值正确。

fp32 全 30 组；fp16 抽查 3 组（第 05 期已全量验证 dtype 路径）。
确定性：策略 C（转置视图）同 Tile 3 次位级一致。
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    backend_sync,
    comparison_metrics,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from stride_layout import RMS_NORM_EPS  # noqa: E402

TILE = 256
FP16_SPOT = [("H1024_r2048", "H1024", 2048), ("H127_r4", "H127", 4), ("H129_r128", "H129", 128)]
PAD = 8  # 行切片 pad（fp32: 32B 对齐；fp16: 16B 对齐）


def load_fixtures() -> tuple[dict[str, np.ndarray], dict]:
    data = np.load(HERE / "fixtures" / "correctness.npz")
    manifest = json.loads((HERE / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    return {k: data[k] for k in data.files}, manifest


def make_views(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """由连续 x 派生三种布局（零拷贝视图；L1 先物化转置副本再视图）。"""
    rows, h = x.shape
    # L1：转置视图。base=[H,rows] 行主序连续 → base.t() 逻辑 [rows,H]、stride=(1,rows)。
    base = x.t().contiguous()          # setup 拷贝（物化 base，保证逻辑值 == x）
    tv = base.t()
    # L2：行切片视图。
    big = torch.zeros(rows, h + PAD, device=x.device, dtype=x.dtype)
    big[:, :h] = x
    rs = big[:, :h]
    return {"L0_contig": x, "L1_transposed": tv, "L2_row_sliced": rs}


def make_kernel_runner(backend: str, device: torch.device):
    import importlib.util

    path = HERE / "implementations" / backend / "build_ext.py"
    spec = importlib.util.spec_from_file_location(f"build_ext_{backend}", path)
    module_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_loader)
    start = time.perf_counter()
    module = module_loader.get_module()
    build_seconds = time.perf_counter() - start
    x = torch.randn(4, 128, device=device, dtype=torch.float32)
    w = torch.randn(128, device=device, dtype=torch.float32)
    y = module.rmsnorm_policy_a(x, w, RMS_NORM_EPS, TILE)
    assert y.shape == x.shape
    return module, round(build_seconds, 1)


def compare_entry(y_actual: torch.Tensor, y_expected: torch.Tensor) -> dict:
    return comparison_metrics(y_actual, y_expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend) if arguments.backend != "cpu" else torch.device("cpu")
    environment = device_environment_record(device)
    write_results_json(HERE / "results" / f"environment-{arguments.backend}.json",
                       {"schema_version": "kernel-chapter07.environment.v1",
                        "backend": arguments.backend, **environment})

    fixtures, manifest = load_fixtures()
    eps = RMS_NORM_EPS
    results: dict = {
        "schema_version": "kernel-chapter07.correctness.v1",
        "backend": arguments.backend,
        "tile": TILE,
        "eps": eps,
        "fixture_seed_base": manifest["seed_base"],
        "entries": [],
        "policy_a_errors": {},
        "boundary": {},
    }
    fp32_tolerance = {"max_abs": 1e-5, "note": "rsqrtf 近似 + 归约顺序（第 02 期口径）"}
    fp16_tolerance = {"max_abs": 2e-2, "note": "dtype 主导（第 01 期口径）"}

    if arguments.backend == "cpu":
        # CPU 列：三策略教学实现 vs fixtures 真值（30 组全量）。
        from stride_layout import StridedTensor, policy_a_reject, policy_b_autocopy, policy_c_strided

        for combo in manifest["combos"]:
            cid, rows, h = combo["id"], combo["rows"], combo["hidden_size"]
            x_np = fixtures[f"x_{cid}"]
            w_np = fixtures[f"w_{cid}"]
            y_np = fixtures[f"y_{cid}"]
            entry = {"combo": cid, "rows": rows, "hidden_size": h, "dtype": "float32"}
            # L0：三策略 vs 真值（naive_rmsnorm_strided 与 fixtures 生成同实现，应位级一致）。
            y_a = policy_a_reject(StridedTensor.row_major(x_np), w_np, eps)
            entry["policyA_L0"] = {"max_abs_error": float(np.abs(y_a.buffer.reshape(rows, h) - y_np).max())}
            y_b = policy_b_autocopy(StridedTensor.transposed_view(x_np), w_np, eps)
            entry["policyB_L1_transposed"] = {
                "max_abs_error": float(np.abs(y_b.buffer.reshape(rows, h) - y_np).max())}
            y_b2 = policy_b_autocopy(StridedTensor.row_sliced_view(x_np, PAD), w_np, eps)
            entry["policyB_L2_row_sliced"] = {
                "max_abs_error": float(np.abs(y_b2.buffer.reshape(rows, h) - y_np).max())}
            y_c1 = policy_c_strided(StridedTensor.transposed_view(x_np), w_np, eps)
            entry["policyC_L1_transposed"] = {
                "max_abs_error": float(np.abs(y_c1.buffer.reshape(rows, h) - y_np).max())}
            y_c2 = policy_c_strided(StridedTensor.row_sliced_view(x_np, PAD), w_np, eps)
            entry["policyC_L2_row_sliced"] = {
                "max_abs_error": float(np.abs(y_c2.buffer.reshape(rows, h) - y_np).max())}
            results["entries"].append(entry)
            print(f"[cpu] {cid}: A_L0={entry['policyA_L0']['max_abs_error']:.2e} "
                  f"B_L1={entry['policyB_L1_transposed']['max_abs_error']:.2e} "
                  f"C_L1={entry['policyC_L1_transposed']['max_abs_error']:.2e}")
    else:
        module, build_seconds = make_kernel_runner(arguments.backend, device)
        results["kernel_build_seconds"] = build_seconds

        def run_combo(cid: str, dtype: torch.dtype, tolerance: dict) -> dict:
            rows = int(cid.split("_r")[1])
            x_np = fixtures[f"x_{cid}"]
            w_np = fixtures[f"w_{cid}"]
            y_np = fixtures[f"y_{cid}"]
            x = torch.from_numpy(x_np).to(device=device, dtype=dtype)
            w = torch.from_numpy(w_np).to(device=device, dtype=dtype)
            y_exp = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)
            views = make_views(x)
            entry = {"combo": cid, "rows": rows, "hidden_size": x.shape[1],
                     "dtype": str(dtype).replace("torch.", "")}
            # L0：三策略。
            entry["policyA_L0"] = compare_entry(module.rmsnorm_policy_a(x, w, eps, TILE), y_exp)
            entry["policyB_L0"] = compare_entry(module.rmsnorm_policy_b(x, w, eps, TILE), y_exp)
            entry["policyC_L0"] = compare_entry(module.rmsnorm_policy_c(x, w, eps, TILE), y_exp)
            # L1 转置视图：A 报错（rows>1）；rows=1 时视图退化连续（torch 对
            # size=1 维不约束 stride），A 合法接受——此时补数值校验。
            if views["L1_transposed"].is_contiguous():
                entry["policyA_L1_transposed"] = {
                    "degenerate_contiguous": True,
                    **compare_entry(module.rmsnorm_policy_a(views["L1_transposed"], w, eps, TILE), y_exp)}
            else:
                try:
                    module.rmsnorm_policy_a(views["L1_transposed"], w, eps, TILE)
                    entry["policyA_L1_transposed"] = {"unexpected": "未按合同报错"}
                except RuntimeError as error:
                    entry["policyA_L1_transposed"] = {"rejected": True, "message": str(error)[:200]}
                    results["policy_a_errors"][f"{cid}_L1"] = str(error)
            entry["policyB_L1_transposed"] = compare_entry(
                module.rmsnorm_policy_b(views["L1_transposed"], w, eps, TILE), y_exp)
            entry["policyC_L1_transposed"] = compare_entry(
                module.rmsnorm_policy_c(views["L1_transposed"], w, eps, TILE), y_exp)
            # L2 行切片视图（rows=1 退化连续同 L1 处理）。
            if views["L2_row_sliced"].is_contiguous():
                entry["policyA_L2_row_sliced"] = {
                    "degenerate_contiguous": True,
                    **compare_entry(module.rmsnorm_policy_a(views["L2_row_sliced"], w, eps, TILE), y_exp)}
            else:
                try:
                    module.rmsnorm_policy_a(views["L2_row_sliced"], w, eps, TILE)
                    entry["policyA_L2_row_sliced"] = {"unexpected": "未按合同报错"}
                except RuntimeError as error:
                    entry["policyA_L2_row_sliced"] = {"rejected": True, "message": str(error)[:200]}
                    results["policy_a_errors"][f"{cid}_L2"] = str(error)
            entry["policyB_L2_row_sliced"] = compare_entry(
                module.rmsnorm_policy_b(views["L2_row_sliced"], w, eps, TILE), y_exp)
            entry["policyC_L2_row_sliced"] = compare_entry(
                module.rmsnorm_policy_c(views["L2_row_sliced"], w, eps, TILE), y_exp)
            # 确定性（策略 C 转置视图同 Tile 3 次位级）。
            y1 = module.rmsnorm_policy_c(views["L1_transposed"], w, eps, TILE)
            det = all(bool(torch.equal(y1, module.rmsnorm_policy_c(views["L1_transposed"], w, eps, TILE)))
                      for _ in range(2))
            entry["determinism_policyC_same_tile_3x_bitwise"] = det
            backend_sync(device)
            # 容差判定汇总。
            worst = max(entry[k]["max_abs_error"] for k in entry
                        if isinstance(entry[k], dict) and "max_abs_error" in entry[k])
            entry["worst_max_abs_error"] = worst
            entry["within_tolerance"] = worst <= tolerance["max_abs"]
            print(f"[{arguments.backend}] {cid} {entry['dtype']}: worst={worst:.3e} "
                  f"(tol {tolerance['max_abs']:.0e}) det={det}")
            return entry

        # fp32 全 30 组。
        for combo in manifest["combos"]:
            results["entries"].append(run_combo(combo["id"], torch.float32, fp32_tolerance))
        # fp16 抽查 3 组。
        for cid, _, _ in FP16_SPOT:
            results["entries"].append(run_combo(cid, torch.float16, fp16_tolerance))

        # 边界行为（GPU 侧）。
        boundary = {}
        x0 = torch.randn(0, 1024, device=device, dtype=torch.float32)
        w0 = torch.randn(1024, device=device, dtype=torch.float32)
        y0 = module.rmsnorm_policy_b(x0, w0, eps, TILE)
        boundary["rows0_returns_empty"] = y0.shape == (0, 1024)
        try:
            module.rmsnorm_policy_a(torch.randn(2, 0, device=device),
                                    torch.randn(0, device=device), eps, TILE)
            boundary["h0_rejected"] = False
        except RuntimeError as error:
            boundary["h0_rejected"] = True
            boundary["h0_message"] = str(error)[:120]
        results["boundary"] = boundary
        print(f"[{arguments.backend}] boundary: {json.dumps(boundary, ensure_ascii=False)}")

    # 汇总判定。
    fp32_entries = [e for e in results["entries"] if e.get("dtype") == "float32"]
    gpu_entries = [e for e in results["entries"] if "policyA_L1_transposed" in e]
    results["summary"] = {
        "fp32_combos": len(fp32_entries),
        "fp32_all_within_tolerance": all(e.get("within_tolerance", e["policyA_L0"]["max_abs_error"] <= 1e-5)
                                         for e in fp32_entries),
        # rows=1 的 L1/L2 视图退化连续（torch size=1 维不约束 stride），策略 A
        # 合法接受并已补数值校验——"全拒绝"只对 rows>1 断言。
        "policy_a_rejects_all_noncontig_rows_gt1": all(
            (e["policyA_L1_transposed"].get("rejected") or e["policyA_L1_transposed"].get("degenerate_contiguous"))
            and (e["policyA_L2_row_sliced"].get("rejected") or e["policyA_L2_row_sliced"].get("degenerate_contiguous"))
            for e in gpu_entries),
        "rows1_degenerate_contiguous_combos": [
            e["combo"] for e in gpu_entries
            if e.get("policyA_L1_transposed", {}).get("degenerate_contiguous")],
    }
    write_results_json(HERE / "results" / f"correctness-results-{arguments.backend}.json", results)
    backend_sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
