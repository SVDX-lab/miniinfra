#!/usr/bin/env python3
"""第 29 期边界补充组（GPU 实测）：H∈{16,32,33} 归约触发边界 + NaN 行。

静态预审 kernel_sim 的边界结论（触发边界=32）用 GPU 实测复证：
  - H∈{16,32}：预测 AI 版正确（全部线程都在 warp 0）；
  - H=33：预测 AI 版错误（warp 1 携带错误部分和）；
  - 每档附 NaN 行探针（H=128，静态预审预判：NaN 行只有 warp0 元素是 NaN）。
真值在 host 侧 numpy FP64 生成（fixtures 口径外的小探针，seed 固定）。

用法：python gpu_acceptance/boundary_extra.py [--out results/boundary-extra.json]
前提：implementations/ai/rmsnorm/ai_rmsnorm 已构建。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

BINARY = ROOT / "implementations/ai/rmsnorm/ai_rmsnorm"


def truth_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """逐行合同语义真值（FP64 累加，NaN 行 → 全 NaN）。"""
    rows, h = x.shape
    out = np.full((rows, h), np.nan, dtype=np.float32)
    for r in range(rows):
        if np.isnan(x[r]).any():
            continue
        sum2 = float(np.dot(x[r].astype(np.float64), x[r].astype(np.float64)))
        out[r] = w * x[r] / np.sqrt(sum2 / h + eps)
    return out


def run_case(tmp: Path, x: np.ndarray, w: np.ndarray, dtype: str) -> np.ndarray | None:
    xp, wp, op_ = tmp / "x.bin", tmp / "w.bin", tmp / "y.bin"
    xa = x.astype(np.float16) if dtype == "fp16" else x.astype(np.float32)
    wa = w.astype(np.float16) if dtype == "fp16" else w.astype(np.float32)
    xa.tofile(xp)
    wa.tofile(wp)
    proc = subprocess.run(
        [str(BINARY), "--run", "--rows", str(x.shape[0]), "--hidden", str(x.shape[1]),
         "--dtype", dtype, "--x", str(xp), "--w", str(wp), "--out", str(op_)],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not op_.exists():
        return None
    dtype_np = np.float16 if dtype == "fp16" else np.float32
    return np.fromfile(op_, dtype=dtype_np).astype(np.float32).reshape(x.shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/boundary-extra.json"))
    args = ap.parse_args()
    if not BINARY.exists():
        print("[boundary-extra] ai_rmsnorm 未构建", file=sys.stderr)
        return 2

    rng = np.random.default_rng(20261121)
    cases = []
    with tempfile.TemporaryDirectory(prefix="ch30bx_") as td:
        tmp = Path(td)
        for h in (16, 32, 33):
            for dtype in ("fp32", "fp16"):
                rows = 2
                x = rng.standard_normal((rows, h)).astype(np.float32)
                w = rng.uniform(0.5, 1.5, h).astype(np.float32)
                truth = truth_rmsnorm(x, w)
                got = run_case(tmp, x, w, dtype)
                if got is None:
                    cases.append({"hidden": h, "dtype": dtype, "run_error": True,
                                  "predicted_ok": h <= 32, "actual_ok": False})
                    continue
                err = float(np.abs(got - truth).max())
                ok = bool(err <= (2e-2 if dtype == "fp16" else 1e-5))
                cases.append({"hidden": h, "dtype": dtype, "rows": rows,
                              "max_abs_error": round(err, 8),
                              "predicted_ok": h <= 32, "actual_ok": ok})

        # NaN 行探针（H=128 fp32，静态预审预判的 GPU 复证）
        h = 128
        x = rng.standard_normal((4, h)).astype(np.float32)
        x[1, 5] = np.nan
        w = rng.uniform(0.5, 1.5, h).astype(np.float32)
        got = run_case(tmp, x, w, "fp32")
        nan_entry: dict = {"hidden": h, "case": "nan_row"}
        if got is None:
            nan_entry["run_error"] = True
        else:
            nan_entry["row1_all_nan"] = bool(np.isnan(got[1]).all())
            nan_entry["row1_nan_fraction"] = round(float(np.isnan(got[1]).mean()), 4)
            nan_entry["other_rows_finite"] = bool(np.isfinite(got[[0, 2, 3]]).all())
            nan_entry["predicted"] = "只有 warp0 的 32 个元素是 NaN（静态预审 kernel_sim）"
        cases.append(nan_entry)

    summary = {
        "schema_version": "kernel-chapter30.boundary-extra.v1",
        "purpose": "未决问题 2（已批准）：H∈{16,32,33} 归约触发边界 GPU 实测",
        "cases": cases,
        "boundary32_confirmed": all(
            c["actual_ok"] == c["predicted_ok"] for c in cases
            if "actual_ok" in c and isinstance(c.get("hidden"), int)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for c in cases:
        print("[boundary-extra]", c)
    print(f"[boundary-extra] boundary32_confirmed={summary['boundary32_confirmed']} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
