"""第 14 期正确性比较（协议 §2：复用第 01 期比较口径）。

对照链（每 fixtures 组）：
1. eager 组合表达式（rmsnorm_expr / add_rmsnorm_separated，FP32）
   vs CPU FP32 串行真值——t 位级（加法无累加）、y 归约顺序差量级；
2. compile(inductor 默认) 输出 vs 同一真值——编译器是否改变数值语义；
3. compile 输出 vs eager 输出——两条框架路径的一致性。

产出 results/correctness-results-cpu.json。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiment_utils import comparison_metrics, write_results_json

CHAPTER = Path(__file__).resolve().parent
SCHEMA_VERSION = "kernel-chapter15.correctness.v1"


def load_fixtures():
    npz = np.load(CHAPTER / "fixtures" / "correctness.npz")
    manifest = json.loads((CHAPTER / "fixtures" / "manifest.json").read_text())
    return npz, manifest


def metrics_vs_truth(actual: torch.Tensor, truth: np.ndarray) -> dict:
    m = comparison_metrics(actual, torch.from_numpy(truth))
    m["bitwise_equal"] = bool(
        actual.detach().cpu().contiguous().numpy().tobytes() == truth.tobytes()
    )
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-variant", default="inductor",
                        help="inductor | dynamo-eager | aot-eager")
    args = parser.parse_args()

    npz, manifest = load_fixtures()
    from implementations.cpu.compiled_objects import (
        add_rmsnorm_separated,
        rmsnorm_expr,
    )
    from implementations.cpu.compile_variants import (
        disable_codegen_caches,
        make_callable,
        reset_compilers,
    )

    torch.set_num_threads(min(6, torch.get_num_threads()))
    entries: list[dict] = []
    compiled_add = None
    compiled_rms = None
    current_variant = None

    for group in manifest["groups"]:
        rows, hidden = group["rows"], group["hidden"]
        prefix = f"{rows}x{hidden}"
        r = torch.from_numpy(npz[f"{prefix}/r"])
        h = torch.from_numpy(npz[f"{prefix}/h"])
        w = torch.from_numpy(npz[f"{prefix}/w"])
        t_true = npz[f"{prefix}/t"]
        y_true = npz[f"{prefix}/y"]

        # 编译实例按 (variant, H) 缓存：同 H 跨 rows 共用一个已编译对象
        # （与真实使用方式一致；重编译行为由 recompile_experiment 单独量化）。
        if current_variant != args.compile_variant or compiled_add is None:
            reset_compilers()
            disable_codegen_caches()
            compiled_add = make_callable("add_rmsnorm", args.compile_variant)
            compiled_rms = make_callable("rmsnorm", args.compile_variant)
            current_variant = args.compile_variant

        entry: dict = {"group": prefix, "rows": rows, "hidden": hidden}

        t_eager, y_eager = add_rmsnorm_separated(r, h, w)
        entry["eager_t_vs_truth"] = metrics_vs_truth(t_eager, t_true)
        entry["eager_y_vs_truth"] = metrics_vs_truth(y_eager, y_true)

        y_eager_rms = rmsnorm_expr(torch.from_numpy(t_true), w)
        entry["eager_rmsnorm_vs_truth"] = metrics_vs_truth(y_eager_rms, y_true)

        t_c, y_c = compiled_add(r, h, w)
        entry["compiled_t_vs_truth"] = metrics_vs_truth(t_c, t_true)
        entry["compiled_y_vs_truth"] = metrics_vs_truth(y_c, y_true)
        entry["compiled_vs_eager_t"] = comparison_metrics(t_c, t_eager)
        entry["compiled_vs_eager_y"] = comparison_metrics(y_c, y_eager)

        y_cr = compiled_rms(torch.from_numpy(t_true), w)
        entry["compiled_rmsnorm_vs_truth"] = metrics_vs_truth(y_cr, y_true)

        entries.append(entry)
        print(
            f"[correctness] {prefix}: eager y max {entry['eager_y_vs_truth']['max_abs_error']:.3e}"
            f" | compiled y max {entry['compiled_y_vs_truth']['max_abs_error']:.3e}"
            f" | compiled t bitwise {entry['compiled_t_vs_truth']['bitwise_equal']}",
            flush=True,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "compile_variant": args.compile_variant,
        "truth": "ch04 separated_add_rmsnorm_serial（CPU FP32，fixtures manifest）",
        "tolerances": {"fp32_y": "1e-5（第 01/02 期合同口径）", "t": "位级"},
        "entries": entries,
    }
    suffix = "cpu" if args.compile_variant == "inductor" else f"cpu-{args.compile_variant}"
    write_results_json(CHAPTER / "results" / f"correctness-results-{suffix}.json", payload)


if __name__ == "__main__":
    main()
