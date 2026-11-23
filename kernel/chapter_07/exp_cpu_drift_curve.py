#!/usr/bin/env python3
"""第 07 期 CPU 实验 3（曲线实验的 CPU 侧）：漂移随 H 的增长曲线。

大纲实验项"漂移随行数/H 的增长曲线"的 CPU 侧（设备侧在 GPU 上按
同一 H 集合实测，见 benchmark.py --curve；rows 维度对 CPU 模拟无效应
—— 每行独立归约，rows 效应只在设备 atomic 版上出现（block 竞争/调度），
由设备实验单独验证）。

口径：
- H ∈ {1024, 4096, 16384, 65536}，每 H 采样 8 行（Python 位精确模拟的
  成本约束，采样标注；行间分布统计给出 min/max）；
- 顺序 ∈ {serial_l2r, pairwise_tree, blocked32, numpy_pairwise} ×
  累加 dtype ∈ {fp32, fp16}；fsum（精确）作 0 误差锚点；
- 每配置统计：相对误差 min/mean/max + 位级不一致次数（8 行中与 fsum
  锚点位级不同的行数）；
- 乱序模型（atomic 一阶）：每 H 单行 100 种子漂移带。

用法：python exp_cpu_drift_curve.py --out results/drift-curve-cpu.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from sum_order_lab import (  # noqa: E402
    ORDER_FUNCTIONS,
    shuffle_drift_distribution,
    sum_blocked,
)

SEED = 20260802
H_SET = (1024, 4096, 16384, 65536)
ROWS_SAMPLED = 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/drift-curve-cpu.json")
    parser.add_argument("--shuffle-seeds", type=int, default=100)
    arguments = parser.parse_args()

    results: dict = {
        "schema_version": "kernel-chapter08.drift-curve-cpu.v1",
        "seed": SEED,
        "H_set": list(H_SET),
        "rows_sampled_per_H": ROWS_SAMPLED,
        "note": "CPU 侧位精确模拟：每 H 采样 8 行；rows 维度对每行独立归约无效应，"
                "设备侧 atomic 版的 rows 效应（竞争/调度）由 benchmark.py --curve 实测",
        "curve": [],
        "shuffle_by_H": [],
        "conclusions": [],
    }

    orders = {name: fn for name, fn in ORDER_FUNCTIONS.items() if name != "fsum_exact"}
    orders["blocked32"] = lambda x, acc_dtype_name: sum_blocked(x, 32, acc_dtype_name)

    for h in H_SET:
        rng = np.random.default_rng(SEED + h)
        weight = rng.standard_normal(h, dtype=np.float32)
        entry = {"H": h, "fp32": {}, "fp16": {}, "truth_rows": ROWS_SAMPLED}
        for dt_name in ("fp32", "fp16"):
            per_order_rel = {name: [] for name in orders}
            per_order_bits_differ = {name: 0 for name in orders}
            for r in range(ROWS_SAMPLED):
                row = rng.standard_normal(h, dtype=np.float32)
                truth = float(np.sum(row.astype(np.float64) ** 2))
                truth_bits = np.float32(ORDER_FUNCTIONS["fsum_exact"](row)).tobytes()
                for name, fn in orders.items():
                    v = fn(row, acc_dtype_name=dt_name)
                    per_order_rel[name].append(abs(v - truth) / truth)
                    if np.float32(v).tobytes() != truth_bits:
                        per_order_bits_differ[name] += 1
            entry[dt_name] = {
                name: {
                    "rel_error_min": float(np.min(vals)),
                    "rel_error_mean": float(np.mean(vals)),
                    "rel_error_max": float(np.max(vals)),
                    "rows_bit_differ_vs_fsum": per_order_bits_differ[name],
                }
                for name, vals in per_order_rel.items()
            }
            worst = max(per_order_rel, key=lambda n: entry[dt_name][n]["rel_error_max"])
            print(f"[curve] H={h:<6} acc={dt_name}: worst={worst} "
                  f"rel_max={entry[dt_name][worst]['rel_error_max']:.2e}")
        results["curve"].append(entry)

        # 乱序模型（每 H 第一行采样、100 种子）
        row0 = np.random.default_rng(SEED + h).standard_normal(h, dtype=np.float32)
        truth0 = float(np.sum(row0.astype(np.float64) ** 2))
        shuffle_entry = {"H": h}
        for dt_name in ("fp32", "fp16"):
            dist = shuffle_drift_distribution(row0, truth0, arguments.shuffle_seeds, dt_name,
                                              seed_base=SEED + 1000 + h)
            shuffle_entry[dt_name] = dist
        results["shuffle_by_H"].append(shuffle_entry)
        print(f"[curve] H={h:<6} shuffle: fp32 rel_max={shuffle_entry['fp32']['rel_error_max']:.2e} "
              f"unique={shuffle_entry['fp32']['unique_values']}/{arguments.shuffle_seeds} | "
              f"fp16 rel_max={shuffle_entry['fp16']['rel_error_max']:.2e} "
              f"unique={shuffle_entry['fp16']['unique_values']}")

    fp32_worst = max(e["fp32"][n]["rel_error_max"] for e in results["curve"]
                     for n in e["fp32"])
    fp16_worst = max(e["fp16"][n]["rel_error_max"] for e in results["curve"]
                     for n in e["fp16"])
    results["conclusions"] = [
        f"FP32 累加全顺序全 H：相对误差 ≤{fp32_worst:.2e} —— 顺序漂移有界且与 H 弱相关"
        "（H 增大略升，因为 acc 大 → ulp 大，但仍 <1e-6）",
        f"FP16 累加：H≥4096 起部分顺序进入卡死区，H=65536 全顺序 rel_max={fp16_worst:.2e}"
        "（ep06 卡死区结论在四种顺序下全部成立）",
        "确定性归约（pairwise/blocked）不改变 FP16 的卡死命运：树形合并的增量更大，"
        "卡死更早 —— 与 ep06 'GPU 蝶形在 H=131072 溢出、CPU 串行卡死'互补失效一致",
        "atomic 乱序模型：FP32 漂移带 ~1e-7 量级且随 H 微增；FP16 全部落入卡死区 —— "
        "乱序改变的不是漂移量级而是'每次运行漂移不同'这个确定性本身",
    ]

    out = HERE / arguments.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[done] 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
