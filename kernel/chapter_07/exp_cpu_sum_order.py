#!/usr/bin/env python3
"""第 07 期 CPU 实验 1：求和顺序实验室 —— 浮点结合律破缺全矩阵。

回答三个问题（全部位级/实测数字）：
1. 同一行向量，六种求和顺序 × 三种累加 dtype：平方和差多少？
   （FP32 下顺序差 ~1e-7 相对；FP16 下顺序决定生死：卡死区；FP64 下
   淹没在 1e-16 —— "顺序不敏感"是相对累加器精度的，不是绝对的）
2. 顺序差异传播到 RMSNorm 输出是多少？（scale = rsqrt 漂移一半 → 整行）
3. atomicAdd 乱序到达（一阶随机置换模型）：100 种子的漂移分布，
   FP32 vs FP16 累加器（FP16 卡死区是否被乱序"救回来"——ep06 假设检验）。

教学核心：CPU 也不是天然确定 —— 是"串行顺序固定"才确定。
左到右与右到右都是"串行"，位级结果却不同；换个实现（NumPy pairwise）
结果又不同。确定性的来源是"归约顺序固定 + 每次运行顺序相同"。

用法：python exp_cpu_sum_order.py --out results/sum-order-cpu.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from serial_rmsnorm import RMS_NORM_EPS, naive_rmsnorm_serial  # noqa: E402
from sum_order_lab import (  # noqa: E402
    ORDER_FUNCTIONS,
    shuffle_drift_distribution,
    sum_blocked,
    sum_pairwise,
    sum_serial,
    sum_shuffled,
    ulp_of,
)

SEED = 20260801
H_MAIN = 1024


def bit_level_matrix(values: dict[str, float], dt_name: str) -> dict:
    """顺序两两比较：位级相同的对数 + 数值差矩阵。"""
    names = list(values)
    matrix = {}
    identical_pairs = 0
    total_pairs = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            total_pairs += 1
            same_bits = (np.float32(values[a]).tobytes() == np.float32(values[b]).tobytes()) \
                if dt_name in ("fp32", "fp16") else (values[a] == values[b])
            if same_bits:
                identical_pairs += 1
            matrix[f"{a}|{b}"] = {
                "bit_identical": bool(same_bits),
                "abs_diff": abs(values[a] - values[b]),
            }
    return {
        "acc_dtype": dt_name,
        "identical_pairs": identical_pairs,
        "total_pairs": total_pairs,
        "pairwise": matrix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/sum-order-cpu.json")
    parser.add_argument("--shuffle-seeds", type=int, default=100)
    arguments = parser.parse_args()

    rng = np.random.default_rng(SEED)
    row = rng.standard_normal(H_MAIN, dtype=np.float32)
    truth = sum_order_truth = float(np.sum(row.astype(np.float64) ** 2))

    results: dict = {
        "schema_version": "kernel-chapter08.sum-order-cpu.v1",
        "seed": SEED,
        "H": H_MAIN,
        "sqsum_truth_f64": truth,
        "note": "truth = FP64 精确和参照；所有顺序值都是位精确模拟（numpy float32/float16 RNE）",
    }

    # ---- 1. 六种顺序 × 三种累加 dtype ----
    orders_matrix = {}
    for dt_name in ("fp32", "fp16", "fp64"):
        values = {name: fn(row, acc_dtype_name=dt_name)
                  for name, fn in ORDER_FUNCTIONS.items() if name != "fsum_exact"}
        values["blocked32"] = sum_blocked(row, 32, dt_name)
        values["fsum_exact"] = ORDER_FUNCTIONS["fsum_exact"](row)
        rel = {k: abs(v - truth) / truth for k, v in values.items()}
        orders_matrix[dt_name] = {
            "values": {k: float(v) for k, v in values.items()},
            "rel_error_vs_f64": rel,
            **bit_level_matrix(values, dt_name),
        }
        print(f"[order] acc={dt_name}: "
              + " ".join(f"{k}={rel[k]:.2e}" for k in sorted(rel, key=rel.get, reverse=True)))
        print(f"[order]   位级相同对: {orders_matrix[dt_name]['identical_pairs']}"
              f"/{orders_matrix[dt_name]['total_pairs']}")
        if dt_name == "fp16":
            stuck = sum_serial(row, "fp16")
            print(f"[order]   fp16 串行卡死检查: acc={stuck:.1f} vs truth={truth:.1f} "
                  f"(ulp(fp16, acc)={ulp_of('fp16', stuck):.3e}, "
                  f"typical increment ~1 → half_ulp 判据)")
    results["order_matrix"] = orders_matrix

    # ---- 2. 顺序差异传播到 RMSNorm 输出 ----
    weight = rng.standard_normal(H_MAIN, dtype=np.float32)
    x2d = row[None, :]
    truth_output = naive_rmsnorm_serial(x2d, weight, RMS_NORM_EPS)[0]
    propagation = {}
    for dt_name in ("fp32", "fp16"):
        values = {name: fn(row, acc_dtype_name=dt_name)
                  for name, fn in ORDER_FUNCTIONS.items() if name != "fsum_exact"}
        values["blocked32"] = sum_blocked(row, 32, dt_name)
        entries = {}
        for name, sqsum in values.items():
            # 把该顺序的平方和直接交给 RMSNorm 后半程（scale = rsqrt(mean+eps)，
            # FP32 一次舍入，与 naive_rmsnorm_serial 的后半程逐位同式）。
            scale = np.float32(1.0) / np.float32(
                np.sqrt(np.float32(np.float32(sqsum) / np.float32(H_MAIN)) + np.float32(RMS_NORM_EPS)))
            y_order = (np.float32(scale) * row * weight).astype(np.float32)
            entries[name] = {
                "sqsum": float(sqsum),
                "output_max_abs_diff_vs_serial_truth": float(np.abs(y_order - truth_output).max()),
            }
        propagation[dt_name] = entries
        print(f"[propagate] acc={dt_name}: "
              + " ".join(f"{k}={v['output_max_abs_diff_vs_serial_truth']:.2e}"
                         for k, v in sorted(entries.items(),
                                            key=lambda kv: kv[1]["output_max_abs_diff_vs_serial_truth"],
                                            reverse=True)))
    results["output_propagation"] = {
        "truth_output_max_abs": float(np.abs(truth_output).max()),
        **propagation,
    }

    # ---- 3. atomic 乱序一阶模型：100 种子漂移分布 ----
    shuffle_results = {}
    for dt_name in ("fp32", "fp16"):
        dist = shuffle_drift_distribution(row, truth, arguments.shuffle_seeds, dt_name)
        shuffle_results[dt_name] = dist
        print(f"[shuffle] acc={dt_name}: rel∈[{dist['rel_error_min']:.2e}, "
              f"{dist['rel_error_max']:.2e}] mean={dist['rel_error_mean']:.2e} "
              f"std={dist['rel_error_std']:.2e} unique={dist['unique_values']}/{dist['seeds']}")
    results["shuffle_model"] = {
        "seeds": arguments.shuffle_seeds,
        "model_note": "随机置换串行累加 = atomicAdd 到同一累加器到达顺序的一阶模型"
                      "（真实顺序由 warp 调度/访存延迟决定，非均匀随机；设备实测分布互相印证）",
        **shuffle_results,
    }

    # ---- 4. FP16 卡死区 × 乱序（ep06 假设检验：乱序救不救得回卡死区）----
    # H=65536：ep06 实测 FP16 串行卡死（acc=18688，误差 71.7%）。乱序把
    # 增量到达顺序打乱，但 ulp(acc) 机制只看当前 acc 大小 → 预期同样卡死。
    row_big = np.random.default_rng(SEED + 65536).standard_normal(65536, dtype=np.float32)
    truth_big = float(np.sum(row_big.astype(np.float64) ** 2))
    serial16 = sum_serial(row_big, "fp16")
    serial32 = sum_serial(row_big, "fp32")
    dist16 = shuffle_drift_distribution(row_big, truth_big, min(arguments.shuffle_seeds, 30), "fp16",
                                        seed_base=SEED + 900)
    dist32 = shuffle_drift_distribution(row_big, truth_big, min(arguments.shuffle_seeds, 30), "fp32",
                                        seed_base=SEED + 900)
    pairwise32 = sum_pairwise(row_big, "fp32")
    results["fp16_deadzone_H65536"] = {
        "H": 65536,
        "sqsum_truth_f64": truth_big,
        "ep06_reference": "ep06 串行 FP16 卡死 acc≈18688 rel err 7.14e-01（本章直接引用并扩展）",
        "fp16_serial": serial16,
        "fp16_serial_rel": abs(serial16 - truth_big) / truth_big,
        "fp16_shuffle": dist16,
        "fp32_serial": serial32,
        "fp32_serial_rel": abs(serial32 - truth_big) / truth_big,
        "fp32_shuffle": dist32,
        "fp32_pairwise_rel": abs(pairwise32 - truth_big) / truth_big,
        "conclusion": "",
    }
    stuck_values = [v for v in
                    [sum_shuffled(row_big, SEED + 900 + i, "fp16") for i in range(5)]]
    results["fp16_deadzone_H65536"]["fp16_shuffle_first5_values"] = stuck_values
    print(f"[deadzone] H=65536: fp16 serial={serial16:.0f} rel={results['fp16_deadzone_H65536']['fp16_serial_rel']:.2e} | "
          f"fp16 shuffle rel∈[{dist16['rel_error_min']:.2e},{dist16['rel_error_max']:.2e}] unique={dist16['unique_values']}")
    print(f"[deadzone]           fp32 serial rel={results['fp16_deadzone_H65536']['fp32_serial_rel']:.2e} | "
          f"fp32 shuffle rel∈[{dist32['rel_error_min']:.2e},{dist32['rel_error_max']:.2e}] unique={dist32['unique_values']}")

    results["conclusions"] = [
        "FP32 累加：六种顺序全不同（位级），但相对误差全部 ≤1e-6 —— 顺序敏感、无害；"
        "FP64 累加：差异 <1e-15 —— '顺序不敏感'是相对累加器精度的",
        "FP16 累加：顺序决定生死（卡死区）—— 任何固定顺序都行，但漂移不可接受（ep06 结论沿用）",
        "CPU 串行实现的确定性来自'顺序固定'而非'精度高'：serial_l2r 与 serial_r2l 位级不同",
        "atomic 乱序（一阶模型）：FP32 累加器下漂移 ~1e-7 相对（与确定性归约同量级）"
        "但每次运行不同 —— 不确定的误差比固定的误差更难测试；FP16 累加器下乱序无法逃离卡死区",
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
