#!/usr/bin/env python3
"""生成第 05 期 RMSNorm 正确性 fixtures（固定 seed 的输入与 FP32 期望输出）。

与第 02 期的关系：seed 规则、Shape 全口径、真值实现完全一致
（seed_base=20260301，30 组 = 6 H × 5 rows，真值 = naive_rmsnorm_serial），
因此生成的 x/w/y 与第 02 期 fixtures 位级相同 —— 跨期对照（第 02/03 期
FP16 误差数字）可直接复用同一份数据口径。schema_version 独立为本期。

产物：
- fixtures/correctness.npz：30 组的 x（float32）、w（float32）、期望输出
  y（float32，naive_rmsnorm_serial）；
- fixtures/manifest.json：seed 规则、逐张量 SHA256、eps、串行/行内实现差。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from naive_rmsnorm import RMS_NORM_EPS, naive_rmsnorm_rowwise, naive_rmsnorm_serial  # noqa: E402

SEED_BASE = 20260301
HIDDEN_SIZES = [127, 128, 129, 1023, 1024, 1025]  # 1024 = Qwen3-0.6B hidden_size
ROW_COUNTS = [1, 4, 128, 512, 2048]

# 实验 4（累加 dtype）的构造输入：大范数行（平方和逼近/越过 FP16 max=65504）
# 与长 H 行（平方和 ≈ H 随 H 增长）。固定 seed，随 fixtures 一起 SHA256 存档。
ACC_SEED_BASE = 20260601
ACC_LARGE_SCALES = [1, 2, 4, 8, 16, 32]      # H=1024，平方和 ≈ 1024·scale²
ACC_LONG_H = [1024, 4096, 16384, 65536, 131072]  # x~N(0,1)，平方和 ≈ H
ACC_WIDE_ROWS = 4                             # 宽动态范围行（bf16 更准构造）


def _acc_weight(hidden_size: int, tag_index: int) -> np.ndarray:
    rng = np.random.default_rng(ACC_SEED_BASE + 777 + tag_index)
    return rng.standard_normal(hidden_size, dtype=np.float32)


def combo_seed(rows: int, hidden_size: int) -> int:
    return SEED_BASE + rows * 100000 + hidden_size


def sha256_of_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fixtures")
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    manifest: dict = {
        "schema_version": "kernel-chapter06.fixtures.v1",
        "seed_base": SEED_BASE,
        "rng": "numpy default_rng (PCG64)，输入 standard_normal float32",
        "eps": RMS_NORM_EPS,
        "truth_implementation": "naive_rmsnorm_serial (FP32 CPU, 逐行逐元素自然顺序累加)",
        "note": "seed 规则与第 02 期一致 —— x/w/y 与 chapter_02 fixtures 位级相同（跨期对照口径）",
        "hidden_sizes": HIDDEN_SIZES,
        "row_counts": ROW_COUNTS,
        "combos": [],
    }

    total_start = time.perf_counter()
    for hidden_size in HIDDEN_SIZES:
        weight_seed = SEED_BASE + hidden_size
        weight_rng = np.random.default_rng(weight_seed)
        weight = weight_rng.standard_normal(hidden_size, dtype=np.float32)
        for rows in ROW_COUNTS:
            seed = combo_seed(rows, hidden_size)
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((rows, hidden_size), dtype=np.float32)
            start = time.perf_counter()
            expected = naive_rmsnorm_serial(x, weight, RMS_NORM_EPS)
            serial_ms = (time.perf_counter() - start) * 1000.0
            rowwise = naive_rmsnorm_rowwise(x, weight, RMS_NORM_EPS)
            order_diff = float(np.abs(expected - rowwise).max())

            key = f"{rows}x{hidden_size}"
            arrays[f"x_{key}"] = x
            arrays[f"w_{hidden_size}"] = weight
            arrays[f"y_{key}"] = expected
            manifest["combos"].append(
                {
                    "rows": rows,
                    "hidden_size": hidden_size,
                    "seed": seed,
                    "weight_seed": weight_seed,
                    "serial_generation_ms": round(serial_ms, 1),
                    "serial_vs_rowwise_max_diff": order_diff,
                    "sha256": {
                        "x": sha256_of_array(x),
                        "w": sha256_of_array(weight),
                        "y": sha256_of_array(expected),
                    },
                }
            )
            print(f"[fixture] {key}: seed={seed} serial={serial_ms:.0f}ms "
                  f"串行vs行内 max_diff={order_diff:.2e}")

    # ------------------------------------------------------------------
    # 实验 4 构造输入：accx_{tag}（x float32）+ accy_{tag}（FP32 真值）
    # + accw_{tag}（权重）。manifest 记录理论平方和（H·scale² 或 ≈H），
    # 供"FP16 累加是否溢出"的先验对照。
    # ------------------------------------------------------------------
    acc_entries = []
    tag_index = 0
    # FP16 两条溢出路径的先验：
    #   element 路径：|x| > sqrt(65504)≈255.96 → x² 本身溢出（任何归约组织都溢出）；
    #   sum 路径：H·scale² > 65504 → 平方和溢出（树形归约必溢出；串行累加在
    #     边界区可能"卡死"在 max 以下而非溢出 —— 实验记录该组织差异）。
    def _fp16_priors(x: np.ndarray, h: float) -> dict:
        return {
            "element_sq_overflow_expected": bool(float(np.abs(x).max()) > 255.96),
            "sum_overflow_expected": None,  # 由调用方按理论平方和填写
        }

    for scale in ACC_LARGE_SCALES:
        h = 1024
        tag = f"norm_s{scale}"
        rng = np.random.default_rng(ACC_SEED_BASE + tag_index)
        x = (rng.standard_normal((4, h)) * scale).astype(np.float32)
        w = _acc_weight(h, tag_index)
        y = naive_rmsnorm_rowwise(x, w, RMS_NORM_EPS)
        arrays[f"accx_{tag}"] = x
        arrays[f"accw_{tag}"] = w
        arrays[f"accy_{tag}"] = y
        priors = _fp16_priors(x, h)
        priors["sum_overflow_expected"] = bool(1024 * scale * scale > 65504)
        acc_entries.append({
            "tag": tag, "kind": "large_norm", "rows": 4, "hidden_size": h,
            "scale": scale, "theoretical_sqsum": float(1024 * scale * scale),
            "fp16_acc_overflow_expected": priors["element_sq_overflow_expected"]
                                           or priors["sum_overflow_expected"],
            **priors,
            "seed": ACC_SEED_BASE + tag_index,
            "sha256": {"x": sha256_of_array(x), "w": sha256_of_array(w), "y": sha256_of_array(y)},
        })
        tag_index += 1
    for h in ACC_LONG_H:
        tag = f"long_h{h}"
        rng = np.random.default_rng(ACC_SEED_BASE + tag_index)
        x = rng.standard_normal((2, h)).astype(np.float32)
        w = _acc_weight(h, tag_index)
        y = naive_rmsnorm_rowwise(x, w, RMS_NORM_EPS)
        arrays[f"accx_{tag}"] = x
        arrays[f"accw_{tag}"] = w
        arrays[f"accy_{tag}"] = y
        priors = _fp16_priors(x, h)
        priors["sum_overflow_expected"] = bool(h > 65504 * 0.99)
        acc_entries.append({
            "tag": tag, "kind": "long_h", "rows": 2, "hidden_size": h,
            "theoretical_sqsum_mean": float(h),
            "fp16_acc_overflow_expected": priors["element_sq_overflow_expected"]
                                           or priors["sum_overflow_expected"],
            **priors,
            "seed": ACC_SEED_BASE + tag_index,
            "sha256": {"x": sha256_of_array(x), "w": sha256_of_array(w), "y": sha256_of_array(y)},
        })
        tag_index += 1
    for i in range(ACC_WIDE_ROWS):
        tag = f"wide_r{i}"
        rng = np.random.default_rng(ACC_SEED_BASE + 900 + i)
        big = rng.standard_normal((1, 512)) * 1e4
        small = rng.standard_normal((1, 512)) * 1e-7
        row = np.concatenate([big, small], axis=1)
        perm_rng = np.random.default_rng(ACC_SEED_BASE + 950 + i)
        perm_rng.shuffle(row, axis=1)
        x = row.astype(np.float32)
        w = _acc_weight(1024, 900 + i)
        y = naive_rmsnorm_rowwise(x, w, RMS_NORM_EPS)
        arrays[f"accx_{tag}"] = x
        arrays[f"accw_{tag}"] = w
        arrays[f"accy_{tag}"] = y
        priors = _fp16_priors(x, h)
        priors["sum_overflow_expected"] = None
        acc_entries.append({
            "tag": tag, "kind": "wide_dynamic", "rows": 1, "hidden_size": 1024,
            "fp16_acc_overflow_expected": priors["element_sq_overflow_expected"],
            "note": "半行 ~1e4 + 半行 ~1e-7：元素平方 1e8 经 element 路径溢出 FP16 累加；"
                    "归一化后小值分支 fp16 丢失 / bf16 保留",
            "seed": ACC_SEED_BASE + 900 + i,
            "sha256": {"x": sha256_of_array(x), "w": sha256_of_array(w), "y": sha256_of_array(y)},
        })
        tag_index += 1
    manifest["acc_inputs"] = acc_entries

    npz_path = out_dir / "correctness.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    total_s = time.perf_counter() - total_start
    print(f"[done] {npz_path}（{npz_path.stat().st_size / 2**20:.1f} MiB）+ manifest.json"
          f"，总耗时 {total_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
