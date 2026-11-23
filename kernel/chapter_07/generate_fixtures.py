#!/usr/bin/env python3
"""生成第 07 期正确性 fixtures（固定 seed 的输入与 FP32 串行期望输出）。

口径沿用第 02 期（30 组 = 6 H × 5 rows，1024 = Qwen3-0.6B hidden_size），
seed 换为本期独立基（20260801），真值 = naive_rmsnorm_serial（FP32 串行，
自然顺序累加 = 确定性参照系）。

额外记录（本期专属）：
- 每组"串行 vs 行内 NumPy pairwise"的归约顺序差（第 02 期口径延续）；
- 每组"串行 vs 成对（pairwise_tree 模型）"的平方和位级差 —— 本期确定性
  主题的 fixtures 级证据：同一输入在不同固定顺序下平方和位级不同但
  输出差异被舍入粒度吞掉的部分，与设备 100 次一致率实验互相印证。

大 H 曲线形状（H ∈ {4096, 16384, 65536}）不进 fixtures 文件：输入由
seed 规则在运行时生成（numpy PCG64 流跨平台稳定，设备侧实验与本地
CPU 模拟共享同一 seed 规则即可保证同一输入），避免 ~GB 级文件。

产物：
- fixtures/correctness.npz：30 组 x/w/y_expected；
- fixtures/manifest.json：seed 规则、逐张量 SHA256、eps、顺序差记录。
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

from serial_rmsnorm import RMS_NORM_EPS, naive_rmsnorm_serial  # noqa: E402
from sum_order_lab import sum_pairwise, sum_serial  # noqa: E402

SEED_BASE = 20260801
HIDDEN_SIZES = [127, 128, 129, 1023, 1024, 1025]
ROW_COUNTS = [1, 4, 128, 512, 2048]


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
        "schema_version": "kernel-chapter08.fixtures.v1",
        "seed_base": SEED_BASE,
        "rng": "numpy default_rng (PCG64)，输入 standard_normal float32",
        "eps": RMS_NORM_EPS,
        "truth_implementation": "naive_rmsnorm_serial (FP32 CPU, 逐行逐元素自然顺序累加——确定性参照系)",
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
            # 本期专属：第一行的平方和顺序差（串行 vs pairwise 树形模型）
            row0 = x[0]
            sq_serial = sum_serial(row0, "fp32")
            sq_pairwise = sum_pairwise(row0, "fp32")

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
                    "row0_sqsum_serial": sq_serial,
                    "row0_sqsum_pairwise_tree": sq_pairwise,
                    "row0_sqsum_bit_identical": (np.float32(sq_serial).tobytes()
                                                 == np.float32(sq_pairwise).tobytes()),
                    "sha256": {
                        "x": sha256_of_array(x),
                        "w": sha256_of_array(weight),
                        "y": sha256_of_array(expected),
                    },
                }
            )
            print(f"[fixture] {key}: seed={seed} serial={serial_ms:.0f}ms "
                  f"sqsum顺序位级同={manifest['combos'][-1]['row0_sqsum_bit_identical']}")

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
