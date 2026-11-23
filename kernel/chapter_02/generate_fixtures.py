#!/usr/bin/env python3
"""生成第 02 期 RMSNorm 正确性 fixtures（固定 seed 的输入与 FP32 期望输出）。

与第 01 期的差异：
1. 合同 Shape 扩到全口径 30 组 = 6 个 H（127/128/129/1023/1024/1025，其中
   1024 = Qwen3-0.6B hidden_size）× 5 个行数（1/4/128/512/2048）；
2. 期望输出由本期新增的"朴素串行实现"（逐行 × 逐元素 FP32 自然顺序累加）
   生成，并对每组额外记录"串行 vs 行内 NumPy 归约"的实现差；
3. seed 用 NumPy default_rng（PCG64）而非 torch.Generator —— 本期 fixtures
   独立生成、numpy-only，可在无 torch 的本地环境与任一服务器重建核对；
   PCG64 流跨 numpy 版本/平台稳定（NEP 19 流兼容承诺），manifest 记录 SHA256
   供两台服务器共用同一份文件核对。

产物：
- fixtures/correctness.npz：30 组的 x（float32）、w（float32）、期望输出
  y（float32，naive_rmsnorm_serial）；
- fixtures/manifest.json：seed 规则、逐张量 SHA256、eps、串行/行内实现差。

性能基准的输入不进 fixtures：benchmark.py 用同一 seed 规则在运行时生成
（性能对照只依赖 Shape/dtype/数值分布，不依赖具体数值的跨平台一致）。
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
        "schema_version": "kernel-chapter03.fixtures.v1",
        "seed_base": SEED_BASE,
        "rng": "numpy default_rng (PCG64)，输入 standard_normal float32",
        "eps": RMS_NORM_EPS,
        "truth_implementation": "naive_rmsnorm_serial (FP32 CPU, 逐行逐元素自然顺序累加)",
        "hidden_sizes": HIDDEN_SIZES,
        "row_counts": ROW_COUNTS,
        "combos": [],
    }

    total_start = time.perf_counter()
    for hidden_size in HIDDEN_SIZES:
        # 权重对同一 H 的所有 rows 共享一份（与第 01 期同一规则）。
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
            # 归约顺序差异的定量证据：串行（自然顺序）vs 行内 NumPy（pairwise）。
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
