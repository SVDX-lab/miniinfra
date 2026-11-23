#!/usr/bin/env python3
"""生成第 03 期 Residual Add + RMSNorm 正确性 fixtures（固定 seed）。

与第 02 期的关系（复用而非重新发明）：
1. 合同 Shape 同口径 30 组 = 6 个 H（127/128/129/1023/1024/1025，1024 =
   Qwen3-0.6B hidden_size）× 5 个行数（1/4/128/512/2048）；
2. 输入从单张量 x 变为二元组 (r, h)：r ~ N(0,1)，h ~ N(0,1)，独立流生成；
3. 期望输出：t = fl32(r + h)（分离路径 Pass 1，加法无累加、串行与向量化
   位级一致）+ y = naive_rmsnorm_serial(t, w)（第 02 期真值实现原样复用，
   生成器对 fused 路径自洽性另做位级核对）；
4. seed 规则沿用 numpy default_rng（PCG64），本期基址 20260401（第 02 期
   20260301），h 流错开 555555 避免同种子；PCG64 流跨版本/平台稳定；
   manifest 记录 SHA256 供两台服务器共用同一份文件核对。

产物：
- fixtures/correctness.npz：30 组的 r/h/w/t/y（全部 float32）；
- fixtures/manifest.json：seed 规则、逐张量 SHA256、eps、自洽性核对。

性能基准输入不进 fixtures：benchmark.py 用同一 seed 规则运行时生成
（DRAM 区大形状 r/h 达数百 MB，不落盘）。
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

from fused_add_rmsnorm import (  # noqa: E402
    RMS_NORM_EPS,
    fused_add_rmsnorm_inplace_serial,
    separated_add_rmsnorm_serial,
)

SEED_BASE = 20260401
HIDDEN_SIZES = [127, 128, 129, 1023, 1024, 1025]  # 1024 = Qwen3-0.6B hidden_size
ROW_COUNTS = [1, 4, 128, 512, 2048]
H_STREAM_OFFSET = 555555  # h 流与 r 流错开（同 base 种子不同流）


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
        "schema_version": "kernel-chapter04.fixtures.v1",
        "seed_base": SEED_BASE,
        "rng": "numpy default_rng (PCG64)，r/h 各自独立流 standard_normal float32",
        "stream_rule": "r: SEED_BASE+rows*100000+H；h: r_seed+555555；w: SEED_BASE+H",
        "eps": RMS_NORM_EPS,
        "truth_implementation": (
            "separated_add_rmsnorm_serial：t=fl32(r+h)（Pass1 加法无累加）；"
            "y=naive_rmsnorm_serial(t,w)（第 02 期真值实现原样复用）"
        ),
        "residual_semantics": (
            "in-place（合同主语义）：融合路径调用后 residual buffer 内容变为 t；"
            "fixtures 记录的 t 即更新后的 residual 真值"
        ),
        "hidden_sizes": HIDDEN_SIZES,
        "row_counts": ROW_COUNTS,
        "combos": [],
    }

    total_start = time.perf_counter()
    selfcheck_bitwise_mismatch = 0
    for hidden_size in HIDDEN_SIZES:
        # 权重对同一 H 的所有 rows 共享一份（与第 01/02 期同一规则）。
        weight_seed = SEED_BASE + hidden_size
        weight_rng = np.random.default_rng(weight_seed)
        weight = weight_rng.standard_normal(hidden_size, dtype=np.float32)
        for rows in ROW_COUNTS:
            seed = combo_seed(rows, hidden_size)
            rng_r = np.random.default_rng(seed)
            r = rng_r.standard_normal((rows, hidden_size), dtype=np.float32)
            rng_h = np.random.default_rng(seed + H_STREAM_OFFSET)
            h = rng_h.standard_normal((rows, hidden_size), dtype=np.float32)

            start = time.perf_counter()
            t, y = separated_add_rmsnorm_serial(r, h, weight, RMS_NORM_EPS)
            serial_ms = (time.perf_counter() - start) * 1000.0

            # 融合路径自洽核对：in-place 融合版（r 克隆）的 y 与 residual 应与
            # 分离真值位级一致（同一 FP32 串行累加顺序，无实现差）。
            r_clone = r.copy()
            r_upd, y_fused = fused_add_rmsnorm_inplace_serial(r_clone, h, weight, RMS_NORM_EPS)
            y_diff = float(np.abs(y_fused - y).max())
            t_diff = float(np.abs(r_upd - t).max())
            if y_diff != 0.0 or t_diff != 0.0:
                selfcheck_bitwise_mismatch += 1

            key = f"{rows}x{hidden_size}"
            arrays[f"r_{key}"] = r
            arrays[f"h_{key}"] = h
            arrays[f"w_{hidden_size}"] = weight
            arrays[f"t_{key}"] = t
            arrays[f"y_{key}"] = y
            manifest["combos"].append(
                {
                    "rows": rows,
                    "hidden_size": hidden_size,
                    "seed_r": seed,
                    "seed_h": seed + H_STREAM_OFFSET,
                    "weight_seed": weight_seed,
                    "serial_generation_ms": round(serial_ms, 1),
                    "fused_vs_separated_bitwise": y_diff == 0.0 and t_diff == 0.0,
                    "sha256": {
                        "r": sha256_of_array(r),
                        "h": sha256_of_array(h),
                        "w": sha256_of_array(weight),
                        "t": sha256_of_array(t),
                        "y": sha256_of_array(y),
                    },
                }
            )
            print(f"[fixture] {key}: seed_r={seed} serial={serial_ms:.0f}ms "
                  f"融合自洽(y差={y_diff:.1e},t差={t_diff:.1e})")

    manifest["fused_vs_separated_selfcheck_bitwise_mismatch_groups"] = selfcheck_bitwise_mismatch
    npz_path = out_dir / "correctness.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    total_s = time.perf_counter() - total_start
    print(f"[done] {npz_path}（{npz_path.stat().st_size / 2**20:.1f} MiB）+ manifest.json"
          f"，总耗时 {total_s:.1f}s，融合自洽位级不一致组数={selfcheck_bitwise_mismatch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
