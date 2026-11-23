#!/usr/bin/env python3
"""生成第 01 期 RMSNorm 正确性 fixtures（固定 seed 的输入与 FP32 期望输出）。

产物：
- fixtures/correctness.npz：18 组合同 Shape（6 个 H × 3 个 rows）的
  x（float32）、w（float32）、期望输出 y（float32，NumPy 朴素参考）；
- fixtures/manifest.json：seed 规则、逐张量 SHA256、eps。

性能基准的输入不进 fixtures：benchmark.py 用同一 seed 规则在运行时生成
（性能对照只依赖 Shape/dtype/数值分布，不依赖具体数值的跨平台一致）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rmsnorm_reference import RMS_NORM_EPS, naive_rmsnorm_numpy

SEED_BASE = 20260101
HIDDEN_SIZES = [127, 128, 129, 1023, 1024, 1025]
ROW_COUNTS = [1, 4, 128]


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
        "schema_version": "kernel-chapter01.fixtures.v1",
        "seed_base": SEED_BASE,
        "eps": RMS_NORM_EPS,
        "truth_implementation": "naive_rmsnorm_numpy (FP32 CPU)",
        "combos": [],
    }

    for hidden_size in HIDDEN_SIZES:
        # 权重对同一 H 的所有 rows 共享一份（不同 rows 各自生成会导致
        # npz 按键覆盖后，期望输出与权重不匹配）。
        weight_seed = SEED_BASE + hidden_size
        weight_generator = torch.Generator().manual_seed(weight_seed)
        weight = torch.randn(hidden_size, generator=weight_generator, dtype=torch.float32)
        for rows in ROW_COUNTS:
            seed = combo_seed(rows, hidden_size)
            generator = torch.Generator().manual_seed(seed)
            x = torch.randn(rows, hidden_size, generator=generator, dtype=torch.float32)
            expected = naive_rmsnorm_numpy(x.numpy(), weight.numpy(), RMS_NORM_EPS)

            key = f"{rows}x{hidden_size}"
            arrays[f"x_{key}"] = x.numpy()
            arrays[f"w_{hidden_size}"] = weight.numpy()
            arrays[f"y_{key}"] = expected
            manifest["combos"].append(
                {
                    "rows": rows,
                    "hidden_size": hidden_size,
                    "seed": seed,
                    "weight_seed": weight_seed,
                    "sha256": {
                        "x": sha256_of_array(x.numpy()),
                        "w": sha256_of_array(weight.numpy()),
                        "y": sha256_of_array(expected),
                    },
                }
            )
            print(f"[fixture] {key}: seed={seed}")

    npz_path = out_dir / "correctness.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"[done] {npz_path}（{npz_path.stat().st_size} 字节）+ manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
