#!/usr/bin/env python3
"""生成第 06 期 RMSNorm 正确性 fixtures（固定 seed 的输入与 FP32 期望输出）。

与第 02 期的关系：RMSNorm 数学与真值口径完全继承（naive 串行 FP32 自然顺序
累加），本期新增的是"同一逻辑数据的多布局视图"——x 连续生成，转置视图 /
行切片视图在实验时零拷贝派生，逻辑值与 x 相同 => 期望输出 y 一份即可覆盖
三种布局（这是本期 fixtures 的设计要点：布局不改逻辑值）。

产物：
- fixtures/correctness.npz：30 组的 x（float32）、w（float32）、y（float32，
  naive_rmsnorm_serial）；
- fixtures/manifest.json：seed 规则、逐张量 SHA256、eps、布局派生规则。
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

from stride_layout import (  # noqa: E402
    RMS_NORM_EPS,
    StridedTensor,
    naive_rmsnorm_strided,
    policy_a_reject,
    policy_b_autocopy,
    verify_stride_semantics,
)

SEED_BASE = 20260701
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
        "schema_version": "kernel-chapter07.fixtures.v1",
        "seed_base": SEED_BASE,
        "rng": "numpy default_rng (PCG64)，输入 standard_normal float32",
        "eps": RMS_NORM_EPS,
        "truth_implementation": "naive_rmsnorm_strided (FP32 CPU, stride 显式寻址 + 自然顺序累加)",
        "layout_rule": "转置视图/行切片视图由 x 零拷贝派生（逻辑值不变 => 期望 y 复用一份）",
        "hidden_sizes": HIDDEN_SIZES,
        "row_counts": ROW_COUNTS,
        "combos": [],
    }

    total_start = time.perf_counter()
    stride_report = verify_stride_semantics()
    for hidden_size in HIDDEN_SIZES:
        weight_seed = SEED_BASE + hidden_size
        weight_rng = np.random.default_rng(weight_seed)
        weight = weight_rng.standard_normal(hidden_size, dtype=np.float32)
        for rows in ROW_COUNTS:
            rng = np.random.default_rng(combo_seed(rows, hidden_size))
            x = rng.standard_normal((rows, hidden_size), dtype=np.float32)
            y = naive_rmsnorm_strided(StridedTensor.row_major(x), weight, RMS_NORM_EPS)
            combo_id = f"H{hidden_size}_r{rows}"
            # 三策略在连续布局下互相一致（真值自检）。
            y_b = policy_b_autocopy(StridedTensor.row_major(x), weight)
            y_c = policy_c_check = naive_rmsnorm_strided(
                StridedTensor.transposed_view(x), weight)  # 策略C 直接吃转置视图
            diff_bc = float(np.abs(y_c.buffer.reshape(rows, hidden_size) - y_b.buffer.reshape(rows, hidden_size)).max())
            arrays[f"x_{combo_id}"] = x
            arrays[f"w_{combo_id}"] = weight
            arrays[f"y_{combo_id}"] = y.buffer.reshape(rows, hidden_size)
            manifest["combos"].append({
                "id": combo_id,
                "rows": rows,
                "hidden_size": hidden_size,
                "seed": combo_seed(rows, hidden_size),
                "sha256": {"x": sha256_of_array(x), "w": sha256_of_array(weight),
                           "y": sha256_of_array(arrays[f"y_{combo_id}"])},
                "policyB_vs_policyC_transposed_view_max_abs": diff_bc,
            })
    elapsed = time.perf_counter() - total_start
    manifest["stride_semantics_report"] = stride_report
    manifest["generation_seconds"] = round(elapsed, 2)

    np.savez_compressed(out_dir / "correctness.npz", **arrays)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    max_diff = max(c["policyB_vs_policyC_transposed_view_max_abs"] for c in manifest["combos"])
    print(f"[fixtures] 30 组生成完毕（{elapsed:.1f}s）：x/w/y 各 30 张量")
    print(f"[fixtures] 策略C(转置视图直接算) vs 策略B(拷贝后算) 全组合 max_abs={max_diff:.3e}（应≈0：布局不改逻辑值）")
    print(f"[fixtures] stride 公式 vs numpy as_strided：all_equal={stride_report['all_equal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
