#!/usr/bin/env python3
"""生成第 16 期 fixtures：9 组 Shape（3 rows × 3 cols）× 5 算子的输入与 CPU FP32 真值。

纯 NumPy 实现（主 agent 环境纪律：本地不装 torch；NumPy Generator 流跨版本稳定，
见 NEP 19——Generator 的比特流与分布方法受版本兼容承诺，跨机 SHA256 校验兜底）。

产物（fixtures/ 目录）：
- correctness.npz：全部输入与真值（FP32）；
- manifest.json：seed 规则、逐张量 SHA256、npz SHA256。

跨机一致性约定（协议 §2）：
- fixtures 只在一台机器生成（本地 WSL2），两台服务器共用同一份；
- 服务器可用 --verify-only 重放同 seed 生成并逐张量比对 SHA256：
  位级一致则等价使用本地 npz（免大文件上传），不一致则必须上传本地文件。
- transpose 真值是纯重排（零浮点运算），位级一致由排列性质保证。

seed 规则：seed = SEED_BASE + rows*100000 + cols（NumPy PCG64）；
silu_gate 的第二输入 up 用 seed+1；softmax 稳定性压力例 x_stress = x_{4x1024} + 100。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "implementations" / "cpu"))

from ops_reference import (  # noqa: E402
    reduce_max_rows_numpy,
    reduce_sum_rows_numpy,
    silu_gate_numpy,
    softmax_rows_numpy,
    scan_inclusive_rows_numpy,
    transpose_numpy,
)

SEED_BASE = 20260117
ROW_COUNTS = [1, 128, 2048]
COL_COUNTS = [127, 1024, 4096]
STRESS_ROWS, STRESS_COLS = 4, 1024
STRESS_OFFSET = 100.0


def combo_seed(rows: int, cols: int) -> int:
    return SEED_BASE + rows * 100000 + cols


def sha256_of_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def random_normal(seed: int, rows: int, cols: int) -> np.ndarray:
    """固定 seed 的 FP32 正态随机矩阵（NumPy PCG64 + ziggurat，跨版本流稳定）。"""
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.standard_normal((rows, cols), dtype=np.float32)


def build_arrays() -> tuple[dict[str, np.ndarray], dict]:
    """生成全部输入与真值（FP32 CPU），返回数组字典与 manifest 骨架。"""
    arrays: dict[str, np.ndarray] = {}
    manifest: dict = {
        "schema_version": "kernel-chapter17.fixtures.v1",
        "seed_base": SEED_BASE,
        "rng": "numpy PCG64 + standard_normal(ziggurat) FP32，流跨版本兼容（NEP 19）",
        "truth_implementation": "implementations/cpu/ops_reference.py（NumPy FP32 真值锚点）",
        "truth_dtype": "float32",
        "shapes": {"rows": ROW_COUNTS, "cols": COL_COUNTS},
        "theoretical_bytes_note": "各算子理论字节公式见 ops_harness.py theoretical_bytes()",
        "combos": [],
    }
    for rows in ROW_COUNTS:
        for cols in COL_COUNTS:
            seed = combo_seed(rows, cols)
            x = random_normal(seed, rows, cols)
            up = random_normal(seed + 1, rows, cols)

            key = f"{rows}x{cols}"
            arrays[f"x_{key}"] = x
            arrays[f"up_{key}"] = up
            arrays[f"y_softmax_{key}"] = softmax_rows_numpy(x)
            arrays[f"y_scan_{key}"] = scan_inclusive_rows_numpy(x)
            arrays[f"y_transpose_{key}"] = transpose_numpy(x)
            arrays[f"sum_{key}"] = reduce_sum_rows_numpy(x)
            arrays[f"max_{key}"] = reduce_max_rows_numpy(x)
            arrays[f"y_silu_gate_{key}"] = silu_gate_numpy(x, up)

            manifest["combos"].append(
                {
                    "rows": rows,
                    "cols": cols,
                    "seed": seed,
                    "up_seed": seed + 1,
                    "sha256": {
                        "x": sha256_of_array(x),
                        "up": sha256_of_array(up),
                        "y_softmax": sha256_of_array(arrays[f"y_softmax_{key}"]),
                        "y_scan": sha256_of_array(arrays[f"y_scan_{key}"]),
                        "y_transpose": sha256_of_array(arrays[f"y_transpose_{key}"]),
                        "sum": sha256_of_array(arrays[f"sum_{key}"]),
                        "max": sha256_of_array(arrays[f"max_{key}"]),
                        "y_silu_gate": sha256_of_array(arrays[f"y_silu_gate_{key}"]),
                    },
                }
            )
            print(f"[fixture] {key}: seed={seed} elements={rows * cols}")

    # softmax 数值稳定性压力例：x + 100 使朴素 softmax 溢出（exp(100) 超 FP32 上限）
    stress = random_normal(SEED_BASE + 777, STRESS_ROWS, STRESS_COLS)
    stress = (stress + np.float32(STRESS_OFFSET)).astype(np.float32)
    arrays["x_stress"] = stress
    arrays["y_softmax_stress"] = softmax_rows_numpy(stress)
    manifest["stress_case"] = {
        "rows": STRESS_ROWS,
        "cols": STRESS_COLS,
        "seed": SEED_BASE + 777,
        "offset": STRESS_OFFSET,
        "sha256": {
            "x": sha256_of_array(stress),
            "y_softmax": sha256_of_array(arrays["y_softmax_stress"]),
        },
    }
    return arrays, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "fixtures"))
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="不写 npz：重放同 seed 生成并与 manifest.json 逐张量比对 SHA256",
    )
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    arrays, manifest = build_arrays()

    if arguments.verify_only:
        if not manifest_path.exists():
            print("[verify] manifest.json 不存在，无法校验")
            return 2
        reference_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = []
        checked = 0
        for combo in reference_manifest["combos"]:
            key = f"{combo['rows']}x{combo['cols']}"
            for name in combo["sha256"]:
                arrays_key = {
                    "x": f"x_{key}", "up": f"up_{key}", "y_softmax": f"y_softmax_{key}",
                    "y_scan": f"y_scan_{key}", "y_transpose": f"y_transpose_{key}",
                    "sum": f"sum_{key}", "max": f"max_{key}", "y_silu_gate": f"y_silu_gate_{key}",
                }[name]
                digest = sha256_of_array(arrays[arrays_key])
                checked += 1
                if digest != combo["sha256"][name]:
                    mismatches.append(f"{arrays_key}: 期望 {combo['sha256'][name][:16]}… 实得 {digest[:16]}…")
        for name, digest in reference_manifest["stress_case"]["sha256"].items():
            arrays_key = {"x": "x_stress", "y_softmax": "y_softmax_stress"}[name]
            local_digest = sha256_of_array(arrays[arrays_key])
            checked += 1
            if local_digest != digest:
                mismatches.append(f"{arrays_key}: 期望 {digest[:16]}… 实得 {local_digest[:16]}…")
        if mismatches:
            print(f"[verify] 失败：{len(mismatches)} 个张量与 manifest 不一致（RNG 跨版本漂移）：")
            for item in mismatches:
                print("  -", item)
            return 1
        print(f"[verify] 通过：{checked} 个张量位级一致（本机可与生成机共用同一份 fixtures）")
        return 0

    npz_path = out_dir / "correctness.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"[done] {npz_path}（{npz_path.stat().st_size / 2**20:.1f} MiB）+ manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
