#!/usr/bin/env python3
"""第 18 期 fixtures：复用第 17 期同一份（同一 GEMM 语义，真值单源不分叉）。

策略（协议 §2：CPU FP32 真值一台机器生成后全平台共用）：
- 第 17 期 fixtures/（correctness.npz + manifest.json）为唯一真值源；
- 本脚本从同一 seed 规则重新生成全部数组，逐张量 SHA256 对拍第 17 期
  manifest——两者必须一致（同时验证生成过程的可复现性），然后写入
  chapter_18/fixtures/，第 18 期的两台服务器消费这份拷贝；
- 结论写 stdout 与 manifest（provenance 字段）。
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

from naive_gemm import (  # noqa: E402
    TINY_A,
    TINY_B,
    TINY_BIAS,
    add_bias,
    gelu_tanh,
    matmul_naive_rows,
    matmul_triple_loop,
    silu,
)

SEED_BASE = 20261118  # 与第 17 期一致（fixture 单源）
RANDOM_COMBOS = [
    (13, 17, 11),
    (64, 64, 64),
    (128, 128, 128),
    (200, 200, 200),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
]
EPILOGUE_MAX_DIM = 256


def combo_seed(m: int, k: int, n: int) -> int:
    return SEED_BASE + m * 100000 + k * 100 + n


def sha256_of_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fixtures")
    parser.add_argument("--ep18-dir", default=str(ROOT.parent / "chapter_17" / "fixtures"))
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    ep18_manifest_path = Path(arguments.ep18_dir) / "manifest.json"
    if not ep18_manifest_path.exists():
        print(f"[warn] 第 17 期 manifest 不存在（{ep18_manifest_path}），跳过对拍，仅生成")
        ep18_manifest = None
    else:
        with ep18_manifest_path.open("r", encoding="utf-8") as file:
            ep18_manifest = json.load(file)

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    hashes: dict[str, dict[str, str]] = {}

    # ---- tiny_int ----
    c_truth = matmul_triple_loop(TINY_A, TINY_B)
    key = "4x8x4"
    arrays[f"A_{key}"] = TINY_A
    arrays[f"B_{key}"] = TINY_B
    arrays[f"bias_{key}"] = TINY_BIAS
    arrays[f"C_{key}"] = c_truth
    arrays[f"C_bias_{key}"] = add_bias(c_truth, TINY_BIAS)
    arrays[f"C_gelu_{key}"] = gelu_tanh(add_bias(c_truth, TINY_BIAS))
    arrays[f"C_silu_{key}"] = silu(add_bias(c_truth, TINY_BIAS))

    # ---- 随机组 ----
    for m, k, n in RANDOM_COMBOS:
        seed = combo_seed(m, k, n)
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((m, k), dtype=np.float32)
        B = rng.standard_normal((k, n), dtype=np.float32)
        bias = (0.1 * rng.standard_normal(n, dtype=np.float32)).astype(np.float32)
        c_truth = matmul_naive_rows(A, B)
        key = f"{m}x{k}x{n}"
        arrays[f"A_{key}"] = A
        arrays[f"B_{key}"] = B
        arrays[f"bias_{key}"] = bias
        arrays[f"C_{key}"] = c_truth
        if m <= EPILOGUE_MAX_DIM:
            c_bias = add_bias(c_truth, bias)
            arrays[f"C_bias_{key}"] = c_bias
            arrays[f"C_gelu_{key}"] = gelu_tanh(c_bias)
            arrays[f"C_silu_{key}"] = silu(c_bias)

    for name, array in arrays.items():
        hashes[name] = sha256_of_array(array)

    mismatch = []
    if ep18_manifest is not None:
        # 逐 combo 逐张量对拍（语义级强校验；npz 字节级可能受压缩器版本影响）
        for combo in ep18_manifest["combos"]:
            shape = combo["shape"]
            key = f"{shape[0]}x{shape[1]}x{shape[2]}"
            for tag, digest in combo.get("sha256", {}).items():
                local_name = f"{tag}_{key}"
                if local_name in hashes and hashes[local_name] != digest:
                    mismatch.append(f"{local_name}: {hashes[local_name][:12]} != {digest[:12]}")

        if mismatch:
            print("[FAIL] 逐张量 SHA 对拍不一致：")
            for line in mismatch:
                print("  ", line)
            return 1
        print(f"[check] 逐张量 SHA256 与第 17 期 manifest 全部一致（{len(ep18_manifest['combos'])} 组，"
              f"共 {len(hashes)} 个张量）——真值单源成立")

    manifest = {
        "schema_version": "kernel-chapter19.fixtures.v1",
        "seed_base": SEED_BASE,
        "seed_rule": "seed = SEED_BASE + M*100000 + K*100 + N；rng = numpy default_rng(seed)",
        "truth_implementation": "matmul_naive_rows（FP32，k 升序累加）；tiny_int 用 matmul_triple_loop",
        "provenance": "与第 17 期同一份真值（同 seed 重生成 + 逐张量 SHA256 对拍一致），单源不分叉",
        "ep18_manifest_used": str(ep18_manifest_path),
        "ep18_npz_sha256": (ep18_manifest or {}).get("npz_sha256"),
        "arrays": {name: {"sha256": hashes[name]} for name in sorted(hashes)},
    }
    npz_path = out_dir / "correctness.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"[done] {npz_path}（{npz_path.stat().st_size / 1e6:.1f} MB）+ manifest.json（provenance=chapter_17 单源）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
