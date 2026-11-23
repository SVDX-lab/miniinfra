#!/usr/bin/env python3
"""第 19 期 fixtures（与第 17 期 GEMM 同语义同 seed，单源真值策略）。

第 19 期的算子语义与第 17 期完全一致（同一 GEMM C[M,N]=A[M,K]·B[K,N]），
ep19 的先例：fixtures 复用第 17 期（真值单源，避免两份"相同语义"的 fixtures
漂移）。本脚本的职责：
1. 若 ../chapter_17/fixtures/correctness.npz 存在：校验其 manifest 的
   npz_sha256 并把可复算的 A/B/C 摘要写进本期的 fixtures/PROVENANCE.json
   （不复制 npz 本体——运行时 compare_correctness.py 直接读 ch18 路径）；
2. 否则（本期目录独立部署时）：按与 ch18 完全相同的 seed 规则重新生成
   A/B/C（无 epilogue 列——本期不涉及 epilogue 实验），写
   fixtures/correctness.npz + manifest.json。

真值策略（ch18 同源）：tiny_int 手算 + matmul_naive_rows（FP32 k 升序确定性）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "implementations" / "cpu"))

from naive_gemm import TINY_A, TINY_B, matmul_naive_rows, matmul_triple_loop  # noqa: E402

SEED_BASE = 20261118

# 与 ch18 相同的随机组合（epilogue 列不进本期 fixtures）
RANDOM_COMBOS = [
    (13, 17, 11),
    (64, 64, 64),
    (128, 128, 128),
    (200, 200, 200),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
]


def combo_seed(m: int, k: int, n: int) -> int:
    return SEED_BASE + m * 100000 + k * 100 + n


def sha256_of_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def npz_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fixtures")
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ch18_npz = Path(__file__).resolve().parent.parent / "chapter_17" / "fixtures" / "correctness.npz"
    ch18_manifest = ch18_npz.with_name("manifest.json")

    if ch18_npz.exists() and ch18_manifest.exists():
        manifest = json.loads(ch18_manifest.read_text(encoding="utf-8"))
        digest = npz_sha256(ch18_npz)
        provenance = {
            "schema_version": "kernel-chapter20.fixtures.v1",
            "mode": "reuse_chapter18",
            "source_npz": str(ch18_npz.resolve()),
            "source_manifest_schema": manifest.get("schema_version"),
            "source_npz_sha256": digest,
            "source_manifest_npz_sha256": manifest.get("npz_sha256"),
            "sha256_match": manifest.get("npz_sha256") == digest,
            "seed_base": manifest.get("seed_base"),
            "seed_rule": manifest.get("seed_rule"),
            "truth_implementation": manifest.get("truth_implementation"),
            "note": "第 19 期 GEMM 语义与第 17 期一致（载体复用），fixtures 单源复用；"
                    "A/B 可由 seed 规则复算核对（generate --regenerate 亦按同规则）",
        }
        (out_dir / "PROVENANCE.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[fixtures] 复用 ch18：{ch18_npz}")
        print(f"[fixtures] SHA256 一致 = {provenance['sha256_match']}")
        return 0 if provenance["sha256_match"] else 1

    # ---- 独立部署回退：按 ch18 同规则重新生成 A/B/C ----
    arrays: dict[str, np.ndarray] = {}
    manifest: dict = {
        "schema_version": "kernel-chapter20.fixtures.v1",
        "seed_base": SEED_BASE,
        "seed_rule": "seed = SEED_BASE + M*100000 + K*100 + N；rng = numpy default_rng(seed)",
        "truth_implementation": "matmul_naive_rows（FP32，k 升序累加）；tiny_int 用 matmul_triple_loop",
        "combos": [],
    }
    m, k, n = TINY_A.shape[0], TINY_A.shape[1], TINY_B.shape[1]
    key = f"{m}x{k}x{n}"
    arrays[f"A_{key}"] = TINY_A
    arrays[f"B_{key}"] = TINY_B
    arrays[f"C_{key}"] = matmul_triple_loop(TINY_A, TINY_B)
    manifest["combos"].append({"shape": [m, k, n], "kind": "tiny_int_hand_checkable"})
    for mm, kk, nn in RANDOM_COMBOS:
        seed = combo_seed(mm, kk, nn)
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((mm, kk), dtype=np.float32)
        B = rng.standard_normal((kk, nn), dtype=np.float32)
        c_truth = matmul_naive_rows(A, B)
        key = f"{mm}x{kk}x{nn}"
        arrays[f"A_{key}"] = A
        arrays[f"B_{key}"] = B
        arrays[f"C_{key}"] = c_truth
        manifest["combos"].append(
            {"shape": [mm, kk, nn], "kind": "random", "seed": seed,
             "sha256": {"A": sha256_of_array(A), "B": sha256_of_array(B),
                        "C": sha256_of_array(c_truth)}})
        print(f"[fixture] {key}")
    npz_path = out_dir / "correctness.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest["npz_sha256"] = npz_sha256(npz_path)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[fixtures] 重新生成：{npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
