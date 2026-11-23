#!/usr/bin/env python3
"""生成第 17 期 GEMM 正确性 fixtures（固定 seed 输入 + CPU FP32 真值）。

产物：
- fixtures/correctness.npz：各组合的 A、B（部分含 bias）与 FP32 期望输出
  C（朴素三重循环真值）；epilogue 组合额外含 C_bias / C_gelu / C_silu；
- fixtures/manifest.json：seed 规则、逐张量 SHA256、真值实现与公式。

真值策略：
- tiny_int（4×8 · 8×4）：整数矩阵，期望值可手算核对（见 smoke_test.py 与
  implementations/cpu/naive_gemm.py 顶部注释）；真值 = 字面三重循环；
- 随机组组：真值 = matmul_naive_rows（k 升序 FP32 累加，确定性）；
  本地生成后随代码一起上传，两台服务器共用同一份（协议 §2）。

性能基准输入不进 fixtures：benchmark.py 用同一 seed 规则在运行时生成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "implementations" / "cpu"))

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

SEED_BASE = 20261118

# (M, K, N) 随机组合：覆盖非整除（guard 路径）、整除边界、大矩阵锚点
RANDOM_COMBOS = [
    (13, 17, 11),      # 全维非整除（任何 tile 都触发 guard）
    (64, 64, 64),
    (128, 128, 128),
    (200, 200, 200),   # 8 的倍数但非 16/32/64 倍数（部分 tile ragged）
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
]
EPILOGUE_MAX_DIM = 256  # epilogue 真值只做到 256（大矩阵 epilogue 由性能实验覆盖）


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
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    manifest: dict = {
        "schema_version": "kernel-chapter18.fixtures.v1",
        "seed_base": SEED_BASE,
        "seed_rule": "seed = SEED_BASE + M*100000 + K*100 + N；rng = numpy default_rng(seed)",
        "truth_implementation": "matmul_naive_rows（FP32，k 升序累加）；tiny_int 用 matmul_triple_loop",
        "gelu_formula": "0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))（tanh 近似式，Kernel 侧同式）",
        "silu_formula": "x/(1+exp(-x))",
        "combos": [],
    }

    # ---- tiny_int：手算可核对 ----
    m, k, n = TINY_A.shape[0], TINY_A.shape[1], TINY_B.shape[1]
    key = f"{m}x{k}x{n}"
    c_truth = matmul_triple_loop(TINY_A, TINY_B)
    c_check = matmul_naive_rows(TINY_A, TINY_B)
    if not np.allclose(c_truth, c_check, atol=0.0, rtol=0.0):
        raise SystemExit("tiny_int：字面三重循环与 naive rows 不一致（整数域应位级一致）")
    arrays[f"A_{key}"] = TINY_A
    arrays[f"B_{key}"] = TINY_B
    arrays[f"bias_{key}"] = TINY_BIAS
    arrays[f"C_{key}"] = c_truth
    arrays[f"C_bias_{key}"] = add_bias(c_truth, TINY_BIAS)
    arrays[f"C_gelu_{key}"] = gelu_tanh(add_bias(c_truth, TINY_BIAS))
    arrays[f"C_silu_{key}"] = silu(add_bias(c_truth, TINY_BIAS))
    manifest["combos"].append(
        {
            "shape": [m, k, n],
            "kind": "tiny_int_hand_checkable",
            "seed": None,
            "epilogue_truth": True,
            "sha256": {name.split("_")[0]: sha256_of_array(arrays[name]) for name in list(arrays) if key in name},
        }
    )
    print(f"[fixture] {key} (tiny_int 手算可核对)")

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

        entry = {
            "shape": [m, k, n],
            "kind": "random",
            "seed": seed,
            "epilogue_truth": m <= EPILOGUE_MAX_DIM,
            "sha256": {
                "A": sha256_of_array(A),
                "B": sha256_of_array(B),
                "bias": sha256_of_array(bias),
                "C": sha256_of_array(c_truth),
            },
        }
        if m <= EPILOGUE_MAX_DIM:
            c_bias = add_bias(c_truth, bias)
            arrays[f"C_bias_{key}"] = c_bias
            arrays[f"C_gelu_{key}"] = gelu_tanh(c_bias)
            arrays[f"C_silu_{key}"] = silu(c_bias)
            entry["sha256"].update(
                {
                    "C_bias": sha256_of_array(arrays[f"C_bias_{key}"]),
                    "C_gelu": sha256_of_array(arrays[f"C_gelu_{key}"]),
                    "C_silu": sha256_of_array(arrays[f"C_silu_{key}"]),
                }
            )
        manifest["combos"].append(entry)
        print(f"[fixture] {key}: seed={seed} C绝对值范围=[{float(c_truth.min()):.2f}, {float(c_truth.max()):.2f}]")

    npz_path = out_dir / "correctness.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"[done] {npz_path}（{npz_path.stat().st_size / 1e6:.1f} MB）+ manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
