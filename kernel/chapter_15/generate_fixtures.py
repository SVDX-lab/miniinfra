#!/usr/bin/env python3
"""生成第 15 期量化 GEMM 正确性 fixtures（固定 seed + CPU FP32 真值 + 误差分层各层输出）。

产物：
- fixtures/correctness.npz：各组合的 A16、W16、q(int8)、scale(fp32) 与误差分层
  各层真值输出（y_truth / y_a1 / y_a2 / y_a3，FP32）；
- fixtures/manifest.json：seed 规则、逐张量 SHA256、量化方案、分层定义。

真值与分层定义（大纲 §9 行 476：量化误差与算子实现误差分开归因）：
  原始权重 = FP16 checkpoint 值 W16（服务化流程：checkpoint 为 FP16），
  W32 = float32(W16)，A32 = float32(A16)：
    y_truth = A32 @ W32                       （L0 锚点）
    y_a1    = A32 @ dequant(q)                （La1 仅量化，FP32 算术）
    y_a2    = engine_fp16(A16, fp16(dequant(q)))（La2 量化+FP16 权重存储，引擎参考）
    y_a3    = engine_fp16(A16, W16)           （La3 FP16 无量化对照，引擎参考）
  GPU 侧 Lb（int8_fused kernel）与 Lc（fp16 kernel）在服务器上对 y_truth 测
  误差后，与 La1/La3 的差 = kernel 实现误差叠加量。
  GEMM 真值用 torch FP32（BLAS，一次生成随代码上传共用，协议 §2）；tiny 组
  另做字面三重循环对拍。

fp16 GEMM（y_a2/y_a3）引擎口径说明：torch CPU half matmul（fp16 输入、FP32
累加语义由引擎实现决定）——这正是"用引擎参考实现"的含义（大纲 §9）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "implementations" / "cpu"))

from quant_reference import (  # noqa: E402
    QMAX,
    dequantize_weight,
    error_layer_report,
    gemm_engine_fp16,
    gemm_fp32_truth,
    gemm_triple_loop,
    quantize_weight_perchannel,
)

SEED_BASE = 20261121

# (M, K, N)：decode 档（权重主导）+ prefill 档 + 边界组
COMBOS = [
    ("tiny_hand", 4, 8, 4),        # 手算可核对，字面三重循环真值
    ("ragged", 13, 17, 11),        # 全维非整除（tile guard 全触发）
    ("decode_m1_1k", 1, 1024, 1024),
    ("decode_m1_4k", 1, 4096, 4096),   # 权重 16.8MB(int8)/33.6MB(fp16) 锚点
    ("decode_m4_4k1k", 4, 4096, 1024),
    ("prefill_256", 256, 1024, 1024),
    ("prefill_1024", 1024, 1024, 1024),
]


def combo_seed(m: int, k: int, n: int) -> int:
    return SEED_BASE + m * 100000 + k * 100 + n


def sha256_of_array(array: np.ndarray) -> str:
    import hashlib

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
        "schema_version": "kernel-chapter16.fixtures.v1",
        "seed_base": SEED_BASE,
        "seed_rule": "seed = SEED_BASE + M*100000 + K*100 + N；rng = numpy default_rng(seed)",
        "weight_convention": "原始权重 = FP16 checkpoint 值 W16；W32 = float32(W16)；"
        "量化 scale 自 FP16 权重计算（服务化标准流程）",
        "quant_scheme": f"per-channel symmetric INT8, scale=amax/127, q in [-{QMAX},{QMAX}], "
        "按输出通道（W[K,N] 的 N 列）",
        "truth_implementation": "y_truth = torch FP32 matmul(A32, W32)（本地一次生成共用）；"
        "tiny_hand 另做字面三重循环对拍",
        "layer_definition": {
            "L0_y_truth": "A32 @ W32",
            "La1_y_a1": "A32 @ dequant(q)（仅量化，FP32 算术）",
            "La2_y_a2": "engine_fp16(A16, fp16(dequant(q)))（量化+FP16 权重存储，引擎参考）",
            "La3_y_a3": "engine_fp16(A16, W16)（FP16 无量化对照）",
            "Lb": "int8_fused kernel 输出 vs L0（服务器实测）",
            "Lc": "fp16 kernel 输出 vs L0（服务器实测）",
        },
        "combos": [],
    }

    for name, m, k, n in COMBOS:
        rng = np.random.default_rng(combo_seed(m, k, n))
        # 权重：FP16 checkpoint 值（正态 ×0.05 量级——贴近真实权重分布；
        # randn float32 → cast fp16 → 这就是"原始权重"）
        w32 = (rng.standard_normal((k, n)) * 0.05).astype(np.float32)
        w16 = w32.astype(np.float16)
        a16 = (rng.standard_normal((m, k)) * 0.5).astype(np.float16)

        w32_from_fp16 = w16.astype(np.float32)
        a32 = a16.astype(np.float32)

        q, scale = quantize_weight_perchannel(w32_from_fp16)
        w_hat32 = dequantize_weight(q, scale)
        w_hat16 = w_hat32.astype(np.float16)

        y_truth = gemm_fp32_truth(a32, w32_from_fp16)
        y_a1 = gemm_fp32_truth(a32, w_hat32)
        y_a2 = gemm_engine_fp16(a16, w_hat16)
        y_a3 = gemm_engine_fp16(a16, w16)

        key = f"{name}_{m}x{k}x{n}"
        arrays[f"{key}__A16"] = a16
        arrays[f"{key}__W16"] = w16
        arrays[f"{key}__q"] = q
        arrays[f"{key}__scale"] = scale
        arrays[f"{key}__y_truth"] = y_truth
        arrays[f"{key}__y_a1"] = y_a1
        arrays[f"{key}__y_a2"] = y_a2
        arrays[f"{key}__y_a3"] = y_a3

        entry = {
            "combo": name,
            "m": m,
            "k": k,
            "n": n,
            "seed": combo_seed(m, k, n),
            "sha256": {
                base: sha256_of_array(arrays[f"{key}__{base}"])
                for base in ["A16", "W16", "q", "scale", "y_truth", "y_a1", "y_a2", "y_a3"]
            },
            "scale_stats": {
                "min": float(scale.min()),
                "max": float(scale.max()),
                "mean": float(scale.mean()),
            },
        }

        # tiny_hand：字面三重循环对拍（整数级锚点）
        if name == "tiny_hand":
            y_loop = gemm_triple_loop(a32, w32_from_fp16)
            if not np.array_equal(y_loop, y_truth):
                # BLAS 累加序差异允许微小 FP32 差；tiny 尺寸下应位级一致
                diff = float(np.abs(y_loop - y_truth).max())
                entry["tiny_triple_loop_max_abs_diff"] = diff
                if diff > 1e-5:
                    raise SystemExit(f"tiny_hand 三重循环与 BLAS 真值差异过大: {diff}")
            else:
                entry["tiny_triple_loop_max_abs_diff"] = 0.0
        manifest["combos"].append(entry)
        print(
            f"[fixtures] {key}: A{a16.shape} W{w16.shape} "
            f"scale[{scale.min():.2e},{scale.max():.2e}] 完成"
        )

    np.savez_compressed(out_dir / "correctness.npz", **arrays)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    size_mb = (out_dir / "correctness.npz").stat().st_size / 1e6
    print(f"[fixtures] 共 {len(arrays)} 张量，npz {size_mb:.1f} MB → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
