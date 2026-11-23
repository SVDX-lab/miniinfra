#!/usr/bin/env python3
"""生成第 22 期 fixtures（固定 seed，逐张量 SHA256 manifest）。

内容：
- correctness.npz：正确性矩阵输入（seq ∈ {1, 64, 512, 2048} 的 q/k fp32，
  seed 20261123 + seq 编码；seq=32768 体积过大（402MB/张）不入 npz，由
  compare_correctness.py/benchmark.py 以同一 seed 现场再生，值经 checksum
  核对——manifest 记录两种来源的校验和口径）；
- cache 表 fp64 锚点（max_seq=40960，Qwen3 max_position_embeddings 上限）；
- mu_hash 校验和 demo（与 MUSA/CUDA standalone 二进制的 hash-RNG 同算法
  同 seed，float64 顺序和逐位可复算——跨平台数据一致性抽查锚点）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import experiment_utils as eu
import rope_config as cfg
from implementations.cpu import rope_reference as ref



def sha256_of_array(arr: np.ndarray) -> str:
    """numpy 版 SHA256（与 experiment_utils.sha256_of_tensor 同口径：shape+dtype+bytes）。"""
    contiguous = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()

def gen_qk(seq: int, seed_base: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed_base + seq)
    q = rng.standard_normal((seq, cfg.HQ, cfg.HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((seq, cfg.HKV, cfg.HEAD_DIM)).astype(np.float32)
    return q, k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fixtures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    manifest: dict = {
        "schema_version": "kernel-chapter23.fixtures.v1",
        "seed_base": cfg.FIXTURES_SEED,
        "rng": "numpy PCG64 standard_normal（q/k 输入）；cache 表 = FP64 公式直算",
        "rope_params": {
            "theta": cfg.ROPE_THETA,
            "head_dim": cfg.HEAD_DIM,
            "hq": cfg.HQ,
            "hkv": cfg.HKV,
            "layout": "compact cos/sin 各 [max_seq, 64]；HF duplicated 可 expand_dup 无损展开",
        },
        "tensors": {},
    }

    # 1) q/k 输入（seq ≤ 2048；32768 现场再生口径在 manifest 单独声明）
    for seq in [1, 64, 512, 2048]:
        q, k = gen_qk(seq, cfg.FIXTURES_SEED)
        arrays[f"q_seq{seq}"] = q
        arrays[f"k_seq{seq}"] = k
        manifest["tensors"][f"q_seq{seq}"] = sha256_of_array(q)
        manifest["tensors"][f"k_seq{seq}"] = sha256_of_array(k)

    # seq=32768 再生口径校验和（现场再生时核对）
    q_big, k_big = gen_qk(32768, cfg.FIXTURES_SEED)
    manifest["regen_32768"] = {
        "method": "np.random.default_rng(20261123+32768).standard_normal, fp32",
        "checksum_q_f64": cfg.checksum_f64(q_big.reshape(-1)[::4096]),
        "checksum_k_f64": cfg.checksum_f64(k_big.reshape(-1)[::4096]),
        "note": "抽样步长 4096 的 float64 顺序和（现场再生后应逐位一致）",
    }

    # 2) FP64 cache 锚点（max_seq = Qwen3 max_position_embeddings）
    cos64, sin64 = ref.build_cache(40960, cfg.ROPE_THETA, cfg.HEAD_DIM, "float64")
    arrays["cache_cos_fp64"] = cos64
    arrays["cache_sin_fp64"] = sin64
    manifest["tensors"]["cache_cos_fp64"] = sha256_of_array(cos64)
    manifest["tensors"]["cache_sin_fp64"] = sha256_of_array(sin64)

    np.savez_compressed(out_dir / "correctness.npz", **arrays)

    # 3) mu_hash 校验和 demo（MUSA/CUDA standalone 对拍锚点）
    demo_len = 1000
    demo_q = [cfg.mu_value(i, cfg.SEED_Q) for i in range(demo_len)]
    demo_k = [cfg.mu_value(i, cfg.SEED_K) for i in range(demo_len)]
    manifest["mu_hash_demo"] = {
        "algorithm": "x = idx*0x9E3779B9 ^ seed; x ^= x>>16; x *= 0x85EBCA6B; "
                     "x ^= x>>13; x *= 0xC2B2AE35; x ^= x>>16; val = (x%20001)/10000-1",
        "seeds": {"q": hex(cfg.SEED_Q), "k": hex(cfg.SEED_K)},
        "length": demo_len,
        "checksum_q_f64": cfg.checksum_f64(demo_q),
        "checksum_k_f64": cfg.checksum_f64(demo_k),
        "note": "二进制侧 double 顺序和对拍容差 1e-9",
    }

    manifest["environment"] = eu.device_environment_record(torch.device("cpu"))
    eu.write_results_json(out_dir / "manifest.json", manifest)
    print(f"[fixtures] {len(arrays)} tensors → {out_dir/'correctness.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
