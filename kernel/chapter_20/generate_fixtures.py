#!/usr/bin/env python3
"""第 20 期 fixtures 生成（固定 seed，逐张量 SHA256 manifest，跨服务器共用）。

内容：
  - 正确性矩阵输入：prefill seq {64,512,2048} × batch {1,4}、decode 同矩阵
    的 Q/K/V（Qwen3-0.6B 维度 HQ=16/HKV=8/D=128），FP32 存储（两服务器共用）；
  - 真值：CPU FP32 参考输出（全矩阵）+ FP64 锚点（S ∈ {64,512}，控制体积）；
  - GQA 映射抽查数据（S=64 B=1）。

RNG：numpy PCG64，seed = 20261121 + shape 编码；输入 N(0,1)——softmax scale
=1/sqrt(128) 已把分数压回 O(1)（ch25"N(0,1) 退化 one-hot"教训的 Attention 本体
版：scale 自带归一，不需要 He 缩放，manifest 显式记录该判断）。

MUSA standalone 二进制数据一致性：manifest 附 mu_generator 段（hash-RNG 算法
说明 + 本地复算的逐张量加法校验和），用于核对 S4000 侧 .mu 自生成数据与
fixtures 的分布一致性（不是同一份数据，是同分布同算法——correctness 的
truth 在 .mu 内部用同算法重生成，见 EXPERIMENT_LOG 口径说明）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import attention_config as cfg  # noqa: E402
from attention_reference import (  # noqa: E402
    gqa_decode_reference,
    gqa_decode_truth64,
    gqa_mapping_spotcheck,
    gqa_prefill_reference,
    gqa_prefill_truth64,
)

SEED_BASE = 20261121
SEQ_LENS = cfg.CORRECTNESS_SEQ_LENS       # [64, 512, 2048]
BATCHES = cfg.CORRECTNESS_BATCHES         # [1, 4]


def tensor_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def gen_input(shape: tuple[int, ...], tag: str) -> np.ndarray:
    """确定性 seed（不用内置 hash——PYTHONHASHSEED 跨进程不稳定）。"""
    tag_map = {"q": 0, "k": 1, "v": 2, "dq": 3, "dk": 4, "dv": 5, "gq": 6, "gk": 7, "gv": 8}
    seed = SEED_BASE + tag_map[tag.split("_")[0]] * 1000003 + sum(shape) * 7 + len(shape)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="只校验现有 fixtures（manifest SHA256），不重新生成")
    arguments = parser.parse_args()
    fixtures_dir = HERE / "fixtures"
    npz_path = fixtures_dir / "correctness.npz"
    manifest_path = fixtures_dir / "manifest.json"

    if arguments.verify_only:
        data = np.load(npz_path)
        manifest = json.loads(manifest_path.read_text())
        mismatches = 0
        checked = 0
        for key in data.files:
            if key in manifest["tensors"]:
                want = manifest["tensors"][key]["sha256"]
                got = tensor_sha256(data[key])
                checked += 1
                if want != got:
                    mismatches += 1
                    print(f"[verify] {key}: SHA256 不一致（portable 差异，见 manifest 注记）")
        print(f"[verify] {checked} 张量核对完成，{mismatches} 不一致")
        return 0 if mismatches == 0 else 1

    tensors: dict[str, np.ndarray] = {}
    manifest: dict = {
        "schema_version": "kernel-chapter21.fixtures.v1",
        "seed_base": SEED_BASE,
        "rng": "numpy default_rng(PCG64)，standard_normal，FP32 存储；"
               "输入 N(0,1)：softmax scale=1/sqrt(128) 已把分数压回 O(1)，"
               "softmax 非退化（ch25 教训的 Attention 本体版，无需 He 缩放）",
        "config": {
            "hq": cfg.HQ, "hkv": cfg.HKV, "d": cfg.D,
            "scale": cfg.SOFTMAX_SCALE, "layers": cfg.NUM_HIDDEN_LAYERS,
        },
        "tensors": {},
    }

    def register(name: str, array: np.ndarray):
        tensors[name] = array
        manifest["tensors"][name] = {
            "sha256": tensor_sha256(array), "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    # ---- prefill 输入（全矩阵复用同一组 per-shape 输入；decode 共用 k/v，
    #      单独的 q 为每头单 query）----
    for s in SEQ_LENS:
        for b in BATCHES:
            q = gen_input((b, cfg.HQ, s, cfg.D), f"q_{b}_{s}")
            k = gen_input((b, cfg.HKV, s, cfg.D), f"k_{b}_{s}")
            v = gen_input((b, cfg.HKV, s, cfg.D), f"v_{b}_{s}")
            register(f"p_s{s}_b{b}_q", q)
            register(f"p_s{s}_b{b}_k", k)
            register(f"p_s{s}_b{b}_v", v)
            # CPU FP32 真值
            out32 = gqa_prefill_reference(
                np2torch(q), np2torch(k), np2torch(v), cfg.SOFTMAX_SCALE).numpy()
            register(f"p_s{s}_b{b}_truth32", out32.astype(np.float32))
            # FP64 锚点（体积控制：S ≤ 512）
            if s <= 512:
                out64 = gqa_prefill_truth64(
                    np2torch(q), np2torch(k), np2torch(v), cfg.SOFTMAX_SCALE).numpy()
                register(f"p_s{s}_b{b}_truth64", out64.astype(np.float32))
            print(f"[fixtures] prefill S={s} B={b} 完成（truth32{' + truth64' if s <= 512 else ''}）")

    for n in SEQ_LENS:
        for b in BATCHES:
            dq = gen_input((b, cfg.HQ, cfg.D), f"dq_{b}_{n}")
            dk = gen_input((b, cfg.HKV, n, cfg.D), f"dk_{b}_{n}")
            dv = gen_input((b, cfg.HKV, n, cfg.D), f"dv_{b}_{n}")
            register(f"d_n{n}_b{b}_q", dq)
            register(f"d_n{n}_b{b}_k", dk)
            register(f"d_n{n}_b{b}_v", dv)
            out32 = gqa_decode_reference(
                np2torch(dq), np2torch(dk), np2torch(dv), cfg.SOFTMAX_SCALE).numpy()
            register(f"d_n{n}_b{b}_truth32", out32.astype(np.float32))
            if n <= 512:
                out64 = gqa_decode_truth64(
                    np2torch(dq), np2torch(dk), np2torch(dv), cfg.SOFTMAX_SCALE).numpy()
                register(f"d_n{n}_b{b}_truth64", out64.astype(np.float32))
            print(f"[fixtures] decode N={n} B={b} 完成")

    # ---- GQA 映射抽查数据 ----
    gq = gen_input((1, cfg.HQ, 64, cfg.D), "gq")
    gk = gen_input((1, cfg.HKV, 64, cfg.D), "gk")
    gv = gen_input((1, cfg.HKV, 64, cfg.D), "gv")
    register("gqa_q", gq)
    register("gqa_k", gk)
    register("gqa_v", gv)
    spot = gqa_mapping_spotcheck(np2torch(gq), np2torch(gk), np2torch(gv), cfg.SOFTMAX_SCALE)
    manifest["gqa_spotcheck_local"] = spot
    print(f"[fixtures] GQA 抽查（本地预跑）: {spot}")

    # ---- MUSA 二进制生成器口径说明 + 本地复算校验和（小形状演示可复算）----
    mu_note = {
        "algorithm": "x = idx*0x9E3779B9 ^ seed; x ^= x>>16; x *= 0x85EBCA6B; "
                     "x ^= x>>13; x *= 0xC2B2AE35; x ^= x>>16; val = (x%20001-10000)/10000",
        "seeds": {"q": hex(cfg.MU_HASH_SEED_Q), "k": hex(cfg.MU_HASH_SEED_K), "v": hex(cfg.MU_HASH_SEED_V)},
        "note": "S4000 侧 correctness truth 由 .mu 内部用本算法重生成（同分布同算法）；"
                "与 fixtures 的 SHA256 清单是两套独立数据，不互相校验 SHA256",
    }
    # 本地复算一组校验和（S=64 B=1 的 q）验证 Python/C++ hash 实现一致性
    # 复刻 C++ mu_value 的 fp32 算术（(h%20001)/10000.0f - 1.0f）——两语言对拍必须
    # 连舍入路径一起复刻（fp64 复刻首跑失配 1e-3 级，见 EXPERIMENT_LOG）
    def mu_value_f32(idx, seed):
        h = cfg.mu_hash(idx, seed) % 20001
        return float(np.float32(np.float32(h) / np.float32(10000.0)) - np.float32(1.0))

    idx_sum = 0.0
    for i in range(1 * cfg.HQ * 64 * cfg.D):
        idx_sum += np.float64(np.float32(mu_value_f32(i, cfg.MU_HASH_SEED_Q)))
    mu_note["checksum_demo_q_s64_b1_float64_sequential"] = idx_sum
    manifest["mu_generator"] = mu_note
    print(f"[fixtures] mu hash 复算校验和（q, S=64 B=1）= {idx_sum:.6f}")

    np.savez_compressed(npz_path, **tensors)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    size_mb = npz_path.stat().st_size / 1e6
    print(f"[fixtures] 写入 {npz_path}（{size_mb:.1f} MB，{len(tensors)} 张量）")
    return 0


def np2torch(array: np.ndarray):
    import torch

    return torch.from_numpy(array.astype(np.float64 if array.dtype == np.float64 else np.float32))


if __name__ == "__main__":
    raise SystemExit(main())
