#!/usr/bin/env python3
"""第 21 期 fixtures 生成（固定 seed，逐张量 SHA256 manifest，跨服务器共用）。

口径（与 ch21 同法、本期改两点）：
  1. fixtures = **锚点子集**：decode N ∈ {65, 128, 511} × B ∈ {1, 4} ×
     bs ∈ {16, 128} × 块表 {identity, rand} + prefill S ∈ {130, 512}（bs=128,
     rand，部分块/恰整块各一）——FP32 输入 + CPU FP32 真值（+ FP64 锚点，N ≤ 128）。
  2. **大形状（N ∈ {2048, 8191}）不进 npz**（B=4 时 K/V fp32 各 67/134 MB）：
     按与 fixtures 完全相同的 seed 规则在服务器运行期生成，CPU FP32 真值在
     服务器上现算（128 核），manifest 记录 seed 规则与大形状张量的预期
     SHA256（本地预生成后仅记录哈希、不落盘）——上服务器后 `--verify-only`
     核对锚点子集，大形状哈希由 correctness 流程现场对账。

RNG：numpy PCG64，seed = 20261122 + tag 编码（gen_input 与 benchmark 运行期
生成共用同一函数——单源，避免 fixtures 与运行期数据漂移）。
块表：build_block_table（identity / rand，seed 独立编码），池由 scatter_to_pool
构造（ fixtures 存逻辑连续 K/V + 表；池构造在消费端）。
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

import paged_config as cfg  # noqa: E402
from paged_kv_reference import (  # noqa: E402
    build_block_table,
    contiguous_decode_reference,
    contiguous_decode_truth64,
    contiguous_prefill_reference,
    contiguous_prefill_truth64,
)

SEED_BASE = 20261122
TAG_MAP = {"q": 0, "k": 1, "v": 2, "pq": 3, "pk": 4, "pv": 5}

# fixtures 锚点子集
FIXTURE_DECODE_LENS = [65, 128, 511]      # 65/128: 单块~跨块边界；511: 部分块+padding
FIXTURE_DECODE_BATCHES = [1, 4]
FIXTURE_BLOCK_SIZES = [16, 128]
FIXTURE_PREFILL_LENS = [130, 512]         # 130: 部分块；512: 恰整块
# 大形状（运行期生成，不落盘 npz；manifest 记录预期 SHA256）
RUNTIME_DECODE_LENS = [2048, 8191]


def gen_input(shape: tuple[int, ...], tag: str) -> np.ndarray:
    """确定性 seed（与 benchmark 运行期生成共用——单源）。"""
    seed = SEED_BASE + TAG_MAP[tag] * 1000003 + sum(shape) * 7 + len(shape)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def table_seed(seq_lens: list[int], bs: int, mode: str) -> int:
    return SEED_BASE + 77 + sum(seq_lens) * 31 + bs * 101 + (hash(mode) % 997)


def tensor_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def decode_case(seq_lens: list[int], b: int):
    """decode 用例输入（逻辑连续 K/V + q），torch 张量。"""
    import torch

    n_max = max(seq_lens)
    q = torch.from_numpy(gen_input((b, cfg.HQ, cfg.D), "q"))
    k = torch.from_numpy(gen_input((b, cfg.HKV, n_max, cfg.D), "k"))
    v = torch.from_numpy(gen_input((b, cfg.HKV, n_max, cfg.D), "v"))
    return q, k, v


def prefill_case(s: int, b: int):
    import torch

    q = torch.from_numpy(gen_input((b, cfg.HQ, s, cfg.D), "pq"))
    k = torch.from_numpy(gen_input((b, cfg.HKV, s, cfg.D), "pk"))
    v = torch.from_numpy(gen_input((b, cfg.HKV, s, cfg.D), "pv"))
    return q, k, v


def build_case_tables(seq_lens: list[int], bs: int, num_blocks: int):
    """三个对照组的块表 + seq_lens（torch int32）。"""
    tables = {}
    for mode in ("identity", "rand"):
        tables[mode] = build_block_table(seq_lens, bs, num_blocks, mode,
                                         seed=table_seed(seq_lens, bs, mode))
    return tables


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
        mismatches = checked = 0
        for key in data.files:
            if key in manifest["tensors"]:
                checked += 1
                if manifest["tensors"][key]["sha256"] != tensor_sha256(data[key]):
                    mismatches += 1
                    print(f"[verify] {key}: SHA256 不一致")
        print(f"[verify] {checked} 张量核对完成，{mismatches} 不一致")
        return 0 if mismatches == 0 else 1

    import torch

    tensors: dict[str, np.ndarray] = {}
    manifest: dict = {
        "schema_version": "kernel-chapter22.fixtures.v1",
        "seed_base": SEED_BASE,
        "rng": "numpy default_rng(PCG64) standard_normal FP32；seed = 20261122 "
               "+ tag*1000003 + sum(shape)*7 + ndim（gen_input 单源，运行期大形状同规则）",
        "config": {"hq": cfg.HQ, "hkv": cfg.HKV, "d": cfg.D,
                   "scale": cfg.SOFTMAX_SCALE, "layers": cfg.NUM_HIDDEN_LAYERS,
                   "block_sizes": cfg.BLOCK_SIZES, "table_pad": cfg.TABLE_PAD},
        "contract_note": "块表语义合同见 paged_config.py；identity: B=1 时 p==t、"
                         "B>1 请求独占连续段；rand: 全局置换切片（跨请求不重叠）",
        "runtime_large_shapes": {
            "note": "N ∈ {2048, 8191} × B ∈ {1,4} 不入 npz；服务器运行期用同一 "
                    "gen_input 规则生成，CPU FP32 真值现场算；下表为预期 SHA256",
            "tensors": {},
        },
        "tensors": {},
    }

    def register(name: str, array: np.ndarray):
        tensors[name] = array
        manifest["tensors"][name] = {"sha256": tensor_sha256(array),
                                     "shape": list(array.shape), "dtype": str(array.dtype)}

    # ---- decode 锚点子集（逻辑 K/V 每用例注册一次；表/真值按 bs×模式）----
    for n in FIXTURE_DECODE_LENS:
        for b in FIXTURE_DECODE_BATCHES:
            q, k, v = decode_case([n] * b, b)
            seq_lens = [n] * b
            key = f"d_n{n}_b{b}"
            register(f"{key}_q", q.numpy())
            register(f"{key}_k", k.numpy())
            register(f"{key}_v", v.numpy())
            register(f"{key}_seqlens", np.array(seq_lens, dtype=np.int32))
            truth32 = contiguous_decode_reference(q, k, v, seq_lens,
                                                  cfg.SOFTMAX_SCALE).numpy()
            register(f"{key}_truth32", truth32.astype(np.float32))
            if n <= 128:
                truth64 = contiguous_decode_truth64(q, k, v, seq_lens,
                                                    cfg.SOFTMAX_SCALE).numpy()
                register(f"{key}_truth64", truth64.astype(np.float32))
            for bs in FIXTURE_BLOCK_SIZES:
                total_nb = sum(cfg.blocks_for(n_, bs) for n_ in seq_lens)
                num_blocks = total_nb + (2 if bs == 128 else 1)  # 留 padding 余量
                tables = build_case_tables(seq_lens, bs, num_blocks)
                for mode, table in tables.items():
                    register(f"{key}_bs{bs}_{mode}_table",
                             table.numpy().astype(np.int32))
            print(f"[fixtures] decode N={n} B={b} 完成（2 bs × 2 模式表 + 真值）")

    # ---- prefill 锚点 ----
    for s in FIXTURE_PREFILL_LENS:
        b = 1
        q, k, v = prefill_case(s, b)
        bs = 128
        num_blocks = cfg.blocks_for(s, bs) + 1
        tables = build_case_tables([s], bs, num_blocks)
        for mode, table in tables.items():
            key = f"p_s{s}_b{b}_bs{bs}_{mode}"
            register(f"{key}_q", q.numpy())
            register(f"{key}_k", k.numpy())
            register(f"{key}_v", v.numpy())
            register(f"{key}_table", table.numpy().astype(np.int32))
            truth32 = contiguous_prefill_reference(q, k, v, cfg.SOFTMAX_SCALE).numpy()
            register(f"{key}_truth32", truth32.astype(np.float32))
        print(f"[fixtures] prefill S={s} 完成")

    # ---- 变长 batch（引擎口径锚点：每请求独立 seq_len + padding 表项）----
    seq_lens = [1, 17, 255, 511]
    b = len(seq_lens)
    bs = 32
    n_max = max(seq_lens)
    q, k, v = decode_case(seq_lens, b)
    max_blocks = max(cfg.blocks_for(n_, bs) for n_ in seq_lens)
    num_blocks = sum(cfg.blocks_for(n_, bs) for n_ in seq_lens)
    table = build_block_table(seq_lens, bs, num_blocks, "rand", seed=table_seed(seq_lens, bs, "rand"))
    table = torch.cat([table, torch.full((b, max_blocks - table.shape[1]),
                                         cfg.TABLE_PAD, dtype=torch.int32)], dim=1)
    key = "d_varlen_b4_bs32_rand"
    register(f"{key}_q", q.numpy())
    register(f"{key}_k", k.numpy())
    register(f"{key}_v", v.numpy())
    register(f"{key}_table", table.numpy().astype(np.int32))
    register(f"{key}_seqlens", np.array(seq_lens, dtype=np.int32))
    register(f"{key}_truth32", contiguous_decode_reference(
        q, k, v, seq_lens, cfg.SOFTMAX_SCALE).numpy().astype(np.float32))
    print(f"[fixtures] decode varlen {seq_lens} 完成")

    # ---- 大形状预期 SHA256（本地生成仅记哈希，不落盘）----
    for n in RUNTIME_DECODE_LENS:
        for b in FIXTURE_DECODE_BATCHES:
            _, k, v = decode_case([n] * b, b)
            q_big, _, _ = decode_case([n] * b, b)
            for name, arr in ((f"rt_n{n}_b{b}_q", q_big.numpy()),
                              (f"rt_n{n}_b{b}_k", k.numpy()),
                              (f"rt_n{n}_b{b}_v", v.numpy())):
                manifest["runtime_large_shapes"]["tensors"][name] = {
                    "sha256": tensor_sha256(arr), "shape": list(arr.shape)}

    np.savez_compressed(npz_path, **tensors)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    size_mb = npz_path.stat().st_size / 1e6
    print(f"[fixtures] 写入 {npz_path}（{size_mb:.1f} MB，{len(tensors)} 张量）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
