#!/usr/bin/env python3
"""第 04 期 fixtures：复用第 03 期 30 组算子级 fixtures（不重新发明）。

复用规则（协议 §2"复用第 01 期口径"）：
- 算子级 fixtures = 第 03 期同一份（H ∈ {127,128,129,1023,1024,1025} ×
  rows ∈ {1,4,128,512,2048}，r/h/w/t/y，float32）；本期注册层不改 kernel 数学，
  fixtures 逐张量 SHA256 必须与第 03 期 manifest 完全一致（一致性即"注册层
  未改变被注册算子语义"的证据）；
- 模型级真值 = 第 01 期 truth/（probes.npz + truth-manifest.json，服务器
  chapter_01/ 下已就位），本期不生成、只消费。

本脚本行为：
1. 若 ../chapter_03/fixtures/correctness.npz 存在 → 原样复制到本期 fixtures/
   （npz 文件 SHA256 连同逐张量 SHA256 一起核对）；
2. 否则按第 03 期同一 seed 规则重新生成（串行真值实现逐字复制自
   chapter_03/implementations/cpu/fused_add_rmsnorm.py，保证逐张量 SHA256
   与 manifest 一致；npz 容器字节因 zip 时间戳不同而不同，属正常，
   逐张量核对以 manifest 为准）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EP04_FIXTURES = HERE.parent / "chapter_03" / "fixtures"

RMS_NORM_EPS = 1e-6
SEED_BASE = 20260401
HIDDEN_SIZES = [127, 128, 129, 1023, 1024, 1025]
ROW_COUNTS = [1, 4, 128, 512, 2048]
H_STREAM_OFFSET = 555555


# ---------------------------------------------------------------------------
# 串行真值实现（逐字复制自 chapter_03/implementations/cpu/fused_add_rmsnorm.py，
# 协议：不跨期 import；复制以保证与第 03 期 manifest 逐张量 SHA256 一致）。
# ---------------------------------------------------------------------------
def _validate_pair(r, h, weight):
    r = np.asarray(r)
    h = np.asarray(h)
    weight = np.asarray(weight)
    if r.ndim != 2 or h.ndim != 2:
        raise ValueError("r/h 必须是 [rows, H] 二维数组")
    if r.shape != h.shape:
        raise ValueError("r 与 h 形状必须一致")
    if weight.ndim != 1 or weight.shape[0] != r.shape[1]:
        raise ValueError("weight 必须是 [H] 一维数组且与 r/h 的 H 一致")
    if r.dtype != np.float32 or h.dtype != np.float32 or weight.dtype != np.float32:
        raise ValueError("真值路径只接受 float32 输入")
    if r.shape[1] == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错")
    return r, h, weight


def naive_rmsnorm_serial(x, weight, eps=RMS_NORM_EPS):
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x 必须是 [rows, H] 二维数组")
    if x.dtype != np.float32:
        raise ValueError("真值路径只接受 float32 输入")
    rows, hidden = x.shape
    output = np.empty((rows, hidden), dtype=np.float32)
    f32 = np.float32
    for i in range(rows):
        row = x[i]
        acc = f32(0.0)
        for j in range(hidden):
            v = row[j]
            acc = f32(acc + f32(v * v))
        scale = f32(1.0) / f32(np.sqrt(f32(acc / f32(hidden)) + f32(eps)))
        w_row = weight
        out_row = output[i]
        for j in range(hidden):
            out_row[j] = f32(f32(row[j] * scale) * w_row[j])
    return output


def separated_add_rmsnorm_serial(r, h, weight, eps=RMS_NORM_EPS):
    r, h, weight = _validate_pair(r, h, weight)
    t = (r + h).astype(np.float32)
    y = naive_rmsnorm_serial(t, weight, eps)
    return t, y


def sha256_of_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def verify_against_manifest(arrays: dict, manifest: dict) -> int:
    """逐张量 SHA256 核对（150 = 30 组 × 5 张量）。"""
    checked = 0
    for combo in manifest["combos"]:
        rows, hidden = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden}"
        for name in ("r", "h", "t", "y"):
            array = arrays[f"{name}_{key}"]
            digest = sha256_of_array(array)
            if digest != combo["sha256"][name]:
                raise RuntimeError(f"SHA256 不一致：{name}_{key}（注册层/生成器漂移？）")
            checked += 1
        weight = arrays[f"w_{hidden}"]
        if sha256_of_array(weight) != combo["sha256"]["w"]:
            raise RuntimeError(f"SHA256 不一致：w_{hidden}")
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fixtures")
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_manifest = json.loads(
        (HERE / "fixtures" / "manifest.json").read_text(encoding="utf-8"))

    source_npz = EP04_FIXTURES / "correctness.npz"
    if source_npz.is_file():
        shutil.copyfile(source_npz, out_dir / "correctness.npz")
        with np.load(out_dir / "correctness.npz") as data:
            arrays = {key: data[key] for key in data.files}
        checked = verify_against_manifest(arrays, reference_manifest)
        source_sha = hashlib.sha256((out_dir / "correctness.npz").read_bytes()).hexdigest()
        same_file = source_sha == reference_manifest.get("npz_sha256")
        print(f"[reuse] 复制第 03 期 fixtures → {out_dir / 'correctness.npz'}"
              f"（逐张量 SHA256 {checked}/150 一致；npz 文件字节与 ep04 manifest"
              f" 一致={same_file}）")
        return 0

    # 重新生成路径（第 03 期同一 seed 规则 + 同一串行真值实现）。
    print("[regen] ../chapter_03/fixtures 不存在，按第 03 期 seed 规则重新生成")
    arrays: dict[str, np.ndarray] = {}
    start_all = time.perf_counter()
    for hidden_size in HIDDEN_SIZES:
        weight_rng = np.random.default_rng(SEED_BASE + hidden_size)
        weight = weight_rng.standard_normal(hidden_size, dtype=np.float32)
        for rows in ROW_COUNTS:
            seed = SEED_BASE + rows * 100000 + hidden_size
            r = np.random.default_rng(seed).standard_normal(
                (rows, hidden_size), dtype=np.float32)
            h = np.random.default_rng(seed + H_STREAM_OFFSET).standard_normal(
                (rows, hidden_size), dtype=np.float32)
            t, y = separated_add_rmsnorm_serial(r, h, weight, RMS_NORM_EPS)
            key = f"{rows}x{hidden_size}"
            arrays[f"r_{key}"] = r
            arrays[f"h_{key}"] = h
            arrays[f"w_{hidden_size}"] = weight
            arrays[f"t_{key}"] = t
            arrays[f"y_{key}"] = y
            print(f"[fixture] {key} 生成完成")
    checked = verify_against_manifest(arrays, reference_manifest)
    np.savez_compressed(out_dir / "correctness.npz", **arrays)
    print(f"[done] {out_dir / 'correctness.npz'}"
          f"（{ (out_dir / 'correctness.npz').stat().st_size / 2**20:.1f} MiB），"
          f"逐张量 SHA256 {checked}/150 一致，总耗时 {time.perf_counter()-start_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
