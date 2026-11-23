#!/usr/bin/env python3
"""第 08 期正确性比较（复用第 01 期比较口径）。

两条正确性链：
  A. RMSNorm kernel vs CPU 参考真值（fixtures/rmsnorm-fixtures.npz，FP32 朴素
     参考语义与第 02 期一致）：max/mean 绝对误差、相对误差、余弦相似度、
     NaN/Inf 计数 —— kernel 本体在第 02 期已全量验证，此处为管线级的回归锚；
  B. 重叠流水线 vs 单流串行：payload 32MB / chunk 4MB / K=6，逐 chunk 位级
     比较（memcmp）—— 本期核心正确性结论（实验 4）。

输出：results/correctness-{backend}.json
"""

from __future__ import annotations

import argparse
import ctypes
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_overlap import Pipeline, SEED  # 复用流水线（K/事件/深缓冲逻辑单源）
from experiment_utils import SCHEMA_VERSION, utc_now_iso, write_results_json
from transfer_bench_lib import KIND_D2H, KIND_H2D, memcmp_equal, TransferOps

ROOT = Path(__file__).resolve().parent
MIB = 1024 * 1024


def compare_block(name: str, got: np.ndarray, ref: np.ndarray) -> dict:
    got64 = got.astype(np.float64)
    ref64 = ref.astype(np.float64)
    diff = np.abs(got64 - ref64)
    dot = float(np.sum(got64 * ref64))
    denom = math.sqrt(float(np.sum(got64 ** 2)) * float(np.sum(ref64 ** 2))) or 1.0
    return {
        "name": name,
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "max_rel_err": float((diff / (np.abs(ref64) + 1e-30)).max()),
        "cosine_similarity": dot / denom,
        "nan_count": int(np.isnan(got).sum()),
        "inf_count": int(np.isinf(got).sum()),
        "shape": list(got.shape),
        "dtype": str(got.dtype),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["cuda", "musa"])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--fixtures", default=str(ROOT / "fixtures" / "rmsnorm-fixtures.npz"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ops = TransferOps(args.lib)
    ops.init()
    h = 1024

    blocks: list[dict] = []

    # ---- A. kernel vs fixtures 真值 ------------------------------------------
    fx_path = Path(args.fixtures)
    if fx_path.exists():
        fx = np.load(fx_path)
        x, w, y_ref = fx["x"], fx["w"], fx["y_ref"]
        rows = x.shape[0]
        nbytes = rows * h * 4
        x_host = ops.malloc_pinned_default(nbytes)
        y_host = ops.malloc_pinned_default(nbytes)
        w_host = ops.malloc_pinned_default(h * 4)
        x_dev, y_dev, w_dev = (ops.malloc_device(nbytes), ops.malloc_device(nbytes),
                               ops.malloc_device(h * 4))
        ctypes.memmove(ctypes.c_void_p(x_host), np.ascontiguousarray(x, dtype=np.float32).tobytes(), nbytes)
        ctypes.memmove(ctypes.c_void_p(w_host), np.ascontiguousarray(w, dtype=np.float32).tobytes(), h * 4)
        ops.memcpy_sync(x_dev, x_host, nbytes, KIND_H2D)
        ops.memcpy_sync(w_dev, w_host, h * 4, KIND_H2D)
        ops.rmsnorm(x_dev, w_dev, y_dev, rows, h, 1e-6, 256, 0)
        ops.device_sync()
        ops.memcpy_sync(y_host, y_dev, nbytes, KIND_D2H)
        y = np.frombuffer((ctypes.c_char * nbytes).from_address(y_host), dtype=np.float32).reshape(rows, h)
        blocks.append(compare_block("kernel_vs_cpu_ref", y, y_ref))
        for p in (x_host, y_host, w_host):
            ops.free_pinned(p)
        for p in (x_dev, y_dev, w_dev):
            ops.free_device(p)
    else:
        blocks.append({"name": "kernel_vs_cpu_ref", "status": "SKIP",
                       "detail": f"fixtures 不存在: {fx_path}"})

    # ---- B. 重叠流水线位级（实验 4 主正确性） ---------------------------------
    payload = 32 * MIB
    chunk = 4 * MIB
    w_host = ops.malloc_pinned_default(h * 4)
    w_np = np.random.default_rng(SEED + 1).uniform(0.5, 2.0, h).astype(np.float32)
    ctypes.memmove(ctypes.c_void_p(w_host), w_np.tobytes(), h * 4)
    w_dev = ops.malloc_device(h * 4)
    ops.memcpy_sync(w_dev, w_host, h * 4, KIND_H2D)
    ops.free_pinned(w_host)

    sa, sb = ops.stream_create(), ops.stream_create()
    pipe = Pipeline(ops, payload, chunk, h, w_dev)
    rows = chunk // (h * 4)
    k = 6
    ops.device_sync()
    pipe.run_serial(sa, k, rows)
    ops.device_sync()
    pipe.run_overlap(sa, sb, k, rows)
    ops.device_sync()
    mismatches = []
    for i in range(pipe.n):
        if not memcmp_equal(pipe.host_dst_serial + i * pipe.chunk,
                            pipe.host_dst_overlap + i * pipe.chunk, pipe.chunk):
            mismatches.append(i)
    pipe.destroy()
    ops.stream_destroy(sa)
    ops.stream_destroy(sb)
    ops.free_device(w_dev)

    blocks.append({
        "name": "overlap_vs_serial_bitwise",
        "bitwise_equal_all_chunks": not mismatches,
        "mismatch_chunks": mismatches[:10],
        "payload_mb": payload / MIB, "chunk_mb": chunk / MIB, "n_chunks": payload // chunk,
        "k_rounds": k,
        "method": "libc memcmp（零拷贝位级比较）",
    })

    result = {
        "schema_version": SCHEMA_VERSION + ".correctness",
        "created_at_utc": utc_now_iso(),
        "backend": args.backend,
        "lib": str(Path(args.lib).resolve()),
        "comparator": "ch01 口径（max/mean 绝对、相对、余弦、NaN/Inf）+ 位级 memcmp",
        "blocks": blocks,
    }
    write_results_json(args.out, result)
    for b in blocks:
        print(f"[correctness] {b['name']}: {b if 'status' in b else {k: v for k, v in b.items() if k != 'name'}}")
    has_failure = any((b.get("status") == "SKIP") or (b.get("bitwise_equal_all_chunks") is False)
                      or (b.get("max_rel_err", 0) > 1e-4 and b["name"] == "kernel_vs_cpu_ref")
                      for b in blocks)
    if has_failure:
        raise SystemExit(1)
    print("[correctness] 全部通过")


if __name__ == "__main__":
    main()
