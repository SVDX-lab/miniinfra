#!/usr/bin/env python3
"""第 22 期 CPU 基准（统一计时口径；本地 WSL2 12 核）。

口径：
- CPU 无异步执行：perf_counter 直接包围 numpy 调用（ch01 OperatorTimer CPU 分支）；
- 分母 = 本地大数组 copy 实测带宽（现场测量，不假设）；
- 报告 P50/P95/min + 理想字节（unique/issued 双口径）+ 有效带宽 + 达成率；
- 形态切换开销：prefill+decode 组合 vs 单独之和；
- cache 生成吞吐（一次性成本）与 fp16 存储档对比；
- decode cache-free 变体（无表现算）对照。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import rope_config as cfg  # noqa: E402
import rope_reference as ref  # noqa: E402

WARMUP = 3


def measure(fn, repeats: int) -> dict:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    ordered = sorted(samples)
    return {
        "repeats": repeats, "warmup": WARMUP,
        "p50_ms": round(ordered[len(ordered) // 2], 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(round(len(ordered) * 0.95)) - 1)], 4),
        "min_ms": round(ordered[0], 4),
        "mean_ms": round(float(np.mean(ordered)), 4),
        "samples_ms": [round(v, 4) for v in ordered],
    }


def io_bytes(tokens: int, item: int = 4, citem: int = 4) -> tuple[int, int]:
    qkv = 2 * (tokens * cfg.HQ * cfg.HEAD_DIM + tokens * cfg.HKV * cfg.HEAD_DIM) * item
    cache = 2 * tokens * cfg.HALF_DIM * citem
    cache_issued = cache * (cfg.HQ + cfg.HKV)
    return qkv + cache, qkv + cache_issued


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/benchmark-results-cpu.json")
    args = parser.parse_args()

    # 分母：本地拷贝带宽（512MB fp32 copy）
    big = np.empty(128 * 1024 * 1024, dtype=np.float32)
    dst = np.empty_like(big)
    t0 = time.perf_counter()
    for _ in range(3):
        np.copyto(dst, big)
    copy_bw = 2 * big.nbytes / (time.perf_counter() - t0) / 3 / 1e9

    report = {
        "schema_version": "kernel-chapter23.benchmark.cpu.v1",
        "platform": "cpu",
        "cpu_copy_bandwidth_gb_s": round(copy_bw, 2),
        "timing": "perf_counter 直包围 numpy 调用（CPU 天然同步）",
        "theta": cfg.ROPE_THETA,
        "entries": [],
    }
    cos32, sin32 = ref.build_cache(40960, cfg.ROPE_THETA, cfg.HEAD_DIM, "float32")
    cos64, sin64 = ref.build_cache(40960, cfg.ROPE_THETA, cfg.HEAD_DIM, "float64")
    entries = report["entries"]

    # ---- prefill 扫描（fp32 io）----
    for tokens in (1, 4, 128, 512, 2048, 8192, 32768):
        q, k = gen_qk(tokens)
        reps = 30 if tokens <= 512 else (10 if tokens <= 8192 else 5)
        t = measure(lambda: ref.apply_prefill(q, k, cos32, sin32, 0), reps)
        bu, bi = io_bytes(tokens)
        entries.append({"form": "prefill", "tokens": tokens, "dtype": "fp32",
                        "bytes_unique": bu, "bytes_issued": bi,
                        "gb_s_unique": round(bu / (t["p50_ms"] * 1e-3) / 1e9, 2),
                        "bw_achieved_vs_copy": round(bu / (t["p50_ms"] * 1e-3) / 1e9 / copy_bw, 4),
                        **t})
        print(f"[bench/prefill] T={tokens:>6}: p50={t['p50_ms']:.4f} ms ({bu / (t['p50_ms'] * 1e-3) / 1e9:.2f} GB/s)")

    # ---- decode 扫描（位置平坦性）----
    q1, k1 = gen_qk(1)
    for pos in (0, 2047, 32767):
        t = measure(lambda: ref.apply_decode(q1[0], k1[0], cos32, sin32, pos), 100)
        bu, bi = io_bytes(1)
        entries.append({"form": "decode", "pos": pos, "dtype": "fp32",
                        "bytes_unique": bu, "bytes_issued": bi,
                        "gb_s_unique": round(bu / (t["p50_ms"] * 1e-3) / 1e9, 4), **t})
        print(f"[bench/decode] pos={pos:>6}: p50={t['p50_ms']:.4f} ms")

    # decode cache-free 变体
    t = measure(lambda: ref.apply_decode_cache_free(q1[0], k1[0], cfg.ROPE_THETA, cfg.HEAD_DIM, 2047), 100)
    entries.append({"form": "decode-cache-free", "pos": 2047, "dtype": "fp32", **t})
    print(f"[bench/decode-cache-free] p50={t['p50_ms']:.4f} ms")

    # decode fp16 io 对照
    t = measure(lambda: ref.apply_decode(q1[0], k1[0], cos32, sin32, 2047, io_dtype=np.float16), 100)
    entries.append({"form": "decode", "pos": 2047, "dtype": "fp16", **t})
    print(f"[bench/decode-fp16] p50={t['p50_ms']:.4f} ms")

    # ---- cache 生成吞吐（一次性成本）----
    for max_seq in (2048, 32768, 131072):
        for dtype, fn in (("fp32", ref.build_cache), ("fp16", ref.build_cache)):
            t = measure(lambda: fn(max_seq, cfg.ROPE_THETA, cfg.HEAD_DIM, dtype),
                        10 if max_seq <= 32768 else 5)
            item = 4 if dtype == "fp32" else 2
            nbytes = 2 * max_seq * cfg.HALF_DIM * item
            entries.append({"form": "build-cache", "max_seq": max_seq, "dtype": dtype,
                            "bytes_written": nbytes,
                            "gb_s": round(nbytes / (t["p50_ms"] * 1e-3) / 1e9, 2),
                            "mem_footprint_mb": round(nbytes / 1048576.0, 2), **t})
            print(f"[bench/build-cache] max_seq={max_seq:>6} {dtype}: p50={t['p50_ms']:.4f} ms ({nbytes / (t['p50_ms'] * 1e-3) / 1e9:.2f} GB/s)")

    # ---- 形态切换：prefill(2048) + decode 组合 vs 之和 ----
    q, k = gen_qk(2048)
    tp = measure(lambda: ref.apply_prefill(q, k, cos32, sin32, 0), 10)
    td = measure(lambda: ref.apply_decode(q1[0], k1[0], cos32, sin32, 2048), 30)
    tc = measure(lambda: (ref.apply_prefill(q, k, cos32, sin32, 0),
                          ref.apply_decode(q1[0], k1[0], cos32, sin32, 2048)), 10)
    entries.append({"form": "switch-prefill+decode", "tokens": 2048,
                    "prefill_p50_ms": tp["p50_ms"], "decode_p50_ms": td["p50_ms"],
                    "combined_p50_ms": tc["p50_ms"],
                    "switch_overhead_ms": round(tc["p50_ms"] - tp["p50_ms"] - td["p50_ms"], 4)})
    print(f"[bench/switch] prefill={tp['p50_ms']:.4f} decode={td['p50_ms']:.4f} "
          f"combined={tc['p50_ms']:.4f} overhead={tc['p50_ms'] - tp['p50_ms'] - td['p50_ms']:.4f} ms")

    # fp64 锚点路径成本（口径参考：锚点不被用于生产）
    t = measure(lambda: ref.apply_prefill(q.astype(np.float64), k.astype(np.float64), cos64, sin64, 0), 5)
    entries.append({"form": "prefill-fp64-anchor", "tokens": 2048, "dtype": "fp64", **t})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[benchmark-cpu] {len(entries)} entries → {out}")
    return 0


def gen_qk(seq: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.FIXTURES_SEED + seq)
    q = rng.standard_normal((seq, cfg.HQ, cfg.HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((seq, cfg.HKV, cfg.HEAD_DIM)).astype(np.float32)
    return q, k


if __name__ == "__main__":
    raise SystemExit(main())
