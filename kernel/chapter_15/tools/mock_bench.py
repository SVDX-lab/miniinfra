#!/usr/bin/env python3
"""第 15 期 mock 二进制（本地驱动链 dry-run 用，不上服务器跑 GPU 实验）。

模拟 int8_gemm_bench 的 run/materialize/attrs/bench/decodeblock 五个模式的
CLI 与输出格式（bench/decodeblock 的延迟为占位数字，仅用于驱动脚本联调）；
run/materialize 用 numpy float32 仿真 kernel 语义（int8_fused = q×scale 后
GEMM），让 compare_correctness.py 的完整链路（文件 I/O、尺寸校验、指标、
确定性、物化位级）在本地无 GPU 环境走通。
"""

from __future__ import annotations

import json
import sys

import numpy as np


def parse_args(argv: list[str]) -> dict:
    args = {}
    for i in range(1, len(argv) - 1):
        if argv[i].startswith("--"):
            args[argv[i]] = argv[i + 1]
    return args


def read_raw(path: str, dtype: np.dtype) -> np.ndarray:
    return np.fromfile(path, dtype=dtype)


def main() -> int:
    mode = sys.argv[1]
    a = parse_args(sys.argv)
    m, k, n = int(a.get("--m", 1)), int(a.get("--k", 1024)), int(a.get("--n", 1024))

    if mode == "run":
        kernel = a.get("--kernel", "fp16")
        a16 = read_raw(a["--in-a"], np.float16).reshape(m, k)
        if kernel == "fp16":
            w = read_raw(a["--in-w"], np.float16).reshape(k, n).astype(np.float32)
        else:
            q = read_raw(a["--in-w"], np.int8).reshape(k, n)
            scale = read_raw(a["--in-scale"], np.float32)
            w = q.astype(np.float32) * scale[None, :]
        c = a16.astype(np.float32) @ w
        c.astype(np.float16).tofile(a["--out-c"])
        return 0

    if mode == "materialize":
        q = read_raw(a["--in-w"], np.int8).reshape(k, n)
        scale = read_raw(a["--in-scale"], np.float32)
        w = (q.astype(np.float32) * scale[None, :]).astype(np.float16)
        w.tofile(a["--out-w"])
        return 0

    if mode == "attrs":
        print(json.dumps({"mode": "attrs", "configs": [{"name": "mock", "num_regs": 72}]}))
        return 0

    if mode in ("bench", "decodeblock"):
        kernel = a.get("--kernel", "fp16")
        tile, reg = a.get("--tile", "?"), a.get("--reg", "?")
        reps = int(a.get("--repeats", 10))
        if mode == "bench":
            bytes_w = k * n * (2 if kernel == "fp16" else 1) + (n * 4 if kernel != "fp16" else 0)
            total = m * k * 2 + bytes_w + m * n * 2
        else:
            shapes = [(1024, 2048), (1024, 1024), (1024, 1024), (4096, 1024), (1024, 6144), (3072, 1024)]
            total = sum(m * kk * 2 + kk * nn * (2 if kernel == "fp16" else 1) + m * nn * 2 for kk, nn in shapes)
        p50 = 0.001 + total / 711.28e9 * 1000
        entry = {
            "mode": mode,
            "backend_binary": "mock",
            "kernel": kernel,
            "tile": tile,
            "reg": reg,
            "m": m, "k": k, "n": n,
            "bytes": {"total_nominal": total},
            "warmup": int(a.get("--warmup", 10)),
            "repeats": reps,
            "p50_ms": round(p50, 6),
            "p95_ms": round(p50 * 1.1, 6),
            "mean_ms": round(p50 * 1.02, 6),
            "min_ms": round(p50 * 0.98, 6),
            "samples_ms": [round(p50, 6)] * reps,
            "effective_gb_s_p50": round(total / (p50 / 1000.0) / 1e9, 2),
            "tflops_p50": 0.0,
            "device_name": "MOCK-LOCAL",
            "sm_count": 0,
            "warp_size": 32,
            "mock": True,
        }
        print(json.dumps(entry))
        return 0

    print(f"mock: 未知模式 {mode}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
