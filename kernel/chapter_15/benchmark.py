#!/usr/bin/env python3
"""第 15 期性能基准（统一计时口径）。

口径（沿用第 01/17 期合同）：
- GPU kernel 延迟：int8_gemm_bench 二进制内部事件对计时（warmup 后逐样本
  record→launch→record，末尾一次同步；P50/P95/min/mean + 全样本保留）；
- 分离路径 vs 融合路径（大纲核心实验）：
    物化 = dequant 一次性成本（kernel=dequant，加载期账本）
         + fp16 kernel 稳态延迟（权重 2 字节/元素）；
    融合 = int8_fused 稳态延迟（权重 1 字节/元素，tile 加载端反量化）；
- 名义字节（与二进制 bytes 字段单源）：
    fp16: A(MK×2)+W(KN×2)+C(MN×2)；int8: A(MK×2)+W(KN×1)+scale(N×4)+C(MN×2)；
- decodeblock：Qwen3-0.6B 单 Block 6 投影链（Decode 单步端到端代理）；
- 峰值分母引用第 01 期实测：S4000 711.28 GB/s；4090 918.8 GB/s（待修复列）；
  GEMM 参照系用第 17 期朴素最优 tile 实测值（S4000 FP16 1024³ 64x64x8_reg4x4
  ≈ 2.94 TF / ~0.72ms）。

实验段：
1. tile 扫描：decode 档 {1,4,8}×{1024,4096}²×{fp16,int8_fused}×5 配置；
2. prefill 档 {256,1024}×1024²×两 kernel×5 配置；
3. 物化反量化一次性成本（dequant kernel）per 权重形状；
4. decodeblock 端到端链 M∈{1,4,8}×两 kernel；
5. CPU 列：参考 GEMM 三版计时（FP32 真值 / 引擎 FP16 / 反量化参考，信息性）。

用法：
    python benchmark.py --backend cpu
    python benchmark.py --backend musa --binary implementations/musa/int8_gemm_bench
    python benchmark.py --backend cuda --binary implementations/cuda/int8_gemm_bench
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    SCHEMA_VERSION,
    device_environment_record,
    parse_binary_json,
    write_results_json,
)

CONFIGS = [
    ("16x16x16", "2x2"),
    ("32x32x8", "2x2"),
    ("32x32x8", "4x4"),
    ("64x64x8", "2x2"),
    ("64x64x8", "4x4"),
]

DECODE_SHAPES = [(m, k, n) for m in (1, 4, 8) for k in (1024, 4096) for n in (1024, 4096)]
PREFILL_SHAPES = [(256, 1024, 1024), (1024, 1024, 1024)]
KERNELS = ["fp16", "int8_fused"]

SEED = 20261121
# 峰值分母（第 01 期实测，不重复测量）
MEASURED_PEAK = {"cuda": 918.8, "musa": 711.28, "cpu": None}
# 第 17 期朴素最优 tile 参照系（S4000：64x64x8_reg4x4 FP16 1024³）
EP18_REFERENCE = {
    "musa_fp16_1024_best_p50_ms": 0.73008,  # 64x64x8_reg4x4, 2.9414 TF（ep18 实测原值）
    "musa_fp32_1024_best_p50_ms": 0.726537,  # 同配置 FP32
    "note": "chapter_17/results/benchmark-results-musa.json 1024³ 最优（参照系，同 tile 家族）",
}


def repeats_for(m: int) -> int:
    return 100 if m <= 8 else (50 if m == 256 else 30)


def warmup_for(m: int) -> int:
    return 20 if m <= 8 else 10


def run_bench(binary: Path, extra: list[str]) -> dict:
    result = subprocess.run(
        [str(binary), "bench", *extra], capture_output=True, text=True, cwd=str(binary.parent)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"bench 失败 {extra}\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )
    return parse_binary_json(result.stdout)


def run_decodeblock(binary: Path, extra: list[str]) -> dict:
    result = subprocess.run(
        [str(binary), "decodeblock", *extra], capture_output=True, text=True, cwd=str(binary.parent)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"decodeblock 失败 {extra}\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )
    return parse_binary_json(result.stdout)


def benchmark_gpu(arguments) -> dict:
    binary = Path(arguments.binary).resolve()
    if not binary.exists():
        raise SystemExit(f"[FATAL] 二进制不存在：{binary}")
    backend = arguments.backend

    try:
        env = device_environment_record(__import__("torch").device(backend))
    except Exception as exc:
        env = {"error": str(exc)}

    entries: list[dict] = []

    # ---- 1. tile 扫描（decode + prefill 档）----
    for (m, k, n) in DECODE_SHAPES + PREFILL_SHAPES:
        for kernel in KERNELS:
            for tile, reg in CONFIGS:
                entry = run_bench(
                    binary,
                    [
                        "--m", str(m), "--k", str(k), "--n", str(n),
                        "--kernel", kernel, "--tile", tile, "--reg", reg,
                        "--warmup", str(warmup_for(m)), "--repeats", str(repeats_for(m)),
                        "--seed", str(SEED),
                    ],
                )
                entry["section"] = "tile_scan"
                entry["bucket"] = "decode" if m <= 8 else "prefill"
                entry["peak_gb_s_reference"] = MEASURED_PEAK.get(backend)
                entries.append(entry)
                print(
                    f"[{backend}] {kernel:10s} {tile}_reg{reg} {m}x{k}x{n}: "
                    f"p50={entry['p50_ms']:.4f}ms bw={entry['effective_gb_s_p50']:.1f}GB/s"
                )

    # ---- 2. 物化反量化一次性成本（per 权重形状；v1/v2/v3 三变体诊断）----
    # v1 = size_t 取模（首测异常）；v2 = 2D 网格无取模；v3 = 32 位取模。
    seen_wn = set()
    for (_, k, n) in DECODE_SHAPES + PREFILL_SHAPES:
        if (k, n) in seen_wn:
            continue
        seen_wn.add((k, n))
        for variant in ("dequant", "dequant2", "dequant3"):
            entry = run_bench(
                binary,
                [
                    "--m", "1", "--k", str(k), "--n", str(n),
                    "--kernel", variant, "--tile", "16x16x16", "--reg", "2x2",
                    "--warmup", "10", "--repeats", "50", "--seed", str(SEED),
                ],
            )
            entry["section"] = "dequant_materialize"
            entry["peak_gb_s_reference"] = MEASURED_PEAK.get(backend)
            entries.append(entry)
            print(
                f"[{backend}] {variant} {k}x{n}: p50={entry['p50_ms']:.4f}ms "
                f"bw={entry['effective_gb_s_p50']:.1f}GB/s"
            )

    # ---- 3. decodeblock 端到端链 ----
    for m in (1, 4, 8):
        for kernel in KERNELS:
            for tile, reg in CONFIGS:
                entry = run_decodeblock(
                    binary,
                    [
                        "--m", str(m),
                        "--kernel", kernel, "--tile", tile, "--reg", reg,
                        "--warmup", "20", "--repeats", "50", "--seed", str(SEED),
                    ],
                )
                entry["section"] = "decodeblock"
                entry["peak_gb_s_reference"] = MEASURED_PEAK.get(backend)
                entries.append(entry)
                print(
                    f"[{backend}] block {kernel:10s} {tile}_reg{reg} m={m}: "
                    f"p50={entry['p50_ms']:.4f}ms bw={entry['effective_gb_s_p50']:.1f}GB/s"
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "timing": "binary 内部事件对（warmup 后逐样本 record→launch→record，末尾一次同步）",
        "seed": SEED,
        "nominal_bytes_rule": {
            "fp16": "A(MK*2)+W(KN*2)+C(MN*2)",
            "int8_fused": "A(MK*2)+W(KN*1)+scale(N*4)+C(MN*2)",
            "dequant": "W(KN*1)+scale(N*4)+Wout(KN*2)",
        },
        "peak_reference": MEASURED_PEAK.get(backend),
        "ep18_reference": EP18_REFERENCE,
        "environment": env,
        "entries": entries,
    }


def benchmark_cpu() -> dict:
    """CPU 列：参考 GEMM 三版计时（信息性——CPU BLAS 多线程，口径单独标注）。"""
    sys.path.insert(0, str(ROOT / "implementations" / "cpu"))
    import torch

    from quant_reference import dequantize_weight, gemm_engine_fp16, quantize_weight_perchannel

    try:
        env = device_environment_record(torch.device("cpu"))
    except Exception as exc:
        env = {"error": str(exc)}

    torch.set_num_threads(8)
    entries = []
    for (m, k, n) in [(1, 1024, 1024), (256, 1024, 1024), (1024, 1024, 1024)]:
        rng = np.random.default_rng(SEED + m)
        w32 = (rng.standard_normal((k, n)) * 0.05).astype(np.float32)
        a32 = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
        q, scale = quantize_weight_perchannel(w32)
        w_hat = dequantize_weight(q, scale)
        w16, a16 = w32.astype(np.float16), a32.astype(np.float16)

        def timed(fn, warmup=3, repeats=10):
            for _ in range(warmup):
                fn()
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            return (time.perf_counter() - t0) * 1000.0 / repeats

        t32 = timed(lambda: torch.from_numpy(a32) @ torch.from_numpy(w32))
        t16 = timed(lambda: torch.from_numpy(a16) @ torch.from_numpy(w16))
        t_mat = timed(lambda: torch.from_numpy(a32) @ torch.from_numpy(w_hat))
        # 分离路径 CPU 版：反量化（一次性，numpy elementwise）+ FP16 GEMM
        t_dq = timed(lambda: dequantize_weight(q, scale), warmup=1, repeats=5)
        # 融合路径 CPU 语义 = 数学上与分离等价（FP32 算术），不单独计时——记 note
        flops = 2.0 * m * k * n
        entries.append(
            {
                "section": "cpu_reference_gemm",
                "m": m, "k": k, "n": n,
                "truth_fp32_p50_ms": round(t32, 3),
                "engine_fp16_p50_ms": round(t16, 3),
                "quant_dequant_fp32_p50_ms": round(t_mat, 3),
                "dequant_materialize_only_ms": round(t_dq, 3),
                "tflops_fp32": round(flops / t32 / 1e9, 3),
                "tflops_fp16_engine": round(flops / t16 / 1e9, 3),
                "note": "CPU BLAS 多线程 perf_counter 口径，信息性参考；"
                "融合 vs 分离在 CPU FP32 算术下数学等价，无单独融合路径",
                "torch_threads": torch.get_num_threads(),
            }
        )
        print(f"[cpu] {m}x{k}x{n}: fp32={t32:.1f}ms fp16={t16:.1f}ms dq+fp32={t_mat:.1f}ms")

    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "cpu",
        "environment": env,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--binary", default="")
    parser.add_argument("--out", default="")
    arguments = parser.parse_args()

    if arguments.backend == "cpu":
        payload = benchmark_cpu()
    else:
        if not arguments.binary:
            print("[FATAL] gpu 后端需要 --binary")
            return 2
        payload = benchmark_gpu(arguments)

    out_path = (
        Path(arguments.out)
        if arguments.out
        else ROOT / "results" / f"benchmark-results-{arguments.backend}.json"
    )
    payload["created_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_results_json(out_path, payload)
    print(f"[benchmark] 完成：{len(payload['entries'])} 条目 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
