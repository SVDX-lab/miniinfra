#!/usr/bin/env python3
"""第 29 期性能对照（AI 版 vs 本篇人工手写版，同口径）。

--backend cuda：
  - 形状矩阵（合同 invariants）：rmsnorm rows{1,4,128,512,2048}×H1024、
    gemm {256,512,1024}³、attention decode N{64,512,2048,8192}×B{1,4}；
  - AI 二进制 --bench（二进制内部 cudaEvent 对：warmup 后逐样本
    record→launch→record、末尾一次同步——与 ch18/21 二进制口径一致）；
  - P50/P95/mean/min + 全样本；有效带宽/算力 + 达成率双分母
    （ch02 实测 918.8 GB/s、FP32 55.6 TF、FP16 171.63 TF 主口径；
     规格 1008/82.6/165.2 对照列）；
  - 人工基线：动态读取 ../chapter_0{3,18,21}/results/benchmark-results-cuda.json
    同形状最优（人工版多配置时取 min P50），算 AI/人工比值；
  - attention N=32768 上限探针记录（预期 launch 失败 → AI-SCALING）。

用法：python benchmark.py --backend cuda [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def _baseline_root(chapter: str, rel: str) -> Path:
    """人工基线 JSON 解析：CH29_BASELINES_ROOT（仓库镜像或平铺）优先。"""
    env_root = os.environ.get("CH29_BASELINES_ROOT") or os.environ.get("CH30_BASELINES_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root) / chapter / rel)
        candidates.append(Path(env_root) / Path(rel).name)
    candidates.append(REPO_ROOT / chapter / rel)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"基线未找到: {chapter}/{rel}（尝试 " +
                            "; ".join(str(c) for c in candidates) + "）")

BINARIES = {
    "rmsnorm": HERE / "implementations/ai/rmsnorm/ai_rmsnorm",
    "gemm": HERE / "implementations/ai/gemm/ai_gemm",
    "attention": HERE / "implementations/ai/attention/ai_decode_attn",
}

# 达成率分母（合同 peak_denominators：ch02 实测主口径 + 规格对照列）
DENOMS = {
    "copy_gb_s_measured": 918.8, "copy_gb_s_spec": 1008.0,
    "fp32_tf_measured": 55.6, "fp32_tf_spec": 82.6,
    "fp16_tf_measured": 171.63, "fp16_tf_spec": 165.2,
}

# AI 版（单配置）warmup/repeats：与人工版同形状最接近的档位
PLAN = {
    "rmsnorm": {
        "warmup": 10, "repeats": 50,
        "shapes": [{"rows": r, "hidden": 1024} for r in (1, 4, 128, 512, 2048)],
        "dtypes": ("fp32", "fp16"),
    },
    "gemm": {
        "warmup": 10, "repeats": 30,
        "shapes": [{"m": n, "k": n, "n": n} for n in (256, 512, 1024)],
        "dtypes": ("fp32", "fp16"),
    },
    "attention": {
        "warmup": 3, "repeats": 30,
        "shapes": [{"batch": b, "kv_len": n} for n in (64, 512, 2048, 8192)
                   for b in (1, 4)],
        "dtypes": ("fp32", "fp16"),
    },
}


def parse_time_ms(stdout: str) -> list[float]:
    vals = [float(m.group(1)) for m in re.finditer(r"time_ms\s+([0-9.eE+-]+)", stdout)]
    return vals


def run_bench(op: str, dtype: str, shape: dict, warmup: int, repeats: int) -> dict:
    binp = BINARIES[op]
    if op == "rmsnorm":
        args = ["--bench", "--rows", str(shape["rows"]), "--hidden", str(shape["hidden"])]
    elif op == "gemm":
        args = ["--bench", "--m", str(shape["m"]), "--k", str(shape["k"]), "--n", str(shape["n"])]
    else:
        args = ["--bench", "--batch", str(shape["batch"]), "--kv-len", str(shape["kv_len"])]
    args += ["--dtype", dtype, "--warmup", str(warmup), "--repeats", str(repeats)]
    proc = subprocess.run([str(binp)] + args, capture_output=True, text=True,
                          timeout=600)
    samples = parse_time_ms(proc.stdout)
    entry: dict = {"operator": op, "dtype": dtype, "shape": shape,
                   "warmup": warmup, "repeats": repeats,
                   "exit_code": proc.returncode}
    if proc.returncode != 0:
        entry["error"] = proc.stderr.strip().splitlines()[-5:]
        return entry
    if not samples:
        entry["error"] = ["no time_ms lines parsed", proc.stdout.strip().splitlines()[-3:]]
        return entry
    entry["samples_ms"] = [round(v, 6) for v in samples]
    entry["p50_ms"] = round(statistics.median(samples), 6)
    entry["p95_ms"] = round(sorted(samples)[int(len(samples) * 0.95) - 1 if len(samples) > 1 else 0], 6)
    entry["mean_ms"] = round(statistics.fmean(samples), 6)
    entry["min_ms"] = round(min(samples), 6)
    return entry


def rmsnorm_bytes(rows: int, hidden: int, dtype: str) -> int:
    item = 2 if dtype == "fp16" else 4
    return rows * hidden * 2 * item + hidden * item


def gemm_flops_bytes(m: int, k: int, n: int, dtype: str):
    item = 2 if dtype == "fp16" else 4
    flops = 2 * m * k * n
    nom_bytes = (m * k + k * n + m * n) * item
    return flops, nom_bytes


def attn_bytes(batch: int, kv_len: int, dtype: str) -> int:
    item = 2 if dtype == "fp16" else 4
    qo = 2 * batch * 16 * 128 * item      # q 读 + o 写
    kv = 2 * batch * 8 * kv_len * 128 * item  # k+v 读
    return qo + kv


def derive_metrics(entry: dict) -> dict:
    """有效带宽 / TFLOPS / 达成率（写入 entry 原地）。"""
    op, dtype, shape = entry["operator"], entry["dtype"], entry["shape"]
    p50 = entry.get("p50_ms")
    if not p50:
        return entry
    if op == "rmsnorm":
        b = rmsnorm_bytes(shape["rows"], shape["hidden"], dtype)
        entry["theoretical_bytes"] = b
        entry["effective_bandwidth_gb_s"] = round(b / (p50 * 1e-3) / 1e9, 2)
        entry["achievement_ratio_measured"] = round(
            entry["effective_bandwidth_gb_s"] / DENOMS["copy_gb_s_measured"], 4)
        entry["achievement_ratio_spec"] = round(
            entry["effective_bandwidth_gb_s"] / DENOMS["copy_gb_s_spec"], 4)
    elif op == "gemm":
        flops, nom = gemm_flops_bytes(shape["m"], shape["k"], shape["n"], dtype)
        entry["flops"] = flops
        entry["nominal_bytes"] = nom
        entry["effective_tflops"] = round(flops / (p50 * 1e-3) / 1e12, 4)
        key = "fp32" if dtype == "fp32" else "fp16"
        entry["compute_achievement_measured"] = round(
            entry["effective_tflops"] / DENOMS[f"{key}_tf_measured"], 4)
        entry["compute_achievement_spec"] = round(
            entry["effective_tflops"] / DENOMS[f"{key}_tf_spec"], 4)
        entry["effective_bandwidth_gb_s"] = round(nom / (p50 * 1e-3) / 1e9, 2)
    else:
        b = attn_bytes(shape["batch"], shape["kv_len"], dtype)
        entry["theoretical_bytes_unique"] = b
        entry["effective_bandwidth_gb_s"] = round(b / (p50 * 1e-3) / 1e9, 2)
        entry["achievement_ratio_measured"] = round(
            entry["effective_bandwidth_gb_s"] / DENOMS["copy_gb_s_measured"], 4)
    return entry


# ---------------------------------------------------------------------------
# 人工基线读取（动态，不硬编码数字）
# ---------------------------------------------------------------------------

def manual_baseline(op: str, dtype: str, shape: dict) -> dict | None:
    dtype_alias = {"fp16": "float16", "fp32": "float32"}[dtype]
    try:
        if op == "rmsnorm":
            data = json.loads(_baseline_root("chapter_02", "results/benchmark-results-cuda.json").read_text())
            best = None
            for section in (f"kernel_{dtype}",):
                for e in data.get(section, {}).get("entries", []):
                    if (e.get("rows") == shape["rows"]
                            and e.get("hidden_size") == shape["hidden"]
                            and e.get("dtype") in (dtype, dtype_alias)):
                        if best is None or e["p50_ms"] < best["p50_ms"]:
                            best = {"p50_ms": e["p50_ms"], "tile": e.get("tile"),
                                    "source": f"chapter_02 {section}"}
            return best
        if op == "gemm":
            data = json.loads(_baseline_root("chapter_17", "results/benchmark-results-cuda.json").read_text())
            scan = data.get(f"tile_scan_{dtype}", [])
            if isinstance(scan, dict):  # 兼容可能的 {entries: [...]} 包装
                scan = scan.get("entries", [])
            best = None
            for e in scan:
                e_shape = e.get("shape") or f"{e.get('m')}x{e.get('k')}x{e.get('n')}"
                target = f"{shape['m']}x{shape['k']}x{shape['n']}"
                if e_shape == target and e.get("dtype") == dtype:
                    if best is None or e["p50_ms"] < best["p50_ms"]:
                        best = {"p50_ms": e["p50_ms"], "config": e.get("config"),
                                "source": f"chapter_17 tile_scan_{dtype}"}
            return best
        if op == "attention":
            data = json.loads(_baseline_root("chapter_20", "results/benchmark-results-cuda.json").read_text())
            best = None
            for e in data.get("entries", []):
                if (e.get("form") == "decode" and e.get("kv_len") == shape["kv_len"]
                        and e.get("batch") == shape["batch"] and e.get("dtype") == dtype):
                    if best is None or e["p50_ms"] < best["p50_ms"]:
                        best = {"p50_ms": e["p50_ms"], "source": "chapter_20 decode"}
            return best
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return {"error": f"人工基线读取失败: {exc}"}
    return None


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("cuda",), default="cuda")
    ap.add_argument("--quick", action="store_true",
                    help="缩减 repeats（冒烟用，正式数据不用 --quick）")
    args = ap.parse_args()

    for op, p in BINARIES.items():
        if not p.exists():
            raise RuntimeError(f"AI 二进制未构建: {p}（先 bash implementations/ai/*/build.sh）")

    entries: list[dict] = []
    for op, plan in PLAN.items():
        for dtype in plan["dtypes"]:
            for shape in plan["shapes"]:
                repeats = plan["repeats"] if not args.quick else 5
                entry = run_bench(op, dtype, shape, plan["warmup"], repeats)
                entry = derive_metrics(entry)
                base = manual_baseline(op, dtype, shape)
                entry["manual_baseline"] = base
                if base and "p50_ms" in base and entry.get("p50_ms"):
                    entry["ai_over_manual_p50"] = round(
                        entry["p50_ms"] / base["p50_ms"], 4)
                entries.append(entry)
                ms = entry.get("p50_ms", entry.get("error"))
                ratio = entry.get("ai_over_manual_p50")
                print(f"[bench] {op:9s} {dtype} {shape} p50={ms}"
                      + (f" ai/manual={ratio}" if ratio else ""))
            # attention 缩放上限探针
            if op == "attention" and dtype == "fp16":
                probe = run_bench(op, dtype, {"batch": 1, "kv_len": 32768},
                                  warmup=0, repeats=1)
                probe["classification"] = "AI-SCALING 探针（预期 launch 失败）"
                entries.append(probe)
                print(f"[bench] attention N=32768 probe exit={probe.get('exit_code')}")

    results = {
        "schema_version": "kernel-chapter30.benchmark.v1",
        "backend": "cuda",
        "timing_policy": "AI 二进制内部 cudaEvent 对（warmup 后逐样本 record→"
                         "launch→record、末尾一次同步）——ch18/21 二进制口径",
        "denominators": DENOMS,
        "manual_baseline_policy": "动态读取 ch03/18/21 results/benchmark-results-cuda.json "
                                  "同形状最优 P50（人工版多配置取 min）",
        "quick_mode": args.quick,
        "entries": entries,
    }
    out = HERE / "results" / "benchmark-results-cuda.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"[results] 已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
