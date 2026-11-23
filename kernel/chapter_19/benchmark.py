#!/usr/bin/env python3
"""第 19 期两平台（cuda/musa）异步流水 GEMM 性能基准（统一计时口径）。

无 CPU 性能列（期首声明：设备侧数据供给组织概念）。

口径（与第 17/18 期合同一脉相承）：
- kernel 延迟：async_gemm_bench 二进制内部事件对计时（warmup 后逐样本
  record→launch→record，末尾一次同步；P50/P95/min/mean + 全样本保留）；
- 厂商库（torch.matmul）：OperatorTimer 事件对口径（ch01 合同）；
- 峰值分母引用 ch02 实测（4090 拷贝 918.8 GB/s、FP32 cuBLAS 55.6 TF；
  S4000 711.28 / 12.53）——不重复测量（roofline_gemm.MEASURED_PEAK 单源）；
- 跨期对照：同 tile 同形状的 ep18 记录（tile_scan_fp32）与 ep19 记录
  （wmma_scan）从 chapter_17/19 results JSON 直接引用（协议"前序基线
  直接引用不重测"），本期同会话的 sync 级重测值作一致性互证。

实验段（对应大纲第 19 期实验清单）：
E1 三级阶梯：sync → cpasync1 → cpasync2（8 配置 × {256,512,1024}³，FP32）；
E2 shape 交叉点：小 tile/小矩阵扫描（{64,128,256,512}³ × 5 tile × 3 级）；
E3 K 深度交叉：m=n=256，k ∈ {8,64,256,1024}（流水级数 1/8/32/128）；
E4 v16 向量化变体（cp.async.cg 16B vs 4B issue 效率）；
E5 warp specialization（生产者/消费者分工）；
E6 wmma_pipe 组合核（Tensor Core × 流水两个正交维度，FP16 输入 FP32 累加）；
E7 attrs：三级各自寄存器/片上占用（resource_usage）。

用法：
    python benchmark.py --backend cuda
    python benchmark.py --backend musa
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiment_utils import (  # noqa: E402
    SCHEMA_VERSION,
    device_environment_record,
    parse_binary_json,
    resolve_device,
    write_results_json,
)
from async_gemm_configs import (  # noqa: E402
    CONFIGS,
    SCALAR_LEVELS,
    V16_CONFIGS,
    block_dims,
    grid_dims,
    smem_bytes,
    smem_bytes_double,
)
from roofline_gemm import (  # noqa: E402
    MEASURED_PEAK,
    PLATFORM_SPEC,
    classify_kernel_bound,
    effective_bandwidth_gb_s,
    effective_tflops,
    gemm_arithmetic_intensity,
    theoretical_bytes,
    theoretical_flops,
)

SEED = 20261118
WARMUP = 10


def repeats_for(n: int) -> int:
    # 与 ep18 同策略；本期扩展的小形状取更宽样本
    if n <= 64:
        return 300
    if n <= 128:
        return 200
    return 100 if n == 256 else (50 if n == 512 else 30)


def run_bench(binary: Path, extra: list[str]) -> dict:
    result = subprocess.run(
        [str(binary), "bench", *extra], capture_output=True, text=True, cwd=str(binary.parent)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"bench 失败 {extra}\nstdout: {result.stdout[-400:]}\nstderr: {result.stderr[-500:]}"
        )
    return parse_binary_json(result.stdout)


def try_bench(binary: Path, extra: list[str]) -> dict | None:
    """探测式 bench：kernel 未编译（MUSA 平台边界）时返回 None 并打印报错尾。"""
    result = subprocess.run(
        [str(binary), "bench", *extra], capture_output=True, text=True, cwd=str(binary.parent)
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-2:]
        print(f"[boundary] bench 失败 {' '.join(extra[:6])}...: {tail}")
        return None
    return parse_binary_json(result.stdout)


def enrich_roofline(entry: dict, backend: str) -> dict:
    m, n, k = entry["m"], entry["n"], entry["k"]
    itemsize = 2 if entry["dtype"] == "fp16" else 4
    entry["itemsize"] = itemsize
    entry["theoretical_bytes"] = theoretical_bytes(m, n, k, itemsize)
    entry["theoretical_flops"] = theoretical_flops(m, n, k)
    entry["arithmetic_intensity"] = round(gemm_arithmetic_intensity(m, n, k, itemsize), 2)
    entry["effective_bandwidth_gb_s"] = round(
        effective_bandwidth_gb_s(entry["theoretical_bytes"], entry["p50_ms"]), 1)
    entry["effective_tflops_p50"] = round(
        effective_tflops(entry["theoretical_flops"], entry["p50_ms"]), 3)
    spec = PLATFORM_SPEC.get(backend)
    if spec and spec.get("peak_bandwidth_gb_s"):
        verdict = classify_kernel_bound(
            entry["theoretical_bytes"], entry["theoretical_flops"],
            spec["peak_bandwidth_gb_s"] * 1e9,
            (spec["peak_fp16_tflops"] if entry["dtype"] == "fp16" else spec["peak_fp32_tflops"]) * 1e12)
        entry["bound_class_spec"] = verdict["kernel_bound_class"]
        entry["bound_ms_spec"] = verdict["roofline_bound_ms"]
        entry["achievement_vs_spec"] = round(verdict["roofline_bound_ms"] / entry["p50_ms"], 4)
    measured = MEASURED_PEAK.get(backend)
    if measured:
        entry["copy_peak_gb_s_measured"] = measured["copy_bandwidth_gb_s"]
        entry["bandwidth_over_copy_peak"] = round(
            entry["effective_bandwidth_gb_s"] / measured["copy_bandwidth_gb_s"], 3)
    return entry


def load_prior_results() -> dict:
    """引用前序期结果（协议：前序基线直接引用不重测）。"""
    out: dict = {"ep18_tile_scan_fp32": {}, "ep19_wmma_fp16": {}, "sources": []}
    p18 = ROOT.parent / "chapter_17" / "results" / "benchmark-results-cuda.json"
    if p18.exists():
        data = json.loads(p18.read_text(encoding="utf-8"))
        for e in data.get("tile_scan_fp32", []):
            if e.get("dtype") == "fp32" and e.get("m") == e.get("n") == e.get("k"):
                out["ep18_tile_scan_fp32"][f"{e['config']}@{e['m']}"] = {
                    "p50_ms": e["p50_ms"], "tflops": e.get("effective_tflops_p50")}
        out["sources"].append(str(p18.name))
    p19 = ROOT.parent / "chapter_18" / "results" / "benchmark-results-cuda.json"
    if p19.exists():
        data = json.loads(p19.read_text(encoding="utf-8"))
        for e in data.get("wmma_scan", []):
            if e.get("dtype") == "fp16" and e.get("m") == e.get("n") == e.get("k"):
                out["ep19_wmma_fp16"][f"{e['config']}@{e['m']}"] = {
                    "p50_ms": e["p50_ms"], "tflops": e.get("effective_tflops_p50")}
        out["sources"].append(str(p19.name))
    return out


def vendor_matmul_chain(device, backend: str) -> dict:
    """同形状 FP32 厂商库 GEMM 锚点（FP16 锚点沿用 ep19 同会话记录）。"""
    import torch

    from experiment_utils import OperatorTimer

    module = torch.cuda if backend == "cuda" else torch.musa
    entries = []
    for n in [256, 512, 1024]:
        generator = torch.Generator().manual_seed(SEED + n)
        A = torch.randn(n, n, generator=generator).to(device=device, dtype=torch.float32)
        B = torch.randn(n, n, generator=generator).to(device=device, dtype=torch.float32)
        warmup = 20 if backend == "musa" else 10
        timer = OperatorTimer(device, warmup=warmup, repeats=repeats_for(n))
        timing = timer.measure(lambda: torch.matmul(A, B))  # noqa: E731
        flops = theoretical_flops(n, n, n)
        entry = {"shape": n, "dtype": "fp32", "library": "torch.matmul", **timing,
                 "effective_tflops_p50": round(flops / (timing["p50_ms"] / 1000.0) / 1e12, 3)}
        entries.append(entry)
        print(f"[vendor/{backend}] fp32 {n}³: p50={timing['p50_ms']:.3f}ms "
              f"tflops={entry['effective_tflops_p50']}")
        del A, B
    module.empty_cache()
    return {"entries": entries}


def ladder_chain(binary: Path, backend: str, prior: dict) -> dict:
    """E1 三级阶梯：3 级 × 8 配置 × {256,512,1024}³（FP32）。"""
    entries = []
    for n in [256, 512, 1024]:
        for level in SCALAR_LEVELS:
            for config in CONFIGS:
                name = config["name"]
                tile, reg = name.split("_reg")
                bx, by = block_dims(config)
                gx, gy = grid_dims(config, n, n)
                entry = run_bench(binary, [
                    "--dtype", "fp32", "--kernel", level, "--tile", tile, "--reg", reg,
                    "--m", str(n), "--n", str(n), "--k", str(n),
                    "--warmup", str(WARMUP), "--repeats", str(repeats_for(n)), "--seed", str(SEED),
                ])
                entry["level"] = level
                entry["config"] = name
                entry["block_calculated"] = [bx, by]
                entry["grid_calculated"] = [gx, gy]
                entry["smem_bytes_table"] = (smem_bytes_double(config)
                                             if level != "sync" else smem_bytes(config))
                ref = prior["ep18_tile_scan_fp32"].get(f"{name}@{n}")
                if ref:
                    entry["ep18_sync_p50_ms_reference"] = ref["p50_ms"]
                    entry["ep18_speedup_vs_record"] = round(ref["p50_ms"] / entry["p50_ms"], 3)
                enrich_roofline(entry, backend)
                entries.append(entry)
                print(f"[E1/{backend}] {level:>9} {config} {n}³: p50={entry['p50_ms']:.4f}ms "
                      f"tflops={entry['effective_tflops_p50']}"
                      + (f" ep18ref={ref['p50_ms']:.4f}" if ref else ""))
    return {"entries": entries}


def parse_like(config: str) -> dict:
    for c in CONFIGS:
        if c["name"] == config:
            return c
    raise ValueError(config)


def crossover_shape_chain(binary: Path, backend: str) -> dict:
    """E2 形状交叉点：小矩阵 × 小 tile × 3 级（流水开销反噬扫描）。"""
    entries = []
    tiles = ["16x16x16_reg2x2", "16x16x16_reg4x4", "32x32x8_reg2x2", "32x32x8_reg4x4", "64x64x8_reg4x4"]
    for n in [64, 128, 256, 512]:
        for level in SCALAR_LEVELS:
            for config in tiles:
                tile, reg = config.split("_reg")
                entry = run_bench(binary, [
                    "--dtype", "fp32", "--kernel", level, "--tile", tile, "--reg", reg,
                    "--m", str(n), "--n", str(n), "--k", str(n),
                    "--warmup", str(WARMUP), "--repeats", str(repeats_for(n)), "--seed", str(SEED),
                ])
                entry.update({"level": level, "config": config, "scan": "shape_crossover"})
                enrich_roofline(entry, backend)
                entries.append(entry)
                print(f"[E2/{backend}] {level:>9} {config} {n}³: p50={entry['p50_ms']:.4f}ms "
                      f"tflops={entry['effective_tflops_p50']}")
    return {"entries": entries}


def crossover_kdepth_chain(binary: Path, backend: str) -> dict:
    """E3 K 深度交叉：m=n=256 固定，k ∈ {8,64,256,1024}（流水级数 1/8/32/128）。"""
    entries = []
    tiles = ["32x32x8_reg4x4", "64x64x8_reg4x4"]
    for k in [8, 64, 256, 1024]:
        for level in SCALAR_LEVELS:
            for config in tiles:
                tile, reg = config.split("_reg")
                entry = run_bench(binary, [
                    "--dtype", "fp32", "--kernel", level, "--tile", tile, "--reg", reg,
                    "--m", "256", "--n", "256", "--k", str(k),
                    "--warmup", str(WARMUP), "--repeats", "100", "--seed", str(SEED),
                ])
                entry.update({"level": level, "config": config, "scan": "k_depth_crossover"})
                enrich_roofline(entry, backend)
                entries.append(entry)
                print(f"[E3/{backend}] {level:>9} {config} 256²x{k}: p50={entry['p50_ms']:.4f}ms "
                      f"steps={entry.get('k_steps')}")
    return {"entries": entries}


def variant_chain(binary: Path, backend: str) -> dict:
    """E4/E5：v16 向量化变体 + warp specialization（256/512/1024 × 实例化配置）。"""
    entries = []
    for n in [256, 512, 1024]:
        for kernel, config in [("cpasync2v16", "32x32x8_reg4x4"), ("cpasync2v16", "64x64x8_reg4x4"),
                               ("warpspec", "32x32x8_reg4x4"), ("warpspec", "64x64x8_reg4x4")]:
            tile, reg = config.split("_reg")
            got = try_bench(binary, [
                "--dtype", "fp32", "--kernel", kernel, "--tile", tile, "--reg", reg,
                "--m", str(n), "--n", str(n), "--k", str(n),
                "--warmup", str(WARMUP), "--repeats", str(repeats_for(n)), "--seed", str(SEED),
            ])
            if got is None:
                entries.append({"kernel": kernel, "config": config, "m": n, "status": "unavailable"})
                continue
            got.update({"kernel": kernel, "config": config})
            enrich_roofline(got, backend)
            entries.append(got)
            print(f"[E45/{backend}] {kernel:>12} {config} {n}³: p50={got['p50_ms']:.4f}ms "
                  f"tflops={got['effective_tflops_p50']}")
    return {"entries": entries}


def wmma_combo_chain(binary: Path, backend: str, prior: dict) -> dict:
    """E6 组合核：WMMA(64x64x16, fp16 in / fp32 acc) × cp.async 双缓冲。"""
    entries = []
    for n in [256, 512, 1024]:
        got = try_bench(binary, [
            "--dtype", "fp16", "--kernel", "wmma_pipe", "--tile", "64x64x16", "--reg", "4x4",
            "--m", str(n), "--n", str(n), "--k", str(n),
            "--warmup", str(WARMUP), "--repeats", str(repeats_for(n)), "--seed", str(SEED),
        ])
        if got is None:
            entries.append({"kernel": "wmma_pipe", "config": "64x64x16", "m": n,
                            "status": "unavailable"})
            continue
        got.update({"kernel": "wmma_pipe", "config": "64x64x16"})
        enrich_roofline(got, backend)
        ref = prior["ep19_wmma_fp16"].get("wm64x64@%d" % n)
        if ref:
            got["ep19_wmma_sync_p50_ms_reference"] = ref["p50_ms"]
            got["speedup_vs_ep19_wmma"] = round(ref["p50_ms"] / got["p50_ms"], 3)
        entries.append(got)
        ref_s = f" ep19wmma={ref['p50_ms']:.4f}({got.get('speedup_vs_ep19_wmma')}x)" if ref else ""
        print(f"[E6/{backend}] wmma_pipe 64x64x16 {n}³: p50={got['p50_ms']:.4f}ms "
              f"tflops={got['effective_tflops_p50']}{ref_s}")
    return {"entries": entries}


def attrs_chain(binary: Path) -> dict:
    result = subprocess.run([str(binary), "attrs"], capture_output=True, text=True,
                            cwd=str(binary.parent))
    if result.returncode != 0:
        return {"error": result.stderr[-300:]}
    return parse_binary_json(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    arguments = parser.parse_args()
    backend = arguments.backend

    device = resolve_device(backend)
    memory_api = getattr(__import__("torch"), backend)
    memory_api.empty_cache()

    binary = ROOT / "implementations" / backend / "async_gemm_bench"
    if not binary.exists():
        raise SystemExit(f"[ERROR] 找不到 {binary}——先运行 implementations/{backend}/build.sh")

    # 环境快照（每平台一份）
    environment = device_environment_record(device)
    write_results_json(ROOT / "results" / f"environment-{backend}.json",
                       {"schema_version": SCHEMA_VERSION + ".environment", **environment})

    prior = load_prior_results()
    attrs = attrs_chain(binary)
    vendor = vendor_matmul_chain(device, backend)
    vendor_lookup = {e["shape"]: e["p50_ms"] for e in vendor["entries"]}

    ladder = ladder_chain(binary, backend, prior)
    crossover_shape = crossover_shape_chain(binary, backend)
    crossover_kdepth = crossover_kdepth_chain(binary, backend)
    variants = variant_chain(binary, backend)
    wmma_combo = wmma_combo_chain(binary, backend, prior)

    # 三级阶梯的增量归因表（相对同会话 sync 的 speedup 与 vs ep18 记录的互证）
    attribution = []
    ladder_by = {(e["level"], e["config"], e["m"]): e for e in ladder["entries"]}
    for n in [256, 512, 1024]:
        for config in [c["name"] for c in CONFIGS]:
            row = {"config": config, "shape": n}
            base = ladder_by.get(("sync", config, n))
            if not base:
                continue
            row["sync_p50_ms"] = base["p50_ms"]
            for level in ["cpasync1", "cpasync2"]:
                e = ladder_by.get((level, config, n))
                if e:
                    row[f"{level}_p50_ms"] = e["p50_ms"]
                    row[f"{level}_speedup_vs_sync"] = round(base["p50_ms"] / e["p50_ms"], 3)
            ref = prior["ep18_tile_scan_fp32"].get(f"{config}@{n}")
            if ref:
                row["ep18_record_p50_ms"] = ref["p50_ms"]
                row["sync_session_vs_ep18_record"] = round(base["p50_ms"] / ref["p50_ms"], 4)
            attribution.append(row)

    spec = PLATFORM_SPEC[backend]
    measured = MEASURED_PEAK[backend]
    results = {
        "schema_version": SCHEMA_VERSION + ".benchmark",
        "backend": backend,
        "timing_policy": {
            "kernel_timer": "binary 内部事件对（warmup 后逐样本 record→launch→record，末尾一次同步）",
            "vendor_timer": "OperatorTimer 事件对（ch01 口径）",
            "warmup": WARMUP,
            "repeats": {"64": repeats_for(64), "128": repeats_for(128), "256": repeats_for(256),
                        "512": repeats_for(512), "1024": repeats_for(1024)},
            "note": "第 17/18 期同口径；本期只改数据供给组织（kernel 级别）",
        },
        "vendor_matmul_fp32": vendor,
        "E1_ladder_fp32": ladder,
        "E2_shape_crossover": crossover_shape,
        "E3_k_depth_crossover": crossover_kdepth,
        "E45_variants": variants,
        "E6_wmma_combo_fp16": wmma_combo,
        "attribution_vs_sync": attribution,
        "prior_results_reference": prior,
        "resource_usage": attrs,
        "roofline_platform": {
            "spec": spec,
            "measured_peak_source": measured["source"],
        },
        "environment": environment,
    }
    write_results_json(ROOT / "results" / f"benchmark-results-{backend}.json", results)
    memory_api.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
