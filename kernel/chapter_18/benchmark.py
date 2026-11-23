#!/usr/bin/env python3
"""第 18 期性能基准：WMMA vs 朴素版（同 tile 配置、同形状、同计时口径）。

口径（与第 17 期完全一致，见第 17 期合同/EXPERIMENT_LOG）：
- kernel 延迟：二进制内部事件对（warmup 后逐样本 record→launch→record，末尾一次同步）；
- 厂商库：OperatorTimer 事件对（同会话重测，S4000 分母注意形状效率——同口径重测）；
- 达成率分母：ch02 实测峰（4090 FP16 GEMM 171.63 TF @4096³；S4000 64.55 TF）+ 同形状厂商库；
- roofline：roofline_gemm.py 单源。

实验段：
1. wmma_scan：5 tile（wm16x16…wm64x64，KSTEP=16）× {256,512,1024} × acc=fp32；
   另加 1024³ × acc=fp16（累加 dtype 的性能对照）；
2. naive_rescan：第 17 期同一扫描在本会话重跑（11 配置 × 3 形状 × 2 dtype +
   零组织基线）——WMMA vs 朴素的同会话对照，并与 ep18 结果做漂移检查；
3. epilogue traffic probe：sep2 额外读写随形状放大（256³→4096³），分离
   "L2 容量"与"elementwise kernel 效率"两个解释（回应 ep18 未决问题；
   4096³ 为探针形状，超出合同扫描范围，仅定性用）；
4. vendor_matmul / vendor_epilogue：同会话重测。

用法：python3 benchmark.py --backend cuda|musa
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiment_utils import (  # noqa: E402
    SCHEMA_VERSION,
    device_environment_record,
    parse_binary_json,
    resolve_device,
    write_results_json,
)
from gemm_configs import (  # noqa: E402
    CONFIGS,
    WMMA_CONFIGS,
    block_dims,
    grid_dims,
    parse_wmma_config,
    smem_bytes,
    wmma_block_threads,
    wmma_smem_bytes,
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

SHAPES = [256, 512, 1024]
PROBE_SHAPES = [256, 512, 1024, 4096]  # sep2 traffic probe（4096³ 定性用）
SEED = 20261118
WARMUP = 10


def repeats_for(n: int, kernel: str = "tiled") -> int:
    if kernel == "naive":
        return 10
    if n >= 4096:
        return 5
    return 100 if n == 256 else (50 if n == 512 else 30)


def run_bench(binary: Path, extra: list[str]) -> dict:
    result = subprocess.run([str(binary), "bench", *extra], capture_output=True, text=True,
                            cwd=str(binary.parent))
    if result.returncode != 0:
        raise RuntimeError(f"bench 失败 {extra}\nstdout: {result.stdout[-400:]}\nstderr: {result.stderr[-500:]}")
    return parse_binary_json(result.stdout)


def wmma_available(binary: Path) -> tuple[bool, str]:
    """探测二进制是否编译了 WMMA（MUSA 平台边界的运行级判定）。"""
    probe = subprocess.run(
        [str(binary), "bench", "--dtype", "fp16", "--kernel", "wmma", "--tile", "16x16x16",
         "--acc", "fp32", "--m", "16", "--n", "16", "--k", "16", "--warmup", "1", "--repeats", "2"],
        capture_output=True, text=True, cwd=str(binary.parent))
    if probe.returncode == 0:
        return True, "wmma 可用（探测运行通过）"
    detail = (probe.stderr or probe.stdout).strip().splitlines()
    reason = detail[-1] if detail else "unknown"
    return False, f"wmma 不可用：{reason[:200]}"


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
        peak_tf = (spec["peak_fp16_tflops"] if entry["dtype"] == "fp16" else spec["peak_fp32_tflops"])
        verdict_spec = classify_kernel_bound(
            entry["theoretical_bytes"], entry["theoretical_flops"], spec["peak_bandwidth_gb_s"] * 1e9,
            peak_tf * 1e12)
        entry["bound_class_spec"] = verdict_spec["kernel_bound_class"]
        entry["bound_ms_spec"] = verdict_spec["roofline_bound_ms"]
        entry["achievement_vs_spec"] = round(verdict_spec["roofline_bound_ms"] / entry["p50_ms"], 4)

    measured = MEASURED_PEAK.get(backend)
    if measured:
        entry["copy_peak_gb_s_measured"] = measured["copy_bandwidth_gb_s"]
        entry["bandwidth_over_copy_peak"] = round(
            entry["effective_bandwidth_gb_s"] / measured["copy_bandwidth_gb_s"], 3)
        gemm_peak = (measured["fp16_gemm_tflops"] if entry["dtype"] == "fp16"
                     else measured["fp32_gemm_tflops"])
        entry["vendor_gemm_peak_tflops_measured"] = gemm_peak
        entry["tflops_vs_vendor_peak"] = round(entry["effective_tflops_p50"] / gemm_peak, 4)
    return entry


def vendor_matmul_chain(device, backend: str) -> dict:
    import torch

    from experiment_utils import OperatorTimer

    module = torch.cuda if backend == "cuda" else torch.musa
    entries = []
    for n in SHAPES:
        for dtype in [torch.float32, torch.float16]:
            dtype_name = "fp32" if dtype == torch.float32 else "fp16"
            generator = torch.Generator().manual_seed(SEED + n)
            A = torch.randn(n, n, generator=generator).to(device=device, dtype=dtype)
            B = torch.randn(n, n, generator=generator).to(device=device, dtype=dtype)
            warmup = 20 if backend == "musa" else 10
            timer = OperatorTimer(device, warmup=warmup, repeats=repeats_for(n))
            timing = timer.measure(lambda: torch.matmul(A, B))  # noqa: E731
            flops = theoretical_flops(n, n, n)
            entry = {
                "shape": n, "dtype": dtype_name, "library": "torch.matmul",
                "note": "同会话重测（第 18 期分母）；ep02 为 4096³ 峰值口径",
                **timing,
                "effective_tflops_p50": round(flops / (timing["p50_ms"] / 1000.0) / 1e12, 3),
                "theoretical_flops": flops,
            }
            entries.append(entry)
            print(f"[vendor/{backend}] {dtype_name} {n}³: p50={timing['p50_ms']:.3f}ms "
                  f"tflops={entry['effective_tflops_p50']}")
            del A, B
    module.empty_cache()
    return {"entries": entries}


def wmma_scan_chain(binary: Path, backend: str, vendor_lookup: dict) -> dict:
    entries = []
    for n in SHAPES:
        for config in WMMA_CONFIGS:
            tile = f"{config['tm']}x{config['tn']}x16"
            entry = run_bench(binary, [
                "--dtype", "fp16", "--kernel", "wmma", "--tile", tile, "--acc", "fp32",
                "--m", str(n), "--n", str(n), "--k", str(n),
                "--warmup", str(WARMUP), "--repeats", str(repeats_for(n)), "--seed", str(SEED),
            ])
            entry["config"] = config["name"]
            entry["threads"] = wmma_block_threads(config)
            entry["smem_bytes_table"] = wmma_smem_bytes(config, acc_float=True)
            vendor_p50 = vendor_lookup.get(("fp16", n))
            if vendor_p50:
                entry["vendor_p50_ms"] = vendor_p50
                entry["naive_vs_vendor_ratio"] = round(entry["p50_ms"] / vendor_p50, 2)
            enrich_roofline(entry, backend)
            entries.append(entry)
            print(f"[wmma/{backend}] {config['name']} fp32acc {n}³: p50={entry['p50_ms']:.3f}ms "
                  f"tflops={entry['effective_tflops_p50']} 朴素/厂商={entry.get('naive_vs_vendor_ratio', '-')}×")
    # 累加 dtype 对照（1024³）
    acc16_entries = []
    for config in WMMA_CONFIGS:
        tile = f"{config['tm']}x{config['tn']}x16"
        entry = run_bench(binary, [
            "--dtype", "fp16", "--kernel", "wmma", "--tile", tile, "--acc", "fp16",
            "--m", "1024", "--n", "1024", "--k", "1024",
            "--warmup", str(WARMUP), "--repeats", str(repeats_for(1024)), "--seed", str(SEED),
        ])
        entry["config"] = config["name"]
        entry["threads"] = wmma_block_threads(config)
        entry["smem_bytes_table"] = wmma_smem_bytes(config, acc_float=False)
        vendor_p50 = vendor_lookup.get(("fp16", 1024))
        if vendor_p50:
            entry["vendor_p50_ms"] = vendor_p50
            entry["naive_vs_vendor_ratio"] = round(entry["p50_ms"] / vendor_p50, 2)
        enrich_roofline(entry, backend)
        acc16_entries.append(entry)
        print(f"[wmma/{backend}] {config['name']} fp16acc 1024³: p50={entry['p50_ms']:.3f}ms "
              f"tflops={entry['effective_tflops_p50']}")
    return {"entries": entries, "acc_fp16_entries": acc16_entries}


def naive_rescan_chain(binary: Path, backend: str, vendor_lookup: dict) -> dict:
    """第 17 期同一扫描在本会话重跑（同会话对照 + 与 ep18 漂移检查）。"""
    sections = {"tile_scan_fp32": [], "tile_scan_fp16": [], "naive_baseline": []}
    for n in SHAPES:
        for config in CONFIGS:
            bx, by = block_dims(config)
            gx, gy = grid_dims(config, n, n)
            for dtype in ["fp32", "fp16"]:
                entry = run_bench(binary, [
                    "--dtype", dtype, "--kernel", "tiled", "--tile",
                    f"{config['tm']}x{config['tn']}x{config['tk']}",
                    "--reg", f"{config['rm']}x{config['rn']}",
                    "--m", str(n), "--n", str(n), "--k", str(n),
                    "--warmup", str(WARMUP), "--repeats", str(repeats_for(n)), "--seed", str(SEED),
                ])
                entry["config"] = config["name"]
                entry["block_calculated"] = [bx, by]
                entry["grid_calculated"] = [gx, gy]
                vendor_p50 = vendor_lookup.get((dtype, n))
                if vendor_p50:
                    entry["vendor_p50_ms"] = vendor_p50
                    entry["naive_vs_vendor_ratio"] = round(entry["p50_ms"] / vendor_p50, 1)
                enrich_roofline(entry, backend)
                sections[f"tile_scan_{dtype}"].append(entry)
            print(f"[naive-rescan/{backend}] {config['name']} {n}³ fp32 p50="
                  f"{sections['tile_scan_fp32'][-1]['p50_ms']:.3f}ms")
        entry = run_bench(binary, [
            "--dtype", "fp32", "--kernel", "naive",
            "--m", str(n), "--n", str(n), "--k", str(n),
            "--warmup", "3", "--repeats", "10", "--seed", str(SEED),
        ])
        entry["config"] = "naive_untiled"
        vendor_p50 = vendor_lookup.get(("fp32", n))
        if vendor_p50:
            entry["vendor_p50_ms"] = vendor_p50
            entry["naive_vs_vendor_ratio"] = round(entry["p50_ms"] / vendor_p50, 1)
        enrich_roofline(entry, backend)
        sections["naive_baseline"].append(entry)
    return sections


def epilogue_traffic_probe(binary: Path, backend: str) -> dict:
    """sep2 额外读写随形状放大：区分 L2 容量 vs elementwise kernel 效率。

    逻辑：若额外读+写 C 的流量在 C 完全 L2 驻留（256³/512³/1024³）时就显著慢于
    L2 速度，则第 17 期 S4000 侧 +20% 的解释不是 L2 容量（S4000 L2=24MB，
    chapter_02 C-API 实测），而是 elementwise kernel 自身的效率；
    4096³（C=64MB > L2 24MB）给出 DRAM 化的参照点。
    """
    entries = []
    config_name = "32x32x8_reg2x2"
    tile, reg = config_name.split("_reg")
    for n in PROBE_SHAPES:
        tag = "probe" if n >= 4096 else "contract"
        gemm_entry = run_bench(binary, [
            "--dtype", "fp32", "--kernel", "tiled", "--tile", tile, "--reg", reg,
            "--epilogue", "0", "--m", str(n), "--n", str(n), "--k", str(n),
            "--warmup", "5", "--repeats", str(repeats_for(n)), "--seed", str(SEED),
        ])
        sep_entry = run_bench(binary, [
            "--dtype", "fp32", "--kernel", "tiled", "--tile", tile, "--reg", reg,
            "--epilogue", "1", "--sep", "2", "--m", str(n), "--n", str(n), "--k", str(n),
            "--warmup", "5", "--repeats", str(repeats_for(n)), "--seed", str(SEED),
        ])
        c_bytes = n * n * 4
        extra = sep_entry["p50_ms"] - gemm_entry["p50_ms"]
        traffic = 2 * c_bytes  # sep2 = 读 C + 写 C 各一遍
        entry = {
            "shape": n, "tag": tag, "config": config_name,
            "gemm_only_p50_ms": gemm_entry["p50_ms"], "sep2_bias_p50_ms": sep_entry["p50_ms"],
            "extra_ms": round(extra, 4),
            "extra_pct": round((sep_entry["p50_ms"] / gemm_entry["p50_ms"] - 1.0) * 100.0, 1),
            "extra_pass_traffic_bytes": traffic,
            "extra_pass_effective_gb_s": round(traffic / (extra / 1000.0) / 1e9, 1),
            "c_bytes": c_bytes,
            "c_fits_l2_musa_24mb": c_bytes <= 24 * 1024 * 1024,
            "c_fits_l2_cuda_72mb": c_bytes <= 72 * 1024 * 1024,
        }
        entries.append(entry)
        print(f"[epiprobe/{backend}] {n}³({tag}): extra={extra:.4f}ms "
              f"({entry['extra_pct']}%), 有效 {entry['extra_pass_effective_gb_s']}GB/s")
    return {"entries": entries, "note": "S4000 L2=24MB 为 chapter_02 C-API 实测（musaDeviceProp.l2CacheSize=25165824）；4090 L2=72MB 公开规格"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    arguments = parser.parse_args()
    backend = arguments.backend

    device = resolve_device(backend)
    memory_api = getattr(__import__("torch"), backend)
    memory_api.empty_cache()

    binary = ROOT / "implementations" / backend / "naive_gemm_bench"
    if not binary.exists():
        raise SystemExit(f"[ERROR] 找不到 {binary}——先运行 implementations/{backend}/build.sh")

    attr_result = subprocess.run([str(binary), "attrs"], capture_output=True, text=True,
                                 cwd=str(binary.parent))
    attrs = parse_binary_json(attr_result.stdout) if attr_result.returncode == 0 else {
        "error": attr_result.stderr[-300:]}

    has_wmma, wmma_note = wmma_available(binary)
    print(f"[wmma-check/{backend}] {wmma_note}")

    vendor = vendor_matmul_chain(device, backend)
    vendor_lookup = {(e["dtype"], e["shape"]): e["p50_ms"] for e in vendor["entries"]}

    wmma = (wmma_scan_chain(binary, backend, vendor_lookup)
            if has_wmma else {"entries": [], "acc_fp16_entries": [],
                              "unavailable": wmma_note})
    naive = naive_rescan_chain(binary, backend, vendor_lookup)
    probe = epilogue_traffic_probe(binary, backend)

    spec = PLATFORM_SPEC[backend]
    measured = MEASURED_PEAK[backend]
    results = {
        "schema_version": SCHEMA_VERSION + ".benchmark",
        "backend": backend,
        "timing_policy": {
            "kernel_timer": "二进制内部事件对（与第 17 期同口径）",
            "vendor_timer": "OperatorTimer 事件对",
            "warmup": WARMUP,
            "repeats": {str(n): repeats_for(n) for n in SHAPES},
            "note": "同会话内 WMMA 与朴素版对照；vendor 同会话重测",
        },
        "vendor_matmul_same_session": vendor,
        "wmma_available": {"available": has_wmma, "note": wmma_note},
        "wmma_scan": wmma["entries"],
        "wmma_acc_fp16": wmma["acc_fp16_entries"],
        "naive_rescan": naive,
        "epilogue_traffic_probe": probe,
        "resource_usage": attrs,
        "roofline_platform": {
            "spec": spec,
            "measured_peak_source": measured["source"],
            "ridge_point_spec_fp16": round(
                spec["peak_fp16_tflops"] * 1e12 / (spec["peak_bandwidth_gb_s"] * 1e9), 1),
            "ridge_point_measured_fp16": round(
                measured["fp16_gemm_tflops"] * 1e12 / (measured["copy_bandwidth_gb_s"] * 1e9), 1),
        },
        "environment": device_environment_record(device),
    }
    write_results_json(ROOT / "results" / f"benchmark-results-{backend}.json", results)
    memory_api.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
