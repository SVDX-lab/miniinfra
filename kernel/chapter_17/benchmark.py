#!/usr/bin/env python3
"""第 17 期三平台 GEMM 性能基准（统一计时口径）。

口径（与第 01/01 期合同一脉相承，为第 18 期 WMMA 对照固定）：
- GPU kernel 延迟：naive_gemm_bench 二进制内部事件对计时（warmup 后逐样本
  record→launch→record，末尾一次同步；P50/P95/min/mean + 全样本保留）；
- 厂商库（torch.matmul / BLAS）：OperatorTimer 事件对口径，同 warmup/repeats
  规则——两者同口径，比值 = "朴素版的位置标尺"；
- 实测峰值分母（ep02 情报）：4090 D2D 918.8 GB/s / FP16 GEMM 171.63 TF /
  FP32 GEMM 55.6 TF；S4000 711.28 / 64.55 / 12.53（roofline_gemm.MEASURED_PEAK）；
- roofline：roofline_gemm.py 单源（理论字节/FLOPs/AI/bound/下限）。

实验段：
1. tile 大小扫描：11 配置 × {256,512,1024} × {fp32,fp16} + 零组织基线（naive）；
2. 达成率与厂商库比值（每 shape 同 dtype 的 torch.matmul 同口径对照）；
3. epilogue 融合：bias/gelu/silu 融合进 epilogue vs 分离 kernel（sep2/sep3）；
4. CPU 列：字面三重循环 / naive rows / numpy BLAS / torch（如有）四档对照。

用法：
    python benchmark.py --backend cuda
    python benchmark.py --backend musa
    python benchmark.py --backend cpu
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
    resolve_device,
    write_results_json,
)
from gemm_configs import CONFIGS, block_dims, grid_dims, smem_bytes  # noqa: E402
from naive_gemm import matmul_naive_rows, matmul_numpy, matmul_triple_loop  # noqa: E402
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
SEED = 20261118

# warmup / repeats 策略（记录进结果；ep19 沿用）
def repeats_for(n: int, kernel: str = "tiled") -> int:
    if kernel == "naive":
        return 10
    return 100 if n == 256 else (50 if n == 512 else 30)


WARMUP = 10
WARMUP_NAIVE = 3


def run_bench(binary: Path, extra: list[str]) -> dict:
    result = subprocess.run(
        [str(binary), "bench", *extra], capture_output=True, text=True, cwd=str(binary.parent)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"bench 失败 {extra}\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )
    return parse_binary_json(result.stdout)


def enrich_roofline(entry: dict, backend: str) -> dict:
    """按 roofline_gemm 单源补齐 AI/bound/达成率字段（就地修改并返回）。"""
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
        verdict_spec = classify_kernel_bound(
            entry["theoretical_bytes"], entry["theoretical_flops"],
            spec["peak_bandwidth_gb_s"] * 1e9,
            (spec["peak_fp16_tflops"] if entry["dtype"] == "fp16" else spec["peak_fp32_tflops"]) * 1e12)
        entry["bound_class_spec"] = verdict_spec["kernel_bound_class"]
        entry["bound_ms_spec"] = verdict_spec["roofline_bound_ms"]
        entry["achievement_vs_spec"] = round(
            verdict_spec["roofline_bound_ms"] / entry["p50_ms"], 4)

    measured = MEASURED_PEAK.get(backend)
    if measured:
        # 相对实测拷贝峰值的带宽对账：effective > copy 峰 ⇒ 该 kernel 不可能被
        # DRAM 访存解释（compute-bound 侧证据，无计数器条件下的对账法）
        entry["copy_peak_gb_s_measured"] = measured["copy_bandwidth_gb_s"]
        entry["bandwidth_over_copy_peak"] = round(
            entry["effective_bandwidth_gb_s"] / measured["copy_bandwidth_gb_s"], 3)
        gemm_peak = (measured["fp16_gemm_tflops"] if entry["dtype"] == "fp16"
                     else measured["fp32_gemm_tflops"])
        entry["vendor_gemm_peak_tflops_measured"] = gemm_peak
        entry["tflops_vs_vendor_peak"] = round(entry["effective_tflops_p50"] / gemm_peak, 4)
    return entry


def vendor_matmul_chain(device, backend: str) -> dict:
    """同形状同 dtype 的厂商库 GEMM（torch.matmul），OperatorTimer 事件对口径。"""
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
            warmup = 20 if backend == "musa" else 10  # MUSA 首轮有 JIT 编译开销（ch02 记录）
            timer = OperatorTimer(device, warmup=warmup, repeats=repeats_for(n))
            timing = timer.measure(lambda: torch.matmul(A, B))  # noqa: E731
            flops = theoretical_flops(n, n, n)
            entry = {
                "shape": n, "dtype": dtype_name, "library": "torch.matmul",
                **timing,
                "effective_tflops_p50": round(
                    flops / (timing["p50_ms"] / 1000.0) / 1e12, 3),
                "theoretical_flops": flops,
            }
            entries.append(entry)
            print(f"[vendor/{backend}] {dtype_name} {n}³: p50={timing['p50_ms']:.3f}ms "
                  f"tflops={entry['effective_tflops_p50']}")
            del A, B
    return {"entries": entries}


def vendor_epilogue_chain(device, backend: str) -> dict:
    """框架侧 epilogue 对照：matmul+bias、matmul+bias+gelu 的 kernel 链延迟。"""
    import torch

    from experiment_utils import OperatorTimer

    entries = []
    for n in [256, 1024]:
        generator = torch.Generator().manual_seed(SEED + n)
        A = torch.randn(n, n, generator=generator).to(device=device, dtype=torch.float32)
        B = torch.randn(n, n, generator=generator).to(device=device, dtype=torch.float32)
        bias = torch.randn(n, generator=generator).to(device=device, dtype=torch.float32)
        timer = OperatorTimer(device, warmup=10, repeats=repeats_for(n))
        t_bias = timer.measure(lambda: torch.nn.functional.linear(A, B, bias))  # noqa: E731
        t_gelu = timer.measure(
            lambda: torch.nn.functional.gelu(torch.nn.functional.linear(A, B, bias),
                                             approximate="tanh"))  # noqa: E731
        entries.append({"shape": n, "dtype": "fp32", "mode": "linear+bias(2 kernel)",
                        **t_bias})
        entries.append({"shape": n, "dtype": "fp32", "mode": "linear+bias+gelu(3 kernel)",
                        **t_gelu})
        print(f"[vendor-epi/{backend}] {n}³: +bias={t_bias['p50_ms']:.3f}ms "
              f"+bias+gelu={t_gelu['p50_ms']:.3f}ms")
    return {"entries": entries}


def tile_scan_chain(binary: Path, backend: str, vendor_lookup: dict) -> dict:
    """tile 大小扫描：11 配置 × 3 形状 × 2 dtype + 零组织基线。"""
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
                entry["smem_bytes_table"] = smem_bytes(config)
                entry["accumulators_per_thread"] = config["rm"] * config["rn"]
                vendor_p50 = vendor_lookup.get((dtype, n))
                if vendor_p50:
                    entry["vendor_p50_ms"] = vendor_p50
                    # 比值方向统一为"朴素/厂商"（≥1：朴素比厂商库慢几倍；同 shape 同 dtype 同口径）
                    entry["naive_vs_vendor_ratio"] = round(entry["p50_ms"] / vendor_p50, 1)
                enrich_roofline(entry, backend)
                sections[f"tile_scan_{dtype}"].append(entry)
                print(f"[scan/{backend}] {config['name']} {dtype} {n}³: "
                      f"p50={entry['p50_ms']:.3f}ms tflops={entry['effective_tflops_p50']} "
                      f"朴素/厂商={entry.get('naive_vs_vendor_ratio', '-')}×")
        # 零组织基线（naive kernel，仅 FP32；零复用组织，重复读 A 行/B 列）
        entry = run_bench(binary, [
            "--dtype", "fp32", "--kernel", "naive",
            "--m", str(n), "--n", str(n), "--k", str(n),
            "--warmup", str(WARMUP_NAIVE), "--repeats", str(repeats_for(n, "naive")),
            "--seed", str(SEED),
        ])
        entry["config"] = "naive_untiled"
        vendor_p50 = vendor_lookup.get(("fp32", n))
        if vendor_p50:
            entry["vendor_p50_ms"] = vendor_p50
            entry["naive_vs_vendor_ratio"] = round(entry["p50_ms"] / vendor_p50, 1)
        enrich_roofline(entry, backend)
        sections["naive_baseline"].append(entry)
        print(f"[scan/{backend}] naive_untiled fp32 {n}³: p50={entry['p50_ms']:.3f}ms "
              f"eff_bw={entry['effective_bandwidth_gb_s']}GB/s")
    return sections


def epilogue_bench_chain(binary: Path, backend: str, vendor_lookup: dict) -> dict:
    """epilogue 融合 vs 分离：同 GEMM kernel，只挪 epilogue 位置。"""
    entries = []
    epi_configs = ["32x32x8_reg2x2", "64x64x8_reg4x4"]
    for n in [256, 1024]:
        for config_name in epi_configs:
            tile, reg = config_name.split("_reg")
            for dtype in ["fp32", "fp16"]:
                cases = [
                    ("gemm_only", 0, 0),
                    ("fused_bias", 1, 0),
                    ("fused_bias_gelu", 2, 0),
                    ("fused_bias_silu", 3, 0),
                    ("sep2_bias", 1, 2),
                    ("sep2_bias_gelu", 2, 2),
                    ("sep3_bias_gelu", 2, 3),
                    ("sep2_bias_silu", 3, 2),
                    ("sep3_bias_silu", 3, 3),
                ]
                for label, epi, sep in cases:
                    entry = run_bench(binary, [
                        "--dtype", dtype, "--kernel", "tiled", "--tile", tile, "--reg", reg,
                        "--epilogue", str(epi), "--sep", str(sep),
                        "--m", str(n), "--n", str(n), "--k", str(n),
                        "--warmup", str(WARMUP), "--repeats", str(repeats_for(n)), "--seed", str(SEED),
                    ])
                    entry["config"] = config_name
                    entry["case"] = label
                    gemm_only = next((e for e in entries
                                      if e["config"] == config_name and e["m"] == n
                                      and e["dtype"] == dtype and e["case"] == "gemm_only"), None)
                    if gemm_only and label != "gemm_only":
                        entry["extra_ms_vs_gemm_only"] = round(entry["p50_ms"] - gemm_only["p50_ms"], 4)
                        entry["extra_pct_vs_gemm_only"] = round(
                            (entry["p50_ms"] / gemm_only["p50_ms"] - 1.0) * 100.0, 1)
                    entries.append(entry)
                print(f"[epilogue/{backend}] {config_name} {dtype} {n}³: 9 cases 完成")
    return {"entries": entries}


# ---------------------------------------------------------------------------
# CPU 列（本地机 + 服务器 CPU 自洽口径）
# ---------------------------------------------------------------------------
def cpu_bench_chain() -> dict:
    import torch

    sections: dict = {"entries": []}
    shapes = [64, 128, 256, 512, 1024]
    rng = np.random.default_rng(SEED)
    for n in shapes:
        A = rng.standard_normal((n, n), dtype=np.float32)
        B = rng.standard_normal((n, n), dtype=np.float32)
        flops = theoretical_flops(n, n, n)

        # 1) 字面三重循环（只测 64/128——纯 Python O(N³)，更大形状不可行）
        if n <= 128:
            repeats = 5
            samples = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                matmul_triple_loop(A, B)
                samples.append((time.perf_counter() - t0) * 1000.0)
            samples.sort()
            sections["entries"].append({
                "shape": n, "impl": "triple_loop_pure_python", "dtype": "fp32",
                "warmup": 0, "repeats": repeats, "p50_ms": round(samples[len(samples) // 2], 2),
                "min_ms": round(samples[0], 2),
                "effective_tflops_p50": round(flops / (samples[len(samples) // 2] / 1000.0) / 1e12, 6),
            })

        # 2) naive rows（numpy 内层 FMA，教学主体）
        repeats = {64: 20, 128: 10, 256: 5, 512: 3, 1024: 2}[n]
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            matmul_naive_rows(A, B)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        naive_entry = {
            "shape": n, "impl": "naive_rows_np_fma", "dtype": "fp32",
            "warmup": 1, "repeats": repeats, "p50_ms": round(samples[len(samples) // 2], 3),
            "min_ms": round(samples[0], 3),
            "effective_tflops_p50": round(flops / (samples[len(samples) // 2] / 1000.0) / 1e12, 4),
        }

        # 3) numpy BLAS（CPU 侧厂商库之一）
        samples = []
        for _ in range(max(repeats, 10)):
            t0 = time.perf_counter()
            matmul_numpy(A, B)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        blas_entry = {
            "shape": n, "impl": "numpy_matmul_blas", "dtype": "fp32",
            "warmup": 2, "repeats": max(repeats, 10), "p50_ms": round(samples[len(samples) // 2], 3),
            "min_ms": round(samples[0], 3),
            "effective_tflops_p50": round(flops / (samples[len(samples) // 2] / 1000.0) / 1e12, 3),
        }
        naive_entry["vendor_np_blas_p50_ms"] = blas_entry["p50_ms"]
        naive_entry["naive_vs_vendor_ratio"] = round(
            naive_entry["p50_ms"] / blas_entry["p50_ms"], 1)

        # 4) torch CPU（如可用，第二个厂商库锚点）
        torch_entry = None
        try:
            tA = torch.from_numpy(A)
            tB = torch.from_numpy(B)
            for _ in range(3):
                torch.matmul(tA, tB)
            samples = []
            for _ in range(max(repeats, 10)):
                t0 = time.perf_counter()
                torch.matmul(tA, tB)
                samples.append((time.perf_counter() - t0) * 1000.0)
            samples.sort()
            torch_entry = {
                "shape": n, "impl": "torch_matmul_cpu", "dtype": "fp32",
                "warmup": 3, "repeats": max(repeats, 10),
                "p50_ms": round(samples[len(samples) // 2], 3),
                "min_ms": round(samples[0], 3),
                "effective_tflops_p50": round(
                    flops / (samples[len(samples) // 2] / 1000.0) / 1e12, 3),
                "torch_version": torch.__version__,
            }
        except Exception as error:  # noqa: BLE001
            torch_entry = {"error": f"{type(error).__name__}: {error}"}

        sections["entries"].extend([naive_entry, blas_entry])
        if torch_entry:
            sections["entries"].append(torch_entry)
        ratio = naive_entry["naive_vs_vendor_ratio"]
        print(f"[cpu] {n}³: naive={naive_entry['p50_ms']:.1f}ms "
              f"npblas={blas_entry['p50_ms']:.1f}ms（朴素/厂商={ratio}×）"
              + (f" torch={torch_entry['p50_ms']}ms" if torch_entry and "p50_ms" in torch_entry else ""))

        del A, B

    # CPU epilogue：融合 vs 分离（256²）
    n = 256
    C = rng.standard_normal((n, n), dtype=np.float32)
    bias = rng.standard_normal(n, dtype=np.float32)
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = C + bias[None, :]
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    sections["entries"].append({
        "shape": n, "impl": "epilogue_bias_separate_array", "dtype": "fp32",
        "repeats": 20, "p50_ms": round(samples[len(samples) // 2], 4),
        "theoretical_bytes": C.nbytes * 2 + n * 4,
    })
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    arguments = parser.parse_args()
    backend = arguments.backend

    if backend == "cpu":
        cpu_sections = cpu_bench_chain()
        results = {
            "schema_version": SCHEMA_VERSION + ".benchmark",
            "backend": "cpu",
            "timing_policy": {"clock": "time.perf_counter", "unit": "ms"},
            "cpu_chain": cpu_sections,
            "environment": {"platform": sys.platform, "note": "CPU 列不计峰值达成率（大纲 §3.4）"},
        }
        write_results_json(ROOT / "results" / "benchmark-results-cpu.json", results)
        return 0

    device = resolve_device(backend)
    memory_api = getattr(__import__("torch"), backend)
    memory_api.empty_cache()

    binary = ROOT / "implementations" / backend / "naive_gemm_bench"
    if not binary.exists():
        raise SystemExit(f"[ERROR] 找不到 {binary}——先运行 implementations/{backend}/build.sh")

    # attrs：寄存器/共享内存占用（寄存器占用估算证据）
    attr_result = subprocess.run([str(binary), "attrs"], capture_output=True, text=True,
                                 cwd=str(binary.parent))
    attrs = parse_binary_json(attr_result.stdout) if attr_result.returncode == 0 else {
        "error": attr_result.stderr[-300:]}

    # 厂商库先行（tile_scan 的比值分母）
    vendor = vendor_matmul_chain(device, backend)
    vendor_lookup = {(e["dtype"], e["shape"]): e["p50_ms"] for e in vendor["entries"]}

    scan = tile_scan_chain(binary, backend, vendor_lookup)
    epilogue = epilogue_bench_chain(binary, backend, vendor_lookup)
    vendor_epi = vendor_epilogue_chain(device, backend)

    spec = PLATFORM_SPEC[backend]
    measured = MEASURED_PEAK[backend]
    results = {
        "schema_version": SCHEMA_VERSION + ".benchmark",
        "backend": backend,
        "timing_policy": {
            "kernel_timer": "binary 内部事件对（warmup 后逐样本 record→launch→record，末尾一次同步）",
            "vendor_timer": "OperatorTimer 事件对（ch01 口径）",
            "warmup_tiled": WARMUP, "warmup_naive": WARMUP_NAIVE,
            "repeats": {str(n): repeats_for(n) for n in SHAPES},
            "note": "第 18 期 WMMA 对照沿用同一口径与配置",
        },
        "vendor_matmul": vendor,
        "tile_scan_fp32": scan["tile_scan_fp32"],
        "tile_scan_fp16": scan["tile_scan_fp16"],
        "naive_baseline": scan["naive_baseline"],
        "epilogue_fusion": epilogue,
        "vendor_epilogue_chain": vendor_epi,
        "resource_usage": attrs,
        "roofline_platform": {
            "spec": spec,
            "measured_peak_source": measured["source"],
            "ridge_point_spec_fp32": round(
                spec["peak_fp32_tflops"] * 1e12 / (spec["peak_bandwidth_gb_s"] * 1e9), 1),
            "ridge_point_measured_fp32": round(
                measured["fp32_gemm_tflops"] * 1e12 / (measured["copy_bandwidth_gb_s"] * 1e9), 1),
        },
        "environment": device_environment_record(device),
    }
    write_results_json(ROOT / "results" / f"benchmark-results-{backend}.json", results)
    memory_api.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
