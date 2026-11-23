#!/usr/bin/env python3
"""第 03 期性能测量（三后端统一口径）：分离 vs 融合的 Residual Add + RMSNorm。

口径（延续第 01/01/02 期）：
- 被测对象：(a) 分离路径 = add kernel + 第 02 期 rmsnorm kernel（2 launch/次），
  (b) 融合路径 = 单 kernel（1 launch/次，residual 原位更新）；
  (c) torch 组合表达式分离路径（平台原生参照，仅 launch 计数与延迟数量级）；
- 形状：合同 rows ∈ {1,4,128,512,2048} × H = 1024 + DRAM 区探针
  （4090 L2=72MB：工作集 > 72MB 才是真实 DRAM 带宽证据，ep03 方法学结论；
  S4000 L2=24MB，合同大形状已接近/超出）；
- 计时：OperatorTimer（warmup + 事件对、末尾一次同步，P50/P95/mean/min）；
- 理论字节（合同口径，itemsize = dtype 字节数，n = rows*H）：
    separated = (5n + H) × itemsize  （读r+读h+写t）+（读t+读w+写y）
    fused     = (4n + H) × itemsize  （读r+读h+读w+写y+写r 原位）
  理论加速比上限 = bytes_sep / bytes_fused（n >> H 时 → 5/4 = 1.25）；
- Kernel Launch 数：torch profiler 计数（cuda: CUDA activity kernel 事件；
  musa: musaLaunchKernel CPU 侧事件，第 01 期实测可计数），每配置 10 次迭代；
- 兑现率（本期核心）：实测加速比 / 理论加速比上限，逐形状记录，
  稀释因素用 launch 地板（rows=1 档）与有效带宽对账归因。

用法：
    python benchmark.py --backend cuda
    python benchmark.py --backend musa
    python benchmark.py --backend cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    OperatorTimer,
    backend_sync,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from fused_add_rmsnorm import (  # noqa: E402
    RMS_NORM_EPS,
    composite_separated_add_rmsnorm_torch,
    fused_add_rmsnorm_inplace_rowwise,
    fused_add_rmsnorm_inplace_serial,
    separated_add_rmsnorm_rowwise,
    separated_add_rmsnorm_serial,
)

HIDDEN_SIZE = 1024
ROW_COUNTS = [1, 4, 128, 512, 2048]
# DRAM 区探针（工作集 = r+h+(t)+y 四张量驻留显存；> 72MB 才能摆脱 4090 L2 驻留）
DRAM_PROBES = [
    {"rows": 8192, "hidden_size": 1024, "dtype": "fp32"},    # 4×32MB = 128MB
    {"rows": 32768, "hidden_size": 1024, "dtype": "fp32"},   # 4×128MB = 512MB
    {"rows": 32768, "hidden_size": 1024, "dtype": "fp16"},   # 4×64MB = 256MB
    {"rows": 65536, "hidden_size": 1024, "dtype": "fp16"},   # 4×128MB = 512MB
]
PEAK_BANDWIDTH_BYTES_PER_SEC = {"cuda": 918.8e9, "musa": 711.3e9, "cpu": None}  # ch02 实测
SPEC_BANDWIDTH_BYTES_PER_SEC = {"cuda": 1008e9, "musa": 768e9, "cpu": None}     # 大纲规格
PEAK_SOURCE = {"cuda": "chapter_01/performance 实测 D2D 拷贝峰值 918.8 GB/s",
               "musa": "chapter_01/performance 实测 D2D 拷贝峰值 711.3 GB/s", "cpu": "n/a"}
L2_BYTES = {"cuda": 75_497_472, "musa": 25_165_824, "cpu": None}  # 实测：ep03/ch03 收尾
SCAN_TILES = [128, 256, 512]
SCAN_SHAPE = {"rows": 2048, "hidden_size": 1024}
DEFAULT_TILE = 256
SEED_BASE = 20260401
H_STREAM_OFFSET = 555555


def theoretical_bytes(rows: int, hidden_size: int, itemsize: int, mode: str) -> int:
    n = rows * hidden_size
    factor = 5 if mode == "separated" else 4
    return (factor * n + hidden_size) * itemsize


def seeded_pair(rows: int, hidden_size: int, device: torch.device,
                dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """与 fixtures 同一 seed 规则（numpy PCG64）生成 (r, h)。"""
    seed = SEED_BASE + rows * 100000 + hidden_size
    r = np.random.default_rng(seed).standard_normal((rows, hidden_size), dtype=np.float32)
    h = np.random.default_rng(seed + H_STREAM_OFFSET).standard_normal(
        (rows, hidden_size), dtype=np.float32)
    return (torch.from_numpy(r).to(device=device, dtype=dtype),
            torch.from_numpy(h).to(device=device, dtype=dtype))


def seeded_weight(hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w = np.random.default_rng(SEED_BASE + hidden_size).standard_normal(
        hidden_size, dtype=np.float32)
    return torch.from_numpy(w).to(device=device, dtype=dtype)


def repeats_for(rows: int) -> int:
    return 200 if rows <= 128 else (100 if rows == 512 else (50 if rows == 2048 else 30))


# ---------------------------------------------------------------------------
# Kernel 路径（ctypes 直传；与 compare_correctness.py 同一加载逻辑）
# ---------------------------------------------------------------------------
class KernelRunner:
    name = "ctypes-sharedlib"

    def __init__(self, backend: str, device: torch.device):
        import importlib.util

        path = HERE / "implementations" / backend / "ops_binding.py"
        spec = importlib.util.spec_from_file_location(f"ops_binding_{backend}", path)
        module_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_loader)
        lib_name = "libmini_fused_cuda.so" if backend == "cuda" else "libmini_fused_musa.so"
        self.kernels = module_loader.load_kernels(backend, lib_name)
        self.warp_size = self.kernels.probe_warp_size(device)
        self.device = device

    def make_separated_op(self, r: torch.Tensor, h: torch.Tensor, w: torch.Tensor,
                          tile: int):
        t = torch.empty_like(r)
        y = torch.empty_like(r)

        def operation() -> None:
            self.kernels.separated(r, h, w, t, y, RMS_NORM_EPS, tile, self.warp_size)

        return operation

    def make_fused_op(self, r: torch.Tensor, h: torch.Tensor, w: torch.Tensor,
                      tile: int):
        y = torch.empty_like(r)

        def operation() -> None:
            # 融合 in-place：r 每次被更新（累计 r+k*h；std ~ sqrt(k)，无溢出，
            # 数值不影响访存量与计时）
            self.kernels.fused_add_rmsnorm(r, h, w, y, RMS_NORM_EPS, tile, self.warp_size)

        return operation


# ---------------------------------------------------------------------------
# Kernel Launch 数测量（torch profiler；MUSA 走 musaLaunchKernel CPU 侧事件）
# ---------------------------------------------------------------------------
def count_kernel_launches(operation, iterations: int, device: torch.device) -> dict:
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    else:
        if hasattr(ProfilerActivity, "PrivateUse1"):
            activities.append(ProfilerActivity.PrivateUse1)

    backend_sync(device)
    try:
        with profile(activities=activities) as prof:
            for _ in range(iterations):
                operation()
            backend_sync(device)
    except Exception as error:  # noqa: BLE001
        return {"available": False, "error": f"{type(error).__name__}: {error}"[:300]}

    kernel_events = {}
    runtime_launch_events = {}
    for event in prof.key_averages():
        name = event.key
        device_type = str(getattr(event, "device_type", ""))
        if "DeviceType.CUDA" in device_type:
            kernel_events[name] = kernel_events.get(name, 0) + event.count
        elif "LaunchKernel" in name or "launchkernel" in name.lower():
            runtime_launch_events[name] = runtime_launch_events.get(name, 0) + event.count

    kernel_total = sum(kernel_events.values())
    runtime_total = sum(runtime_launch_events.values())
    return {
        "available": True,
        "iterations": iterations,
        "kernel_events": kernel_events,
        "kernel_launches_total": kernel_total,
        "kernel_launches_per_iter": round(kernel_total / iterations, 3) if iterations else None,
        "runtime_launch_events": runtime_launch_events,
        "runtime_launches_per_iter": round(runtime_total / iterations, 3) if iterations else None,
    }


def launch_count_chain(runner: KernelRunner, device: torch.device, arguments) -> dict:
    """Launch 数对比：分离（2）vs 融合（1）vs torch 组合表达式分离链（多）。"""
    rows, hidden = SCAN_SHAPE["rows"], SCAN_SHAPE["hidden_size"]
    r, h = seeded_pair(rows, hidden, device, torch.float16)
    w = seeded_weight(hidden, device, torch.float16)
    iterations = 10

    results = {}
    sep_op = runner.make_separated_op(r, h, w, arguments.tile)
    results["separated"] = count_kernel_launches(sep_op, iterations, device)
    r2, h2 = seeded_pair(rows, hidden, device, torch.float16)
    fused_op = runner.make_fused_op(r2, h2, w, arguments.tile)
    results["fused"] = count_kernel_launches(fused_op, iterations, device)

    # torch 组合表达式分离链（框架对照：这条链发射多少个 kernel）
    try:
        t_buf = {}

        def composite_op():
            t, y = composite_separated_add_rmsnorm_torch(r, h, w, RMS_NORM_EPS)
            t_buf["t"] = t
            return y

        results["composite_separated"] = count_kernel_launches(composite_op, iterations, device)
    except Exception as error:  # noqa: BLE001
        results["composite_separated"] = {"available": False,
                                          "error": str(error)[:200]}

    for mode in ("separated", "fused", "composite_separated"):
        entry = results[mode]
        if entry.get("available"):
            print(f"[launch/{device.type}] {mode}: kernel/迭代 = "
                  f"{entry.get('kernel_launches_per_iter')} "
                  f"(runtime API/迭代 = {entry.get('runtime_launches_per_iter')})")
        else:
            print(f"[launch/{device.type}] {mode}: profiler 不可用 —— {entry.get('error')}")
    return {"shape": SCAN_SHAPE, "dtype": "fp16", "tile": arguments.tile,
            "iterations": iterations, **results}


def perf_entry(rows: int, hidden_size: int, dtype_str: str, tile: int,
               sep_timing: dict, fused_timing: dict, working_set_bytes: int,
               dram_region: bool) -> dict:
    """组装单形状的分离 vs 融合对照条目（含理论/实测/兑现率）。"""
    itemsize = 2 if dtype_str == "fp16" else 4
    bytes_sep = theoretical_bytes(rows, hidden_size, itemsize, "separated")
    bytes_fused = theoretical_bytes(rows, hidden_size, itemsize, "fused")
    theory_speedup = bytes_sep / bytes_fused
    measured_speedup = (sep_timing["p50_ms"] / fused_timing["p50_ms"]
                        if fused_timing["p50_ms"] > 0 else None)
    entry = {
        "rows": rows,
        "hidden_size": hidden_size,
        "dtype": dtype_str,
        "tile": tile,
        "working_set_bytes": working_set_bytes,
        "dram_region": dram_region,
        "theoretical_bytes_separated": bytes_sep,
        "theoretical_bytes_fused": bytes_fused,
        "theoretical_bytes_saved": bytes_sep - bytes_fused,
        "theoretical_speedup_ceiling": round(theory_speedup, 4),
        "separated": sep_timing,
        "fused": fused_timing,
        "measured_speedup_p50": round(measured_speedup, 4) if measured_speedup else None,
        "realization_ratio": (round(measured_speedup / theory_speedup, 4)
                              if measured_speedup else None),
    }
    if PEAK_BANDWIDTH_BYTES_PER_SEC.get(ENTRY_BACKEND) is not None:
        for mode, timing in (("separated", sep_timing), ("fused", fused_timing)):
            effective = bytes_sep if mode == "separated" else bytes_fused
            eff_bw = effective / (timing["p50_ms"] / 1000.0)
            entry[f"effective_bandwidth_{mode}_gb_s"] = round(eff_bw / 1e9, 2)
            entry[f"achievement_{mode}"] = round(
                eff_bw / PEAK_BANDWIDTH_BYTES_PER_SEC[ENTRY_BACKEND], 4)
    return entry


ENTRY_BACKEND = "cuda"  # 由 main 设置（带宽分母按平台单源）


def gpu_shape_chain(runner: KernelRunner, device: torch.device, arguments,
                    dtype: torch.dtype) -> list[dict]:
    """合同形状 + DRAM 探针：逐形状分离/融合计时与兑现率。"""
    dtype_str = "fp16" if dtype == torch.float16 else "fp32"
    itemsize = 2 if dtype == torch.float16 else 4
    entries = []
    plans = [{"rows": rows, "hidden_size": HIDDEN_SIZE} for rows in ROW_COUNTS]
    plans += [p for p in DRAM_PROBES if p["dtype"] == dtype_str]

    for plan in plans:
        rows, hidden = plan["rows"], plan["hidden_size"]
        r, h = seeded_pair(rows, hidden, device, dtype)
        w = seeded_weight(hidden, device, dtype)
        repeats = repeats_for(rows)
        timer = OperatorTimer(device, warmup=arguments.warmup, repeats=repeats)

        sep_op = runner.make_separated_op(r, h, w, arguments.tile)
        sep_timing = timer.measure(sep_op)
        r2, h2 = seeded_pair(rows, hidden, device, dtype)
        fused_op = runner.make_fused_op(r2, h2, w, arguments.tile)
        fused_timing = timer.measure(fused_op)

        working_set = rows * hidden * itemsize * 4  # r + h + y + (t 或 r 更新复用)
        dram_region = working_set > (L2_BYTES.get(device.type) or 0)
        entry = perf_entry(rows, hidden, dtype_str, arguments.tile,
                           sep_timing, fused_timing, working_set, dram_region)
        entries.append(entry)
        print(f"[perf/{device.type}/{dtype_str}] rows={rows:<6} "
              f"sep={entry['separated']['p50_ms']:.4f}ms fused={entry['fused']['p50_ms']:.4f}ms "
              f"加速比={entry['measured_speedup_p50']} 理论上限="
              f"{entry['theoretical_speedup_ceiling']} 兑现率={entry['realization_ratio']} "
              f"{'[DRAM区]' if dram_region else '[L2区]'}")
        del r, h, r2, h2
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "musa":
            torch.musa.empty_cache()
    return entries


def tile_scan_chain(runner: KernelRunner, device: torch.device, arguments) -> list[dict]:
    """Tile 受控子实验：{128,256,512} × 分离/融合 @2048×1024 fp16。"""
    rows, hidden = SCAN_SHAPE["rows"], SCAN_SHAPE["hidden_size"]
    r, h = seeded_pair(rows, hidden, device, torch.float16)
    w = seeded_weight(hidden, device, torch.float16)
    timer = OperatorTimer(device, warmup=arguments.warmup, repeats=50)
    entries = []
    for tile in SCAN_TILES:
        sep_op = runner.make_separated_op(r, h, w, tile)
        sep_timing = timer.measure(sep_op)
        r2, h2 = seeded_pair(rows, hidden, device, torch.float16)
        fused_op = runner.make_fused_op(r2, h2, w, tile)
        fused_timing = timer.measure(fused_op)
        entry = perf_entry(rows, hidden, "fp16", tile, sep_timing, fused_timing,
                           rows * hidden * 2 * 4, False)
        entries.append(entry)
        print(f"[scan/{device.type}] tile={tile}: sep={sep_timing['p50_ms']:.4f}ms "
              f"fused={fused_timing['p50_ms']:.4f}ms 加速比={entry['measured_speedup_p50']}")
    return entries


def cpu_chain(arguments) -> dict:
    """CPU 列：三层对照（串行/行内向量化/torch 组合表达式）。

    - 串行（结构对应 Kernel 组织）：分离 vs 融合逐元素循环——融合把加法
      挪进 Python 循环，失去 NumPy 向量化，实测反而更慢（诚实记录）；
    - 行内向量化（同向量化等级的公平对照）：分离 vs 融合——差异只剩
      "中间 t 是否物化为完整矩阵 + 遍历次数"；
    - torch 组合表达式分离链（平台原生参照）。
    """
    entries = []
    for rows in ROW_COUNTS:
        seed = SEED_BASE + rows * 100000 + HIDDEN_SIZE
        r = np.random.default_rng(seed).standard_normal((rows, HIDDEN_SIZE), dtype=np.float32)
        h = np.random.default_rng(seed + H_STREAM_OFFSET).standard_normal(
            (rows, HIDDEN_SIZE), dtype=np.float32)
        w = np.random.default_rng(SEED_BASE + HIDDEN_SIZE).standard_normal(
            HIDDEN_SIZE, dtype=np.float32)
        repeats = 3 if rows <= 128 else (2 if rows == 512 else 1)

        def run_variant(fn):
            samples = []
            for _ in range(repeats):
                r_copy = r.copy()
                start = time.perf_counter()
                fn(r_copy, h, w, RMS_NORM_EPS)
                samples.append((time.perf_counter() - start) * 1000.0)
            ordered = sorted(samples)
            return {"warmup": 0, "repeats": repeats,
                    "p50_ms": round(ordered[len(ordered) // 2], 4),
                    "p95_ms": round(ordered[-1], 4),
                    "min_ms": round(ordered[0], 4)}

        sep_serial = run_variant(separated_add_rmsnorm_serial)
        fused_serial = run_variant(fused_add_rmsnorm_inplace_serial)
        sep_rowwise = run_variant(separated_add_rmsnorm_rowwise)
        fused_rowwise = run_variant(fused_add_rmsnorm_inplace_rowwise)

        itemsize = 4
        bytes_sep = theoretical_bytes(rows, HIDDEN_SIZE, itemsize, "separated")
        bytes_fused = theoretical_bytes(rows, HIDDEN_SIZE, itemsize, "fused")
        ceiling = bytes_sep / bytes_fused

        def speedup(a_ms, b_ms):
            return round(a_ms / b_ms, 4)

        entry = {
            "rows": rows, "hidden_size": HIDDEN_SIZE, "dtype": "fp32",
            "theoretical_bytes_separated": bytes_sep,
            "theoretical_bytes_fused": bytes_fused,
            "theoretical_speedup_ceiling": round(ceiling, 4),
            "serial": {"separated": sep_serial, "fused": fused_serial,
                       "measured_speedup_p50": speedup(sep_serial["p50_ms"],
                                                       fused_serial["p50_ms"]),
                       "note": "纯 Python 逐元素循环：融合把加法挪进解释器循环，"
                               "失去 NumPy 向量化——负收益如实记录"},
            "rowwise": {"separated": sep_rowwise, "fused": fused_rowwise,
                        "measured_speedup_p50": speedup(sep_rowwise["p50_ms"],
                                                        fused_rowwise["p50_ms"]),
                        "realization_ratio": round(
                            speedup(sep_rowwise["p50_ms"], fused_rowwise["p50_ms"]) / ceiling, 4),
                        "note": "同向量化等级公平对照：NumPy 每条语句都写回内存，"
                                "融合无法把 t 留在寄存器——收益仍被物化稀释"},
        }
        entries.append(entry)
        print(f"[perf/cpu] rows={rows:<6} serial sep={sep_serial['p50_ms']:.1f}ms "
              f"fused={fused_serial['p50_ms']:.1f}ms ({entry['serial']['measured_speedup_p50']}×) | "
              f"rowwise sep={sep_rowwise['p50_ms']:.2f}ms "
              f"fused={fused_rowwise['p50_ms']:.2f}ms ({entry['rowwise']['measured_speedup_p50']}×)")

    # torch 组合表达式分离链（平台原生参照）
    composite_entries = []
    device = torch.device("cpu")
    for rows in ROW_COUNTS:
        r, h = seeded_pair(rows, HIDDEN_SIZE, device, torch.float32)
        w = seeded_weight(HIDDEN_SIZE, device, torch.float32)
        repeats = 5 if rows <= 128 else 3
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            composite_separated_add_rmsnorm_torch(r, h, w, RMS_NORM_EPS)
            samples.append((time.perf_counter() - start) * 1000.0)
        ordered = sorted(samples)
        composite_entries.append({"rows": rows, "hidden_size": HIDDEN_SIZE,
                                  "dtype": "fp32", "warmup": 0, "repeats": repeats,
                                  "p50_ms": round(ordered[len(ordered) // 2], 4),
                                  "min_ms": round(ordered[0], 4)})
        print(f"[composite/cpu] rows={rows:<6} p50={ordered[len(ordered) // 2]:.3f}ms")
    return {"serial_entries": entries, "composite_entries": composite_entries}


def main() -> int:
    global ENTRY_BACKEND
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--skip-scan", action="store_true")
    arguments = parser.parse_args()
    ENTRY_BACKEND = arguments.backend

    device = resolve_device(arguments.backend) if arguments.backend != "cpu" else torch.device("cpu")
    if device.type != "cpu":
        memory_api = getattr(torch, device.type)
        memory_api.empty_cache()
        if hasattr(memory_api, "reset_peak_memory_stats"):
            memory_api.reset_peak_memory_stats()

    environment = device_environment_record(device)
    environment["l2_cache_bytes_measured"] = L2_BYTES[device.type]
    environment["l2_source"] = ("ep03 收尾 C API 实测（results/server-logs/musa-deviceprop-l2.txt）"
                                if device.type == "musa" else
                                "ep03 torch.cuda.get_device_properties 实测 75,497,472 B")
    write_results_json(HERE / "results" / f"environment-{arguments.backend}.json",
                       {"schema_version": "kernel-chapter04.environment.v1",
                        "backend": arguments.backend, **environment})

    results = {
        "schema_version": "kernel-chapter04.benchmark.v1",
        "backend": arguments.backend,
        "tile_default": arguments.tile,
        "peak_bandwidth_source": PEAK_SOURCE[arguments.backend],
        "l2_bytes": L2_BYTES[device.type],
        "environment": environment,
    }

    if arguments.backend == "cpu":
        results["cpu"] = cpu_chain(arguments)
    else:
        runner = KernelRunner(arguments.backend, device)
        results["kernel_runner"] = runner.name
        results["probe_warp_size"] = runner.warp_size
        results["launch_counts"] = launch_count_chain(runner, device, arguments)
        results["perf_fp32"] = gpu_shape_chain(runner, device, arguments, torch.float32)
        results["perf_fp16"] = gpu_shape_chain(runner, device, arguments, torch.float16)
        if not arguments.skip_scan:
            results["tile_scan"] = tile_scan_chain(runner, device, arguments)

    output = HERE / "results" / f"benchmark-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
