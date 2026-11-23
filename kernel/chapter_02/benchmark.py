#!/usr/bin/env python3
"""第 02 期手写 RMSNorm Kernel 性能测量（三后端统一口径）。

口径（延续第 01/01 期）：
- 被测对象：(a) 手写 Kernel（本期产物，torch 扩展或独立 bench 路径），
  (b) PyTorch 组合表达式（第 01/01 期 baseline，同口径重测以便直接对照）；
- 形状：rows ∈ {1, 4, 128, 512, 2048} × H = 1024（合同 benchmark_shapes）；
- 计时：OperatorTimer（warmup + 事件对、末尾一次同步，报告 P50/P95/mean/min）；
  GPU 另测 Graph Replay 口径（单 Kernel launch 开销的量化，达成率取最优者
  并标注 mode，MUSA 可能 Graph 慢于 eager —— 第 01 期实测）；
- 理论字节 = (rows*H 读 x + rows*H 写 y + H 读 w) × dtype 字节数
  （Kernel 把 x 行缓存进 shared，无中间重读 —— 与组合表达式的 FP32 中间
  物化对比是第 03 期融合动机的定量素材）；
- 达成率 = 有效带宽 / 平台峰值带宽（cuda: 1008 GB/s, musa: 768 GB/s）；
- Tile 扫描（受控子实验）：tiles {64,128,256,512,1024} × (H=1024, rows=2048,
  fp16 主口径 + fp32 参照)，+ 非整 Tile H=127/129 抽查 —— 最优配置与敏感度曲线。

用法：
    python benchmark.py --backend cuda
    python benchmark.py --backend musa --kernel-backend auto
    python benchmark.py --backend cpu          # CPU 串行参考（数量级对照）
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
    measure_with_graph,
    resolve_device,
    write_results_json,
)
from naive_rmsnorm import RMS_NORM_EPS, composite_rmsnorm_torch, naive_rmsnorm_serial  # noqa: E402

HIDDEN_SIZE = 1024
ROW_COUNTS = [1, 4, 128, 512, 2048]
# 达成率分母单源（主 agent 指示）：复用第 01 期实测峰值（D2D 拷贝对账口径），
# 大纲 §3.3 规格值另存对照列 achievement_ratio_spec。
PEAK_BANDWIDTH_BYTES_PER_SEC = {"cuda": 918.8e9, "musa": 711.3e9, "cpu": None}   # ch02 实测
SPEC_BANDWIDTH_BYTES_PER_SEC = {"cuda": 1008e9, "musa": 768e9, "cpu": None}      # 大纲规格
PEAK_SOURCE = {"cuda": "chapter_01/performance 实测 D2D 拷贝峰值 918.8 GB/s",
               "musa": "chapter_01/performance 实测 D2D 拷贝峰值 711.3 GB/s", "cpu": "n/a"}
SCAN_TILES = [64, 128, 256, 512, 1024]
SCAN_SHAPE = {"rows": 2048, "hidden_size": 1024}
SCAN_SPOT_HIDDEN = [127, 129]  # 非整 Tile 抽查（标量路径）
DEFAULT_TILE = 256


def theoretical_bytes(rows: int, hidden_size: int, itemsize: int) -> int:
    return (rows * hidden_size * 2 + hidden_size) * itemsize


def seeded_input(rows: int, hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """与 fixtures 同一 seed 规则（numpy PCG64）生成基准输入（运行时，不落盘）。"""
    seed = 20260301 + rows * 100000 + hidden_size
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, hidden_size), dtype=np.float32)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def seeded_weight(hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    seed = 20260301 + hidden_size
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(hidden_size, dtype=np.float32)
    return torch.from_numpy(w).to(device=device, dtype=dtype)


def repeats_for(rows: int) -> int:
    return 200 if rows <= 128 else (100 if rows == 512 else 50)


# ---------------------------------------------------------------------------
# Kernel 路径选择（与 compare_correctness.py 同一套逻辑）
# ---------------------------------------------------------------------------
def make_kernel_runner(backend: str, preference: str, device: torch.device):
    import importlib.util

    if preference == "bench":
        return _make_bench_runner(backend)
    try:
        path = HERE / "implementations" / backend / "build_ext.py"
        spec = importlib.util.spec_from_file_location(f"build_ext_{backend}", path)
        module_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_loader)
        start = time.perf_counter()
        module = module_loader.get_module()
        build_seconds = time.perf_counter() - start
        x = torch.randn(4, 128, device=device, dtype=torch.float16)
        w = torch.randn(128, device=device, dtype=torch.float16)
        module.rmsnorm(x, w, RMS_NORM_EPS, DEFAULT_TILE)
        return {"kind": "ext", "module": module, "build_seconds": build_seconds}
    except Exception as error:  # noqa: BLE001
        if preference == "ext":
            raise
        print(f"[kernel] torch 扩展路径不可用（{type(error).__name__}: {error}），"
              f"退到独立可执行程序 + 文件 IO 对比路径")
        return {"kind": "bench", "runner": _make_bench_runner(backend)}


def _make_bench_runner(backend: str):
    binary = HERE / "implementations" / backend / "build" / "rmsnorm_bench"
    if not binary.is_file():
        raise RuntimeError(f"rmsnorm_bench 未构建：{binary}（先跑 build.sh）")
    return {"binary": str(binary)}


def bench_measure(binary: str, rows: int, h: int, dtype: str, tiles: list[int],
                  warmup: int, repeats: int, backend_peak: str = "cuda") -> list[dict]:
    """独立 bench 进程测量：返回每 tile 一条结果（与 OperatorTimer 同口径）。"""
    peak_gbs = {"cuda": 918.8, "musa": 711.3}.get(backend_peak, 918.8)
    command = [
        binary, "--h", str(h), "--rows", str(rows), "--dtype", dtype,
        "--tiles", ",".join(str(t) for t in tiles),
        "--warmup", str(warmup), "--repeats", str(repeats),
        "--peak-gbs", str(peak_gbs),
        "--check", "false",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"rmsnorm_bench 失败（rc={result.returncode}）：{result.stderr[-2000:]}")
    entries = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            entries.append(json.loads(line))
    return entries


def kernel_chain_ext(module, arguments, device: torch.device, dtype: torch.dtype) -> dict:
    tile = arguments.tile
    weight = seeded_weight(HIDDEN_SIZE, device, dtype)
    entries = []
    for rows in ROW_COUNTS:
        x = seeded_input(rows, HIDDEN_SIZE, device, dtype)
        repeats = repeats_for(rows)
        timer = OperatorTimer(device, warmup=arguments.warmup, repeats=repeats)
        operation = lambda: module.rmsnorm(x, weight, RMS_NORM_EPS, tile)  # noqa: E731
        timing = timer.measure(operation)
        timing["mode"] = "eager"
        graph_timing = measure_with_graph(device, warmup=3, repeats=repeats, operation=operation)

        itemsize = x.element_size()
        total_bytes = theoretical_bytes(rows, HIDDEN_SIZE, itemsize)
        entry = {
            "rows": rows,
            "hidden_size": HIDDEN_SIZE,
            "dtype": str(dtype).replace("torch.", ""),
            "tile": tile,
            "theoretical_bytes": total_bytes,
        }
        entry.update(timing)
        if graph_timing:
            entry["graph_replay"] = graph_timing
        effective_source = timing
        if graph_timing and graph_timing["p50_ms"] < timing["p50_ms"]:
            effective_source = graph_timing
            entry["mode"] = "graph_replay"
        if PEAK_BANDWIDTH_BYTES_PER_SEC[device.type] is not None:
            effective = total_bytes / (effective_source["p50_ms"] / 1000.0)
            entry["effective_bandwidth_gb_s"] = round(effective / 1e9, 2)
            entry["peak_bandwidth_gb_s_measured"] = PEAK_BANDWIDTH_BYTES_PER_SEC[device.type] / 1e9
            entry["peak_bandwidth_gb_s_spec"] = SPEC_BANDWIDTH_BYTES_PER_SEC[device.type] / 1e9
            entry["peak_source"] = PEAK_SOURCE[device.type]
            entry["achievement_ratio"] = round(
                effective / PEAK_BANDWIDTH_BYTES_PER_SEC[device.type], 4)
            entry["achievement_ratio_spec"] = round(
                effective / SPEC_BANDWIDTH_BYTES_PER_SEC[device.type], 4)
            entry["bandwidth_mode"] = entry["mode"]
        entries.append(entry)
        print(f"[kernel/{arguments.backend}/{entry['dtype']}] rows={rows:<5} tile={tile} "
              f"p50={entry['p50_ms']:.3f}ms "
              + (f"有效带宽={entry['effective_bandwidth_gb_s']}GB/s "
                 f"达成率={entry['achievement_ratio']:.2%}（{entry['bandwidth_mode']}）"
                 if "achievement_ratio" in entry else ""))
    return {"entries": entries}


def kernel_chain_bench(bench_runner, arguments, dtype_str: str) -> dict:
    binary = bench_runner["binary"]
    entries = []
    for rows in ROW_COUNTS:
        warmup = arguments.warmup
        repeats = repeats_for(rows)
        for measured in bench_measure(binary, rows, HIDDEN_SIZE, dtype_str, [arguments.tile],
                                      warmup, repeats, backend_peak=arguments.backend):
            entry = {
                "rows": rows,
                "hidden_size": measured["h"],
                "dtype": measured["dtype"],
                "tile": measured["tile"],
                "theoretical_bytes": measured["theoretical_bytes"],
                "warmup": measured["warmup"],
                "repeats": measured["repeats"],
                "p50_ms": measured["p50_ms"],
                "p95_ms": measured["p95_ms"],
                "mean_ms": measured["mean_ms"],
                "min_ms": measured["min_ms"],
                "mode": "standalone-bench",
                "effective_bandwidth_gb_s": measured["effective_bandwidth_gb_s"],
                "peak_bandwidth_gb_s_spec": measured["peak_bandwidth_gb_s_spec"],
                "achievement_ratio": measured["achievement_ratio"],
                "bandwidth_mode": "standalone-bench",
            }
            entries.append(entry)
            print(f"[kernel/{arguments.backend}/{dtype_str}] rows={rows:<5} tile={measured['tile']} "
                  f"p50={entry['p50_ms']:.3f}ms 有效带宽={entry['effective_bandwidth_gb_s']}GB/s "
                  f"达成率={entry['achievement_ratio']:.2%}（standalone-bench）")
    return {"entries": entries}


def tile_scan_chain(arguments, device: torch.device, kernel_state) -> dict:
    """Tile 扫描（受控子实验）：主形状 H=1024×rows=2048 + 非整 Tile H=127/129 抽查。"""
    plans = [
        {"rows": SCAN_SHAPE["rows"], "hidden_size": SCAN_SHAPE["hidden_size"], "dtype_str": "fp16"},
        {"rows": SCAN_SHAPE["rows"], "hidden_size": SCAN_SHAPE["hidden_size"], "dtype_str": "fp32"},
    ]
    for spot in SCAN_SPOT_HIDDEN:
        plans.append({"rows": SCAN_SHAPE["rows"], "hidden_size": spot, "dtype_str": "fp16"})

    results = []
    if kernel_state["kind"] == "ext":
        module = kernel_state["module"]
        for plan in plans:
            dtype = torch.float16 if plan["dtype_str"] == "fp16" else torch.float32
            weight = seeded_weight(plan["hidden_size"], device, dtype)
            x = seeded_input(plan["rows"], plan["hidden_size"], device, dtype)
            timer = OperatorTimer(device, warmup=arguments.warmup, repeats=50)
            for tile in SCAN_TILES:
                entry = {"rows": plan["rows"], "hidden_size": plan["hidden_size"],
                         "dtype": plan["dtype_str"], "tile": tile}
                try:
                    operation = lambda: module.rmsnorm(x, weight, RMS_NORM_EPS, tile)  # noqa: E731
                    timing = timer.measure(operation)
                    itemsize = x.element_size()
                    total = theoretical_bytes(plan["rows"], plan["hidden_size"], itemsize)
                    entry.update({
                        "theoretical_bytes": total,
                        "p50_ms": timing["p50_ms"],
                        "p95_ms": timing["p95_ms"],
                        "min_ms": timing["min_ms"],
                        "warmup": timing["warmup"],
                        "repeats": timing["repeats"],
                    })
                    if PEAK_BANDWIDTH_BYTES_PER_SEC[device.type] is not None:
                        effective = total / (timing["p50_ms"] / 1000.0)
                        entry["effective_bandwidth_gb_s"] = round(effective / 1e9, 2)
                        entry["achievement_ratio"] = round(
                            effective / PEAK_BANDWIDTH_BYTES_PER_SEC[device.type], 4)
                        entry["achievement_ratio_spec"] = round(
                            effective / SPEC_BANDWIDTH_BYTES_PER_SEC[device.type], 4)
                    results.append(entry)
                    print(f"[scan] H={plan['hidden_size']} rows={plan['rows']} "
                          f"{plan['dtype_str']} tile={tile}: p50={timing['p50_ms']:.4f}ms "
                          + (f"达成率={entry['achievement_ratio']:.2%}"
                             if "achievement_ratio" in entry else ""))
                except Exception as error:  # noqa: BLE001 —— Tile 被平台拒绝（如 MUSA TILE=64）记录为平台边界
                    entry["error"] = str(error)[:200]
                    entry["platform_boundary"] = True
                    results.append(entry)
                    print(f"[scan] H={plan['hidden_size']} {plan['dtype_str']} tile={tile}: "
                          f"平台拒绝（{type(error).__name__}）—— 平台边界")
    else:
        binary = kernel_state["runner"]["binary"]
        for plan in plans:
            try:
                measured_list = bench_measure(binary, plan["rows"], plan["hidden_size"],
                                              plan["dtype_str"], SCAN_TILES,
                                              arguments.warmup, 50,
                                              backend_peak=arguments.backend)
            except Exception as error:  # noqa: BLE001 —— bench 进程级失败（如 tile 被拒）记平台边界
                results.append({"rows": plan["rows"], "hidden_size": plan["hidden_size"],
                                "dtype": plan["dtype_str"], "error": str(error)[:300],
                                "platform_boundary": True})
                print(f"[scan] H={plan['hidden_size']} {plan['dtype_str']}: bench 失败（平台边界）")
                continue
            for measured in measured_list:
                entry = {key: measured[key] for key in (
                    "rows", "h", "dtype", "tile", "theoretical_bytes",
                    "p50_ms", "p95_ms", "min_ms", "warmup", "repeats",
                    "effective_bandwidth_gb_s", "achievement_ratio")}
                entry["hidden_size"] = entry.pop("h")
                results.append(entry)
                print(f"[scan] H={entry['hidden_size']} rows={entry['rows']} "
                      f"{entry['dtype']} tile={entry['tile']}: p50={entry['p50_ms']:.4f}ms "
                      f"达成率={entry['achievement_ratio']:.2%}（standalone-bench）")

    # 最优配置汇总
    best = {}
    for entry in results:
        if "error" in entry:
            continue
        key = f"{entry['dtype']}/H{entry['hidden_size']}"
        if key not in best or entry["p50_ms"] < best[key]["p50_ms"]:
            best[key] = entry
    summary = {key: {"tile": value["tile"], "p50_ms": value["p50_ms"],
                     "achievement_ratio": value.get("achievement_ratio")}
               for key, value in best.items()}
    print(f"[scan] 最优配置：{json.dumps(summary, ensure_ascii=False)}")
    return {"plans": plans, "tiles": SCAN_TILES, "entries": results, "best": summary}


def composite_chain(arguments, device: torch.device, dtype: torch.dtype) -> dict:
    """组合表达式 baseline（第 01 期口径重测，供直接对照）。"""
    weight = seeded_weight(HIDDEN_SIZE, device, dtype)
    entries = []
    for rows in ROW_COUNTS:
        x = seeded_input(rows, HIDDEN_SIZE, device, dtype)
        repeats = repeats_for(rows)
        timer = OperatorTimer(device, warmup=arguments.warmup, repeats=repeats)
        operation = lambda: composite_rmsnorm_torch(x, weight, RMS_NORM_EPS)  # noqa: E731
        timing = timer.measure(operation)
        timing["mode"] = "eager"
        graph_timing = measure_with_graph(device, warmup=3, repeats=repeats, operation=operation)
        itemsize = x.element_size()
        total_bytes = theoretical_bytes(rows, HIDDEN_SIZE, itemsize)
        entry = {
            "rows": rows,
            "hidden_size": HIDDEN_SIZE,
            "dtype": str(dtype).replace("torch.", ""),
            "theoretical_bytes": total_bytes,
        }
        entry.update(timing)
        if graph_timing:
            entry["graph_replay"] = graph_timing
        effective_source = timing
        if graph_timing and graph_timing["p50_ms"] < timing["p50_ms"]:
            effective_source = graph_timing
            entry["mode"] = "graph_replay"
        if PEAK_BANDWIDTH_BYTES_PER_SEC[device.type] is not None:
            effective = total_bytes / (effective_source["p50_ms"] / 1000.0)
            entry["effective_bandwidth_gb_s"] = round(effective / 1e9, 2)
            entry["peak_bandwidth_gb_s_measured"] = PEAK_BANDWIDTH_BYTES_PER_SEC[device.type] / 1e9
            entry["peak_bandwidth_gb_s_spec"] = SPEC_BANDWIDTH_BYTES_PER_SEC[device.type] / 1e9
            entry["peak_source"] = PEAK_SOURCE[device.type]
            entry["achievement_ratio"] = round(
                effective / PEAK_BANDWIDTH_BYTES_PER_SEC[device.type], 4)
            entry["achievement_ratio_spec"] = round(
                effective / SPEC_BANDWIDTH_BYTES_PER_SEC[device.type], 4)
            entry["bandwidth_mode"] = entry["mode"]
        entries.append(entry)
        print(f"[composite/{arguments.backend}] rows={rows:<5} p50={entry['p50_ms']:.3f}ms "
              + (f"达成率={entry['achievement_ratio']:.2%}" if "achievement_ratio" in entry else ""))
    return {"dtype": str(dtype).replace("torch.", ""), "entries": entries}


def cpu_serial_chain(arguments) -> dict:
    """CPU 朴素串行版计时（数量级参照；Python 逐元素循环，repeats 极小）。"""
    entries = []
    for rows in ROW_COUNTS:
        rng = np.random.default_rng(20260301 + rows * 100000 + HIDDEN_SIZE)
        x = rng.standard_normal((rows, HIDDEN_SIZE), dtype=np.float32)
        weight_rng = np.random.default_rng(20260301 + HIDDEN_SIZE)
        weight = weight_rng.standard_normal(HIDDEN_SIZE, dtype=np.float32)
        repeats = 3 if rows <= 128 else (2 if rows == 512 else 1)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            naive_rmsnorm_serial(x, weight, RMS_NORM_EPS)
            samples.append((time.perf_counter() - start) * 1000.0)
        ordered = sorted(samples)
        entry = {
            "rows": rows,
            "hidden_size": HIDDEN_SIZE,
            "dtype": "fp32",
            "warmup": 0,
            "repeats": repeats,
            "p50_ms": round(ordered[len(ordered) // 2], 4),
            "p95_ms": round(ordered[-1], 4),
            "min_ms": round(ordered[0], 4),
            "note": "纯 Python 逐元素循环，repeat 数极少 —— 仅作串行数量级参照",
        }
        entries.append(entry)
        print(f"[serial/cpu] rows={rows:<5} p50={entry['p50_ms']:.1f}ms")
    return {"dtype": "fp32", "entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--kernel-backend", default="auto", choices=["auto", "ext", "bench"])
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--skip-scan", action="store_true")
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend) if arguments.backend != "cpu" else torch.device("cpu")
    if device.type != "cpu":
        memory_api = getattr(torch, device.type)
        memory_api.empty_cache()
        if hasattr(memory_api, "reset_peak_memory_stats"):
            memory_api.reset_peak_memory_stats()

    environment = device_environment_record(device)
    write_results_json(HERE / "results" / f"environment-{arguments.backend}.json",
                       {"schema_version": "kernel-chapter03.environment.v1",
                        "backend": arguments.backend, **environment})

    results = {
        "schema_version": "kernel-chapter03.benchmark.v1",
        "backend": arguments.backend,
        "kernel_tile_default": arguments.tile,
        "cpu_serial": cpu_serial_chain(arguments) if arguments.backend == "cpu" else None,
        "environment": environment,
    }
    if arguments.backend == "cpu":
        results["composite"] = composite_chain(arguments, device, torch.float32)
    else:
        results["composite_fp32"] = composite_chain(arguments, device, torch.float32)
        results["composite"] = composite_chain(arguments, device, torch.float16)

    if arguments.backend in ("cuda", "musa"):
        kernel_state = make_kernel_runner(arguments.backend, arguments.kernel_backend, device)
        results["kernel_runner"] = kernel_state["kind"]
        if kernel_state["kind"] == "ext":
            results["kernel_build_seconds"] = round(kernel_state["build_seconds"], 1)
            results["kernel_fp32"] = kernel_chain_ext(kernel_state["module"], arguments, device,
                                                      torch.float32)
            results["kernel_fp16"] = kernel_chain_ext(kernel_state["module"], arguments, device,
                                                      torch.float16)
        else:
            results["kernel_fp32"] = kernel_chain_bench(kernel_state["runner"], arguments, "fp32")
            results["kernel_fp16"] = kernel_chain_bench(kernel_state["runner"], arguments, "fp16")
        if not arguments.skip_scan:
            results["tile_scan"] = tile_scan_chain(arguments, device, kernel_state)

    output = HERE / "results" / f"benchmark-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
