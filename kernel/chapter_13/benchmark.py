#!/usr/bin/env python3
"""第 13 期性能基准：Triton 路径 vs 手写路径（同形状同口径同会话对照）。

口径（延续 ch01–04）：
- OperatorTimer（事件对 + 末尾一次同步，warmup/repeats 入结果，P50/P95/mean/min）；
- GPU 另测 Graph Replay 口径，达成率取最优者并标注 mode；
- 达成率分母 = ch02 实测 D2D 拷贝峰值（cuda 918.8 / musa 711.3 GB/s）；
- 理论字节 = (rows*H 读 x + rows*H 写 y + H 读 w) × dtype 字节数；
- 边际带宽差分（ch02/ch03 口径）用于 bound 分类。

链：
  A. triton_rmsnorm（SP 生产写法，BLOCK=next_pow2(H)）fp16/fp32 × rows {1,4,128,512,2048}；
  B. triton_tile_scan：looped kernel × BLOCK {64,128,256,512,1024,2048}（手动扫描，
     与 ch03 TILE 扫描同口径同空间）@2048×1024 fp16/fp32 + H=127/129 抽查；
  C. DRAM 区（工作集 > L2）：rows {16384, 32768} fp32 + rows {32768, 65536} fp16；
  D. composite torch baseline（同会话）；
  E. fused：rows {1,4,128,512,2048,32768}（fp32）/{1,4,128,512,2048,32768,65536}（fp16）
     fused_triton vs triton_separated vs torch eager 分离链（同会话 DSL 融合收益）；
  F. 手写对照（--with-handwritten）：ch03 RMSNorm + ch04 融合同口径重测（同会话），
     与 ch03/ch04 期 results/ 历史 JSON 交叉核对；
  G. 首次运行成本：冷/热缓存已在 compile_artifacts.py，此处记录 kernel 首调
     （进程内）时间到结果头。

用法：
    python benchmark.py --backend cuda
    python benchmark.py --backend cuda --with-handwritten --skip-scan
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations"))
sys.path.insert(0, str(HERE.parent / "chapter_02" / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    OperatorTimer,
    backend_sync,
    device_environment_record,
    measure_with_graph,
    resolve_device,
    write_results_json,
)
from naive_rmsnorm import RMS_NORM_EPS, composite_rmsnorm_torch  # noqa: E402
import triton  # noqa: E402

HIDDEN_SIZE = 1024
ROW_COUNTS = [1, 4, 128, 512, 2048]
PEAK_BANDWIDTH_BYTES_PER_SEC = {"cuda": 918.8e9, "musa": 711.3e9}
SPEC_BANDWIDTH_BYTES_PER_SEC = {"cuda": 1008e9, "musa": 768e9}
PEAK_SOURCE = {"cuda": "chapter_01/performance 实测 D2D 拷贝峰值 918.8 GB/s",
               "musa": "chapter_01/performance 实测 D2D 拷贝峰值 711.3 GB/s"}
SCAN_TILES = [64, 128, 256, 512, 1024, 2048]
SCAN_SHAPE = {"rows": 2048, "hidden_size": 1024}
SCAN_SPOT_HIDDEN = [127, 129]
DRAM_ROWS = {"fp32": [16384, 32768], "fp16": [32768, 65536]}
FUSED_ROWS = {"fp32": [1, 4, 128, 512, 2048, 32768],
              "fp16": [1, 4, 128, 512, 2048, 32768, 65536]}
L2_BYTES = {"cuda": 72 * 1024 * 1024, "musa": 24 * 1024 * 1024}


def theoretical_bytes(rows: int, hidden_size: int, itemsize: int) -> int:
    return (rows * hidden_size * 2 + hidden_size) * itemsize


def seeded_input(rows: int, hidden_size: int, device, dtype) -> torch.Tensor:
    seed = 20260301 + rows * 100000 + hidden_size
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, hidden_size), dtype=np.float32)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def seeded_weight(hidden_size: int, device, dtype) -> torch.Tensor:
    rng = np.random.default_rng(20260301 + hidden_size)
    w = rng.standard_normal(hidden_size, dtype=np.float32)
    return torch.from_numpy(w).to(device=device, dtype=dtype)


def repeats_for(rows: int) -> int:
    return 200 if rows <= 128 else (100 if rows <= 2048 else 30)


def attach_bandwidth(entry: dict, rows, hidden_size, itemsize, timing, device_type, mode) -> None:
    total_bytes = theoretical_bytes(rows, hidden_size, itemsize)
    entry["theoretical_bytes"] = total_bytes
    # graph_replay 口径的计时字典只有 p50/p95/min（无 mean）——用 get 容忍两口径
    for key in ("warmup", "repeats", "p50_ms", "p95_ms", "mean_ms", "min_ms"):
        if timing.get(key) is not None:
            entry[key] = timing[key]
    entry["mode"] = mode
    if device_type in PEAK_BANDWIDTH_BYTES_PER_SEC:
        effective = total_bytes / (entry["p50_ms"] / 1000.0)
        entry["effective_bandwidth_gb_s"] = round(effective / 1e9, 2)
        entry["peak_bandwidth_gb_s_measured"] = PEAK_BANDWIDTH_BYTES_PER_SEC[device_type] / 1e9
        entry["peak_bandwidth_gb_s_spec"] = SPEC_BANDWIDTH_BYTES_PER_SEC[device_type] / 1e9
        entry["peak_source"] = PEAK_SOURCE[device_type]
        entry["achievement_ratio"] = round(effective / PEAK_BANDWIDTH_BYTES_PER_SEC[device_type], 4)
        entry["achievement_ratio_spec"] = round(effective / SPEC_BANDWIDTH_BYTES_PER_SEC[device_type], 4)


def timed_op(arguments, device, rows, hidden_size, dtype, operation, label) -> dict:
    timer = OperatorTimer(device, warmup=arguments.warmup, repeats=repeats_for(rows))
    timing = timer.measure(operation)
    timing["mode"] = "eager"
    graph_timing = measure_with_graph(device, warmup=3, repeats=repeats_for(rows),
                                      operation=operation)
    entry = {"rows": rows, "hidden_size": hidden_size,
             "dtype": str(dtype).replace("torch.", ""), "label": label}
    if graph_timing:
        entry["graph_replay"] = graph_timing
    effective_source = timing
    mode = "eager"
    if graph_timing and graph_timing["p50_ms"] < timing["p50_ms"]:
        effective_source = graph_timing
        mode = "graph_replay"
    attach_bandwidth(entry, rows, hidden_size, 2 if dtype == torch.float16 else 4,
                     effective_source, device.type, mode)
    print(f"[{label}/{device.type}] rows={rows:<6} p50={entry['p50_ms']:.4f}ms "
          + (f"eff={entry.get('effective_bandwidth_gb_s')}GB/s "
             f"ach={entry.get('achievement_ratio'):.2%}（{mode}）"
             if "achievement_ratio" in entry else ""))
    return entry


def triton_chain(arguments, device, dtype) -> list[dict]:
    entries = []
    weight = seeded_weight(HIDDEN_SIZE, device, dtype)
    for rows in ROW_COUNTS:
        x = seeded_input(rows, HIDDEN_SIZE, device, dtype)
        entry = timed_op(arguments, device, rows, HIDDEN_SIZE, dtype,
                         lambda: rmsnorm(x, weight, RMS_NORM_EPS), "triton-SP")
        entry["kernel"] = "rmsnorm_fwd_kernel(SP)"
        entries.append(entry)
    return entries


def composite_chain(arguments, device, dtype) -> list[dict]:
    entries = []
    weight = seeded_weight(HIDDEN_SIZE, device, dtype)
    for rows in ROW_COUNTS:
        x = seeded_input(rows, HIDDEN_SIZE, device, dtype)
        entry = timed_op(arguments, device, rows, HIDDEN_SIZE, dtype,
                         lambda: composite_rmsnorm_torch(x, weight, RMS_NORM_EPS),
                         "composite-torch")
        entries.append(entry)
    return entries


def tile_scan_chain(arguments, device) -> dict:
    """手动 Tile 扫描（looped kernel，BLOCK 显式逐个测——autotune 实验的对照基线）。"""
    plans = [
        {"rows": SCAN_SHAPE["rows"], "hidden_size": SCAN_SHAPE["hidden_size"], "dtype": torch.float16},
        {"rows": SCAN_SHAPE["rows"], "hidden_size": SCAN_SHAPE["hidden_size"], "dtype": torch.float32},
    ]
    for spot in SCAN_SPOT_HIDDEN:
        plans.append({"rows": SCAN_SHAPE["rows"], "hidden_size": spot, "dtype": torch.float16})
    results = []
    for plan in plans:
        dtype = plan["dtype"]
        weight = seeded_weight(plan["hidden_size"], device, dtype)
        x = seeded_input(plan["rows"], plan["hidden_size"], device, dtype)
        timer = OperatorTimer(device, warmup=arguments.warmup, repeats=50)
        for tile in SCAN_TILES:
            if tile < plan["hidden_size"]:
                operation = lambda: rmsnorm(x, weight, RMS_NORM_EPS, block_size=tile,
                                            looped=True)  # noqa: E731
            else:
                # BLOCK >= H 时 looped 与 SP 等价，但保持同一 kernel 路径可比
                operation = lambda: rmsnorm(x, weight, RMS_NORM_EPS, block_size=tile,
                                            looped=True)  # noqa: E731
            entry = {"rows": plan["rows"], "hidden_size": plan["hidden_size"],
                     "dtype": "fp16" if dtype == torch.float16 else "fp32", "block": tile}
            try:
                timing = timer.measure(operation)
                attach_bandwidth(entry, plan["rows"], plan["hidden_size"],
                                 2 if dtype == torch.float16 else 4, timing, device.type, "eager")
            except Exception as error:  # noqa: BLE001
                entry["error"] = str(error)[:200]
                entry["platform_boundary"] = True
            results.append(entry)
            ach = f" ach={entry.get('achievement_ratio'):.2%}" if "achievement_ratio" in entry else ""
            print(f"[scan/{device.type}] H={plan['hidden_size']} "
                  f"{entry['dtype']} BLOCK={tile}: p50={entry.get('p50_ms')}ms{ach}")
    best = {}
    for entry in results:
        if "error" in entry:
            continue
        key = f"{entry['dtype']}/H{entry['hidden_size']}"
        if key not in best or entry["p50_ms"] < best[key]["p50_ms"]:
            best[key] = {"block": entry["block"], "p50_ms": entry["p50_ms"],
                         "achievement_ratio": entry.get("achievement_ratio")}
    print(f"[scan/{device.type}] 最优配置：{json.dumps(best, ensure_ascii=False)}")
    return {"tiles": SCAN_TILES, "entries": results, "best": best}


def dram_chain(arguments, device, dtype_str: str) -> list[dict]:
    dtype = torch.float16 if dtype_str == "fp16" else torch.float32
    entries = []
    weight = seeded_weight(HIDDEN_SIZE, device, dtype)
    for rows in DRAM_ROWS[dtype_str]:
        x = seeded_input(rows, HIDDEN_SIZE, device, dtype)
        working_set = rows * HIDDEN_SIZE * (2 if dtype == torch.float16 else 4)
        entry = timed_op(arguments, device, rows, HIDDEN_SIZE, dtype,
                         lambda: rmsnorm(x, weight, RMS_NORM_EPS), "triton-SP-DRAM")
        entry["working_set_bytes"] = working_set
        entry["dram_region"] = working_set > L2_BYTES[device.type]
        entry["kernel"] = "rmsnorm_fwd_kernel(SP)"
        entries.append(entry)
    return entries


def fused_chain(arguments, device, dtype_str: str, hw_fused=None) -> dict:
    """融合三路对照（同会话）：triton 融合 / triton 分离 / torch eager 分离。

    理论字节口径 = 第 03 期（ch04 benchmark.py）：
      fused     = (4n + H) × itemsize（读 r + 读 h + 写 y + 写 r 原位 + 读 w）；
      separated = (5n + H) × itemsize（读 r 读 h 写 t 读 t 写 y + 读 w）；
      理论加速比上限 = 5/4 = 1.25。
    in-place 计时语义 = ch04（make_fused_op 不 clone：r 每迭代被更新为
    r+k*h，数值漂移不影响访存量与计时）。
    """
    dtype = torch.float16 if dtype_str == "fp16" else torch.float32
    itemsize = 2 if dtype == torch.float16 else 4
    weight = seeded_weight(HIDDEN_SIZE, device, dtype)
    entries = []
    for rows in FUSED_ROWS[dtype_str]:
        r0 = seeded_input(rows, HIDDEN_SIZE, device, dtype)
        rng = np.random.default_rng(20260301 + rows * 100000 + HIDDEN_SIZE + 7)
        h_in = torch.from_numpy(
            rng.standard_normal((rows, HIDDEN_SIZE), dtype=np.float32)).to(device=device,
                                                                          dtype=dtype)
        n = rows * HIDDEN_SIZE

        # 计时语义对齐 ch04（make_fused_op）：y/t 预分配在计时外，op 只含 kernel。
        y_pre = torch.empty_like(r0)
        t_pre = torch.empty_like(r0)
        block = max(1, triton.next_power_of_2(HIDDEN_SIZE))

        def op_fused():
            fused_add_rmsnorm_kernel[(rows,)](r0, h_in, weight, y_pre, HIDDEN_SIZE,
                                              RMS_NORM_EPS, BLOCK=block, num_warps=4)

        def op_triton_sep():
            torch.add(r0, h_in, out=t_pre)
            rmsnorm(t_pre, weight, RMS_NORM_EPS)

        def op_eager_sep():
            t = torch.add(r0, h_in)
            var = t.float().pow(2).mean(dim=-1, keepdim=True)
            return weight * (t.float() * torch.rsqrt(var + RMS_NORM_EPS)).to(dtype)

        fused_timing = OperatorTimer(device, warmup=arguments.warmup,
                                     repeats=repeats_for(rows)).measure(op_fused)
        triton_sep_timing = OperatorTimer(device, warmup=arguments.warmup,
                                          repeats=repeats_for(rows)).measure(op_triton_sep)
        eager_timing = OperatorTimer(device, warmup=arguments.warmup,
                                     repeats=repeats_for(rows)).measure(op_eager_sep)

        entry = {"rows": rows, "hidden_size": HIDDEN_SIZE,
                 "dtype": dtype_str, "working_set_bytes": n * itemsize * 3,
                 "dram_region": n * itemsize * 3 > L2_BYTES[device.type]}
        fused_bytes = (n * 4 + HIDDEN_SIZE) * itemsize
        separated_bytes = (n * 5 + HIDDEN_SIZE) * itemsize
        entry["theoretical_bytes_fused"] = fused_bytes
        entry["theoretical_bytes_separated"] = separated_bytes
        entry["theoretical_speedup_ceiling"] = round(separated_bytes / fused_bytes, 4)
        entry["fused_triton"] = {"p50_ms": fused_timing["p50_ms"], "p95_ms": fused_timing["p95_ms"],
                                 "min_ms": fused_timing["min_ms"], "warmup": fused_timing["warmup"],
                                 "repeats": fused_timing["repeats"]}
        entry["fused_triton"]["effective_bandwidth_gb_s"] = round(
            fused_bytes / (fused_timing["p50_ms"] / 1000.0) / 1e9, 2)
        entry["fused_triton"]["achievement_ratio"] = round(
            fused_bytes / (fused_timing["p50_ms"] / 1000.0) / PEAK_BANDWIDTH_BYTES_PER_SEC[device.type], 4)
        entry["separated_triton"] = {
            "p50_ms": triton_sep_timing["p50_ms"],
            "effective_bandwidth_gb_s": round(
                separated_bytes / (triton_sep_timing["p50_ms"] / 1000.0) / 1e9, 2)}
        entry["separated_eager_torch"] = {"p50_ms": eager_timing["p50_ms"]}
        entry["speedup_fused_vs_triton_sep"] = round(
            triton_sep_timing["p50_ms"] / fused_timing["p50_ms"], 4)
        entry["speedup_fused_vs_eager"] = round(
            eager_timing["p50_ms"] / fused_timing["p50_ms"], 4)
        entry["realization_ratio"] = round(
            entry["speedup_fused_vs_triton_sep"] / entry["theoretical_speedup_ceiling"], 4)
        if hw_fused is not None:
            def op_hw_fused():
                hw_fused.run(r0, h_in, weight, RMS_NORM_EPS, 256)
            hw_timing = OperatorTimer(device, warmup=arguments.warmup,
                                      repeats=repeats_for(rows)).measure(op_hw_fused)
            entry["fused_handwritten_ch04"] = {
                "p50_ms": hw_timing["p50_ms"], "tile": 256,
                "effective_bandwidth_gb_s": round(
                    fused_bytes / (hw_timing["p50_ms"] / 1000.0) / 1e9, 2),
                "achievement_ratio": round(
                    fused_bytes / (hw_timing["p50_ms"] / 1000.0) / PEAK_BANDWIDTH_BYTES_PER_SEC[device.type], 4)}
            entry["speedup_handwritten_vs_triton_fused"] = round(
                fused_timing["p50_ms"] / hw_timing["p50_ms"], 4)
        entries.append(entry)
        print(f"[fused/{device.type}/{dtype_str}] rows={rows:<6} "
              f"fused={fused_timing['p50_ms']:.4f}ms "
              f"triton_sep={triton_sep_timing['p50_ms']:.4f}ms "
              f"eager={eager_timing['p50_ms']:.4f}ms "
              f"speedup_vs_sep={entry['speedup_fused_vs_triton_sep']}"
              + (f" 手写={entry['fused_handwritten_ch04']['p50_ms']:.4f}ms "
                 f"手写/Triton={entry['speedup_handwritten_vs_triton_fused']}"
                 if hw_fused is not None else ""))
    return {"dtype": dtype_str, "entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-dram", action="store_true")
    parser.add_argument("--with-handwritten", action="store_true")
    arguments = parser.parse_args()

    # DSL 实现模块按 backend 子目录动态导入（cuda/ 与 musa/ 同名同结构）
    global rmsnorm, fused_add_rmsnorm_kernel
    impl_dir = HERE / "implementations" / arguments.backend
    if str(impl_dir) not in sys.path:
        sys.path.insert(0, str(impl_dir))
    from rmsnorm_triton import rmsnorm as _rmsnorm  # noqa: PLC0415
    from fused_triton import fused_add_rmsnorm_kernel as _fused_kernel  # noqa: PLC0415
    rmsnorm = _rmsnorm
    fused_add_rmsnorm_kernel = _fused_kernel

    device = resolve_device(arguments.backend)
    memory_api = getattr(torch, device.type)
    memory_api.empty_cache()
    if hasattr(memory_api, "reset_peak_memory_stats"):
        memory_api.reset_peak_memory_stats()

    environment = device_environment_record(device)
    environment["triton"] = _triton_env()
    write_results_json(HERE / "results" / f"environment-{arguments.backend}.json",
                       {"schema_version": "kernel-chapter14.environment.v1",
                        "backend": arguments.backend, **environment})

    results = {
        "schema_version": "kernel-chapter14.benchmark.v1",
        "backend": arguments.backend,
        "environment": environment,
    }

    # 手写对照（后台构建先于本链完成；失败按 fallback 记录）
    hw_rms = None
    hw_fused = None
    if arguments.with_handwritten:
        try:
            from reference_runners import HandwrittenFused, HandwrittenRMSNorm  # noqa: PLC0415
            hw_rms = HandwrittenRMSNorm(arguments.backend)
            results["handwritten_rms_build_seconds"] = hw_rms.build_seconds
            hw_fused = HandwrittenFused(arguments.backend)
            results["handwritten_fused_build_seconds"] = hw_fused.build_seconds
            results["handwritten_warp_size"] = hw_fused._ensure_warp_size(device)
        except Exception as error:  # noqa: BLE001
            results["handwritten_runner_error"] = f"{type(error).__name__}: {str(error)[-400:]}"
            hw_rms = hw_fused = None

    # A. 主口径链
    results["triton_rmsnorm_fp16"] = triton_chain(arguments, device, torch.float16)
    results["triton_rmsnorm_fp32"] = triton_chain(arguments, device, torch.float32)
    results["composite_fp32"] = composite_chain(arguments, device, torch.float32)
    results["composite_fp16"] = composite_chain(arguments, device, torch.float16)

    # F. 手写同会话链（RMSNorm 主口径 + DRAM 区）
    if hw_rms is not None:
        for dtype, key in ((torch.float16, "handwritten_rmsnorm_fp16"),
                           (torch.float32, "handwritten_rmsnorm_fp32")):
            entries = []
            weight = seeded_weight(HIDDEN_SIZE, device, dtype)
            for rows in ROW_COUNTS:
                x = seeded_input(rows, HIDDEN_SIZE, device, dtype)
                entry = timed_op(arguments, device, rows, HIDDEN_SIZE, dtype,
                                 lambda: hw_rms.run(x, weight, RMS_NORM_EPS, 256),
                                 "handwritten-ch03")
                entry["tile"] = 256
                entries.append(entry)
            results[key] = entries
        # DRAM 区手写（fp32 16384/32768）
        entries = []
        weight = seeded_weight(HIDDEN_SIZE, device, torch.float32)
        for rows in DRAM_ROWS["fp32"]:
            x = seeded_input(rows, HIDDEN_SIZE, device, torch.float32)
            entry = timed_op(arguments, device, rows, HIDDEN_SIZE, torch.float32,
                             lambda: hw_rms.run(x, weight, RMS_NORM_EPS, 256),
                             "handwritten-ch03-DRAM")
            entry["tile"] = 256
            entry["working_set_bytes"] = rows * HIDDEN_SIZE * 4
            entry["dram_region"] = entry["working_set_bytes"] > L2_BYTES[device.type]
            entries.append(entry)
        results["handwritten_rmsnorm_fp32_dram"] = entries
        # 手写 Tile 扫描（2048×1024 fp16/fp32，与 ch03 SCAN_TILES 同口径重测）
        scan_results = []
        timer = OperatorTimer(device, warmup=arguments.warmup, repeats=50)
        for dtype in (torch.float16, torch.float32):
            weight = seeded_weight(HIDDEN_SIZE, device, dtype)
            x = seeded_input(SCAN_SHAPE["rows"], HIDDEN_SIZE, device, dtype)
            for tile in [64, 128, 256, 512, 1024]:
                entry = {"rows": SCAN_SHAPE["rows"], "hidden_size": HIDDEN_SIZE,
                         "dtype": "fp16" if dtype == torch.float16 else "fp32", "tile": tile}
                try:
                    timing = timer.measure(
                        lambda: hw_rms.run(x, weight, RMS_NORM_EPS, tile))  # noqa: E731
                    attach_bandwidth(entry, SCAN_SHAPE["rows"], HIDDEN_SIZE,
                                     2 if dtype == torch.float16 else 4, timing,
                                     device.type, "eager")
                except Exception as error:  # noqa: BLE001
                    entry["error"] = str(error)[:200]
                scan_results.append(entry)
                print(f"[hw-scan/{device.type}] {entry['dtype']} tile={tile}: "
                      f"p50={entry.get('p50_ms')}ms")
        results["handwritten_tile_scan"] = scan_results

    # B. Triton Tile 扫描
    if not arguments.skip_scan:
        results["triton_tile_scan"] = tile_scan_chain(arguments, device)

    # C. DRAM 区
    if not arguments.skip_dram:
        results["triton_dram_fp32"] = dram_chain(arguments, device, "fp32")
        results["triton_dram_fp16"] = dram_chain(arguments, device, "fp16")

    # E. 融合链
    results["fused_fp32"] = fused_chain(arguments, device, "fp32", hw_fused)
    results["fused_fp16"] = fused_chain(arguments, device, "fp16", hw_fused)

    output = HERE / "results" / f"benchmark-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(device)
    return 0


def _triton_env() -> dict:
    try:
        import triton  # noqa: PLC0415
        from triton.runtime import driver  # noqa: PLC0415
        try:
            target = repr(driver.active.get_current_target())
        except Exception as error:  # noqa: BLE001
            target = f"unavailable: {error}"
        return {"version": triton.__version__, "target": target}
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    raise SystemExit(main())
