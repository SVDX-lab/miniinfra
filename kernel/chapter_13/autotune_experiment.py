#!/usr/bin/env python3
"""第 13 期 auto-tuning 实验：triton.autotune 自动搜索 vs 手动 Tile 扫描。

实验设计（大纲实验清单第 4 项 + 验收问题 2）：
  主形状 2048×1024（ch03 tile_scan 同款），fp16 + fp32 两档。

  A. parity 手动扫描：BLOCK {64,128,256,512,1024} × num_warps=4
     —— 与第 02 期手动 TILE 扫描同一搜索空间同一 config 数（5）；
  B. wide 手动扫描：BLOCK {64,128,256,512,1024,2048} × num_warps {1,2,4,8}
     —— Triton 的额外维度（ch03 手写 TILE 是编译期 block 维度，num_warps
     由 TILE 决定；Triton 里两者独立，搜索空间扩大 4 倍）；
  C. SP 手动扫描：单遍 kernel BLOCK {1024, 2048} × num_warps {2,4,8,16}
     —— 生产写法（BLOCK >= H）的真实空间；
  D. autotune parity / wide / SP：同一空间的 triton.autotune 自动搜索，
     记录首调墙钟（含全部编译与测时）、选中配置、稳态 p50；
  E. 口径交叉核对：同一配置分别用 OperatorTimer（事件对）与
     triton.testing.do_bench（Triton 内部测时，默认清 L2 + 中位数）测量，
     记录两口径差异（autotune 的选择依据是 do_bench 口径）。

回答：autotune 找到的配置离（同空间）全局最优多远、搜索开销多少、
do_bench 口径与实验口径差多少。

用法：python autotune_experiment.py --backend cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import triton

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations"))

from experiment_utils import (  # noqa: E402
    OperatorTimer,
    backend_sync,
    resolve_device,
    write_results_json,
)

ROWS = 2048
HIDDEN = 1024
PEAK = {"cuda": 918.8e9, "musa": 711.3e9}


def theoretical_bytes(rows: int, hidden_size: int, itemsize: int) -> int:
    return (rows * hidden_size * 2 + hidden_size) * itemsize


def seeded(dtype, device):
    import numpy as np
    rng = np.random.default_rng(20260301 + ROWS * 100000 + HIDDEN)
    x = torch.from_numpy(rng.standard_normal((ROWS, HIDDEN), dtype=np.float32)).to(
        device=device, dtype=dtype)
    rng_w = np.random.default_rng(20260301 + HIDDEN)
    w = torch.from_numpy(rng_w.standard_normal(HIDDEN, dtype=np.float32)).to(
        device=device, dtype=dtype)
    return x, w


def config_label(config) -> str:
    kwargs = config.kwargs if hasattr(config, "kwargs") else config[0]
    num_warps = getattr(config, "num_warps", 4)
    return f"B={kwargs.get('BLOCK')},nw={num_warps}"


def manual_scan(arguments, device, dtype, tiles, num_warps_list, kernel_mode) -> dict:
    """手动逐配置扫描（OperatorTimer 口径 + do_bench 口径双记录）。"""
    x, w = seeded(dtype, device)
    eps = 1e-6
    timer = OperatorTimer(device, warmup=arguments.warmup, repeats=50)
    entries = []
    scan_start = time.perf_counter()
    for tile in tiles:
        for num_warps in num_warps_list:
            entry = {"block": tile, "num_warps": num_warps, "kernel": kernel_mode}
            try:
                start_compile = time.perf_counter()
                rmsnorm(x, w, eps, block_size=tile, num_warps=num_warps, looped=(kernel_mode == "looped"))
                backend_sync(device)
                entry["compile_plus_first_s"] = round(time.perf_counter() - start_compile, 4)
                timing = timer.measure(
                    lambda: rmsnorm(x, w, eps, block_size=tile, num_warps=num_warps,
                                    looped=(kernel_mode == "looped")))  # noqa: E731
                entry["p50_ms"] = timing["p50_ms"]
                entry["p95_ms"] = timing["p95_ms"]
                itemsize = 2 if dtype == torch.float16 else 4
                entry["effective_bandwidth_gb_s"] = round(
                    theoretical_bytes(ROWS, HIDDEN, itemsize)
                    / (timing["p50_ms"] / 1000.0) / 1e9, 2)
                entry["achievement_ratio"] = round(
                    entry["effective_bandwidth_gb_s"] * 1e9 / PEAK[device.type], 4)
                entry["do_bench_mean_ms"] = round(
                    float(triton.testing.do_bench(
                        lambda: rmsnorm(x, w, eps, block_size=tile, num_warps=num_warps,
                                        looped=(kernel_mode == "looped")))), 4)  # noqa: E731
            except Exception as error:  # noqa: BLE001
                entry["error"] = str(error)[:200]
            entries.append(entry)
            print(f"[manual/{device.type}/{kernel_mode}] {entry.get('block')}x"
                  f"nw{entry.get('num_warps')}: p50={entry.get('p50_ms')}ms "
                  f"do_bench={entry.get('do_bench_mean_ms')}ms "
                  f"ach={entry.get('achievement_ratio')}")
    scan_seconds = time.perf_counter() - scan_start
    valid = [e for e in entries if "p50_ms" in e]
    best = min(valid, key=lambda e: e["p50_ms"]) if valid else None
    best_do = min(valid, key=lambda e: e.get("do_bench_mean_ms", 9e9)) if valid else None
    print(f"[manual/{device.type}/{kernel_mode}] {len(valid)} 配置有效，"
          f"扫描墙钟 {scan_seconds:.1f}s，最优(eager) {best}，最优(do_bench) {best_do}")
    return {"space": f"{len(tiles)}x{len(num_warps_list)}", "kernel": kernel_mode,
            "entries": entries, "best": best, "best_do_bench": best_do,
            "scan_wall_seconds": round(scan_seconds, 1)}


def manual_scan_sp(arguments, device, dtype, configs) -> dict:
    """SP kernel 手动扫描（显式 num_warps）。"""
    x, w = seeded(dtype, device)
    eps = 1e-6
    timer = OperatorTimer(device, warmup=arguments.warmup, repeats=50)
    entries = []
    scan_start = time.perf_counter()
    for config in configs:
        tile = config.kwargs["BLOCK"]
        num_warps = config.num_warps
        entry = {"block": tile, "num_warps": num_warps, "kernel": "SP"}
        try:
            start_compile = time.perf_counter()
            rmsnorm(x, w, eps, num_warps=num_warps)
            backend_sync(device)
            entry["compile_plus_first_s"] = round(time.perf_counter() - start_compile, 4)
            timing = timer.measure(lambda: rmsnorm(x, w, eps, num_warps=num_warps))  # noqa: E731
            entry["p50_ms"] = timing["p50_ms"]
            entry["p95_ms"] = timing["p95_ms"]
            itemsize = 2 if dtype == torch.float16 else 4
            entry["effective_bandwidth_gb_s"] = round(
                theoretical_bytes(ROWS, HIDDEN, itemsize) / (timing["p50_ms"] / 1000.0) / 1e9, 2)
            entry["achievement_ratio"] = round(
                entry["effective_bandwidth_gb_s"] * 1e9 / PEAK[device.type], 4)
            entry["do_bench_mean_ms"] = round(
                float(triton.testing.do_bench(
                    lambda: rmsnorm(x, w, eps, num_warps=num_warps))), 4)  # noqa: E731
        except Exception as error:  # noqa: BLE001
            entry["error"] = str(error)[:200]
        entries.append(entry)
        print(f"[manual/{device.type}/SP] B{tile}xnw{num_warps}: p50={entry.get('p50_ms')}ms "
              f"do_bench={entry.get('do_bench_mean_ms')}ms")
    scan_seconds = time.perf_counter() - scan_start
    valid = [e for e in entries if "p50_ms" in e]
    best = min(valid, key=lambda e: e["p50_ms"]) if valid else None
    best_do = min(valid, key=lambda e: e.get("do_bench_mean_ms", 9e9)) if valid else None
    return {"space": f"{len(configs)}", "kernel": "SP", "entries": entries,
            "best": best, "best_do_bench": best_do, "scan_wall_seconds": round(scan_seconds, 1)}


def autotune_run(arguments, device, dtype, kernel_body, configs, label) -> dict:
    """autotune 包装：首调墙钟（含全配置编译+测时）→ 选中配置 → 稳态 p50。"""
    x, w = seeded(dtype, device)
    eps = 1e-6
    fn = _make_autotuned(kernel_body, configs)
    rows = x.shape[0]
    y = torch.empty_like(x)

    def launch():
        fn[(rows,)](x, w, y, HIDDEN, eps)

    start = time.perf_counter()
    launch()
    backend_sync(device)
    first_call_s = time.perf_counter() - start
    # 读选中配置（3.7 实测：Autotuner.best_config 直出 + configs_timings 逐配置样本）
    chosen = getattr(fn, "best_config", None)
    cache_attr_used = "best_config_attr" if chosen is not None else "none"
    if chosen is None:
        # 3.2 fork：无 best_config 属性，cache = {device: {key: Config}} 两层嵌套
        caches = getattr(fn, "cache", None) or {}
        cache_attr_used = "cache-descend"
        for entry in caches.values():
            if isinstance(entry, triton.Config):
                chosen = entry
                break
            if hasattr(entry, "values"):
                for inner in entry.values():
                    if isinstance(inner, triton.Config):
                        chosen = inner
                        break
                if chosen is not None:
                    break
        # configs_timings 在 3.2 缺失（3.7 才有）——记 None 由 JSON 如实反映
    search_timings = None
    try:
        timings = getattr(fn, "configs_timings", None)
        if timings:
            search_timings = [
                {"config": f"B={c.kwargs.get('BLOCK')},nw={getattr(c, 'num_warps', '?')}",
                 "min_ms": round(min(v), 4), "samples_ms": [round(s, 4) for s in v]}
                for c, v in timings.items()]
    except Exception as error:  # noqa: BLE001
        search_timings = [{"error": str(error)[:200]}]
    # 稳态计时（OperatorTimer 同口径）
    timing = OperatorTimer(device, warmup=arguments.warmup, repeats=50).measure(launch)
    itemsize = 2 if dtype == torch.float16 else 4
    steady = {
        "p50_ms": timing["p50_ms"], "p95_ms": timing["p95_ms"],
        "effective_bandwidth_gb_s": round(
            theoretical_bytes(ROWS, HIDDEN, itemsize) / (timing["p50_ms"] / 1000.0) / 1e9, 2),
        "achievement_ratio": round(
            theoretical_bytes(ROWS, HIDDEN, itemsize) / (timing["p50_ms"] / 1000.0) / 1e9
            / PEAK[device.type], 4),
    }
    # do_bench 口径（autotune 内部依据）
    try:
        import triton  # noqa: PLC0415
        do_bench_ms = triton.testing.do_bench(launch)
    except Exception as error:  # noqa: BLE001
        do_bench_ms = f"unavailable: {error}"
    result = {"label": label, "n_configs": len(configs),
              "first_call_incl_search_s": round(first_call_s, 2),
              "autotuner_bench_time_s": getattr(fn, "bench_time", None),
              "cache_attr_used": cache_attr_used,
              "chosen_config": config_label(chosen) if chosen is not None else None,
              "chosen_config_repr": repr(chosen) if chosen is not None else None,
              "search_per_config_timings": search_timings,
              "steady_state": steady,
              "do_bench_mean_ms": do_bench_ms if isinstance(do_bench_ms, float)
              else round(do_bench_ms, 4) if isinstance(do_bench_ms, (int, float)) else do_bench_ms}
    print(f"[autotune/{device.type}/{label}] 首调 {first_call_s:.1f}s "
          f"选中={result['chosen_config']} 稳态 p50={steady['p50_ms']}ms "
          f"ach={steady['achievement_ratio']:.2%} do_bench={result['do_bench_mean_ms']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    parser.add_argument("--warmup", type=int, default=10)
    arguments = parser.parse_args()

    # DSL 实现模块按 backend 子目录动态导入（cuda/ 与 musa/ 同名同结构）
    global PARITY_CONFIGS, SP_CONFIGS, WIDE_CONFIGS, _make_autotuned, rmsnorm, rmsnorm_fwd_kernel, rmsnorm_fwd_looped
    impl_dir = HERE / "implementations" / arguments.backend
    if str(impl_dir) not in sys.path:
        sys.path.insert(0, str(impl_dir))
    import rmsnorm_triton as _rt  # noqa: PLC0415
    PARITY_CONFIGS = _rt.PARITY_CONFIGS
    SP_CONFIGS = _rt.SP_CONFIGS
    WIDE_CONFIGS = _rt.WIDE_CONFIGS
    _make_autotuned = _rt._make_autotuned
    rmsnorm = _rt.rmsnorm
    rmsnorm_fwd_kernel = _rt.rmsnorm_fwd_kernel
    rmsnorm_fwd_looped = _rt.rmsnorm_fwd_looped

    device = resolve_device(arguments.backend)
    results = {"schema_version": "kernel-chapter14.autotune.v1",
               "backend": arguments.backend, "shape": f"{ROWS}x{HIDDEN}"}

    for dtype, dtype_name in ((torch.float16, "fp16"), (torch.float32, "fp32")):
        section = {}
        section["manual_parity"] = manual_scan(arguments, device, dtype,
                                               [64, 128, 256, 512, 1024], [4], "looped")
        section["manual_wide"] = manual_scan(arguments, device, dtype,
                                             [64, 128, 256, 512, 1024, 2048],
                                             [1, 2, 4, 8], "looped")
        section["manual_sp"] = manual_scan_sp(arguments, device, dtype, SP_CONFIGS)
        section["autotune_parity"] = autotune_run(arguments, device, dtype,
                                                  __import__("rmsnorm_triton").rmsnorm_fwd_looped,
                                                  PARITY_CONFIGS, "parity")
        section["autotune_wide"] = autotune_run(arguments, device, dtype,
                                                __import__("rmsnorm_triton").rmsnorm_fwd_looped,
                                                WIDE_CONFIGS, "wide")
        section["autotune_sp"] = autotune_run(arguments, device, dtype,
                                              rmsnorm_fwd_kernel, SP_CONFIGS, "SP")
        # 差距分析
        def gap(autotune_entry, manual_section):
            best = manual_section["best"]
            if best is None or autotune_entry["chosen_config"] is None:
                return None
            chosen = autotune_entry["steady_state"]["p50_ms"]
            best_p50 = best["p50_ms"]
            out = {"autotune_p50_ms": chosen, "manual_best_p50_ms": best_p50,
                   "manual_best_config": f"B={best['block']},nw={best['num_warps']}",
                   "gap_ratio_vs_manual_best": round(chosen / best_p50, 4) if best_p50 else None}
            # do_bench 口径（autotune 的原生选择依据）：搜索逐配置 min vs 手动扫描逐配置 min
            chosen_do = autotune_entry.get("search_per_config_timings")
            if chosen_do:
                chosen_min = min((r["min_ms"] for r in chosen_do if "min_ms" in r), default=None)
                manual_do = [e.get("do_bench_mean_ms") for e in manual_section["entries"]
                             if e.get("do_bench_mean_ms")]
                manual_min = min(manual_do) if manual_do else None
                if chosen_min and manual_min:
                    out["do_bench_chosen_min_ms"] = chosen_min
                    out["do_bench_manual_min_ms"] = manual_min
                    out["do_bench_gap_ratio"] = round(chosen_min / manual_min, 4)
            return out
        section["analysis"] = {
            "parity": gap(section["autotune_parity"], section["manual_parity"]),
            "wide": gap(section["autotune_wide"], section["manual_wide"]),
            "sp": gap(section["autotune_sp"], section["manual_sp"]),
        }
        results[dtype_name] = section

    # 汇总打印
    print("\n===== autotune 差距汇总 =====")
    for dtype_name in ("fp16", "fp32"):
        for space in ("parity", "wide", "sp"):
            analysis = results[dtype_name]["analysis"][space]
            if analysis:
                print(f"[{dtype_name}/{space}] autotune={analysis['autotune_p50_ms']}ms "
                      f"vs 手动最优 {analysis['manual_best_config']} "
                      f"{analysis['manual_best_p50_ms']}ms → 差距 "
                      f"{analysis['gap_ratio_vs_manual_best']}x")

    output = HERE / "results" / f"autotune-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
