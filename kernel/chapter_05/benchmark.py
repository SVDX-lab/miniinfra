#!/usr/bin/env python3
"""第 05 期性能测量：三 dtype 延迟/带宽对照 + 理论字节比例兑现实验（核心）。

口径（延续第 01/02 期）：
- 被测对象：(a) 手写 Kernel 三 dtype 编译路径（torch 扩展，acc=fp32），
  (b) PyTorch 组合表达式三 dtype（平台原生路径）；
- 合同 Shape：rows ∈ {1,4,128,512,2048} × H=1024 × dtype ∈ {fp32,fp16,bf16}；
- 计时：OperatorTimer（warmup + 事件对、末尾一次同步，P50/P95/mean/min）；
  GPU 另测 Graph Replay 口径，带宽取最优者并标注 mode；
- 理论字节 = (rows*H 读 x + rows*H 写 y + H 读 w) × itemsize
  （FP32:FP16:BF16 = 4:2:2 —— 同 Shape 下 FP16/BF16 的理论字节是 FP32 的一半）。

实验 3（理论字节比例是否兑现，大纲 §10.4 行 533）：
- byte_ratio_probe：L2 区（rows=2048, H=2048）与 DRAM 区（rows∈{8192,32768},
  H=2048，working set 134/537MB > 4090 L2=72MB / S4000 L2=24MB）三 dtype 各测；
- 同 Shape 跨 dtype：t_fp32/t_fp16 应≈2（若字节减半兑现为时间减半）；
- 同字节跨 Shape 对照：fp32@32768×2048（536,879,104 B）vs fp16@65536×2048
  （536,875,008 B）—— 理论字节几乎相同，DRAM-bound 时延迟应几乎相同；
- 先验（第 02 期 M-12）：S4000 的 FP16 路径瓶颈在转换/shared 管线而非 DRAM，
  字节减半不自动兑现为时间减半；4090 DRAM 区 fp16=fp32 带宽（ch03 实测 916.6
  vs 915.0 GB/s）—— 本实验把该对照扩到三 dtype。

Tensor Core 支持探针（"为什么选 FP16"的证据之一）：
- matmul_probe：M=N=K=4096 的 fp32/fp16/bf16 GEMM TFLOPS（库路径），
  对照第 18 期结论（WMMA 仅 fp16 输入）。

用法：
    python benchmark.py --backend cpu
    python benchmark.py --backend cuda
    python benchmark.py --backend musa --kernel-backend auto
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
    measure_with_graph,
    resolve_device,
    write_results_json,
)
from naive_rmsnorm import RMS_NORM_EPS  # noqa: E402

SCHEMA = "kernel-chapter06.benchmark.v1"
HIDDEN_SIZE = 1024
ROW_COUNTS = [1, 4, 128, 512, 2048]
DTYPES = ["fp32", "fp16", "bf16"]
TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# 达成率分母单源（主 agent 指示，第 01 期实测 D2D 拷贝峰值）
PEAK_BANDWIDTH_BYTES_PER_SEC = {"cuda": 918.8e9, "musa": 711.3e9, "cpu": None}
SPEC_BANDWIDTH_BYTES_PER_SEC = {"cuda": 1008e9, "musa": 768e9, "cpu": None}
PEAK_SOURCE = {"cuda": "chapter_01/performance 实测 D2D 拷贝峰值 918.8 GB/s",
               "musa": "chapter_01/performance 实测 D2D 拷贝峰值 711.3 GB/s", "cpu": "n/a"}
# 平台 L2 容量（C-API 实锤：4090=72MB ch03 L2 复核；S4000=24MB ch03 musa l2cache）
PLATFORM_L2_BYTES = {"cuda": 72 * 1024 * 1024, "musa": 24 * 1024 * 1024, "cpu": None}

DEFAULT_TILE = 256


def theoretical_bytes(rows: int, hidden_size: int, itemsize: int) -> int:
    return (rows * hidden_size * 2 + hidden_size) * itemsize


def seeded_input(rows: int, hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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


def make_kernel_runner(backend: str, preference: str, device: torch.device):
    import importlib.util

    if preference == "bench":
        raise RuntimeError("ch06 benchmark 不支持 bench 退路（理论字节实验走 ext 口径）")
    path = HERE / "implementations" / backend / "build_ext.py"
    spec = importlib.util.spec_from_file_location(f"build_ext_{backend}", path)
    module_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_loader)
    start = time.perf_counter()
    module = module_loader.get_module()
    build_seconds = time.perf_counter() - start
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(4, 128, device=device, dtype=dt)
        w = torch.randn(128, device=device, dtype=dt)
        module.rmsnorm(x, w, RMS_NORM_EPS, DEFAULT_TILE)
    return {"kind": "ext", "module": module, "build_seconds": build_seconds}


def annotate_bandwidth(entry: dict, device: torch.device, source: dict) -> None:
    if PEAK_BANDWIDTH_BYTES_PER_SEC[device.type] is None:
        return
    effective = entry["theoretical_bytes"] / (source["p50_ms"] / 1000.0)
    entry["effective_bandwidth_gb_s"] = round(effective / 1e9, 2)
    entry["peak_bandwidth_gb_s_measured"] = PEAK_BANDWIDTH_BYTES_PER_SEC[device.type] / 1e9
    entry["peak_bandwidth_gb_s_spec"] = SPEC_BANDWIDTH_BYTES_PER_SEC[device.type] / 1e9
    entry["peak_source"] = PEAK_SOURCE[device.type]
    entry["achievement_ratio"] = round(
        effective / PEAK_BANDWIDTH_BYTES_PER_SEC[device.type], 4)
    entry["achievement_ratio_spec"] = round(
        effective / SPEC_BANDWIDTH_BYTES_PER_SEC[device.type], 4)


def kernel_chain(arguments, device: torch.device, module, dtype_name: str) -> dict:
    dtype = TORCH_DTYPES[dtype_name]
    weight = seeded_weight(HIDDEN_SIZE, device, dtype)
    entries = []
    for rows in ROW_COUNTS:
        x = seeded_input(rows, HIDDEN_SIZE, device, dtype)
        repeats = repeats_for(rows)
        timer = OperatorTimer(device, warmup=arguments.warmup, repeats=repeats)
        operation = lambda: module.rmsnorm(x, weight, RMS_NORM_EPS, arguments.tile)  # noqa: E731
        timing = timer.measure(operation)
        timing["mode"] = "eager"
        graph_timing = measure_with_graph(device, warmup=3, repeats=repeats, operation=operation)
        itemsize = x.element_size()
        total_bytes = theoretical_bytes(rows, HIDDEN_SIZE, itemsize)
        entry = {
            "rows": rows, "hidden_size": HIDDEN_SIZE, "dtype": dtype_name,
            "acc": "fp32", "tile": arguments.tile, "theoretical_bytes": total_bytes,
            "working_set_mb": round((2 * rows * HIDDEN_SIZE + HIDDEN_SIZE) * itemsize / 2**20, 1),
        }
        entry.update(timing)
        if graph_timing:
            entry["graph_replay"] = graph_timing
        effective_source = timing
        if graph_timing and graph_timing["p50_ms"] < timing["p50_ms"]:
            effective_source = graph_timing
            entry["mode"] = "graph_replay"
        annotate_bandwidth(entry, device, effective_source)
        entry["bandwidth_mode"] = entry["mode"]
        entries.append(entry)
        print(f"[kernel/{arguments.backend}/{dtype_name}] rows={rows:<5} "
              f"p50={entry['p50_ms']:.3f}ms "
              + (f"有效带宽={entry['effective_bandwidth_gb_s']}GB/s "
                 f"达成率={entry['achievement_ratio']:.2%}（{entry['bandwidth_mode']}）"
                 if "achievement_ratio" in entry else ""))
        del x
        if device.type != "cpu":
            getattr(torch, device.type).empty_cache()
    return {"dtype": dtype_name, "entries": entries}


def byte_ratio_probe(arguments, device: torch.device, module) -> dict:
    """实验 3：理论字节比例（4:2:2）是否兑现为延迟/带宽比例。

    H=2048；rows ∈ {2048(L2 区), 8192(过渡), 32768(DRAM 区)}。
    附加同字节对照：fp32@32768 vs fp16@65536 vs bf16@65536（≈537MB）。
    """
    plans = [
        {"rows": 2048, "h": 2048, "region": "L2 区（<平台 L2）"},
        {"rows": 8192, "h": 2048, "region": "过渡区"},
        {"rows": 32768, "h": 2048, "region": "DRAM 区（>平台 L2）"},
        {"rows": 65536, "h": 2048, "region": "DRAM 区（同字节对照，仅 fp16/bf16）"},
    ]
    entries = []
    for plan in plans:
        for dtype_name in DTYPES:
            if plan["rows"] == 65536 and dtype_name == "fp32":
                continue  # fp32@65536 = 1074MB，仅 24G 卡可容；同字节对照由 32768 承担
            dtype = TORCH_DTYPES[dtype_name]
            itemsize = 4 if dtype_name == "fp32" else 2
            total = theoretical_bytes(plan["rows"], plan["h"], itemsize)
            x = seeded_input(plan["rows"], plan["h"], device, dtype)
            w = seeded_weight(plan["h"], device, dtype)
            timer = OperatorTimer(device, warmup=arguments.probe_warmup,
                                  repeats=arguments.probe_repeats)
            timing = timer.measure(lambda: module.rmsnorm(x, w, RMS_NORM_EPS, arguments.tile))  # noqa: E731
            entry = {
                "rows": plan["rows"], "hidden_size": plan["h"], "dtype": dtype_name,
                "region": plan["region"], "tile": arguments.tile,
                "theoretical_bytes": total,
                "working_set_mb": round((2 * plan["rows"] * plan["h"] + plan["h"]) * itemsize / 2**20, 1),
            }
            entry.update(timing)
            annotate_bandwidth(entry, device, timing)
            entries.append(entry)
            print(f"[byte-ratio/{arguments.backend}] {plan['rows']}x{plan['h']} {dtype_name} "
                  f"({entry['working_set_mb']}MB): p50={timing['p50_ms']:.3f}ms "
                  + (f"有效带宽={entry['effective_bandwidth_gb_s']}GB/s "
                     f"达成率={entry['achievement_ratio']:.2%}"
                     if "achievement_ratio" in entry else ""))
            del x, w
            if device.type != "cpu":
                getattr(torch, device.type).empty_cache()

    # ---- 分析：同 Shape 跨 dtype 时间比 vs 理论字节比 2.0 ----
    analysis = []
    for plan in plans[:-1]:
        by_dtype = {e["dtype"]: e for e in entries
                    if e["rows"] == plan["rows"] and e["hidden_size"] == plan["h"]}
        if not {"fp32", "fp16", "bf16"} <= set(by_dtype):
            continue
        t32, t16, tbf = (by_dtype["fp32"]["p50_ms"], by_dtype["fp16"]["p50_ms"],
                         by_dtype["bf16"]["p50_ms"])
        analysis.append({
            "rows": plan["rows"], "hidden_size": plan["h"], "region": plan["region"],
            "theoretical_byte_ratio_fp32_fp16": 2.0,
            "t_fp32_over_t_fp16": round(t32 / t16, 3),
            "t_fp32_over_t_bf16": round(t32 / tbf, 3),
            "t_bf16_over_t_fp16": round(tbf / t16, 3),
            "ratio_realized_fp32_fp16": round(t32 / t16 / 2.0, 3),
            "conclusion": "字节减半兑现为时间减半" if abs(t32 / t16 - 2.0) / 2.0 < 0.15
                          else "未兑现（瓶颈非纯 DRAM 字节）",
        })
    # 同字节跨 dtype/Shape 对照
    same_bytes = {}
    fp32_big = next((e for e in entries if e["rows"] == 32768 and e["dtype"] == "fp32"), None)
    fp16_big = next((e for e in entries if e["rows"] == 65536 and e["dtype"] == "fp16"), None)
    bf16_big = next((e for e in entries if e["rows"] == 65536 and e["dtype"] == "bf16"), None)
    if fp32_big and fp16_big:
        same_bytes["fp32_32768_vs_fp16_65536"] = {
            "bytes_fp32": fp32_big["theoretical_bytes"],
            "bytes_fp16": fp16_big["theoretical_bytes"],
            "bytes_ratio": round(fp32_big["theoretical_bytes"] / fp16_big["theoretical_bytes"], 5),
            "t_fp32_ms": fp32_big["p50_ms"], "t_fp16_ms": fp16_big["p50_ms"],
            "t_ratio": round(fp32_big["p50_ms"] / fp16_big["p50_ms"], 3),
        }
    if bf16_big and fp16_big:
        same_bytes["bf16_65536_vs_fp16_65536"] = {
            "bytes_equal": True,
            "t_bf16_ms": bf16_big["p50_ms"], "t_fp16_ms": fp16_big["p50_ms"],
            "t_ratio": round(bf16_big["p50_ms"] / fp16_big["p50_ms"], 3),
        }
    print(f"[byte-ratio/{arguments.backend}] 分析: {json.dumps(analysis, ensure_ascii=False)}")
    print(f"[byte-ratio/{arguments.backend}] 同字节对照: {json.dumps(same_bytes, ensure_ascii=False)}")
    return {
        "plans": plans, "tile": arguments.tile,
        "probe_warmup": arguments.probe_warmup, "probe_repeats": arguments.probe_repeats,
        "platform_l2_bytes": PLATFORM_L2_BYTES[device.type],
        "entries": entries, "analysis": analysis, "same_bytes": same_bytes,
    }


def matmul_probe(arguments, device: torch.device) -> dict:
    """Tensor Core 支持探针：M=N=K=4096 GEMM 三 dtype TFLOPS（库路径）。"""
    entries = []
    size = 4096
    for dtype_name in DTYPES:
        dtype = TORCH_DTYPES[dtype_name]
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)
        timer = OperatorTimer(device, warmup=3, repeats=10)
        timing = timer.measure(lambda: torch.mm(a, b))  # noqa: E731
        flops = 2.0 * size ** 3
        entry = {"dtype": dtype_name, "size": size, "p50_ms": timing["p50_ms"],
                 "min_ms": timing["min_ms"],
                 "tflops_p50": round(flops / (timing["p50_ms"] / 1000.0) / 1e12, 2)}
        entries.append(entry)
        print(f"[matmul/{arguments.backend}] {dtype_name}@{size}³: "
              f"p50={timing['p50_ms']:.3f}ms tflops={entry['tflops_p50']}")
        del a, b
        if device.type != "cpu":
            getattr(torch, device.type).empty_cache()
    return {"size": size, "entries": entries,
            "note": "对照第 18 期：WMMA 仅 fp16 输入（bf16 fragment 不支持）——"
                    "本探针给库路径的三 dtype 算力对照"}


def composite_chain(arguments, device: torch.device, dtype_name: str) -> dict:
    dtype = TORCH_DTYPES[dtype_name]
    weight = seeded_weight(HIDDEN_SIZE, device, dtype)
    entries = []
    for rows in ROW_COUNTS:
        x = seeded_input(rows, HIDDEN_SIZE, device, dtype)
        repeats = repeats_for(rows) if device.type != "cpu" else min(repeats_for(rows), 20)
        timer = OperatorTimer(device, warmup=arguments.warmup, repeats=repeats)
        operation = lambda: weight * (x.float() * torch.rsqrt(  # noqa: E731
            x.float().pow(2).mean(-1, keepdim=True) + RMS_NORM_EPS)).to(dtype)
        timing = timer.measure(operation)
        itemsize = x.element_size()
        total_bytes = theoretical_bytes(rows, HIDDEN_SIZE, itemsize)
        entry = {"rows": rows, "hidden_size": HIDDEN_SIZE, "dtype": dtype_name,
                 "theoretical_bytes": total_bytes}
        entry.update(timing)
        graph_timing = measure_with_graph(device, warmup=3, repeats=repeats,
                                          operation=operation)
        if graph_timing:
            entry["graph_replay"] = graph_timing
        effective_source = timing
        entry["mode"] = "eager"
        if graph_timing and graph_timing["p50_ms"] < timing["p50_ms"]:
            effective_source = graph_timing
            entry["mode"] = "graph_replay"
        annotate_bandwidth(entry, device, effective_source)
        entry["bandwidth_mode"] = entry["mode"]
        entries.append(entry)
        print(f"[composite/{arguments.backend}/{dtype_name}] rows={rows:<5} "
              f"p50={entry['p50_ms']:.3f}ms "
              + (f"达成率={entry['achievement_ratio']:.2%}" if "achievement_ratio" in entry else ""))
        del x
        if device.type != "cpu":
            getattr(torch, device.type).empty_cache()
    return {"dtype": dtype_name, "entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--kernel-backend", default="auto")
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--probe-warmup", type=int, default=5)
    parser.add_argument("--probe-repeats", type=int, default=30)
    parser.add_argument("--skip-probe", action="store_true")
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend) if arguments.backend != "cpu" else torch.device("cpu")
    if device.type != "cpu":
        memory_api = getattr(torch, device.type)
        memory_api.empty_cache()
        if hasattr(memory_api, "reset_peak_memory_stats"):
            memory_api.reset_peak_memory_stats()

    environment = device_environment_record(device)
    write_results_json(HERE / "results" / f"environment-{arguments.backend}.json",
                       {"schema_version": "kernel-chapter06.environment.v1",
                        "backend": arguments.backend, **environment})

    results = {
        "schema_version": SCHEMA,
        "backend": arguments.backend,
        "kernel_tile_default": arguments.tile,
        "platform_l2_bytes": PLATFORM_L2_BYTES[device.type],
        "environment": environment,
    }
    for dtype_name in DTYPES:
        results[f"composite_{dtype_name}"] = composite_chain(arguments, device, dtype_name)

    if arguments.backend in ("cuda", "musa"):
        kernel_state = make_kernel_runner(arguments.backend, arguments.kernel_backend, device)
        results["kernel_runner"] = kernel_state["kind"]
        results["kernel_build_seconds"] = round(kernel_state["build_seconds"], 1)
        for dtype_name in DTYPES:
            results[f"kernel_{dtype_name}"] = kernel_chain(arguments, device,
                                                           kernel_state["module"], dtype_name)
        if not arguments.skip_probe:
            results["byte_ratio_probe"] = byte_ratio_probe(arguments, device,
                                                           kernel_state["module"])
            results["matmul_probe"] = matmul_probe(arguments, device)

    output = HERE / "results" / f"benchmark-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
