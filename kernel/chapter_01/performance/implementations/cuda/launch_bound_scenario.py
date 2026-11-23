#!/usr/bin/env python3
"""launch-bound 与 memory-bound 场景对比扫描（第 02 期建立，第 10 期复用）。

场景：最简单的逐元素算子 y = x + 1.0，从 16 个元素扫到 64M 个元素。

每个尺寸测量四种口径：
1. tight-loop 紧循环：一次性发射 N 次再同步，单次耗时 ≈ max(CPU 发射吞吐, GPU 执行)；
2. event-pair：事件对口径（第 01 期口径）。**注意（本期实测发现）**：当 GPU
   排空队列（launch-bound 区）时，事件对测到的是 ~15-24µs 的"地板值"而非
   纯 GPU 执行时间——事件口径只在 GPU 忙碌（单 op 时间 > 发射间隔）时可信；
3. graph-replay：单 op 捕获为图后回放，per-op = 回放跨度/次数（发射连续，
   不受 CPU 发射间隔影响）。CUDA 上可靠；MUSA 上 Graph 回放本身有开销
   （ch01 M-02：回放慢于 eager），数字如实记录并标注可信度；
4. sync-per-iter：每次发射后同步，含完整往返（发射→执行→同步）。

判定（第 10 期复用契约）：
- launch_throughput_us = 三个最小尺寸 tight-loop 的中位数（CPU 发射吞吐）；
- tight_loop_us > 1.2 × launch_throughput_us → 该尺寸已进入 GPU-bound 区
  （时间开始随尺寸增长）；否则处于 launch-bound 区（时间与尺寸无关）；
- 分界点 = 最后一个 launch-bound 尺寸 / 第一个 GPU-bound 尺寸。

用法：
    python launch_bound_scenario.py                # 默认 backend=cuda（本平台版）
    python launch_bound_scenario.py --backend musa # 可覆盖
输出：results/launch-bound-{backend}.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# 导入性能诊断公式及第一期共用的评测工具。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiment_utils import (
    backend_sync,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from roofline import per_size_launch_bound

SCHEMA_VERSION = "kernel-chapter02.launch-bound.v2"
DEFAULT_BACKEND = "cuda"

# 第 10 期复用契约：尺寸梯子（元素数）与迭代规则
ELEMENT_COUNTS = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
WARMUP = 20


def iters_rule(elements: int) -> int:
    if elements <= 65536:
        return 1000
    if elements <= 1048576:
        return 200
    return 50


device_used: torch.device  # 模块级引用，tight_loop_us 同步用


def tight_loop_us(operation, iters: int, rounds: int = 5) -> float:
    """紧循环单次耗时（µs）：发射 N 次后同步一次，取各轮中位数。"""
    per_round = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(iters):
            operation()
        backend_sync(device_used)
        per_round.append((time.perf_counter() - start) * 1e6 / iters)
    return sorted(per_round)[rounds // 2]


def graph_replay_us_per_op(module, operation, iters: int) -> dict | None:
    """单 op 捕获为 Graph 后回放：per-op = 事件对跨度/回放次数。

    回放之间发射连续（graph.replay 本身 ~µs 级），GPU 队列不排空，
    测到的是"图回放口径的 per-op GPU 时间"（含图内 launch-to-launch 间隔）。
    """
    if device_used.type == "cpu":
        return None
    try:
        side_stream = module.Stream()
        current = module.current_stream()
        side_stream.wait_stream(current)
        with module.stream(side_stream):
            for _ in range(3):
                operation()
        current.wait_stream(side_stream)
        backend_sync(device_used)

        graph = module.CUDAGraph()
        operation()  # 捕获前的 eager 执行（分配输出缓冲）
        backend_sync(device_used)
        with module.graph(graph):
            operation()
        backend_sync(device_used)
        graph.replay()
        backend_sync(device_used)

        repeats = min(iters, 200)
        start = module.Event(enable_timing=True)
        end = module.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        backend_sync(device_used)
        span_us = start.elapsed_time(end) * 1000.0
        return {
            "per_op_us": round(span_us / repeats, 3),
            "repeats": repeats,
            "note": "图回放 per-op（含图内 launch 间隔）；MUSA 平台 Graph 开销大（ch01 M-02），数值仅供参考",
        }
    except Exception as error:  # noqa: BLE001 —— Graph 不可用时如实记录
        return {"error": f"{type(error).__name__}: {error}"}


def measure_size(device: torch.device, module, elements: int, itemsize: int) -> dict:
    global device_used
    device_used = device
    x = torch.arange(elements, device=device, dtype=torch.float32)
    if device.type != "cpu":
        x = x.to(torch.float16).clamp(-1000, 1000)  # arange 超出 fp16 范围前截断
    operation = lambda: x + 1.0  # noqa: E731 —— 单 Kernel 逐元素算子
    iters = iters_rule(elements)

    tight_us = tight_loop_us(operation, iters)
    # 事件对口径（保留地板值现象的数据；相邻发射重叠假设在排空区不成立）
    repeats = min(iters, 200)
    event_p50_us = None
    if device.type != "cpu":
        start_events = []
        end_events = []
        for _ in range(repeats):
            start_events.append(module.Event(enable_timing=True))
            end_events.append(module.Event(enable_timing=True))
        for index in range(repeats):
            start_events[index].record()
            operation()
            end_events[index].record()
        backend_sync(device)
        event_samples = sorted(
            start_events[i].elapsed_time(end_events[i]) * 1000.0 for i in range(repeats)
        )
        event_p50_us = event_samples[len(event_samples) // 2]

    sync_timing = None
    if device.type != "cpu":
        samples = []
        for _ in range(min(iters, 200)):
            start = time.perf_counter()
            operation()
            backend_sync(device)
            samples.append((time.perf_counter() - start) * 1e6)
        ordered = sorted(samples)
        sync_timing = {"p50_us": round(ordered[len(ordered) // 2], 3), "repeats": len(ordered)}

    graph_timing = graph_replay_us_per_op(module, operation, iters)

    bytes_moved = elements * itemsize * 2  # 读 x + 写 y
    result = {
        "elements": elements,
        "bytes_moved_per_op": bytes_moved,
        "iters": iters,
        "tight_loop_us_per_op": round(tight_us, 3),
        "event_gpu_p50_us": round(event_p50_us, 3) if event_p50_us is not None else None,
        "graph_replay": graph_timing,
        "sync_per_iter_p50_us": sync_timing,
        "tight_loop_bandwidth_gb_s": round(bytes_moved / (tight_us * 1e-6) / 1e9, 2),
    }
    if device.type != "cpu":
        result["classification"] = per_size_launch_bound(tight_us, event_p50_us or 0.0)
        result["classification"]["note"] = "gpu_us 取事件对口径（排空区有地板值，仅供参照；判定见 boundary.tight_constancy）"
    else:
        result["classification"] = {"launch_bound": None, "note": "CPU 无发射概念，仅作数量级参照"}
    print(
        f"[launch-bound/{device.type}] elem={elements:<9} "
        f"tight={result['tight_loop_us_per_op']:.2f}µs "
        f"event={result['event_gpu_p50_us'] or 0:.2f}µs "
        f"graph={((graph_timing or {}).get('per_op_us') if graph_timing else None) or -1:.2f}µs "
        f"tight带宽={result['tight_loop_bandwidth_gb_s']}GB/s"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["cpu", "cuda", "musa"])
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    module = getattr(torch, device.type, None)
    itemsize = 2 if device.type != "cpu" else 4

    if device.type != "cpu":
        module.empty_cache()

    sizes = [measure_size(device, module, count, itemsize) for count in ELEMENT_COUNTS]

    # 分界判定：发射吞吐 = 三个最小尺寸 tight 的中位数；tight > 1.2×吞吐 = 进入 GPU-bound
    launch_throughput = sorted(s["tight_loop_us_per_op"] for s in sizes[:3])[1]
    for entry in sizes:
        entry["envelope"] = (
            "launch-bound" if entry["tight_loop_us_per_op"] <= 1.2 * launch_throughput
            else "gpu-bound (time scales with size)"
        )
    launch_bound_sizes = [s["elements"] for s in sizes if s["envelope"] == "launch-bound"]
    gpu_bound_sizes = [s["elements"] for s in sizes if s["envelope"] != "launch-bound"]
    boundary = {
        "rule": "launch_throughput=最小3尺寸tight中位数；tight>1.2x吞吐 → GPU-bound；否则 launch-bound",
        "launch_throughput_us": round(launch_throughput, 3),
        "launch_bound_max_elements": max(launch_bound_sizes) if launch_bound_sizes else None,
        "first_gpu_bound_elements": min(gpu_bound_sizes) if gpu_bound_sizes else None,
        "sync_roundtrip_us_at_16": sizes[0]["sync_per_iter_p50_us"]["p50_us"] if sizes[0]["sync_per_iter_p50_us"] else None,
        "event_floor_us_note": "事件对口径在 launch-bound 区测到地板值（见各尺寸 event_gpu_p50_us），"
                               "不随尺寸变化——证明该区间 GPU 排空，事件口径失效",
    }
    print(f"[launch-bound/{arguments.backend}] 发射吞吐={boundary['launch_throughput_us']}µs，"
          f"launch-bound 最大尺寸={boundary['launch_bound_max_elements']}，"
          f"首个 GPU-bound 尺寸={boundary['first_gpu_bound_elements']}")

    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "operation": "y = x + 1.0（单 Kernel 逐元素）",
        "elements_ladder": ELEMENT_COUNTS,
        "warmup": WARMUP,
        "iters_rule": "≤65536:1000 / ≤1048576:200 / else 50",
        "sizes": sizes,
        "boundary": boundary,
        "environment": device_environment_record(device),
    }
    output = Path("results") / f"launch-bound-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
