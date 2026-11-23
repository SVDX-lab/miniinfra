#!/usr/bin/env python3
"""第 09 期实验 5（M-02 根因深挖）：Graph 成本模型矩阵 {K}×{shape} + 阶段拆解。

M-02 待解释：ch01 实测 MUSA Graph Replay 0.30ms 慢于 eager 0.17ms @2048×1024
（CUDA 相反快 3.5×）；ch02 实测 MUSA 1-op 图回放 25.5µs/op > eager 发射 16.7µs/op。

成本模型假设（本期用矩阵检验）：
  replay_span(K, shape) ≈ intercept(平台 per-replay 固定成本)
                        + slope(平台 per-kernel-node 边际成本) × K
                        + Σ kernel 执行时间
  - 若 intercept > eager 发射吞吐 → MUSA"回放即负收益"在小 K 段成立（根因①）
  - 若 slope ≫ CUDA 的 slope → 图内每节点调度贵（根因②）
  - 若两参数都正常但 eager 更快 → musaGraph 回放路径本身绕过了发射税优化（根因③）

矩阵：K ∈ {1, 8, 64} × shape ∈ {小=16 元素, 中=2048×1024, 大=2048×8192}（FP16）
每格测（10 轮中位数）：
  eager 链 per-iteration / graph 链回放 per-iteration（事件对）
  阶段拆解：warmup / capture+instantiate / first replay / steady replay dispatch(CPU)
同 K 连续捕获 3 张图的 capture 耗时抖动（实例化成本稳定性）。
CUDA 平台跑同矩阵 → 跨后端对照表（M-02 台账条目的双平台证据）。

输出：results/graph-matrix-{backend}.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]
for _path in (str(CHAPTER), str(CHAPTER / "implementations"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import (  # noqa: E402
    backend_module,
    backend_sync,
    device_environment_record,
    enqueue_wall_us,
    linear_fit,
    resolve_device,
    write_results_json,
)

SCHEMA_VERSION = "kernel-chapter10.graph-matrix.v1"
DEFAULT_BACKEND = "musa"
CHAIN_LENGTHS = [1, 8, 64]
SHAPES = {"small_16": 16, "mid_2048x1024": 2048 * 1024, "large_2048x8192": 2048 * 8192}


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def eager_chain_us(x: torch.Tensor, t: torch.Tensor, chain_k: int, device) -> float:
    def chain():
        for _ in range(chain_k):
            t.add_(1.0)

    samples = []
    for _ in range(5):
        t.copy_(x)
        backend_sync(device)
        start = time.perf_counter()
        for _ in range(200):
            chain()
        backend_sync(device)
        samples.append((time.perf_counter() - start) * 1e6 / 200)
    return median(samples)


def graph_chain_us(x: torch.Tensor, chain_k: int, device) -> dict:
    """捕获 K-op 图并测回放口径 + 全阶段拆解。"""
    module = backend_module(device)
    side_stream = module.Stream()
    current = module.current_stream()
    side_stream.wait_stream(current)
    t_warm = x.clone()
    t0 = time.perf_counter()
    with module.stream(side_stream):
        for _ in range(3):
            for _ in range(chain_k):
                t_warm.add_(1.0)
    current.wait_stream(side_stream)
    backend_sync(device)
    warmup_us = (time.perf_counter() - t0) * 1e6

    graph = module.CUDAGraph()
    t_static = x.clone()
    t0 = time.perf_counter()
    with module.graph(graph):
        for _ in range(chain_k):
            t_static.add_(1.0)
    capture_us = (time.perf_counter() - t0) * 1e6
    backend_sync(device)

    t0 = time.perf_counter()
    graph.replay()
    backend_sync(device)
    first_replay_us = (time.perf_counter() - t0) * 1e6

    event_cls = module.Event
    spans = []
    for _ in range(5):
        start = event_cls(enable_timing=True)
        end = event_cls(enable_timing=True)
        start.record()
        for _ in range(200):
            graph.replay()
        end.record()
        backend_sync(device)
        spans.append(start.elapsed_time(end) * 1000.0 / 200)
    replay_span_us = median(spans)

    dispatch = enqueue_wall_us(graph.replay, repeats=30, device=device)
    return {
        "graph": graph,
        "t_static": t_static,
        "warmup_us": round(warmup_us, 3),
        "capture_plus_instantiate_us": round(capture_us, 3),
        "first_replay_us": round(first_replay_us, 3),
        "replay_span_us": round(replay_span_us, 3),
        "replay_dispatch_cpu": dispatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["cuda", "musa"])
    parser.add_argument("--capture-repeat", type=int, default=3,
                        help="同 K 重复捕获次数（实例化抖动）")
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    module = backend_module(device)
    module.empty_cache()

    matrix = {}
    for shape_name, elements in SHAPES.items():
        x = torch.arange(elements, device=device, dtype=torch.float32)
        x = x.to(torch.float16).clamp(-100.0, 100.0)
        t_eager = x.clone()
        shape_entry = {"elements": elements}
        graph_points = []  # (K, replay_span_us per-iteration)
        eager_points = []
        for chain_k in CHAIN_LENGTHS:
            eager_us = eager_chain_us(x, t_eager, chain_k, device)
            eager_points.append((chain_k, eager_us))
            entry = {"eager_us_per_iter": round(eager_us, 3)}
            captures = []
            for repeat_index in range(arguments.capture_repeat):
                state = graph_chain_us(x, chain_k, device)
                if repeat_index == 0:
                    entry.update({k: v for k, v in state.items() if k not in ("graph", "t_static")})
                captures.append(state["capture_plus_instantiate_us"])
                if repeat_index == arguments.capture_repeat - 1:
                    graph_span = state["replay_span_us"]
                del state["graph"]
            entry["capture_us_repeats"] = captures
            entry["capture_us_jitter_pct"] = round(
                (max(captures) - min(captures)) / median(captures) * 100.0, 2)
            entry["graph_us_per_iter"] = graph_span
            graph_points.append((chain_k, graph_span))
            entry["graph_minus_eager_us_per_iter"] = round(graph_span - eager_us, 3)
            shape_entry[f"K{chain_k}"] = entry
            print(f"[matrix/{arguments.backend}] {shape_name} K={chain_k}: "
                  f"eager={eager_us:.2f}µs graph={state['replay_span_us']:.2f}µs "
                  f"差={state['replay_span_us'] - eager_us:+.2f}µs "
                  f"capture={median(captures) / 1000.0:.2f}ms 抖动={entry['capture_us_jitter_pct']}%")
        shape_entry["graph_fit"] = linear_fit(graph_points)
        shape_entry["eager_fit"] = linear_fit(eager_points)
        shape_entry["fit_note"] = "per-iteration µs = intercept + slope×K；intercept=每图回放固定成本，slope=每 kernel 节点边际成本"
        matrix[shape_name] = shape_entry

    # 跨 shape 的 slope/intercept 汇总（根因判别表）
    summary = {}
    for shape_name, entry in matrix.items():
        summary[shape_name] = {
            "graph_intercept_us": entry["graph_fit"]["intercept"],
            "graph_slope_us_per_node": entry["graph_fit"]["slope"],
            "graph_fit_r2": entry["graph_fit"]["r2"],
            "graph_interpretation": (
                "intercept vs eager 发射吞吐对照：intercept ≥ 吞吐 → 小 K 段回放负收益（根因①）；"
                "slope vs CUDA slope 对照：放大 → 图内节点调度贵（根因②）"),
        }
    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "chain_lengths": CHAIN_LENGTHS,
        "shapes": SHAPES,
        "matrix": matrix,
        "root_cause_summary": summary,
        "m02_reference": {
            "ch01": "MUSA Graph Replay 0.30ms 慢于 eager 0.17ms @2048×1024（CUDA 快 3.5×）",
            "ch02": "MUSA 1-op 图回放 25.5µs/op > eager 发射 16.7µs/op；CUDA 4.2 < 9.3",
        },
        "environment": device_environment_record(device),
    }
    output = Path("results") / f"graph-matrix-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
