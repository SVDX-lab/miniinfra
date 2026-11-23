#!/usr/bin/env python3
"""第 09 期实验 1+2：launch-bound 场景 eager vs Graph + 正常 Shape 抽查。

复用第 01 期 launch-bound 契约 v2 的尺寸梯子（12 档，fp16，y=x+1.0 语义），
每个尺寸测 K∈{1,8,64} 的原地 add 链两种组织方式：

  eager：for _ in range(K): t.add_(1.0)（逐 op 发射，紧循环发射 N 次再同步）
  graph ：把 K 次 add_ 录成一张图，回放（1 次 graph launch 提交整链）

每个尺寸输出：
  - eager_chain_us_per_op / graph_chain_us_per_op（per-op = 每迭代耗时 / K）
  - launch_tax_cut_us = eager − graph（被 Graph 砍掉的发射税，含图内残余间隙）
  - graph_replay_dispatch_us（单次 replay 的 CPU 发射成本）
  - eager_enqueue_us_per_chain（K 次 eager 发射的 CPU 成本）
  - replay_span(K) 线性拟合（intercept=per-replay 固定成本，slope=per-node 边际成本）
  - ch02 契约 v2 分界（K=1 eager）与 Graph 分界（K=1 graph）对照——验证
    "Graph 应把 launch-bound 边界向大尺寸推移"

尾部 normal-shape 抽查（实验 2 两档之 RMSNorm 档）：
  rows=2048 × H=1024 FP16 composite RMSNorm 链（ch05 同语义）与注册融合 op
  （ch05 注册版）各测 eager vs graph——正常 shape 下收益应显著缩小。

输出：results/launch-bound-graph-{backend}.json
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
    GraphCaptureCompat,
    backend_module,
    backend_sync,
    device_environment_record,
    enqueue_wall_us,
    linear_fit,
    resolve_device,
    write_results_json,
)

SCHEMA_VERSION = "kernel-chapter10.launch-bound-graph.v1"
DEFAULT_BACKEND = "cuda"

ELEMENT_COUNTS = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576,
                  4194304, 16777216, 67108864]  # ch02 契约 v2 原梯子
CHAIN_LENGTHS = [1, 8, 64]
WARMUP = 20
X_CLAMP = 100.0  # 原位累加防 fp16 溢出（|x|≤100，最多 +K×iters 上限受控）


def base_iters(elements: int) -> int:
    if elements <= 65536:
        return 1000
    if elements <= 1048576:
        return 200
    return 50


def chain_iters(elements: int, chain_k: int) -> int:
    """链迭代数 = 基础迭代数 / K（总发射量与 K=1 档对齐，下限 20）。"""
    return max(20, base_iters(elements) // chain_k)


def tight_rounds_us(operation, iters: int, rounds: int = 5) -> list[float]:
    """每轮：发射 iters 次再同步一次；返回每轮 per-iteration 微秒列表。"""
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(iters):
            operation()
        backend_sync(DEVICE)
        samples.append((time.perf_counter() - start) * 1e6 / iters)
    return samples


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def graph_replay_span_us(replay_call, repeats: int, rounds: int = 5) -> list[float]:
    """图回放口径：R 次回放连续发射后同步，事件对测 GPU 跨度，返回 per-replay µs 列表。"""
    module = backend_module(DEVICE)
    event_cls = module.Event
    samples = []
    for _ in range(rounds):
        start = event_cls(enable_timing=True)
        end = event_cls(enable_timing=True)
        start.record()
        for _ in range(repeats):
            replay_call()
        end.record()
        backend_sync(DEVICE)
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    return samples


def capture_chain_graph(x: torch.Tensor, chain_k: int):
    """把 K 次 t.add_(1.0) 捕获成图（静态 buffer 原位链，无捕获期分配）。

    返回 (graph, 静态输出 buffer, capture 阶段耗时 ms)。捕获前旁路流 warmup
    （torch Graph 捕获规范：目标算子须先执行过）。
    """
    module = backend_module(DEVICE)
    side_stream = module.Stream()
    current = module.current_stream()
    side_stream.wait_stream(current)

    t_warm = x.clone()
    with module.stream(side_stream):
        for _ in range(3):
            for _ in range(chain_k):
                t_warm.add_(1.0)
    current.wait_stream(side_stream)
    backend_sync(DEVICE)

    graph = module.CUDAGraph()
    t_static = x.clone()
    t0 = time.perf_counter()
    with module.graph(graph):
        for _ in range(chain_k):
            t_static.add_(1.0)
    capture_ms = (time.perf_counter() - t0) * 1000.0
    backend_sync(DEVICE)
    return graph, t_static, capture_ms


def assert_graph_matches_eager(out: torch.Tensor, reference: torch.Tensor, label: str) -> dict:
    """图回放输出 vs eager 输出：优先位级一致（同 kernel 同序），否则容差内记差值。"""
    if torch.equal(out, reference):
        return {"equal_bitwise": True, "max_abs_error": 0.0}
    diff = (out.float() - reference.float()).abs().max().item()
    if diff > 1e-2:
        raise RuntimeError(f"[graph-correctness] {label} 图输出与 eager 偏差 {diff} 超容差")
    return {"equal_bitwise": False, "max_abs_error": diff}


def measure_size(elements: int, itemsize: int) -> dict:
    x = torch.arange(elements, device=DEVICE, dtype=torch.float32)
    x = x.to(torch.float16).clamp(-X_CLAMP, X_CLAMP)
    eager_t = x.clone()

    entry = {"elements": elements, "bytes_moved_per_op": elements * itemsize * 2}
    chain_points_graph = []   # (K, per-iteration µs) 拟合用
    chain_points_eager = []

    for chain_k in CHAIN_LENGTHS:
        iters = chain_iters(elements, chain_k)

        def eager_chain():
            for _ in range(chain_k):
                eager_t.add_(1.0)

        # eager 链：紧循环（发射 N 次同步一次）
        eager_t.copy_(x)
        eager_us_iter = median(tight_rounds_us(eager_chain, iters))
        eager_us_per_op = eager_us_iter / chain_k
        chain_points_eager.append((chain_k, eager_us_iter))

        # eager 链 CPU 发射成本（enqueue-only）
        eager_enqueue = enqueue_wall_us(eager_chain, repeats=min(10, iters), device=DEVICE)

        # graph 链：捕获 K-op 图，回放（正确性：与逐次 eager 链位级一致）
        eager_t.copy_(x)
        for _ in range(chain_k):
            eager_t.add_(1.0)
        eager_chain_reference = eager_t.clone()
        graph, t_static, capture_ms = capture_chain_graph(x, chain_k)
        graph.replay()
        backend_sync(DEVICE)
        entry_correct = assert_graph_matches_eager(t_static, eager_chain_reference,
                                                   f"chain K={chain_k} elem={elements}")
        graph_us_iter = median(graph_replay_span_us(graph.replay, min(iters, 300)))
        graph_us_per_op = graph_us_iter / chain_k
        chain_points_graph.append((chain_k, graph_us_iter))
        replay_dispatch = enqueue_wall_us(graph.replay, repeats=20, device=DEVICE)

        entry[f"K{chain_k}"] = {
            "chain_iters": iters,
            "eager_us_per_iter": round(eager_us_iter, 3),
            "eager_us_per_op": round(eager_us_per_op, 3),
            "graph_us_per_iter": round(graph_us_iter, 3),
            "graph_us_per_op": round(graph_us_per_op, 3),
            "launch_tax_cut_us_per_op": round(eager_us_per_op - graph_us_per_op, 3),
            "graph_correctness": entry_correct,
            "eager_enqueue_cpu": eager_enqueue,
            "graph_replay_dispatch_cpu": replay_dispatch,
            "capture_plus_instantiate_ms": round(capture_ms, 3),
        }
        del graph

    entry["graph_span_fit"] = linear_fit(chain_points_graph)
    entry["eager_span_fit"] = linear_fit(chain_points_eager)
    entry["graph_span_fit"]["note"] = "graph per-iteration µs = intercept(回放固定成本) + slope(每节点边际) × K"
    entry["eager_span_fit"]["note"] = "eager per-iteration µs（紧循环 GPU-bound 时≈执行和；launch-bound 时被 CPU 吞吐拉平，拟合仅参照）"

    # K=1 档 ch02 契约 v2 口径补齐：sync-per-iter 与事件对地板
    iters1 = base_iters(elements)

    def single_op():
        eager_t.add_(1.0)

    eager_t.copy_(x)
    sync_samples = []
    for _ in range(min(iters1, 200)):
        start = time.perf_counter()
        single_op()
        backend_sync(DEVICE)
        sync_samples.append((time.perf_counter() - start) * 1e6)
    entry["sync_per_iter_p50_us"] = round(median(sync_samples), 3)

    module = backend_module(DEVICE)
    eager_t.copy_(x)
    reps = min(iters1, 200)
    starts, ends = [], []
    for _ in range(reps):
        starts.append(module.Event(enable_timing=True))
        ends.append(module.Event(enable_timing=True))
    for index in range(reps):
        starts[index].record()
        single_op()
        ends[index].record()
    backend_sync(DEVICE)
    event_p50 = median([starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(reps)])
    entry["event_gpu_p50_us"] = round(event_p50, 3)
    entry["event_floor_note"] = "事件对口径在 GPU 排空区为地板值（ch02 发现），仅作参照"

    print(f"[launch-bound-graph/{DEVICE.type}] elem={elements:<9} "
          f"K1: eager={entry['K1']['eager_us_per_op']:.2f} graph={entry['K1']['graph_us_per_op']:.2f} "
          f"K8: eager={entry['K8']['eager_us_per_op']:.2f} graph={entry['K8']['graph_us_per_op']:.2f} "
          f"K64: eager={entry['K64']['eager_us_per_op']:.2f} graph={entry['K64']['graph_us_per_op']:.2f} "
          f"fit(图形) slope={entry['graph_span_fit']['slope']:.2f} intercept={entry['graph_span_fit']['intercept']:.2f}")
    return entry


def classify_boundary(sizes: list[dict], key: str) -> dict:
    """ch02 契约 v2 分界规则应用于指定 per-op 口径（K=1）。"""
    per_op = [s[f"K1"][key] for s in sizes]
    throughput = median(per_op[:3])
    labels = []
    for value in per_op:
        labels.append("launch-bound" if value <= 1.2 * throughput else "gpu-bound")
    launch_max = max((s["elements"] for s, lab in zip(sizes, labels) if lab == "launch-bound"), default=None)
    first_gpu = min((s["elements"] for s, lab in zip(sizes, labels) if lab != "launch-bound"), default=None)
    return {
        "rule": "最小3尺寸中位数=吞吐；>1.2×吞吐 → GPU-bound（ch02 契约 v2）",
        "throughput_us": round(throughput, 3),
        "launch_bound_max_elements": launch_max,
        "first_gpu_bound_elements": first_gpu,
        "labels": {str(s["elements"]): lab for s, lab in zip(sizes, labels)},
    }


def normal_shape_spot(router, compat=None) -> dict:
    """实验 2 的 RMSNorm 档：rows=2048×H=1024 FP16，composite 链与注册融合 op 各 eager vs graph。"""
    rows, hidden = 2048, 1024
    seed = 20261121 + rows * 100000 + hidden
    generator = torch.Generator().manual_seed(seed)
    r = torch.randn(rows, hidden, generator=generator).to(device=DEVICE, dtype=torch.float16)
    h = torch.randn(rows, hidden, generator=generator).to(device=DEVICE, dtype=torch.float16)
    w = torch.randn(hidden, generator=generator).to(device=DEVICE, dtype=torch.float16)
    eps = 1e-6
    module = backend_module(DEVICE)

    def composite_chain():
        t = r + h
        float_states = t.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + eps)
        return w * normalized.to(t.dtype)

    def fused_call():
        router.fused_add_rmsnorm(h, r, w, eps)

    def timed(operation, use_graph: bool, compat=None) -> dict:
        # warmup（图捕获前旁路流预热）
        side_stream = module.Stream()
        current = module.current_stream()
        side_stream.wait_stream(current)
        with module.stream(side_stream):
            for _ in range(3):
                operation()
        current.wait_stream(side_stream)
        backend_sync(DEVICE)

        repeats = 100
        if not use_graph:
            starts, ends = [], []
            for _ in range(repeats):
                starts.append(module.Event(enable_timing=True))
                ends.append(module.Event(enable_timing=True))
            for index in range(repeats):
                starts[index].record()
                operation()
                ends[index].record()
            backend_sync(DEVICE)
            samples = [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(repeats)]
            mode = "eager_event_pair"
        else:
            graph = module.CUDAGraph()
            if compat is not None and compat.supported:
                with compat.capture(graph):
                    operation()
            else:
                with module.graph(graph):
                    operation()
            backend_sync(DEVICE)
            graph.replay()
            backend_sync(DEVICE)
            starts = module.Event(enable_timing=True)
            ends = module.Event(enable_timing=True)
            samples = []
            for _ in range(5):
                starts.record()
                for _ in range(repeats):
                    graph.replay()
                ends.record()
                backend_sync(DEVICE)
                samples.append(starts.elapsed_time(ends) * 1000.0 / repeats)
            del graph
            mode = "graph_replay"
        ordered = sorted(samples)
        return {
            "mode": mode,
            "p50_us": round(ordered[len(ordered) // 2], 3),
            "p95_us": round(ordered[min(len(ordered) - 1, int(round(len(ordered) * 0.95)) - 1)], 3),
            "repeats": repeats,
        }

    def correctness_composite() -> dict:
        """composite 链 out-of-place：捕获后单次回放 vs eager 同输入输出，应位级一致。"""
        eager_reference = composite_chain()
        backend_sync(DEVICE)
        graph = module.CUDAGraph()
        with module.graph(graph):
            captured = composite_chain()
        backend_sync(DEVICE)
        graph.replay()
        backend_sync(DEVICE)
        outcome = assert_graph_matches_eager(captured, eager_reference, "spot:composite")
        del graph
        return outcome

    def correctness_fused() -> dict:
        """融合 op in-place：静态 buffer 复位→捕获→回放，vs 新克隆 buffer 上的 eager 调用。"""
        rs = r.clone()
        hs = h.clone()

        def fused_static():
            router.fused_add_rmsnorm(hs, rs, w, eps)

        rc = r.clone()
        hc = h.clone()
        router.fused_add_rmsnorm(hc, rc, w, eps)
        eager_output = hc.clone()  # op 契约：归一化输出写进 input buffer
        backend_sync(DEVICE)
        graph = module.CUDAGraph()
        with module.graph(graph):
            fused_static()
        backend_sync(DEVICE)
        rs.copy_(r)
        hs.copy_(h)
        graph.replay()
        backend_sync(device)
        outcome = assert_graph_matches_eager(hs, eager_output, "spot:fused")
        del graph
        return outcome

    def correctness_fused_compat(compat) -> dict:
        """融合 op in-place（musa 变体 2 兼容路径）：显式捕获流注入 + 复位回放对账。"""
        rs = r.clone()
        hs = h.clone()

        def fused_static():
            router.fused_add_rmsnorm(hs, rs, w, eps)

        rc = r.clone()
        hc = h.clone()
        router.fused_add_rmsnorm(hc, rc, w, eps)
        eager_output = hc.clone()
        backend_sync(DEVICE)
        graph = module.CUDAGraph()
        with compat.capture(graph):
            fused_static()
        backend_sync(DEVICE)
        rs.copy_(r)
        hs.copy_(h)
        graph.replay()
        backend_sync(DEVICE)
        outcome = assert_graph_matches_eager(hs, eager_output, "spot:fused(compat)")
        del graph
        return outcome

    result = {"rows": rows, "hidden_size": hidden, "dtype": "float16", "seed": seed}
    eager = timed(composite_chain, use_graph=False)
    graph = timed(composite_chain, use_graph=True)
    check = eager["p50_us"] / graph["p50_us"]
    result["composite_rmsnorm_chain"] = {
        "eager_event_pair": eager,
        "graph_replay": graph,
        "graph_gain_ratio_p50": round(check, 4),
        "launch_tax_cut_us": round(eager["p50_us"] - graph["p50_us"], 3),
        "correctness": correctness_composite(),
    }
    eager_f = timed(fused_call, use_graph=False)
    graph_f = timed(fused_call, use_graph=True, compat=compat)
    check_f = eager_f["p50_us"] / graph_f["p50_us"]
    result["fused_op_registered"] = {
        "eager_event_pair": eager_f,
        "graph_replay": graph_f,
        "graph_gain_ratio_p50": round(check_f, 4),
        "launch_tax_cut_us": round(eager_f["p50_us"] - graph_f["p50_us"], 3),
        "correctness": (correctness_fused_compat(compat)
                        if compat is not None and compat.explicit_stream_mode
                        else correctness_fused()),
    }
    for name in ("composite_rmsnorm_chain", "fused_op_registered"):
        entry = result[name]
        print(f"[spot/{DEVICE.type}] {name}: eager={entry['eager_event_pair']['p50_us']:.2f}µs "
              f"graph={entry['graph_replay']['p50_us']:.2f}µs 收益={entry['graph_gain_ratio_p50']:.3f}× "
              f"砍掉={entry['launch_tax_cut_us']:.2f}µs "
              f"对账={entry['correctness']}")
    result["fused_call_mutation_note"] = (
        "fused_add_rmsnorm 为原地双 buffer op，计时循环中 r/h 数值漂移但 kernel 工作量与"
        "数值路径不变（ch05 layer1 同口径约定）；正确性对账独立完成（静态 buffer 复位后单次回放）")
    return result


DEVICE: torch.device


def main() -> int:
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["cuda", "musa"])
    parser.add_argument("--skip-spot", action="store_true")
    parser.add_argument("--skip-router", action="store_true",
                        help="跳过融合 op 抽查（无扩展环境时）")
    arguments = parser.parse_args()

    DEVICE = resolve_device(arguments.backend)
    module = backend_module(DEVICE)
    module.empty_cache()
    itemsize = 2

    sizes = [measure_size(count, itemsize) for count in ELEMENT_COUNTS]
    eager_boundary = classify_boundary(sizes, "eager_us_per_op")
    graph_boundary = classify_boundary(sizes, "graph_us_per_op")
    print(f"[launch-bound-graph/{arguments.backend}] eager 分界={eager_boundary['launch_bound_max_elements']} "
          f"graph 分界={graph_boundary['launch_bound_max_elements']} "
          f"(ch02 参考: eager CUDA 4194304 / MUSA 1048576)")

    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "operation": "t.add_(1.0) 原位链（y=x+1.0 语义，静态 buffer）",
        "elements_ladder": ELEMENT_COUNTS,
        "chain_lengths": CHAIN_LENGTHS,
        "warmup": WARMUP,
        "sizes": sizes,
        "boundary_eager": eager_boundary,
        "boundary_graph": graph_boundary,
        "boundary_shift_note": (
            "graph 分界 − eager 分界 = Graph 砍掉发射税后 launch-bound 区的推移"
            "（收益上限区扩大）；大尺寸端 graph 收益应趋于 0"),
    }

    if not arguments.skip_spot:
        router = None
        if not arguments.skip_router:
            try:
                from implementations.backend_setup import setup_backend
                setup = setup_backend(arguments.backend)
                router = setup["router"]
                results["fused_op_registration_evidence"] = setup["evidence"]
            except Exception as error:  # noqa: BLE001 —— 无扩展环境时降级
                results["fused_op_registration_evidence"] = f"unavailable: {type(error).__name__}: {error}"
        if router is not None:
            compat = GraphCaptureCompat(DEVICE, setup.get("pybind_module"))
            results["graph_capture_compat"] = {
                "supported": compat.supported,
                "explicit_stream_mode": compat.explicit_stream_mode,
            }
            results["normal_shape_spot"] = normal_shape_spot(router, compat=compat)
        else:
            print("[spot] 跳过融合 op 抽查（router 不可用）")

    results["environment"] = device_environment_record(DEVICE)
    output = Path("results") / f"launch-bound-graph-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
