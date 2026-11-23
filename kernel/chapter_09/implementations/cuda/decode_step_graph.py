#!/usr/bin/env python3
"""第 09 期实验 3+4 核心：手写 Qwen3 单步 decode 的 Graph 化 + 三级归因阶梯。

模型：chapter_01/qwen3-0.6b 手写实现（第 04 期可切换版，本目录复制品），
权重 FP16，batch=1，seq=46（prompt 20 + 26 生成 token 的定长步——真值
generated_token_ids[26] 作为该步 argmax 的 sanity 参照）。

四种步实现（单变量 = 调用组织方式）：
  eager_native ：模型原生组合表达式链（与 chapter_01 forward 逐字同语义）
  eager_fused  ：56 对相邻 Add+Norm 换成注册融合 op（chapter_04 路径）
  graph_native ：eager_native 整步捕获为图（torch.CUDAGraph 流捕获，静态 buffer）
  graph_fused  ：eager_fused 整步捕获为图

三级归因阶梯（大纲验收核心）：
  T0=eager_native → T1=eager_fused → T2=graph_fused，另测 graph_native 做分解：
    融合砍掉的部分   = T0 − T1（多余 kernel 的执行 + 它们的发射槽）
    Graph 砍掉的部分 = T1 − T2（剩余发射税 + framework 调度）
    总收益          = T0 − T2；与 (T0−T1)+(T1−T2) 对照看两路是否可加
  开销三分解（每模式）：
    framework 调度   = enqueue-only wall（CPU 发射整步不等待 GPU）
    launch 税/间隙   = event 跨度 − kernel 执行和（kernel 和：CUDA 用 kineto
                       设备时长求和；MUSA 无设备侧时长（M-04）→ 用图残差法）
    kernel 执行      = 上述差补项，nsys（CUDA）交叉验证
  ep05 未决 1 复核（CUDA 兑现率 1.32 的"上限非严格上限"）：
    隔离微基准（rows=SEQ 的 composite 链 vs 融合 op，ch05 口径）求和上限 vs
    同两算子图回放口径的"launch-free 上限" vs 端到端实测收益——三者对照。

输出：results/decode-step-graph-{backend}.json
"""

from __future__ import annotations

import argparse
import json
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
    resolve_device,
    write_results_json,
)

SCHEMA_VERSION = "kernel-chapter10.decode-step-graph.v1"
DEFAULT_BACKEND = "cuda"
MODEL_DEFAULT = CHAPTER.parent / "chapter_01" / "models" / "qwen3-0.6b"
TRUTH_DEFAULT = CHAPTER.parent / "chapter_01" / "truth"
OP_CALLS_PER_FORWARD = 56
GENERATED_INDEX = 26  # 前向覆盖 ids[:20+26]，预测真值 generated_token_ids[26]
REPEATS = 50
WARMUP = 8


def profiler_kernel_stats(step_fn, device, iterations: int = 5) -> dict:
    """torch.profiler（kineto）口径：整步发射数与设备 kernel 时长和。

    发射数：所有 launch 类 API（cudaLaunchKernel / cudaLaunchKernelExC /
    cuLaunchKernel* / musaLaunchKernel）——部分 kernel 走 ExC 入口，只数
    cudaLaunchKernel 会漏计（本期实测发现，见 EXPERIMENT_LOG）。
    kernel 时长/实例数：只累加设备侧事件（FunctionEvent.device_type ==
    DeviceType.CUDA / PrivateUse1），排除 aten op 包装层的归属时长
    （包装层与 kernel 双重计入，首版口径 bug，本期修正）。
    MUSA：设备时长全 0（M-04/M-30 平台边界），发射数口径不变。
    """
    module = backend_module(device)
    try:
        from torch.profiler import ProfilerActivity, profile
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        else:
            try:
                activities.append(getattr(ProfilerActivity, "PrivateUse1"))
            except AttributeError:
                pass  # MUSA 无 PrivateUse1 profiler activity 时仅 CPU 侧
        with profile(activities=activities) as prof:
            for _ in range(iterations):
                step_fn()
            backend_sync(device)
        events = prof.key_averages()
        launch_prefixes = ("cudaLaunchKernel", "cuLaunchKernel", "musaLaunchKernel")
        launches = 0
        graph_launches = 0
        kernel_device_us = 0.0
        kernel_events = 0
        memcpy_device_us = 0.0
        kernel_names = {}
        device_type_names = ("DeviceType.CUDA", "DeviceType.PrivateUse1")
        for event in events:
            if event.key.startswith(launch_prefixes):
                launches += event.count
            if event.key in ("cudaGraphLaunch", "musaGraphLaunch", "cuGraphLaunch"):
                graph_launches += event.count
            device_us = float(getattr(event, "self_device_time_total", 0.0) or 0.0)
            event_device_type = str(getattr(event, "device_type", ""))
            if device_us > 0 and event_device_type in device_type_names:
                key = event.key
                if key.startswith("Memcpy") or key.startswith("Memset") or \
                        key.startswith("memcpy") or key.startswith("memset"):
                    memcpy_device_us += device_us
                    continue
                kernel_device_us += device_us
                kernel_events += event.count
                kernel_names[key] = round(event.count / iterations, 1)
        stats = {
            "iterations": iterations,
            "host_launch_api_per_step": round(launches / iterations, 1),
            "graph_launch_api_per_step": round(graph_launches / iterations, 1),
            "kernel_device_us_per_step": round(kernel_device_us / iterations, 2),
            "memcpy_memset_device_us_per_step": round(memcpy_device_us / iterations, 2),
            "kernel_instances_per_step": round(kernel_events / iterations, 1),
            "device_event_filter": device_type_names,
        }
        if kernel_names:
            stats["kernel_names_per_step_top"] = dict(
                sorted(kernel_names.items(), key=lambda kv: -kv[1])[:12])
        return stats
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def build_step(world, router) -> dict:
    """构造四种步实现与静态状态。world 携带 model/router/device/ids。

    模式切换点在 model.model.fused_ops 属性（第 04 期可切换模型约定）：
    graph_native 捕获前显式置 None，graph_fused 捕获前置 router（若可用）；
    eager 步是同一闭包，模式由调用时点的 fused_ops 决定（调用方负责置位）。
    """
    model = world["model"]
    device = world["device"]
    module = backend_module(device)
    seq = world["seq"]
    static_ids = world["static_ids"]

    def step_forward():
        logits = model(static_ids)
        return logits, logits[:, -1, :].argmax(dim=-1, keepdim=True)

    steps = {"eager_native": (step_forward, False), "eager_fused": (step_forward, False)}

    graph_modes = {}
    capture_plan = [("graph_native", None)]
    if router is not None:
        capture_plan.append(("graph_fused", router))
    compat = world.get("compat")
    for mode_name, fused_ops in capture_plan:
        model.model.fused_ops = fused_ops
        try:
            side_stream = module.Stream()
            current = module.current_stream()
            side_stream.wait_stream(current)
            with module.stream(side_stream):
                for _ in range(WARMUP):
                    step_forward()
            current.wait_stream(side_stream)
            backend_sync(device)

            graph = module.CUDAGraph()
            t0 = time.perf_counter()
            if compat is not None and compat.supported:
                with compat.capture(graph):
                    captured_logits, captured_next = step_forward()
            else:
                with module.graph(graph):
                    captured_logits, captured_next = step_forward()
            capture_ms = (time.perf_counter() - t0) * 1000.0
            backend_sync(device)
            graph.replay()
            backend_sync(device)
            graph_modes[mode_name] = {
                "graph": graph,
                "logits": captured_logits,
                "next": captured_next,
                "capture_plus_instantiate_ms": capture_ms,
                "fused_ops_at_capture": fused_ops is not None,
            }
        except Exception as error:  # noqa: BLE001 —— Graph 捕获失败如实记录
            graph_modes[mode_name] = {"error": f"{type(error).__name__}: {error}"}
    model.model.fused_ops = None
    return {"steps": steps, "graph_modes": graph_modes}


def time_eager_step(step_fn, device, repeats: int) -> dict:
    """eager 步：sync-per-step wall + 事件对跨度 + enqueue-only 三口径。"""
    module = backend_module(device)
    for _ in range(WARMUP):
        step_fn()
    backend_sync(device)

    wall = []
    for _ in range(repeats):
        backend_sync(device)
        start = time.perf_counter()
        step_fn()
        backend_sync(device)
        wall.append((time.perf_counter() - start) * 1000.0)

    starts, ends = [], []
    for _ in range(repeats):
        starts.append(module.Event(enable_timing=True))
        ends.append(module.Event(enable_timing=True))
    for index in range(repeats):
        starts[index].record()
        step_fn()
        ends[index].record()
    backend_sync(device)
    event_ms = [starts[i].elapsed_time(ends[i]) for i in range(repeats)]

    enqueue = enqueue_wall_us(step_fn, repeats=min(6, repeats), device=device,
                              queue_drain_every=2)
    ordered = sorted(wall)
    ordered_event = sorted(event_ms)
    return {
        "sync_wall_p50_ms": round(ordered[len(ordered) // 2], 4),
        "sync_wall_p95_ms": round(ordered[min(len(ordered) - 1, int(round(len(ordered) * 0.95)) - 1)], 4),
        "sync_wall_mean_ms": round(sum(wall) / len(wall), 4),
        "event_span_p50_ms": round(ordered_event[len(ordered_event) // 2], 4),
        "enqueue_only_cpu": enqueue,
        "repeats": repeats,
    }


def time_graph_step(state: dict, static_ids: torch.Tensor, device, repeats: int,
                    rotate_ids: list[int]) -> dict:
    """graph 步：纯 replay 口径 + 带输入拷贝/输出同步的现实步口径 + replay dispatch。"""
    module = backend_module(device)
    graph = state["graph"]
    next_buffer = state["next"]
    for _ in range(WARMUP):
        graph.replay()
    backend_sync(device)

    # 纯 replay：sync-per-step（图内含 argmax，读 next buffer 需同步）
    wall_pure = []
    for _ in range(repeats):
        backend_sync(device)
        start = time.perf_counter()
        graph.replay()
        _ = next_buffer.item()  # D2H 同步（图输出读取的真实代价）
        wall_pure.append((time.perf_counter() - start) * 1000.0)

    # 现实步：换输入 ids → replay → 读 next
    id_tensor = torch.tensor([rotate_ids], dtype=torch.long, device=device)
    wall_full = []
    for _ in range(repeats):
        backend_sync(device)
        start = time.perf_counter()
        static_ids.copy_(id_tensor)
        graph.replay()
        _ = next_buffer.item()
        wall_full.append((time.perf_counter() - start) * 1000.0)

    # 纯 replay 的 GPU 跨度（事件对，µs/replay）与 CPU dispatch
    starts = module.Event(enable_timing=True)
    ends = module.Event(enable_timing=True)
    spans = []
    for _ in range(5):
        starts.record()
        for _ in range(20):
            graph.replay()
        ends.record()
        backend_sync(device)
        spans.append(starts.elapsed_time(ends) * 1000.0 / 20)  # elapsed ms→µs，/20 次
    replay_dispatch = enqueue_wall_us(graph.replay, repeats=20, device=device)

    ordered = sorted(wall_pure)
    ordered_full = sorted(wall_full)
    return {
        "replay_only_p50_ms": round(ordered[len(ordered) // 2], 4),
        "replay_only_p95_ms": round(ordered[min(len(ordered) - 1, int(round(len(ordered) * 0.95)) - 1)], 4),
        "replay_plus_io_p50_ms": round(ordered_full[len(ordered_full) // 2], 4),
        "replay_gpu_span_p50_us": round(sorted(spans)[2], 4),
        "replay_dispatch_cpu": replay_dispatch,
        "repeats": repeats,
    }


def microbench_rows_seq(device, router, rows: int, hidden: int, eps: float,
                        compat: GraphCaptureCompat | None = None) -> dict:
    """隔离微基准（ch05 口径）+ 图回放口径：ep05 未决 1 复核的数据源。"""
    module = backend_module(device)
    generator = torch.Generator().manual_seed(20261121 + rows * 100000 + hidden)
    r = torch.randn(rows, hidden, generator=generator).to(device=device, dtype=torch.float16)
    h = torch.randn(rows, hidden, generator=generator).to(device=device, dtype=torch.float16)
    w = torch.randn(hidden, generator=generator).to(device=device, dtype=torch.float16)

    def composite_chain():
        t = r + h
        float_states = t.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + eps)
        return w * normalized.to(t.dtype)

    def fused_call():
        router.fused_add_rmsnorm(h, r, w, eps)

    def event_pair_p50_us(operation, repeats: int = 200) -> float:
        starts, ends = [], []
        for _ in range(repeats):
            starts.append(module.Event(enable_timing=True))
            ends.append(module.Event(enable_timing=True))
        for index in range(repeats):
            starts[index].record()
            operation()
            ends[index].record()
        backend_sync(device)
        samples = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(repeats))
        return samples[len(samples) // 2]

    def graph_replay_p50_us(operation, repeats: int = 100) -> float:
        side_stream = module.Stream()
        current = module.current_stream()
        side_stream.wait_stream(current)
        with module.stream(side_stream):
            for _ in range(3):
                operation()
        current.wait_stream(side_stream)
        backend_sync(device)
        graph = module.CUDAGraph()
        if compat is not None and compat.supported:
            with compat.capture(graph):
                operation()
        else:
            with module.graph(graph):
                operation()
        backend_sync(device)
        graph.replay()
        backend_sync(device)
        starts = module.Event(enable_timing=True)
        ends = module.Event(enable_timing=True)
        spans = []
        for _ in range(5):
            starts.record()
            for _ in range(repeats):
                graph.replay()
            ends.record()
            backend_sync(device)
            spans.append(starts.elapsed_time(ends) * 1000.0 / repeats)
        del graph
        return sorted(spans)[2]

    result = {"rows": rows, "hidden_size": hidden, "dtype": "float16"}
    for name, operation in (("composite_chain", composite_chain), ("fused_op", fused_call)):
        eager_us = event_pair_p50_us(operation)
        graph_us = graph_replay_p50_us(operation)
        result[name] = {
            "eager_event_pair_us": round(eager_us, 3),
            "graph_replay_us": round(graph_us, 3),
            "launch_tax_isolated_us": round(eager_us - graph_us, 3),
        }
        print(f"[micro/{device.type}] rows={rows} {name}: eager={eager_us:.1f}µs "
              f"graph={graph_us:.1f}µs 发射税隔离值={eager_us - graph_us:.1f}µs")
    result["ep05_realization_recheck"] = {
        "isolated_ceiling_ms": round(56 * (result["composite_chain"]["eager_event_pair_us"]
                                           - result["fused_op"]["eager_event_pair_us"]) / 1000.0, 4),
        "graph_free_ceiling_ms": round(56 * (result["composite_chain"]["graph_replay_us"]
                                             - result["fused_op"]["graph_replay_us"]) / 1000.0, 4),
        "note": ("isolated_ceiling = 56×(链 eager − 融合 eager)（ch05 口径）；"
                 "graph_free_ceiling = 56×(链 graph − 融合 graph)（把两侧发射税同时砍掉的口径）"),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["cuda", "musa"])
    parser.add_argument("--model-dir", default=str(MODEL_DEFAULT))
    parser.add_argument("--truth-dir", default=str(TRUTH_DEFAULT))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--skip-router", action="store_true")
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    module = backend_module(device)
    module.empty_cache()

    sys.path.insert(0, str(CHAPTER / "implementations"))
    import qwen3_model_fused as model_lib  # noqa: E402

    model = model_lib.load_handwritten_model(
        arguments.model_dir, device, dtype=torch.float16).eval()
    truth = json.loads((Path(arguments.truth_dir) / "truth-manifest.json").read_text(encoding="utf-8"))
    prompt_ids = truth["prompt_token_ids"]
    generated_ids = truth["generated_token_ids"]
    seq = len(prompt_ids) + GENERATED_INDEX
    expected_next = generated_ids[GENERATED_INDEX]

    full_ids = prompt_ids + generated_ids[:GENERATED_INDEX]
    assert len(full_ids) == seq
    static_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    rotate_ids = (prompt_ids + generated_ids)[:seq]  # 带输入拷贝口径用的变体输入

    router = None
    setup = None
    if not arguments.skip_router:
        from implementations.backend_setup import setup_backend
        setup = setup_backend(arguments.backend)
        router = setup["router"]

    world = {"model": model, "device": device, "seq": seq, "static_ids": static_ids}
    world["pybind_module"] = setup.get("pybind_module") if setup else None
    world["compat"] = GraphCaptureCompat(device, world["pybind_module"])

    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "model_dtype": "float16",
        "seq": seq,
        "generated_index": GENERATED_INDEX,
        "expected_next_token": expected_next,
        "op_calls_per_forward_fused": OP_CALLS_PER_FORWARD,
        "repeats": arguments.repeats,
    }

    # ---- eager 两态 ----
    model.model.fused_ops = None
    steps = build_step(world, router)
    eager_native_timing = time_eager_step(steps["steps"]["eager_native"][0], device, arguments.repeats)
    eager_native_logits = steps["steps"]["eager_native"][0]()[0].detach().clone()
    eager_native_next = int(steps["steps"]["eager_native"][0]()[1].item())
    backend_sync(device)

    if router is not None:
        model.model.fused_ops = router
    else:
        results["fused_path_note"] = "router 不可用，fused/graph_fused 缺席（平台边界）"
    eager_fused_timing = (time_eager_step(steps["steps"]["eager_fused"][0], device, arguments.repeats)
                          if router is not None else None)
    eager_fused_logits = (steps["steps"]["eager_fused"][0]()[0].detach().clone()
                          if router is not None else None)
    eager_fused_next = (int(steps["steps"]["eager_fused"][0]()[1].item())
                        if router is not None else None)
    backend_sync(device)

    # ---- graph 两态 ----
    graph_native_timing = None
    graph_fused_timing = None
    graph_native_next = None
    graph_fused_next = None
    correctness = {}
    if "error" not in steps["graph_modes"]["graph_native"]:
        state_n = steps["graph_modes"]["graph_native"]
        graph_native_timing = time_graph_step(state_n, static_ids, device, arguments.repeats, rotate_ids)
        graph_native_next = int(state_n["next"].item())
        diff_n = (state_n["logits"].float() - eager_native_logits.float()).abs().max().item()
        correctness["graph_native_vs_eager_native"] = {
            "logits_max_abs": round(diff_n, 8),
            "next_token_equal": graph_native_next == eager_native_next,
            "capture_plus_instantiate_ms": round(state_n["capture_plus_instantiate_ms"], 3),
        }
    else:
        correctness["graph_native_vs_eager_native"] = {"error": steps["graph_modes"]["graph_native"]["error"]}
    if router is not None and "error" not in steps["graph_modes"]["graph_fused"]:
        state_f = steps["graph_modes"]["graph_fused"]
        graph_fused_timing = time_graph_step(state_f, static_ids, device, arguments.repeats, rotate_ids)
        graph_fused_next = int(state_f["next"].item())
        diff_f = (state_f["logits"].float() - eager_fused_logits.float()).abs().max().item()
        correctness["graph_fused_vs_eager_fused"] = {
            "logits_max_abs": round(diff_f, 8),
            "next_token_equal": graph_fused_next == eager_fused_next,
            "capture_plus_instantiate_ms": round(state_f["capture_plus_instantiate_ms"], 3),
        }
        diff_fn = (eager_fused_logits.float() - eager_native_logits.float()).abs().max().item()
        correctness["eager_fused_vs_eager_native"] = {
            "logits_max_abs": round(diff_fn, 8),
            "note": "ch05 融合语义容差口径（fp16 两次舍入 vs 单次舍入）",
        }
    elif router is not None:
        correctness["graph_fused_vs_eager_fused"] = {"error": steps["graph_modes"]["graph_fused"]["error"]}

    results["eager_native"] = eager_native_timing
    results["eager_fused"] = eager_fused_timing
    results["graph_native"] = graph_native_timing
    results["graph_fused"] = graph_fused_timing
    results["graph_capture_stages"] = {
        name: ({"capture_plus_instantiate_ms": round(state["capture_plus_instantiate_ms"], 3)}
               if "error" not in state else {"error": state["error"]})
        for name, state in steps["graph_modes"].items()
    }
    results["next_token_sanity"] = {
        "expected_from_truth": expected_next,
        "eager_native": eager_native_next,
        "eager_fused": eager_fused_next,
        "graph_native": graph_native_next,
        "graph_fused": graph_fused_next,
    }
    results["correctness"] = correctness

    # ---- 三级归因阶梯 ----
    if eager_fused_timing and graph_fused_timing and graph_native_timing:
        t0 = eager_native_timing["sync_wall_p50_ms"]
        t1 = eager_fused_timing["sync_wall_p50_ms"]
        t2n = graph_native_timing["replay_only_p50_ms"]
        t2 = graph_fused_timing["replay_only_p50_ms"]
        t2f = graph_fused_timing["replay_plus_io_p50_ms"]
        ladder = {
            "T0_eager_native_ms": t0,
            "T1_eager_fused_ms": t1,
            "T2_graph_native_ms": t2n,
            "T2_graph_fused_ms": t2,
            "T2_graph_fused_full_io_ms": t2f,
            "fusion_cut_T0_minus_T1_ms": round(t0 - t1, 4),
            "graph_cut_on_native_T0_minus_T2n_ms": round(t0 - t2n, 4),
            "graph_cut_on_fused_T1_minus_T2_ms": round(t1 - t2, 4),
            "total_cut_T0_minus_T2_ms": round(t0 - t2, 4),
            "sum_of_cuts_T0_minus_T2_full": round((t0 - t1) + (t1 - t2), 4),
            "gain_ratio_native_to_graph_fused": round(t0 / t2, 4),
            "additivity_note": (
                "fusion_cut + graph_cut_on_fused ≈ total_cut → 两路收益可加（串联）；"
                "graph_cut_on_native vs graph_cut_on_fused 的差 = 融合删掉的 kernel"
                "本也占用的图内发射槽"),
        }
        d0 = eager_native_timing["enqueue_only_cpu"]
        d1 = eager_fused_timing["enqueue_only_cpu"]
        dg = graph_fused_timing["replay_dispatch_cpu"] if graph_fused_timing else None
        ladder["framework_dispatch_cpu_us"] = {
            "eager_native_enqueue_p50": d0,
            "eager_fused_enqueue_p50": d1,
            "graph_fused_replay_p50": dg,
            "note": "enqueue-only = Python+ATen dispatch+发射的 CPU 总成本（GPU 空闲时整步发射耗时）",
        }
        results["attribution_ladder"] = ladder
        print(f"[ladder/{arguments.backend}] T0={t0:.3f} T1={t1:.3f} T2n={t2n:.3f} T2={t2:.3f} ms | "
              f"融合砍 {(t0 - t1):.3f} / Graph 再砍 {(t1 - t2):.3f} / 共 {(t0 - t2):.3f} ms | "
              f"总加速 {t0 / t2:.2f}×")

    # ---- kineto 发射数与 kernel 时长和（三分解证据） ----
    model.model.fused_ops = None
    results["profiler_eager_native"] = profiler_kernel_stats(
        steps["steps"]["eager_native"][0], device)
    if router is not None:
        model.model.fused_ops = router
        results["profiler_eager_fused"] = profiler_kernel_stats(
            steps["steps"]["eager_fused"][0], device)
    model.model.fused_ops = None
    for label in ("graph_native", "graph_fused"):
        state = steps["graph_modes"].get(label)
        if state is not None and "error" not in state:
            results[f"profiler_{label}"] = profiler_kernel_stats(
                lambda: state["graph"].replay(), device)

    # ---- ep05 未决 1 复核（隔离微基准 vs 图口径上限 vs 端到端） ----
    if router is not None:
        eps = float(model.config.rms_norm_eps)
        micro = microbench_rows_seq(device, router, seq, model.config.hidden_size, eps,
                                    compat=world["compat"])
        results["microbench_rows_seq"] = micro
        if eager_fused_timing:
            measured_gain = eager_native_timing["sync_wall_p50_ms"] - eager_fused_timing["sync_wall_p50_ms"]
            recheck = dict(micro["ep05_realization_recheck"])
            recheck["measured_step_gain_ms"] = round(measured_gain, 4)
            iso = recheck["isolated_ceiling_ms"]
            gfree = recheck["graph_free_ceiling_ms"]
            recheck["realization_vs_isolated_ceiling"] = round(measured_gain / iso, 4) if iso > 0 else None
            recheck["realization_vs_graph_free_ceiling"] = round(measured_gain / gfree, 4) if gfree > 0 else None
            recheck["ep05_unresolved_1_note"] = (
                "ep05 兑现率 1.32 的'上限非严格上限'：隔离上限用 eager 事件对口径"
                "（含两侧发射税），图回放口径把两侧发射税同时砍掉后才是 launch-free 真上限")
            results["ep05_realization_recheck"] = recheck

    results["graph_capture_compat"] = {
        "supported": world["compat"].supported,
        "explicit_stream_mode": world["compat"].explicit_stream_mode,
        "note": ("musa 变体 2：default-stream kernel 在 torch_musa 捕获期静默不入图"
                 "（MUSAGraph.cpp:160 仅整图空时告警）→ pybind set_launch_stream 注入捕获流"),
    }
    results["environment"] = device_environment_record(device)
    output = Path("results") / f"decode-step-graph-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
