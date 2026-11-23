"""第 09 期共享工具（复制自第 01 期 experiment_utils.py 改造，协议 §2）。

口径约定（全篇统一，来源于第 01 期合同）：
- 设备区间计时使用后端同步原语（torch.cuda/musa.synchronize）+ Event；
- 端到端时间使用 Wall Clock，并在边界正确同步；
- 延迟报告 P50/P95，重复次数与 Warm-up 由调用方传入并写进结果。

第 09 期新增（Launch 开销与 Graph 主题）：
- enqueue_wall_us：CPU 侧纯发射耗时（不同步）——framework 调度成本口径；
- capture_with_stages：Graph 捕获并把 capture/instantiate 两阶段拆开计时；
- memory_snapshot：torch 后端内存统计快照（graph pool 驻留测量用）；
- linear_fit：replay_span(K) = intercept + slope×K 的最小二乘拟合
  （M-02 根因模型：per-replay 固定成本 vs per-kernel-node 边际成本）。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

SCHEMA_VERSION = "kernel-chapter10.v1"


def git_revision() -> str:
    explicit_revision = os.environ.get("MINI_INFRA_SOURCE_REVISION")
    if explicit_revision:
        return explicit_revision
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text).strip()


def _safe_command(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(arguments, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return _strip_ansi(result.stdout) if result.returncode == 0 else None


def backend_sync(device: torch.device) -> None:
    """按设备类型选择同步原语；CPU 天然同步。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "musa":
        torch.musa.synchronize(device)


def resolve_device(name: str) -> torch.device:
    """把合同中的后端名解析成 torch.device，缺失即显式报错，不做静默回退。"""
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("后端 cuda 不可用：torch.cuda.is_available() 为 False")
        return torch.device("cuda")
    if name == "musa":
        if not hasattr(torch, "musa"):
            raise RuntimeError("后端 musa 不可用：未安装 torch_musa")
        if not torch.musa.is_available():
            raise RuntimeError("后端 musa 不可用：torch.musa.is_available() 为 False")
        return torch.device("musa")
    raise ValueError("未知后端: " + name)


def device_environment_record(device: torch.device) -> dict[str, Any]:
    """采集与本期口径相关的环境与设备信息。"""
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device_type": device.type,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record["gpu"] = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
        }
        record["driver_cuda_version"] = _safe_command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        record["tool_versions"] = {
            "nvcc": _safe_command(["nvcc", "--version"]),
            "compute_sanitizer": _safe_command(["compute-sanitizer", "--version"]),
        }
    elif device.type == "musa":
        properties = torch.musa.get_device_properties(device)
        record["gpu"] = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
        }
        record["driver"] = _safe_command(
            ["mthreads-gmi"],  # 容器内若不可用，留 None 并入 KNOWN_ISSUES
        )
        record["tool_versions"] = {"mcc": _safe_command(["mcc", "--version"])}
    return record


def sha256_of_tensor(tensor: torch.Tensor) -> str:
    """张量字节内容的 SHA256，用于跨机器真值核对。"""
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def write_results_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"[results] 已写入 {path}")


class OperatorTimer:
    """统一计时器：Warm-up 后测 repeats 次，返回 P50/P95（毫秒）。

    device_type == cpu 时直接用 perf_counter（CPU 无异步执行需要同步）。
    """

    def __init__(self, device: torch.device, warmup: int, repeats: int):
        self.device = device
        self.warmup = warmup
        self.repeats = repeats
        if device.type != "cpu" and not hasattr(torch, device.type):
            raise ValueError("未知设备类型: " + device.type)

    def _synchronize(self) -> None:
        backend_sync(self.device)

    def measure(self, operation) -> dict[str, Any]:
        for _ in range(self.warmup):
            operation()
        self._synchronize()

        samples_ms: list[float] = []
        if self.device.type == "cpu":
            import time

            for _ in range(self.repeats):
                start = time.perf_counter()
                operation()
                samples_ms.append((time.perf_counter() - start) * 1000.0)
        else:
            # 事件对按次记录、末尾一次同步：相邻重复的发射延迟与前次执行
            # 重叠，GPU 时间线不被 CPU enqueue 空隙拉长（逐次同步会把
            # launch 开销计入每次测量，小算子时读数被发射延迟淹没）。
            event_class = torch.cuda.Event if self.device.type == "cuda" else torch.musa.Event
            event_pairs = []
            for _ in range(self.repeats):
                start_event = event_class(enable_timing=True)
                end_event = event_class(enable_timing=True)
                start_event.record()
                operation()
                end_event.record()
                event_pairs.append((start_event, end_event))
            self._synchronize()
            samples_ms = [start.elapsed_time(end) for start, end in event_pairs]

        ordered = sorted(samples_ms)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(round(len(ordered) * 0.95)) - 1)]
        return {
            "warmup": self.warmup,
            "repeats": self.repeats,
            "p50_ms": round(p50, 4),
            "p95_ms": round(p95, 4),
            "mean_ms": round(mean(samples_ms), 4),
            "min_ms": round(ordered[0], 4),
            "samples_ms": [round(value, 4) for value in samples_ms],
        }


def measure_with_graph(device: torch.device, warmup: int, repeats: int, operation) -> dict[str, Any] | None:
    """Graph Replay 计时：捕获 operation 为单张图后回放，测纯 GPU 执行时间。

    背景：eager 链路（多 Kernel 组合表达式）在小算子时被 CPU 发射开销主导
    （launch-bound），事件对测出的是 CPU enqueue 时间。Graph Replay 把整条
    链固化为一次发射，测得的才是 GPU 时间线上的真实执行时长。设备或驱动
    不支持 Graph 时返回 None（记平台边界）。
    """
    if device.type == "cpu":
        return None
    module = getattr(torch, device.type, None)
    graph_class = getattr(module, "CUDAGraph", None) or getattr(module, "graph", None)
    if graph_class is None or not hasattr(module, "Stream"):
        return None
    try:
        # 旁路流预热（Graph 捕获要求目标算子先在非默认流上执行过）。
        side_stream = module.Stream()
        current = module.current_stream()
        side_stream.wait_stream(current)
        with module.stream(side_stream):
            for _ in range(warmup):
                operation()
        current.wait_stream(side_stream)
        backend_sync(device)

        graph = graph_class()
        reference_output = operation()  # eager 输出，用于校验捕获正确性
        backend_sync(device)
        with module.graph(graph):
            captured_output = operation()
        backend_sync(device)
        graph.replay()
        backend_sync(device)
        if not torch.allclose(captured_output, reference_output, rtol=1e-3, atol=1e-3):
            raise RuntimeError("Graph Replay 输出与 eager 输出不一致，捕获失败")

        event_class = module.Event
        event_pairs = []
        for _ in range(repeats):
            start_event = event_class(enable_timing=True)
            end_event = event_class(enable_timing=True)
            start_event.record()
            graph.replay()
            end_event.record()
            event_pairs.append((start_event, end_event))
        backend_sync(device)
        samples_ms = [start.elapsed_time(end) for start, end in event_pairs]
        ordered = sorted(samples_ms)
        return {
            "warmup": warmup,
            "repeats": repeats,
            "p50_ms": round(ordered[len(ordered) // 2], 4),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(round(len(ordered) * 0.95)) - 1)], 4),
            "min_ms": round(ordered[0], 4),
        }
    except Exception as error:  # noqa: BLE001 —— Graph 不可用时降级为 eager 口径
        print(f"[graph] {device.type} Graph Replay 不可用（{type(error).__name__}: {error}），降级 eager 口径")
        return None


def comparison_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """统一正确性比较口径：max/mean 绝对误差、相对误差、余弦相似度、NaN/Inf。

    统一把两侧移到 CPU 再比较，允许后端结果与真值跨设备传入。
    """
    actual_float = actual.detach().float().cpu()
    expected_float = expected.detach().float().cpu()
    absolute_error = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(1e-6)
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    return {
        "max_abs_error": round(absolute_error.max().item(), 8),
        "mean_abs_error": round(absolute_error.mean().item(), 8),
        "mean_rel_error": round((absolute_error / denominator).mean().item(), 8),
        "cosine_similarity": round(cosine, 8),
        "nan_count": int(torch.isnan(actual_float).sum().item()),
        "inf_count": int(torch.isinf(actual_float).sum().item()),
        "max_abs_error_position": int(absolute_error.argmax().item()),
    }


# ---------------------------------------------------------------------------
# 第 09 期新增：Graph 实验专用口径
# ---------------------------------------------------------------------------

def backend_module(device: torch.device):
    """返回 torch.cuda / torch.musa 模块（Graph API 探测与调用用）。"""
    return getattr(torch, device.type, None)


def graph_support_level(device: torch.device) -> dict[str, Any]:
    """Graph API 可用性探测（不动 GPU，只查属性，供 probe 脚本复用）。"""
    module = backend_module(device)
    if module is None:
        return {"backend": device.type, "module": None}
    return {
        "backend": device.type,
        "module": str(module.__name__),
        "has_CUDAGraph_class": hasattr(module, "CUDAGraph"),
        "CUDAGraph_class_name": type(module.CUDAGraph).__name__ if hasattr(module, "CUDAGraph") else None,
        "has_graph_context": hasattr(module, "graph"),
        "has_Stream": hasattr(module, "Stream"),
        "has_graph_pool_handle": hasattr(module, "graph_pool_handle"),
        "has_make_graphed_callables": hasattr(module, "make_graphed_callables"),
        "torch_version": torch.__version__,
    }


def capture_with_stages(device: torch.device, operation, warmup: int = 3,
                        check_against_eager: bool = True) -> dict[str, Any]:
    """捕获 operation 为单张 Graph，并拆解 capture / instantiate / 首次回放三阶段。

    torch CUDAGraph 的阶段划分（torch/csrc/cuda/Graph.cpp，MUSA 对应实现同构）：
      capture_begin() → cudaStreamBeginCapture（进入捕获模式，CPU 侧只记录不动 GPU）；
      operation()      → 捕获期执行：kernel 不真正发射，只录进图（CPU 走一遍 dispatch 路径）；
      capture_end()   → cudaStreamEndCapture + cudaGraphInstantiate（实例化，
                         CPU 侧完成图对象到可执行对象的编译/实例化，是捕获阶段最贵一步）；
      replay()         → cudaGraphLaunch（单次 API 调用提交整图）。

    返回各阶段毫秒数与静态输出（供正确性对账）。失败时返回 {"error": ...}。
    """
    module = backend_module(device)
    if device.type == "cpu" or module is None:
        return {"error": "cpu 无 Graph 概念"}
    if not (hasattr(module, "CUDAGraph") and hasattr(module, "graph") and hasattr(module, "Stream")):
        return {"error": "本平台 torch 缺少 CUDAGraph/graph/Stream API"}
    try:
        # 旁路流 warmup（Graph 捕获前要求算子执行过；cuBLAS workspace 等惰性
        # 初始化必须在捕获前完成，否则捕获会失败或把初始化录进图）。
        side_stream = module.Stream()
        current = module.current_stream()
        side_stream.wait_stream(current)
        with module.stream(side_stream):
            for _ in range(warmup):
                operation()
        current.wait_stream(side_stream)
        backend_sync(device)

        eager_reference = operation() if check_against_eager else None
        backend_sync(device)

        graph = module.CUDAGraph()
        t0 = time.perf_counter()
        with module.graph(graph):
            captured_outputs = operation()
        t1 = time.perf_counter()
        backend_sync(device)
        graph.replay()  # 首次回放（可能含 lazy 初始化）
        backend_sync(device)
        t2 = time.perf_counter()
        stage = {
            "capture_plus_instantiate_ms": round((t1 - t0) * 1000.0, 4),
            "capture_includes_instantiate_note": (
                "torch CUDAGraph.capture_end 在 graph 上下文退出时同步执行 "
                "EndCapture+Instantiate，本口径二者合计"),
            "first_replay_ms": round((t2 - t1) * 1000.0, 4),
        }
        if check_against_eager and isinstance(captured_outputs, torch.Tensor):
            stage["eager_vs_captured_max_abs"] = comparison_metrics(
                captured_outputs, eager_reference)["max_abs_error"]
        graph.replay()
        backend_sync(device)
        return {"stages": stage, "graph": graph, "static_outputs": captured_outputs}
    except Exception as error:  # noqa: BLE001 —— Graph 不可用时如实记录
        return {"error": f"{type(error).__name__}: {error}"}


def enqueue_wall_us(operation, repeats: int = 10, device: torch.device | None = None,
                    queue_drain_every: int = 0) -> dict[str, Any] | None:
    """CPU 侧纯发射耗时（µs/次）：发射 repeats 次不同步，Wall Clock 除以次数。

    口径注意：CUDA/MUSA 异步队列有深度上限（launch 队列满时 enqueue 会阻塞
    等待 GPU 消化），单条 forward 的发射数 << 队列深度，测量有效；
    queue_drain_every>0 时每 N 次同步一次防溢出（此时均值含被阻塞窗口，
    结果标注保守）。该口径度量 eager 的 framework 调度成本（Python+ATen
    dispatch+发射），与 Graph 回放的单次 dispatch 形成对照。
    """
    if device is not None and device.type == "cpu":
        return None
    samples = []
    try:
        for index in range(repeats):
            if queue_drain_every and index and index % queue_drain_every == 0 and device is not None:
                backend_sync(device)
            start = time.perf_counter()
            operation()
            samples.append((time.perf_counter() - start) * 1e6)
        backend_sync(device)
        ordered = sorted(samples)
        return {
            "repeats": repeats,
            "p50_us": round(ordered[len(ordered) // 2], 3),
            "min_us": round(ordered[0], 3),
            "max_us": round(ordered[-1], 3),
            "mean_us": round(mean(samples), 3),
        }
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def memory_snapshot(device: torch.device) -> dict[str, Any]:
    """torch 后端内存统计快照（graph pool 显存驻留测量用）。

    torch_musa 若缺 memory_stats API（版本差异），返回 error 字段而非崩溃
    （调用方以 None 差分处理，实验项记平台边界）。
    """
    if device.type == "cuda":
        stats = torch.cuda.memory_stats(device)
    elif device.type == "musa":
        musa_module = getattr(torch, "musa", None)
        memory_stats = getattr(musa_module, "memory_stats", None)
        if memory_stats is None:
            return {"device": "musa", "error": "torch.musa.memory_stats 不可用（平台边界）"}
        stats = memory_stats(device)
    else:
        return {"device": "cpu"}
    return {
        "allocated_bytes_all_current": stats.get("allocated_bytes.all.current"),
        "reserved_bytes_all_current": stats.get("reserved_bytes.all.current"),
        "segment_all_current": stats.get("segment.all.current"),
    }


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    """最小二乘拟合 y = intercept + slope·x；返回 slope/intercept/r²。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(points)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    syy = sum((y - mean_y) ** 2 for y in ys)
    slope = sxy / sxx if sxx else float("nan")
    intercept = mean_y - slope * mean_x
    ss_tot = syy
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return {"slope": round(slope, 4), "intercept": round(intercept, 4), "r2": round(r2, 6)}


class GraphCaptureCompat:
    """Graph 捕获的平台兼容层（第 09 期 MUSA 变体 2 的统一入口）。

    背景（ep10 S4000 实测）：torch_musa CUDAGraph 捕获期内，default-stream
    （0）上的 kernel **静默不入图**——MUSAGraph.cpp:160 只在整图为空时发
    UserWarning，模型级混合图（ATen kernel 入图 + 注册 op kernel 不入图）
    完全无声、回放输出错误（语义破坏）。CUDA 上 c10::cuda::getCurrentCUDAStream()
    在捕获上下文返回捕获流故无此问题；MUSA 侧对应 API 头文件不可编译
    （M-28/M-31 族），因此 ep10 注册扩展增加变体 2：pybind set_launch_stream
    注入捕获流句柄，kernel 按显式流发射入图。

    用法：
        compat = GraphCaptureCompat(device, pybind_ext)   # musa 传扩展 pybind 模块
        with compat.capture(graph, pool=shared_pool):
            operation()                                   # 捕获窗口
        # 捕获窗口结束后覆盖流自动清零，eager 行为与变体 1 一致

    CUDA 后端等价于 with torch.cuda.graph(graph, pool=pool)（无需注入）。
    """

    def __init__(self, device: torch.device, pybind_ext=None):
        self.device = device
        self.module = backend_module(device)
        self.pybind_ext = pybind_ext
        self.supported = (
            device.type != "cpu"
            and self.module is not None
            and hasattr(self.module, "CUDAGraph")
            and hasattr(self.module, "graph")
        )
        # musa 变体 2 检测：pybind 模块带 set_launch_stream 才能注入
        self.explicit_stream_mode = (
            device.type == "musa"
            and pybind_ext is not None
            and hasattr(pybind_ext, "set_launch_stream")
        )
        self._capture_stream = None

    def get_or_create_capture_stream(self):
        """复用同一捕获流（musa 共享 pool + 显式流组合要求跨捕获同流）。"""
        if self._capture_stream is None:
            self._capture_stream = self.module.Stream()
        return self._capture_stream

    def capture(self, graph, pool=None, stream=None):
        """返回捕获上下文（graph 上下文 + musa 变体 2 流注入）。"""
        compat_self = self

        class _Ctx:
            def __enter__(self):
                kwargs = {}
                if pool is not None:
                    kwargs["pool"] = pool
                if compat_self.device.type == "musa" and compat_self.explicit_stream_mode:
                    cap_stream = stream if stream is not None else compat_self.get_or_create_capture_stream()
                    kwargs["stream"] = cap_stream
                    compat_self.pybind_ext.set_launch_stream(int(cap_stream.musa_stream))
                compat_self._cm = compat_self.module.graph(graph, **kwargs)
                compat_self._cm.__enter__()
                return compat_self

            def __exit__(self, exc_type, exc, tb):
                try:
                    compat_self._cm.__exit__(exc_type, exc, tb)
                finally:
                    if compat_self.device.type == "musa" and compat_self.explicit_stream_mode:
                        compat_self.pybind_ext.set_launch_stream(0)
                return False

        return _Ctx()
