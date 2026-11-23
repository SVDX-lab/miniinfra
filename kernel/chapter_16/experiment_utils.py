"""第 16 期共享工具：环境记录、结果 JSON 读写、计时与哈希。

复制自第 01 期 experiment_utils.py 并做本期改造（协议 §2：复制改造，不 import）：
- schema_version 改为 kernel-chapter17.<name>.v1；
- 增加 enqueue_time：GPU 后端"仅发射不同步"的平均 CPU 耗时，用于 launch-bound 归因
  （第 16 期 SiLU+Gate 的 launch 收益、小 shape 的 launch-bound 判定都要用它）。

口径约定（全篇统一，来源于第 01 期合同）：
- 设备区间计时使用后端同步原语（torch.cuda/musa.synchronize）+ Event；
- 延迟报告 P50/P95，重复次数与 Warm-up 由调用方传入并写进结果。
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

SCHEMA_VERSION = "kernel-chapter17.v1"


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


def enqueue_time_ms(device: torch.device, warmup: int, repeats: int, operation) -> float | None:
    """仅发射不同步：平均一次 operation() 的 CPU 端发射耗时（毫秒）。

    用途：launch-bound 归因。若 event 计时 P50 ≈ enqueue 平均耗时，说明
    GPU 时间线完全被 CPU 发射节奏拉满（算子太快，瓶颈在 launch 而非执行）。
    CPU 后端返回 None（无异步执行，发射与执行不可分）。
    """
    if device.type == "cpu":
        return None
    for _ in range(warmup):
        operation()
    backend_sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        operation()
    enqueue_wall_ms = (time.perf_counter() - start) * 1000.0 / repeats
    backend_sync(device)
    return round(enqueue_wall_ms, 4)


def comparison_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """统一正确性比较口径：max/mean 绝对误差、相对误差、余弦相似度、NaN/Inf。

    统一把两侧移到 CPU 再比较，允许后端结果与真值跨设备传入。
    """
    actual_float = actual.detach().double().cpu()
    expected_float = expected.detach().double().cpu()
    absolute_error = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(1e-6)
    # cosine 用 float64 归一：float32 归约的舍入会让完全一致的两个向量得到 cos>1
    # （4090 实测出现过 1.0014），指标口径必须量纲自洽。
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
