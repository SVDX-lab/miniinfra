"""第 17 期共享工具（复制自 chapter_01/experiment_utils.py 改造）。

改造点：
- schema_version 改为 kernel-chapter18.v1；
- torch 延迟导入（本地机只有 numpy 时，哈希/写盘工具仍可用）；
- 新增 numpy 版比较口径 comparison_metrics_np（本地 CPU 链路用）；
- 新增子进程二进制 JSON 解析 helper（GPU 侧基准由 .cu/.mu 二进制内部计时）。

口径约定（沿用第 01 期合同）：
- 设备区间计时使用后端同步原语 + 事件对；
- 延迟报告 P50/P95，重复次数与 Warm-up 写进结果；
- GPU kernel 延迟由 naive_gemm_bench 二进制用事件对计时（与 torch Event
  同口径：warmup 后逐样本 record-start/launch/record-end，末尾一次同步），
  厂商库（torch.matmul）用 OperatorTimer 事件对——两者同口径可比。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

SCHEMA_VERSION = "kernel-chapter18.v1"


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


def _import_torch():
    import torch

    return torch


def backend_sync(device) -> None:
    torch = _import_torch()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "musa":
        torch.musa.synchronize(device)


def resolve_device(name: str):
    """把合同中的后端名解析成 torch.device，缺失即显式报错，不做静默回退。"""
    torch = _import_torch()
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


def device_environment_record(device) -> dict[str, Any]:
    """采集与本期口径相关的环境与设备信息。"""
    torch = _import_torch()
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
            "nsys": _safe_command(["nsys", "--version"]),
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


def sha256_of_array(array) -> str:
    """numpy 数组字节内容的 SHA256，用于跨机器 fixtures 核对。"""
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def write_results_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"[results] 已写入 {path}")


def comparison_metrics_np(actual, expected) -> dict[str, Any]:
    """numpy 版统一正确性口径（与 ch01 torch 版同式）：max/mean 绝对误差、
    相对误差、余弦相似度、NaN/Inf 计数、最大误差位置。"""
    import numpy as np

    actual = np.asarray(actual, dtype=np.float64).ravel()
    expected = np.asarray(expected, dtype=np.float64).ravel()
    absolute_error = np.abs(actual - expected)
    denominator = np.maximum(np.abs(expected), 1e-6)
    norm_product = np.linalg.norm(actual) * np.linalg.norm(expected)
    cosine = float(np.dot(actual, expected) / norm_product) if norm_product > 0 else float("nan")
    return {
        "max_abs_error": round(float(absolute_error.max()), 8),
        "mean_abs_error": round(float(absolute_error.mean()), 8),
        "mean_rel_error": round(float((absolute_error / denominator).mean()), 8),
        "cosine_similarity": round(cosine, 8),
        "nan_count": int(np.isnan(actual).sum()),
        "inf_count": int(np.isinf(actual).sum()),
        "max_abs_error_position": int(absolute_error.argmax()),
    }


def comparison_metrics(actual, expected) -> dict[str, Any]:
    """torch 版统一正确性口径（ch01 原样）：比较前统一移到 CPU FP32。"""
    torch = _import_torch()
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


def parse_binary_json(stdout_text: str) -> dict[str, Any]:
    """naive_gemm_bench 输出的单行 JSON 提取（允许前缀日志行）。"""
    import re

    matches = re.findall(r"\{.*\}", stdout_text)
    if not matches:
        raise ValueError(f"二进制输出中没有 JSON 行：{stdout_text[:500]}")
    return json.loads(matches[-1])


class OperatorTimer:
    """统一计时器（ch01 原样迁移）：Warm-up 后测 repeats 次，返回 P50/P95（毫秒）。

    device_type == cpu 时直接用 perf_counter；GPU 用事件对逐次记录、末尾一次
    同步（相邻重复的发射延迟与前次执行重叠，GPU 时间线不被 CPU enqueue 空隙
    拉长）。与 naive_gemm_bench 内部计时口径一致。
    """

    def __init__(self, device, warmup: int, repeats: int):
        torch = _import_torch()
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
            torch = _import_torch()
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
