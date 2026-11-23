"""第 08 期共享工具：环境记录、结果 JSON 读写、UTC 时间戳。

与第 01 期口径的关系（协议 §2：复制改造，不跨目录 import）：
- schema_version = kernel-chapter09.v1；
- 环境记录沿用 ch01 字段风格（设备/驱动/SDK/Python/torch），补充：
  PCIe 总线号、libc 运行时、传输实验关注的 host 内存总量；
- 计时口径：设备侧用事件对（经 ctypes 库的 t_event_*），Host 侧用
  perf_counter —— 本期"Host 侧只做计时基准"，两种口径分开记录。

注意：本期的数据面（分配/传输/计算）全部经 ctypes 直传（M-17 模式），
torch 仅用于环境记录与 torch.<backend>.Stream 探测（stream_probe.py），
不参与性能路径。
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "kernel-chapter09.v1"

ROOT = Path(__file__).resolve().parent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    explicit_revision = os.environ.get("MINI_INFRA_SOURCE_REVISION")
    if explicit_revision:
        return explicit_revision
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True, cwd=str(ROOT),
    )
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-repo"


def _safe_command(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(arguments, check=False, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _first_line(text: str | None) -> str | None:
    if not text:
        return text
    for line in text.splitlines():
        if line.strip() and "Model name" not in line and ":" not in line[:20]:
            return line.strip()[:120]
        if "Model name" in line:
            return line.split(":", 1)[1].strip()[:120]
    return text.strip()[:120]


def host_record() -> dict[str, Any]:
    """宿主 CPU / 内存 / OS（传输实验的 Host 侧环境）。"""
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, val = line.partition(":")
            meminfo[key.strip()] = val.strip()
    except OSError:
        pass
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _first_line(_safe_command(["lscpu"])),
        "cpu_cores": os.cpu_count(),
        "mem_total": meminfo.get("MemTotal"),
        "mem_available": meminfo.get("MemAvailable"),
        "hugepages_total": meminfo.get("HugePages_Total"),
        "transparent_hugepage": _safe_command(
            ["sh", "-c", "cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | head -c 64"]
        ),
    }


def backend_environment_record(backend: str) -> dict[str, Any]:
    """每平台环境快照（environment-cuda.json / environment-musa.json）。"""
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "git_revision": git_revision(),
        "backend": backend,
        "host": host_record(),
    }
    import torch  # 仅环境记录用，不进性能路径

    record["torch"] = torch.__version__
    if backend == "cuda":
        props = torch.cuda.get_device_properties(0)
        record["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
        }
        record["driver"] = _safe_command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        record["tool_versions"] = {"nvcc": _safe_command(["nvcc", "--version"])}
    elif backend == "musa":
        props = torch.musa.get_device_properties(0)
        record["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
        }
        record["driver"] = _safe_command(["mthreads-gmi"])
        record["tool_versions"] = {"mcc": _safe_command(["mcc", "--version"])}
    return record


def write_results_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def _default(o: Any) -> Any:
        # numpy 标量 / torch 标量（如 Stream.__eq__ 返回 torch.bool）兜底
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_default),
                   encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
