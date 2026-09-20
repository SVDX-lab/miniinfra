"""Environment capture, aggregation, and atomic JSON output for Chapter 42.

Distributed-aware: the record carries world size, the rank-to-device mapping,
per-GPU details, and the interconnect topology that this chapter's
hybrid DP/TP/PP numbers are interpreted against.  Callers pass the arm's
dp/tp/pp degrees; the 3-D rank mapping travels in the chapter payloads.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch


SCHEMA_VERSION = "chapter42.v1"


def git_revision() -> str:
    explicit = os.environ.get("MINI_INFRA_SOURCE_REVISION")
    if explicit:
        return explicit
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _output(arguments: list[str]) -> str:
    result = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return re.sub(r"\x1b\[[0-9;]*m", "", result.stdout).strip()


def environment_record(world_size: int = 1, dp_degree: int = 1, tp_degree: int = 1,
                       pp_degree: int = 1) -> dict[str, Any]:
    memory_kib: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"(MemTotal|MemAvailable):\s+(\d+)\s+kB", line)
            if match:
                memory_kib[match.group(1)] = int(match.group(2))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "world_size": world_size,
        "dp_degree": dp_degree,
        "tp_degree": tp_degree,
        "pp_degree": pp_degree,
        "rank_mapping": {str(rank): f"cuda:{rank}" for rank in range(world_size)},
        "host_memory": {
            "total_bytes": memory_kib.get("MemTotal", 0) * 1024,
            "available_bytes_at_capture": memory_kib.get("MemAvailable", 0) * 1024,
        },
        "environment_variables": {
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE")
            if key in os.environ
        },
    }
    if torch.cuda.is_available():
        devices = []
        for index in range(min(world_size, torch.cuda.device_count())):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
                "driver_version": _output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits", f"--id={index}"]),
            })
        result["gpus"] = devices
        result["gpu"] = devices[0]
        result["nccl_version"] = torch.cuda.nccl.version()
        result["gpu_topology"] = _output(["nvidia-smi", "topo", "-m"])
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]


def aggregate_steps(steps: list[dict[str, float]]) -> dict[str, float]:
    times = [step["step_time_ms"] for step in steps]
    return {
        "step_time_mean_ms": mean(times),
        "step_time_p50_ms": median(times),
        "step_time_p95_ms": percentile(times, 0.95),
        "step_time_max_ms": max(times),
        "loss_first": steps[0]["loss"],
        "loss_last": steps[-1]["loss"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
