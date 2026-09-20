"""Small shared utilities for Chapter 17 experiment records."""

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


SCHEMA_VERSION = "chapter17.v1"


def git_revision() -> str:
    explicit_revision = os.environ.get("MINI_INFRA_SOURCE_REVISION")
    if explicit_revision:
        return explicit_revision
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _command_output(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
    )
    return re.sub(r"\x1b\[[0-9;]*m", "", result.stdout).strip()


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32,
        "environment_variables": {
            name: os.environ[name]
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "NCCL_P2P_DISABLE",
                "NCCL_IB_DISABLE",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            )
            if name in os.environ
        },
        "world_size": 1,
        "dp_degree": 1,
        "tp_degree": 1,
        "pp_degree": 1,
        "rank_mapping": {"0": "cuda:0" if torch.cuda.is_available() else "cpu"},
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        record["gpu"] = {
            "index": device,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(device)),
            "driver_version": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                    f"--id={device}",
                ]
            ),
        }
        record["nccl_version"] = torch.cuda.nccl.version()
        record["gpu_topology"] = _command_output(["nvidia-smi", "topo", "-m"])
    return record


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


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
