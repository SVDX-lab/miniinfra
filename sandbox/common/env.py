"""环境元数据采集：experiment-protocol.md 第 5 节要求的最低元数据。"""
from __future__ import annotations

import datetime
import os
import platform
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cmd(cmd: list[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() or r.stderr.strip()
    except Exception as e:  # 采集失败不阻塞实验
        return f"unavailable({type(e).__name__})"


def git_commit() -> str:
    out = _cmd(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"])
    return out[:12]


def _meminfo() -> dict:
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal", "MemAvailable", "SwapTotal")):
                    k, v = line.split(":", 1)
                    info[k] = v.strip()
    except OSError:
        pass
    return info


def _cgroup_version() -> str:
    try:
        with open("/etc/mtab") as f:
            for line in f:
                if " /sys/fs/cgroup " in line:
                    return "v2" if "cgroup2" in line.split()[2] else "v1/hybrid"
    except OSError:
        pass
    return "unknown"


def _docker_info() -> dict:
    fmt = ("{{json (index . \"ServerVersion\")}}|{{json (index . \"CgroupVersion\")}}|"
           "{{json (index . \"Driver\")}}|{{json (index . \"KernelVersion\")}}")
    out = _cmd(["docker", "info", "-f", fmt])
    parts = out.split("|")
    if len(parts) == 4:
        return {"server_version": parts[0].strip('"'), "cgroup_version": parts[1].strip('"'),
                "storage_driver": parts[2].strip('"'), "kernel": parts[3].strip('"')}
    return {"raw": out}


def _kvm() -> str:
    if not os.path.exists("/dev/kvm"):
        return "absent"
    try:
        with open("/dev/kvm", "rb"):
            return "readable"
    except OSError:
        return "present-no-perm"


def collect_env_metadata(extra: dict | None = None) -> dict:
    """实验结果必带的最低元数据；章节可通过 extra 补充（配额、负载版本、种子等）。"""
    meta = {
        "utc_time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "os": _cmd(["sh", "-c", ". /etc/os-release && echo $PRETTY_NAME"]) or platform.platform(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_logical": os.cpu_count(),
        "memory": _meminfo(),
        "disk_root_free_gb": round(int(_cmd(["sh", "-c", "df -BG --output=avail / | tail -1"]).strip().rstrip("G")), 1),
        "cgroup": _cgroup_version(),
        "docker": _docker_info(),
        "kvm": _kvm(),
    }
    if extra:
        meta["chapter_extra"] = extra
    return meta
