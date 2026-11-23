"""docker CLI 通用封装：只操作带本项目 label 的对象（安全范围）。"""
from __future__ import annotations

import subprocess

LABEL_CHAPTER = "mis.chapter"
LABEL_SESSION = "mis.session"


def docker(args: list[str], timeout: float = 120, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          timeout=timeout, check=check)


def ps_ids(label_filter: str, include_stopped: bool = True) -> list[str]:
    args = ["ps", "-aq" if include_stopped else "-q", "--filter", f"label={label_filter}"]
    r = docker(args)
    return [x for x in r.stdout.split() if x]


def kill_ids(ids: list[str]) -> None:
    for i in ids:
        docker(["kill", i], timeout=60)


def cleanup_chapter(chapter: str) -> dict:
    """删除本章 label 的容器/网络/卷，保证每期从干净环境开始（协议第 1 节）。"""
    report = {}
    ids = ps_ids(f"{LABEL_CHAPTER}={chapter}")
    if ids:
        kill_ids(ids)
        docker(["rm", "-f", *ids], timeout=120)
    report["containers_removed"] = len(ids)

    r = docker(["network", "ls", "-q", "--filter", f"label={LABEL_CHAPTER}={chapter}"])
    nets = [x for x in r.stdout.split() if x]
    for n in nets:
        docker(["network", "rm", n], timeout=60)
    report["networks_removed"] = len(nets)

    r = docker(["volume", "ls", "-q", "--filter", f"label={LABEL_CHAPTER}={chapter}"])
    vols = [x for x in r.stdout.split() if x]
    for v in vols:
        docker(["volume", "rm", v], timeout=60)
    report["volumes_removed"] = len(vols)
    return report
