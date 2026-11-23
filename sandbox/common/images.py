"""固定镜像 digest（大纲第 2 节：固定镜像 digest）。"""
from __future__ import annotations

import subprocess

PINNED = {
    # 主线 shell/系统负载
    "alpine": "alpine:3.19@sha256:6baf43584bcb78f2e5847d1de515f23499913ac9f12bdf834811a3145eb11ca1",
    # Python 负载（计算、内存、确定性轨迹）
    "python": "python:3.12-alpine@sha256:0687a6bc9716edc2a6ee0fbfb0f87e7ee358b262b67c9215de91bc9b2d38ba71",
    # 第 09 期冷启动实验的专用牺牲镜像（其他章节不用，避免破坏共享缓存）
    "busybox": "busybox:1.36@sha256:73aaf090f3d85aa34ee199857f03fa3a95c8ede2ffd4cc2cdb5b94e566b11662",
}


def ensure(ref: str) -> str:
    """确保 digest 固定镜像在本地；缺失则拉取。"""
    r = subprocess.run(["docker", "image", "inspect", ref], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["docker", "pull", ref], check=True, capture_output=True, text=True, timeout=600)
    return ref
