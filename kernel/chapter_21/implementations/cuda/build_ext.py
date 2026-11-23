#!/usr/bin/env python3
"""第 21 期 CUDA torch 扩展构建（ch21/ch25 模式：cpp_extension.load 一次构建）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)


def get_module():
    from torch.utils.cpp_extension import load

    if "pkg22" in sys.modules:
        return sys.modules["pkg22"]
    module = load(
        name="pkg22",
        sources=[str(HERE / "paged_ext.cu")],
        # 不开 -use_fast_math：expf 走精确路径（与 CPU 参考同数值口径）
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(BUILD_DIR),
        verbose=False,
    )
    sys.modules["pkg22"] = module
    return module


if __name__ == "__main__":
    module = get_module()
    print("[build_ext] 构建成功，导出：",
          [name for name in dir(module) if not name.startswith("_")])
