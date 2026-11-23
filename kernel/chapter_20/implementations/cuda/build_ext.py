#!/usr/bin/env python3
"""第 20 期 CUDA torch 扩展构建（ch25 模式：cpp_extension.load，一次构建进程内复用）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)  # cpp_extension file_baton 需要


def get_module():
    from torch.utils.cpp_extension import load

    if "atn21" in sys.modules:
        return sys.modules["atn21"]
    module = load(
        name="atn21",
        sources=[str(HERE / "attention_ext.cu")],
        # 不开 -use_fast_math：expf 走精确路径（数值口径与 CPU 参考一致）
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(BUILD_DIR),
        verbose=False,
    )
    sys.modules["atn21"] = module
    return module


if __name__ == "__main__":
    module = get_module()
    print("[build_ext] 构建成功，导出：", [name for name in dir(module) if not name.startswith("_")])
