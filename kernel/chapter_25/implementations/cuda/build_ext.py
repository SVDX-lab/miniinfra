#!/usr/bin/env python3
"""第 25 期 CUDA torch 扩展构建器（torch.utils.cpp_extension.load 路径）。

选择记录（沿用第 02 期模式）：cpp_extension.load + PYBIND11_MODULE，
TORCH_LIBRARY 注册路径留给第 04 期。构建产物写 implementations/cuda/build/
（服务器本地，上传/回拉前清理）。

benchmark.py / compare_correctness.py 通过 get_module() 复用本文件，
保证构建入口单源。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_NAME = "mini_infra_dsa_cuda"
_MODULE_CACHE: dict[str, object] = {}


def build(arch: str | None = None, verbose: bool = True):
    """构建（或复用缓存）CUDA torch 扩展，返回可调用的 module。"""
    if EXT_NAME in _MODULE_CACHE:
        return _MODULE_CACHE[EXT_NAME]
    import torch
    from torch.utils.cpp_extension import load

    if arch is None:
        arch = os.environ.get("CUDA_ARCH", "89")
    flags = ["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    (HERE / "build").mkdir(parents=True, exist_ok=True)  # cpp_extension 需要已存在的 build 目录（file_baton 锁）
    module = load(
        name=EXT_NAME,
        sources=[str(HERE / "dsa_sparse.cu")],
        extra_cuda_cflags=flags,
        build_directory=str(HERE / "build"),
        verbose=verbose,
    )
    _MODULE_CACHE[EXT_NAME] = module
    return module


def get_module():
    """给实验脚本用：环境变量 CUDA_ARCH 可覆盖 arch（默认 89 = RTX 4090）。"""
    return build(arch=os.environ.get("CUDA_ARCH"), verbose=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", default=True)
    arguments = parser.parse_args()
    module = build(verbose=arguments.verbose)
    print("[build_ext] 构建完成：", EXT_NAME)
    print("[build_ext] 导出函数:", [name for name in dir(module) if not name.startswith("_")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
