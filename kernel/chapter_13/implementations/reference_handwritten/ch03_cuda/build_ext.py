#!/usr/bin/env python3
"""第 02 期 CUDA torch 扩展构建器（torch.utils.cpp_extension.load 路径）。

选择记录（合同条目）：本期用 cpp_extension.load + PYBIND11_MODULE，
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
EXT_NAME = "mini_infra_rmsnorm_cuda"
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
    module = load(
        name=EXT_NAME,
        sources=[str(HERE / "rmsnorm_ext.cu")],
        extra_cuda_cflags=flags,
        build_directory=str(HERE / "build"),
        verbose=verbose,
    )
    _MODULE_CACHE[EXT_NAME] = module
    return module


def get_module():
    """给实验脚本用：环境变量 CUDA_ARCH 可覆盖 arch。"""
    return build(arch=os.environ.get("CUDA_ARCH"), verbose=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=os.environ.get("CUDA_ARCH", "89"))
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    module = build(arch=arguments.arch, verbose=arguments.verbose)
    import torch

    # 3 秒冒烟自检：两种 dtype × 向量化/标量路径各跑一发。
    device = torch.device("cuda")
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(3, 1024, device=device, dtype=dtype)
        w = torch.randn(1024, device=device, dtype=dtype)
        y = module.rmsnorm(x, w, 1e-6, 256)
        ref = (w * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype))
        max_err = (y.float() - ref.float()).abs().max().item()
        print(f"[selftest] dtype={dtype} max_abs_vs_composite={max_err:.3e}")
    print("[selftest] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
