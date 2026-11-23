#!/usr/bin/env python3
"""第 05 期 CUDA torch 扩展构建器（结构沿用第 02 期）。

kernel 三 dtype × 2 累加 × 5 tile × 2 vec = 60 个编译期实例一次编入。
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_NAME = "mini_infra_rmsnorm_dtype"
_MODULE_CACHE: dict[str, object] = {}


def build(verbose: bool = True):
    if EXT_NAME in _MODULE_CACHE:
        return _MODULE_CACHE[EXT_NAME]
    import torch
    from torch.utils.cpp_extension import load

    flags = ["-O3", "-std=c++17"]
    module = load(
        name=EXT_NAME,
        sources=[str(HERE / "rmsnorm_ext.cu")],
        extra_cflags=flags,
        extra_cuda_cflags=flags,
        build_directory=str(HERE / "build"),
        verbose=verbose,
    )
    _MODULE_CACHE[EXT_NAME] = module
    return module


def get_module():
    return build(verbose=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    module = build(verbose=arguments.verbose)
    import torch

    # 冒烟自检：三 dtype × 两 acc 各跑一发（cuda 设备），对照组合表达式。
    device = torch.device("cuda")
    eps = 1e-6
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(3, 1024, device=device, dtype=dtype)
        w = torch.randn(1024, device=device, dtype=dtype)
        for acc in ("fp32", "fp16"):
            y = module.rmsnorm(x, w, eps, 256, acc)
            ref = w * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(dtype)
            max_err = (y.float() - ref.float()).abs().max().item()
            print(f"[selftest] dtype={dtype} acc={acc} max_abs_vs_composite={max_err:.3e}")
    print("[selftest] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
