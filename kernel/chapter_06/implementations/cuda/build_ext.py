#!/usr/bin/env python3
"""第 06 期 CUDA torch 扩展构建器（第 02 期同款，单编译单元链）。

编译单元：rmsnorm_ext.cu → include rmsnorm_stride_kernel.cu → include
rmsnorm_kernel.cu（cpp_extension 单 TU 无 multiple definition 问题，
第 02 期 M-09 口径）。构建产物写 implementations/cuda/build/（服务器本地，
上传/回拉前清理）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_NAME = "mini_infra_rmsnorm_layout_cuda"
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

    # 冒烟自检：三策略 × 两种布局（连续 / 转置视图）各跑一发。
    device = torch.device("cuda")
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(3, 1024, device=device, dtype=dtype)
        tv = x.t().contiguous().t()          # 转置视图（逻辑同 x）
        w = torch.randn(1024, device=device, dtype=dtype)
        y_contig = module.rmsnorm_policy_a(x, w, 1e-6, 256)
        y_b = module.rmsnorm_policy_b(tv, w, 1e-6, 256)
        y_c = module.rmsnorm_policy_c(tv, w, 1e-6, 256)
        ref = (w * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype))
        for name, y in (("A(contig)", y_contig), ("B(tv)", y_b), ("C(tv)", y_c)):
            max_err = (y.float() - ref.float()).abs().max().item()
            print(f"[selftest] dtype={dtype} policy={name} max_abs_vs_composite={max_err:.3e}")
    # 策略 A 对非连续输入必须报错。
    try:
        module.rmsnorm_policy_a(tv, w, 1e-6, 256)
        raise RuntimeError("策略 A 未拒绝非连续输入——合同行为失效")
    except RuntimeError as error:
        assert "策略 A" in str(error), f"策略 A 报错文本异常: {error}"
        print("[selftest] 策略 A 对非连续输入按合同报错 OK")
    print("[selftest] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
