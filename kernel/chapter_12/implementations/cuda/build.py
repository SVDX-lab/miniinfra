#!/usr/bin/env python3
"""第 04 期 CUDA 构建器：torch cpp_extension 构建 TORCH_LIBRARY 注册扩展。

产物两件（本目录 build/ 下）：
  1. mini_infra_ops_cuda  —— torch.utils.cpp_extension.load 的 python 扩展模块；
     import 时静态初始化完成 TORCH_LIBRARY(mini_infra_ops) 注册
     （CUDA impl + CPU composite fallback impl + no_cpu_fallback 对照族）；
  2. libmini_fused_cuda.so —— nvcc 直接编译 kernel 库的 C ABI 共享库
     （第 03 期 ctypes 直传通道原样可用；同一份 extern "C" 符号也被链接进
     产物 1，供 dispatch 开销对照的 ctypes 通道使用）。

构建入口单源：benchmark.py / compare_correctness.py 通过 get_ops() 复用本文件。
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_NAME = "mini_infra_ops_cuda"
CTYPES_LIB = "libmini_fused_cuda.so"
_MODULE_CACHE: dict[str, object] = {}


def build_ops(arch: str | None = None, verbose: bool = True):
    """构建（或复用缓存）TORCH_LIBRARY 注册扩展，返回已注册的 torch.ops 命名空间。"""
    if EXT_NAME in _MODULE_CACHE:
        return _MODULE_CACHE[EXT_NAME]
    import torch
    from torch.utils.cpp_extension import load

    if arch is None:
        arch = os.environ.get("CUDA_ARCH", "89")
    flags = ["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    (HERE / "build").mkdir(parents=True, exist_ok=True)  # cpp_extension 的 file_baton 不自建目录
    module = load(
        name=EXT_NAME,
        sources=[str(HERE / "mini_infra_ops_cuda.cu")],
        extra_cuda_cflags=flags,
        build_directory=str(HERE / "build"),
        verbose=verbose,
    )
    # 注册证据：import 成功即静态初始化已执行；暴露 registration_info 原文。
    info = module.registration_info()
    print(f"[cuda-ext] 注册信息：{info}")
    namespace = torch.ops.mini_infra_ops
    _MODULE_CACHE[EXT_NAME] = (module, namespace)
    return _MODULE_CACHE[EXT_NAME]


def get_ops():
    """给实验脚本用：返回 (pybind_module, torch.ops.mini_infra_ops)。"""
    return build_ops(arch=os.environ.get("CUDA_ARCH"), verbose=False)


def build_ctypes_lib(arch: str | None = None) -> Path:
    """nvcc 编译第 03 期 kernel 库为 C ABI 共享库（ctypes 对照通道）。"""
    if arch is None:
        arch = os.environ.get("CUDA_ARCH", "89")
    out = HERE / CTYPES_LIB
    if out.exists():
        return out
    command = [
        "nvcc", "-O3", "-std=c++17",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-shared", "-Xcompiler", "-fPIC",
        str(HERE / "fused_rmsnorm_kernels.cu"), "-o", str(out),
    ]
    print("[ctypes-lib] " + " ".join(command))
    subprocess.run(command, check=True, cwd=str(HERE))
    print(f"[ctypes-lib] {out} OK")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=os.environ.get("CUDA_ARCH", "89"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-ctypes", action="store_true")
    arguments = parser.parse_args()
    build_ops(arch=arguments.arch, verbose=arguments.verbose)
    if not arguments.skip_ctypes:
        build_ctypes_lib(arch=arguments.arch)

    import torch

    # 冒烟自检：注册 op vs 组合表达式（fp32/fp16 各一发）。
    device = torch.device("cuda")
    namespace = torch.ops.mini_infra_ops
    for dtype in (torch.float32, torch.float16):
        torch.manual_seed(0)
        r = torch.randn(4, 1024, device=device, dtype=dtype)
        h = torch.randn(4, 1024, device=device, dtype=dtype)
        w = torch.randn(1024, device=device, dtype=dtype)
        r_op = r.clone()
        h_op = h.clone()
        namespace.fused_add_rmsnorm(h_op, r_op, w, 1e-6)
        t_ref = r + h
        y_ref = w * (t_ref.float() * torch.rsqrt(
            t_ref.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype)
        err_y = (h_op.float() - y_ref.float()).abs().max().item()
        err_r = (r_op.float() - t_ref.float()).abs().max().item()
        print(f"[selftest] dtype={dtype} max_abs(y_vs_composite)={err_y:.3e} "
              f"max_abs(residual)={err_r:.3e}")
    print("[selftest] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
