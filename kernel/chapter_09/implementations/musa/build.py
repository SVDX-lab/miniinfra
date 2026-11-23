#!/usr/bin/env python3
"""第 04 期 MUSA 构建器：torch_musa 扩展注册路径（PrivateUse1 dispatch）验证。

本期正式验证项（第 03 期遗留）：torch_musa 的 TORCH_LIBRARY 注册路径能否工作。
已知坑（台账 M-08/M-09/M-10，ep03 实测）：
  - _is_musa_file 只认 .mu/.muh → 源复制为 .mu（本文件源码的 include 行已直接
    写成 .mu，无需改写内容，只做后缀重命名）；
  - 多源文件逐个编译会 multiple definition → 只传注册 TU 单编译单元；
  - c10/musa/MUSAStream.h 与 torch 官方头 redefinition → 变体 0（流 API）预期
    失败、变体 1（默认流）为可用路径；两变体都尝试并留证据。

产物两件：
  1. mini_infra_ops_musa_v{0,1} —— musa_extension.load 的 python 扩展模块，
     TORCH_LIBRARY 注册（PrivateUse1 impl + CPU composite fallback impl）；
  2. libmini_fused_musa.so —— mcc 编译 kernel 库的 C ABI 共享库（ep04 ctypes
     对照通道）。

若变体 0/1 都不可行（注册路径失败），回退方案（记录路径差异）：
  pybind 直调函数 + Python 显式 dispatch 封装（ops_router.build_gpu_router_musa_pybind）
  ——"进入框架 dispatch"在 MUSA 上退化为"进入显式注册的 Python 路由"。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_NAME = "mini_infra_ops_musa"
CTYPES_LIB = "libmini_fused_musa.so"
_MODULE_CACHE: dict[str, object] = {}


def _prepare_mu_sources(build_dir: Path) -> list[str]:
    """musa_extension 只认 .mu/.muh（M-08）：注册 TU 复制为 .mu。

    include 行已直接写成 "fused_rmsnorm_kernels.mu"（无 .cu 内容改写，
    与 ep03 相比少一处机械替换）；kernel 库原文件即 .mu，但随注册 TU 一并
    复制进 build/（相对 include 从被编译文件的目录解析）。
    只传注册 TU 单编译单元（M-09，kernel 经 #include 进入同一编译单元）。
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "mini_infra_ops_musa.cu", build_dir / "mini_infra_ops_musa.mu")
    shutil.copyfile(HERE / "fused_rmsnorm_kernels.mu", build_dir / "fused_rmsnorm_kernels.mu")
    shutil.copyfile(HERE / "rmsnorm_kernel_ep03.mu", build_dir / "rmsnorm_kernel_ep03.mu")
    return [str(build_dir / "mini_infra_ops_musa.mu")]


def _build_variant(variant: int, verbose: bool):
    import torch
    from torch_musa.utils.musa_extension import load  # 实机确认存在（ep03）

    build_dir = HERE / "build"
    sources = _prepare_mu_sources(build_dir)
    flags = ["-O3", "-std=c++17", f"-DMINI_INFRA_MUSA_STREAM_VARIANT={variant}"]
    module = load(
        name=f"{EXT_NAME}_v{variant}",
        sources=sources,
        extra_cflags=flags,
        extra_musa_cflags=flags,  # 实机确认参数名（ep03 同款）
        build_directory=str(build_dir),
        verbose=verbose,
        keep_intermediates=False,
    )
    return module


def build(verbose: bool = True):
    """变体 0（c10::musa 流 API）先行（M-28/M-31 证据），失败降级变体 2
    （ep10：默认流 + 显式覆盖流，Graph 捕获兼容），再降级变体 1（纯默认流）。"""
    if EXT_NAME in _MODULE_CACHE:
        return _MODULE_CACHE[EXT_NAME]
    errors = {}
    for variant in (0, 2, 1):
        try:
            module = _build_variant(variant, verbose)
            import torch

            info = module.registration_info()
            print(f"[musa-ext] 变体 {variant} 构建成功：{info}")
            namespace = torch.ops.mini_infra_ops
            _MODULE_CACHE[EXT_NAME] = (module, namespace, variant)
            return _MODULE_CACHE[EXT_NAME]
        except Exception as error:  # noqa: BLE001
            errors[variant] = f"{type(error).__name__}: {str(error)[-800:]}"
            print(f"[musa-ext] 变体 {variant} 失败：{errors[variant]}")
    raise RuntimeError(f"torch_musa 注册扩展三个变体均失败：{errors}")


def get_ops():
    """给实验脚本用：返回 (pybind_module, torch.ops 命名空间, 变体号)。"""
    return build(verbose=False)


def build_ctypes_lib() -> Path:
    """mcc 编译第 03 期 kernel 库为 C ABI 共享库（ctypes 对照通道）。"""
    out = HERE / CTYPES_LIB
    if out.exists():
        return out
    command = [
        "mcc", "-O3", "-std=c++17", "-shared", "-fPIC",
        "--offload-arch=mp_22",  # M-18：mcc CLI 下 -march=mp_22 不可用
        str(HERE / "fused_rmsnorm_kernels.mu"), "-o", str(out),
    ]
    print("[ctypes-lib] " + " ".join(command))
    subprocess.run(command, check=True, cwd=str(HERE))
    print(f"[ctypes-lib] {out} OK")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-ctypes", action="store_true")
    arguments = parser.parse_args()
    module, namespace, variant = build(verbose=arguments.verbose)
    if not arguments.skip_ctypes:
        build_ctypes_lib()

    import torch

    # 冒烟自检：注册 op vs 组合表达式（fp32/fp16 各一发，musa 设备）。
    device = torch.device("musa")
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
    print(f"[selftest] OK（stream_variant={variant}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
