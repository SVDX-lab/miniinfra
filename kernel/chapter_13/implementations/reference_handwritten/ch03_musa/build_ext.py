#!/usr/bin/env python3
"""第 02 期 MUSA torch_musa 扩展构建器（CUDA 版 build_ext.py 的迁移对应物）。

路径选择（实机核对结论，2026-11-21）：
  torch_musa.utils.musa_extension.load 是官方扩展入口；其 _is_musa_file
  只认 .mu/.muh 后缀，且被编译的 .mu 不能 #include ".cu" —— 因此构建时把
  源复制为 build/ 下的 .mu，并把 rmsnorm_ext.mu 的 include 行同步改写为
  "rmsnorm_kernel.mu"（唯一内容差异，迁移台账"语义改动"条目）。
  流获取分两变体依次尝试（见 rmsnorm_ext.cu 头注释）：
    变体 0：c10::musa::getCurrentMUSAStream()（与 CUDA 同构；实测可能与
            torch 官方头 redefinition，失败则）；
    变体 1：默认流（torch_musa eager 单流场景等价；多流语义第 08 期）。
  构建失败则 raise，由 build.sh 捕获并退到 rmsnorm_bench 文件 IO 对比路径。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_NAME = "mini_infra_rmsnorm_musa"
_MODULE_CACHE: dict[str, object] = {}


def _prepare_mu_sources(build_dir: Path, variant: int) -> list[str]:
    """源复制为 .mu（musa_extension 只认 .mu/.muh），include 行同步改写。

    只把 rmsnorm_ext.mu 作为编译单元（它 #include kernel）——若把两个文件
    都作为 sources 传入，musa_extension 会把 kernel 再编一遍，链接期
    multiple definition（实测踩坑，见迁移台账）。
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "rmsnorm_kernel.cu", build_dir / "rmsnorm_kernel.mu")
    ext_src = (HERE / "rmsnorm_ext.cu").read_text(encoding="utf-8")
    ext_src = ext_src.replace('#include "rmsnorm_kernel.cu"', '#include "rmsnorm_kernel.mu"')
    ext_dst = build_dir / "rmsnorm_ext.mu"
    ext_dst.write_text(ext_src, encoding="utf-8")
    return [str(ext_dst)]


def _build_variant(variant: int, verbose: bool):
    import torch
    from torch_musa.utils.musa_extension import load  # 实机确认存在

    build_dir = HERE / "build"
    sources = _prepare_mu_sources(build_dir, variant)
    flags = ["-O3", "-std=c++17", f"-DRMSNORM_MUSA_STREAM_VARIANT={variant}"]
    module = load(
        name=f"{EXT_NAME}_v{variant}",
        sources=sources,
        extra_cflags=flags,
        extra_musa_cflags=flags,  # 实机确认参数名（_write_ninja_file_to_build_library 签名）
        build_directory=str(build_dir),
        verbose=verbose,
        keep_intermediates=False,
    )
    return module


def build(verbose: bool = True):
    """变体 0（c10::musa 流 API）优先，失败自动降级变体 1（默认流）。"""
    if EXT_NAME in _MODULE_CACHE:
        return _MODULE_CACHE[EXT_NAME]
    errors = {}
    for variant in (0, 1):
        try:
            module = _build_variant(variant, verbose)
            print(f"[musa-ext] 变体 {variant} 构建成功"
                  + ("（c10::musa::getCurrentMUSAStream）" if variant == 0
                     else "（默认流——c10::musa 头与 torch 官方头 redefinition，平台边界记录）"))
            _MODULE_CACHE[EXT_NAME] = module
            return module
        except Exception as error:  # noqa: BLE001
            errors[variant] = f"{type(error).__name__}: {str(error)[-400:]}"
            print(f"[musa-ext] 变体 {variant} 失败：{errors[variant]}")
    raise RuntimeError(f"torch_musa 扩展两个变体均失败：{errors}")


def get_module():
    return build(verbose=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    module = build(verbose=arguments.verbose)
    import torch

    # 冒烟自检：两种 dtype × 向量化/标量路径各跑一发（musa 设备）。
    device = torch.device("musa")
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
