#!/usr/bin/env python3
"""第 05 期 MUSA torch_musa 扩展构建器（CUDA 版 build_ext.py 的迁移对应物；
第 02 期实机核对结论全部沿用）：

  torch_musa.utils.musa_extension.load 只认 .mu/.muh 后缀，且被编译的 .mu
  不能 #include ".cu" —— 构建时把源复制为 build/ 下的 .mu，并把
  rmsnorm_ext.mu 的 include 行同步改写为 "rmsnorm_kernel.mu"（唯一内容差异）。
  流获取分两变体依次尝试（变体 0 = c10::musa 流 API；变体 1 = 默认流；
  第 02 期实测变体 0 与 torch 官方头 redefinition，变体 1 成功）。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_NAME = "mini_infra_rmsnorm_dtype_musa"
_MODULE_CACHE: dict[str, object] = {}


def _prepare_mu_sources(build_dir: Path) -> list[str]:
    build_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "rmsnorm_kernel.cu", build_dir / "rmsnorm_kernel.mu")
    ext_src = (HERE / "rmsnorm_ext.cu").read_text(encoding="utf-8")
    ext_src = ext_src.replace('#include "rmsnorm_kernel.cu"', '#include "rmsnorm_kernel.mu"')
    ext_dst = build_dir / "rmsnorm_ext.mu"
    ext_dst.write_text(ext_src, encoding="utf-8")
    return [str(ext_dst)]


def _build_variant(variant: int, verbose: bool):
    import torch
    from torch_musa.utils.musa_extension import load

    build_dir = HERE / "build"
    sources = _prepare_mu_sources(build_dir)
    flags = ["-O3", "-std=c++17", f"-DRMSNORM_MUSA_STREAM_VARIANT={variant}"]
    module = load(
        name=f"{EXT_NAME}_v{variant}",
        sources=sources,
        extra_cflags=flags,
        extra_musa_cflags=flags,
        build_directory=str(build_dir),
        verbose=verbose,
        keep_intermediates=False,
    )
    return module


def build(verbose: bool = True):
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

    # 冒烟自检：三 dtype × 两 acc 各跑一发（musa 设备）。
    device = torch.device("musa")
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
