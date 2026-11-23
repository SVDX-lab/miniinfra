#!/usr/bin/env python3
"""第 06 期 MUSA torch_musa 扩展构建器（第 02 期同款变体机制 + 三文件 include 链）。

编译单元链：rmsnorm_ext → include rmsnorm_stride_kernel → include rmsnorm_kernel
（musa_extension 只认 .mu/.muh 且 .mu 内不能 include .cu——M-08：三个源都复制
为 .mu 并同步改写 include 行；只把 ext 作为编译单元避免 multiple definition
——M-09）。流获取两变体依次尝试（M-10 族，见 rmsnorm_ext.cu 头注释）。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_NAME = "mini_infra_rmsnorm_layout_musa"
_MODULE_CACHE: dict[str, object] = {}


def _prepare_mu_sources(build_dir: Path, variant: int) -> list[str]:
    """三个源复制为 .mu 并改写 include 链（唯一内容差异，迁移台账条目）。"""
    build_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "rmsnorm_kernel.cu", build_dir / "rmsnorm_kernel.mu")
    stride_src = (HERE / "rmsnorm_stride_kernel.cu").read_text(encoding="utf-8")
    stride_src = stride_src.replace('#include "rmsnorm_kernel.cu"',
                                    '#include "rmsnorm_kernel.mu"')
    (build_dir / "rmsnorm_stride_kernel.mu").write_text(stride_src, encoding="utf-8")
    ext_src = (HERE / "rmsnorm_ext.cu").read_text(encoding="utf-8")
    ext_src = ext_src.replace('#include "rmsnorm_stride_kernel.cu"',
                              '#include "rmsnorm_stride_kernel.mu"')
    ext_dst = build_dir / "rmsnorm_ext.mu"
    ext_dst.write_text(ext_src, encoding="utf-8")
    return [str(ext_dst)]


def _build_variant(variant: int, verbose: bool):
    import torch
    from torch_musa.utils.musa_extension import load

    build_dir = HERE / "build"
    sources = _prepare_mu_sources(build_dir, variant)
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
    """变体 0（c10::musa 流 API）优先，失败自动降级变体 1（默认流）。"""
    if EXT_NAME in _MODULE_CACHE:
        return _MODULE_CACHE[EXT_NAME]
    errors = {}
    for variant in (0, 1):
        try:
            module = _build_variant(variant, verbose)
            print(f"[musa-ext] 变体 {variant} 构建成功"
                  + ("（c10::musa::getCurrentMUSAStream）" if variant == 0
                     else "（默认流——c10::musa 头与 torch 官方头冲突，平台边界记录）"))
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

    # 冒烟自检：三策略 × 连续/转置视图（musa 设备）。
    device = torch.device("musa")
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(3, 1024, device=device, dtype=dtype)
        tv = x.t().contiguous().t()
        w = torch.randn(1024, device=device, dtype=dtype)
        y_a = module.rmsnorm_policy_a(x, w, 1e-6, 256)
        y_b = module.rmsnorm_policy_b(tv, w, 1e-6, 256)
        y_c = module.rmsnorm_policy_c(tv, w, 1e-6, 256)
        ref = (w * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype))
        for name, y in (("A(contig)", y_a), ("B(tv)", y_b), ("C(tv)", y_c)):
            max_err = (y.float() - ref.float()).abs().max().item()
            print(f"[selftest] dtype={dtype} policy={name} max_abs_vs_composite={max_err:.3e}")
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
