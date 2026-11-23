#!/usr/bin/env python3
"""第 13 期手写对照 runner：同会话构建并运行 ch03/ch04 手写 Kernel（对照物）。

来源与保真：本目录 reference_handwritten/ 下的源文件从 chapter_02/ 与
chapter_03/ 逐字节复制（cp 后 sha256 已核对），复制目的：不动其他期目录，
构建产物落在 chapter_13 自己的 build/ 下。对照数字与 ch03/ch04 期 results/
的数字同口径可互验（若同会话实测与历史 JSON 有系统偏差，先怀疑会话环境）。

提供两个 runner：
  - HandwrittenRMSNorm(backend)：ch03 rmsnorm torch 扩展（cuda: cpp_extension；
    musa: torch_musa musa_extension，含 M-08/09/10 处理），run(x, w, eps, tile)；
  - HandwrittenFused(backend)：ch04 ctypes 共享库（M-17 模式），
    run(r, h, w, eps, tile) -> (y, r_updated)，in-place 语义与 ch04 相同。

构建耗时（"手写路径的开发成本"对照数据点之一）会记录进返回值与日志。
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent          # chapter_13/implementations/
CHAPTER_ROOT = HERE.parent                      # chapter_13/
REF = HERE / "reference_handwritten"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HandwrittenRMSNorm:
    """第 02 期手写 RMSNorm torch 扩展（对照物，语义=本期 Triton 版对齐目标）。"""

    name = "handwritten-ch03"

    def __init__(self, backend: str):
        start = time.perf_counter()
        ext_dir = REF / f"ch03_{backend}"
        # cpp_extension.load(build_directory=...) 要求目录先存在（file_baton 在其中建锁文件）
        (ext_dir / "build").mkdir(parents=True, exist_ok=True)
        loader = _load_module_from_path(f"ch03_build_ext_{backend}", ext_dir / "build_ext.py")
        self.module = loader.get_module()
        self.build_seconds = round(time.perf_counter() - start, 1)
        self.backend = backend

    def run(self, x: torch.Tensor, w: torch.Tensor, eps: float, tile: int = 256) -> torch.Tensor:
        return self.module.rmsnorm(x, w, eps, tile)


class HandwrittenFused:
    """第 03 期手写融合 Kernel 库（ctypes C ABI，M-17 模式）。"""

    name = "handwritten-ch04"

    def __init__(self, backend: str, arch: str | None = None):
        start = time.perf_counter()
        src_dir = REF / f"ch04_{backend}"
        build_dir = CHAPTER_ROOT / "build" / f"handwritten_ch04_{backend}"
        build_dir.mkdir(parents=True, exist_ok=True)
        if backend == "cuda":
            arch = arch or "89"
            lib_path = build_dir / "libmini_fused_cuda.so"
            command = ["nvcc", "-O3", "-std=c++17",
                       "-gencode", f"arch=compute_{arch},code=sm_{arch}",
                       "-shared", "-Xcompiler", "-fPIC",
                       str(src_dir / "fused_rmsnorm_kernels.cu"), "-o", str(lib_path)]
        else:
            arch = arch or "mp_22"
            lib_path = build_dir / "libmini_fused_musa.so"
            command = ["mcc", "-O3", "-std=c++17", "--offload-arch=" + arch,
                       "-shared", "-Xcompiler", "-fPIC",
                       str(src_dir / "fused_rmsnorm_kernels.mu"), "-o", str(lib_path)]
        if not lib_path.exists():
            proc = subprocess.run(command, capture_output=True, text=True, timeout=1200)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"手写融合库构建失败（{backend}）：{(proc.stderr or proc.stdout)[-1500:]}")
        self.lib_path = str(lib_path)
        binding = _load_module_from_path(f"ch04_ops_binding_{backend}",
                                         src_dir / "ops_binding.py")
        self.kernels = binding.FusedRmsNormKernels(self.lib_path)
        self.build_seconds = round(time.perf_counter() - start, 1)
        self.warp_size: int | None = None
        self.backend = backend

    def _ensure_warp_size(self, device: torch.device) -> int:
        if self.warp_size is None:
            self.warp_size = self.kernels.probe_warp_size(device)
        return self.warp_size

    def run(self, r: torch.Tensor, h: torch.Tensor, w: torch.Tensor, eps: float,
            tile: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        device = r.device
        warp_size = self._ensure_warp_size(device)
        y = torch.empty_like(r)
        self.kernels.fused_add_rmsnorm(r, h, w, y, eps, tile, warp_size)
        return y, r

    def run_separated(self, r: torch.Tensor, h: torch.Tensor, w: torch.Tensor,
                      eps: float, tile: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        device = r.device
        warp_size = self._ensure_warp_size(device)
        t = torch.empty_like(r)
        y = torch.empty_like(r)
        self.kernels.separated(r, h, w, t, y, eps, tile, warp_size)
        return y, t


def cleanup_builds() -> None:
    """清理本期构建产物（协议：服务器产物及时清理）。"""
    build_root = CHAPTER_ROOT / "build"
    if build_root.exists():
        shutil.rmtree(build_root, ignore_errors=True)


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    device = torch.device(backend)
    x = torch.randn(8, 1024, device=device, dtype=torch.float32)
    w = torch.randn(1024, device=device, dtype=torch.float32)
    h = HandwrittenRMSNorm(backend)
    y = h.run(x, w, 1e-6, 256)
    ref = w * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6))
    print(f"[ref/{backend}] RMSNorm build={h.build_seconds}s "
          f"max_abs_vs_composite={(y - ref).abs().max().item():.3e}")
    r = torch.randn(8, 1024, device=device, dtype=torch.float32)
    hin = torch.randn(8, 1024, device=device, dtype=torch.float32)
    f = HandwrittenFused(backend)
    y2, r2 = f.run(r.clone(), hin, w, 1e-6, 256)
    t = (r + hin)
    ref2 = w * (t.float() * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + 1e-6))
    print(f"[ref/{backend}] Fused build={f.build_seconds}s warp={f.warp_size} "
          f"max_abs={(y2 - ref2).abs().max().item():.3e} "
          f"residual_ok={torch.allclose(r2, t)}")
