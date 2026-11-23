"""第 03 期 Kernel 绑定层：ctypes 加载 C ABI 共享库（M-17 直传模式）。

设计动机（继承第 16 期模式，协议 §6 三种迁移模式之一）：
- 不走 torch cpp_extension / musa_extension，显存分配/同步/计时全部留在
  torch 侧；Kernel 库只暴露 extern "C" 启动函数（默认流），Python 通过
  data_ptr() 传设备指针；
- CUDA 与 MUSA 两份绑定文件**逐字节相同**（仅默认库名不同）→ 迁移差异
  全部集中在编译器层（nvcc/cudart vs mcc/libmusart）；
- launch 数契约：add_rows / rmsnorm_rows / fused_add_rmsnorm_rows 每次调用
  恰好发射 1 个 Kernel —— 分离路径（add + rmsnorm，2 次）与融合路径
  （1 次）的 Kernel Launch 数差异由此结构保证，并由 torch profiler /
  nsys 实测计数验证。
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

_PTR_P = ctypes.c_void_p    # 设备指针按 void* 传递（torch data_ptr；C 侧 long long 接收）
_LONG_P = ctypes.c_longlong
_INT_P = ctypes.c_int
_FLOAT_S = ctypes.c_float

STATUS_MESSAGES = {
    1: "tile 非法（必须是 {64,128,256,512,1024} 且为本设备 warpSize 的整数倍）",
    2: "H<=0（合同：H=0 为空 Tensor，显式报错）",
    3: "rows<0 / n<0",
}


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    """取设备指针。要求 tensor 连续且驻留在当前后端设备上。"""
    return ctypes.c_void_p(tensor.data_ptr())


class FusedRmsNormKernels:
    """Residual Add + RMSNorm 三入口绑定：分离两步 / 融合一步。"""

    def __init__(self, lib_path: str | Path):
        self.lib_path = str(lib_path)
        self.lib = ctypes.CDLL(self.lib_path)
        signatures = {
            "warp_size_probe": [_PTR_P],
            "add_rows": [_PTR_P, _PTR_P, _PTR_P, _INT_P, _LONG_P, _INT_P],
            "rmsnorm_rows": [_PTR_P, _PTR_P, _PTR_P, _INT_P, _INT_P, _INT_P,
                             _FLOAT_S, _INT_P, _INT_P],
            "fused_add_rmsnorm_rows": [_PTR_P, _PTR_P, _PTR_P, _PTR_P, _INT_P,
                                       _INT_P, _INT_P, _FLOAT_S, _INT_P, _INT_P],
        }
        for name, argument_types in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = argument_types
            function.restype = ctypes.c_int  # 全部返回状态码（0=OK）

    def probe_warp_size(self, device: torch.device) -> int:
        """用 1×1 probe kernel 实测本设备 warpSize（零 Runtime API 依赖）。"""
        out = torch.zeros(1, dtype=torch.int32, device=device)
        status = self.lib.warp_size_probe(_ptr(out))
        if status != 0:
            raise RuntimeError(f"warp_size_probe 失败（status={status}）")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "musa":
            torch.musa.synchronize(device)
        return int(out.item())

    def add_rows(self, r: torch.Tensor, h: torch.Tensor, t: torch.Tensor) -> None:
        """分离路径第 1 步：t = r + h（1 次 Kernel Launch）。"""
        status = self.lib.add_rows(_ptr(r), _ptr(h), _ptr(t),
                                   1 if r.dtype == torch.float16 else 0,
                                   r.numel(), 256)
        if status != 0:
            raise RuntimeError(f"add_rows 参数被拒绝（status={status}: "
                               f"{STATUS_MESSAGES.get(status, '未知')}）")

    def rmsnorm_rows(self, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor,
                     eps: float, tile: int, warp_size: int) -> None:
        """分离路径第 2 步：y = RMSNorm(x)（第 02 期 Kernel，1 次 Launch）。"""
        status = self.lib.rmsnorm_rows(_ptr(x), _ptr(w), _ptr(y),
                                       1 if x.dtype == torch.float16 else 0,
                                       x.shape[0], x.shape[1], eps, tile, warp_size)
        if status != 0:
            raise RuntimeError(f"rmsnorm_rows 参数被拒绝（status={status}: "
                               f"{STATUS_MESSAGES.get(status, '未知')}）")

    def separated(self, r: torch.Tensor, h: torch.Tensor, w: torch.Tensor,
                  t: torch.Tensor, y: torch.Tensor, eps: float,
                  tile: int, warp_size: int) -> None:
        """分离路径整链：add_rows + rmsnorm_rows（共 2 次 Kernel Launch）。"""
        self.add_rows(r, h, t)
        self.rmsnorm_rows(t, w, y, eps, tile, warp_size)

    def fused_add_rmsnorm(self, r: torch.Tensor, h: torch.Tensor, w: torch.Tensor,
                          y: torch.Tensor, eps: float, tile: int,
                          warp_size: int) -> None:
        """融合路径：单 Kernel，r 原位更新为 r+h（合同 in-place 语义），
        y 独立输出（1 次 Kernel Launch）。"""
        status = self.lib.fused_add_rmsnorm_rows(_ptr(r), _ptr(h), _ptr(w), _ptr(y),
                                                 1 if r.dtype == torch.float16 else 0,
                                                 r.shape[0], r.shape[1], eps, tile,
                                                 warp_size)
        if status != 0:
            raise RuntimeError(f"fused_add_rmsnorm_rows 参数被拒绝（status={status}: "
                               f"{STATUS_MESSAGES.get(status, '未知')}）")


def load_kernels(backend: str, lib_name: str) -> FusedRmsNormKernels:
    """从本目录加载对应平台的共享库，缺失时给出可执行动作的报错。"""
    directory = Path(__file__).resolve().parent
    lib_path = directory / lib_name
    if not lib_path.exists():
        raise FileNotFoundError(
            f"{lib_path} 不存在——请先在本目录执行 bash build.sh 构建（backend={backend}）"
        )
    return FusedRmsNormKernels(lib_path)
