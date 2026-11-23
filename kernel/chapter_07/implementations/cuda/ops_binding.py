"""第 07 期 Kernel 绑定层：ctypes 加载 C ABI 共享库（M-17/M-21 直传模式）。

设计（继承第 03 期模式）：
- 不走 torch cpp_extension / musa_extension；显存分配/同步/计时全留 torch 侧；
- 三入口 launch 数契约（由 extern "C" 启动器结构保证，nsys/profiler 可验）：
    rmsnorm_det_rows            恰好 1 个 Kernel（确定性 warp/block 树形归约）
    rmsnorm_atomic_shared_rows  恰好 1 个 Kernel（shared atomic 交叉归约）
    rmsnorm_atomic_global_rows  恰好 2 个 Kernel（sqsum + scale），
                                且调用前须 zero_acc（1 个 Kernel）→ 整链 3 个；
- CUDA 与 MUSA 两份绑定文件逐字节相同（仅默认库名不同）→ 迁移差异全部
  集中在编译器层（nvcc/cudart vs mcc/libmusart），device 代码零改动。

warp_size 由 probe_warp_size() 实测传入（M-20：device 内建口径，CUDA 32 /
MUSA 32；C API 的 musaDeviceProp.warpSize=128 是"调度 warp"概念，不用作
guard）。
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

_PTR_P = ctypes.c_void_p
_LONG_P = ctypes.c_longlong
_INT_P = ctypes.c_int
_FLOAT_S = ctypes.c_float

STATUS_MESSAGES = {
    1: "tile 非法（须为本设备 warpSize 实测值的整数倍：CUDA 32 / MUSA probe 实测）",
    2: "H<=0（合同：H=0 空 Tensor 显式报错）",
    3: "rows<0",
    4: "ACC=half 被编译开关关闭（EP08_NO_HALF_ATOMIC：本平台 half atomicAdd 缺失的平台边界）",
    5: "splits 非法（1..1024）",
    10: "host dispatch 抛异常（见 C 侧 detail）",
}


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


class DeterminismKernels:
    """第 07 期三实现绑定：det / atomic_shared / atomic_global。"""

    def __init__(self, lib_path: str | Path):
        self.lib_path = str(lib_path)
        self.lib = ctypes.CDLL(self.lib_path)
        signatures = {
            "warp_size_probe": [_PTR_P],
            "rmsnorm_det_rows": [_PTR_P, _PTR_P, _PTR_P, _INT_P, _INT_P, _INT_P,
                                 _FLOAT_S, _INT_P, _INT_P],
            "rmsnorm_atomic_shared_rows": [_PTR_P, _PTR_P, _PTR_P, _INT_P, _INT_P,
                                           _INT_P, _INT_P, _FLOAT_S, _INT_P, _INT_P],
            "rmsnorm_atomic_global_rows": [_PTR_P, _PTR_P, _PTR_P, _PTR_P, _INT_P, _INT_P,
                                           _INT_P, _INT_P, _FLOAT_S, _INT_P, _INT_P, _INT_P],
            "zero_acc": [_PTR_P, _INT_P, _INT_P],
        }
        for name, argument_types in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = argument_types
            function.restype = ctypes.c_int

    def probe_warp_size(self, device: torch.device) -> int:
        out = torch.zeros(1, dtype=torch.int32, device=device)
        status = self.lib.warp_size_probe(_ptr(out))
        if status != 0:
            raise RuntimeError(f"warp_size_probe 失败（status={status}）")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "musa":
            torch.musa.synchronize(device)
        return int(out.item())

    @staticmethod
    def _is_fp16(tensor: torch.Tensor) -> int:
        return 1 if tensor.dtype == torch.float16 else 0

    def det(self, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, eps: float,
            tile: int, warp_size: int) -> None:
        """确定性 warp/block 树形归约（ep03 kernel，1 launch）。"""
        status = self.lib.rmsnorm_det_rows(_ptr(x), _ptr(w), _ptr(y),
                                           self._is_fp16(x), x.shape[0], x.shape[1],
                                           eps, tile, warp_size)
        if status != 0:
            raise RuntimeError(f"rmsnorm_det_rows 被拒绝（status={status}: "
                               f"{STATUS_MESSAGES.get(status, '未知')}）")

    def atomic_shared(self, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, eps: float,
                      tile: int, warp_size: int, acc_is_half: bool = False) -> None:
        """shared atomic 交叉归约（1 launch；每线程局部和原子累加到行累加器）。"""
        status = self.lib.rmsnorm_atomic_shared_rows(
            _ptr(x), _ptr(w), _ptr(y), self._is_fp16(x), 1 if acc_is_half else 0,
            x.shape[0], x.shape[1], eps, tile, warp_size)
        if status != 0:
            raise RuntimeError(f"rmsnorm_atomic_shared_rows 被拒绝（status={status}: "
                               f"{STATUS_MESSAGES.get(status, '未知')}）")

    def atomic_global(self, x: torch.Tensor, w: torch.Tensor, g_acc: torch.Tensor,
                      y: torch.Tensor, eps: float, tile: int, warp_size: int,
                      splits: int, acc_is_half: bool = False) -> None:
        """global atomic 交叉归约（zero_acc + sqsum + scale = 3 launches 整链）。"""
        status_zero = self.lib.zero_acc(_ptr(g_acc), x.shape[0], 1 if acc_is_half else 0)
        if status_zero != 0:
            raise RuntimeError(f"zero_acc 被拒绝（status={status_zero}: "
                               f"{STATUS_MESSAGES.get(status_zero, '未知')}）")
        status = self.lib.rmsnorm_atomic_global_rows(
            _ptr(x), _ptr(w), _ptr(g_acc), _ptr(y), self._is_fp16(x), 1 if acc_is_half else 0,
            x.shape[0], x.shape[1], eps, tile, warp_size, splits)
        if status != 0:
            raise RuntimeError(f"rmsnorm_atomic_global_rows 被拒绝（status={status}: "
                               f"{STATUS_MESSAGES.get(status, '未知')}）")


def load_kernels(backend: str, lib_name: str) -> DeterminismKernels:
    directory = Path(__file__).resolve().parent
    lib_path = directory / lib_name
    if not lib_path.exists():
        raise FileNotFoundError(
            f"{lib_path} 不存在——请先在本目录执行 bash build.sh 构建（backend={backend}）")
    return DeterminismKernels(lib_path)
