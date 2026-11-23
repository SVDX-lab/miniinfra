"""第 16 期 Kernel 绑定层：ctypes 加载 C ABI 共享库，把 torch 设备张量指针传给 Kernel。

设计动机（对应协议 §6 迁移对照）：
- 不走 torch cpp_extension，显存分配/同步/计时全部留在 torch 侧；
- Kernel 库只暴露 extern "C" 启动函数（默认流），Python 通过 data_ptr() 传指针；
- CUDA 与 MUSA 两份绑定文件除默认库名外逐行相同 → 迁移差异集中在编译器层。
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

_FLOAT_P = ctypes.c_void_p  # 设备指针按 void* 传递（torch data_ptr）


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    """取设备指针。要求 tensor 连续且驻留在当前后端设备上。"""
    return ctypes.c_void_p(tensor.data_ptr())


class OpsKernels:
    """五个算子的 Kernel 启动封装。所有函数就地写输出张量，调用方负责同步。"""

    def __init__(self, lib_path: str | Path):
        self.lib_path = str(lib_path)
        self.lib = ctypes.CDLL(self.lib_path)
        int_p = ctypes.c_int
        longlong_p = ctypes.c_longlong
        signatures = {
            "softmax_rows": [_FLOAT_P, _FLOAT_P, int_p, int_p],
            "reduce_sum_rows": [_FLOAT_P, _FLOAT_P, int_p, int_p],
            "reduce_max_rows": [_FLOAT_P, _FLOAT_P, int_p, int_p],
            "reduce_sum_max_rows": [_FLOAT_P, _FLOAT_P, _FLOAT_P, int_p, int_p],
            "transpose_naive": [_FLOAT_P, _FLOAT_P, int_p, int_p],
            "transpose_tiled_padded": [_FLOAT_P, _FLOAT_P, int_p, int_p],
            "transpose_tiled_unpadded": [_FLOAT_P, _FLOAT_P, int_p, int_p],
            "scan_hillis": [_FLOAT_P, _FLOAT_P, int_p, int_p],
            "scan_blelloch": [_FLOAT_P, _FLOAT_P, int_p, int_p],
            "silu_only": [_FLOAT_P, _FLOAT_P, longlong_p],
            "mul_only": [_FLOAT_P, _FLOAT_P, _FLOAT_P, longlong_p],
            "silu_gate_fused": [_FLOAT_P, _FLOAT_P, _FLOAT_P, longlong_p],
            "silu_gate_fused_v4": [_FLOAT_P, _FLOAT_P, _FLOAT_P, longlong_p],
        }
        for name, argument_types in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = argument_types
            function.restype = None
        self.lib.silu_gate_fused_v4.restype = ctypes.c_int  # 返回对齐检查状态码

    # -- 1. softmax ----------------------------------------------------------

    def softmax_rows(self, x: torch.Tensor, y: torch.Tensor) -> None:
        rows, cols = x.shape
        self.lib.softmax_rows(_ptr(x), _ptr(y), rows, cols)

    # -- 2. per-row 归约 -------------------------------------------------------

    def reduce_sum_rows(self, x: torch.Tensor, out: torch.Tensor) -> None:
        rows, cols = x.shape
        self.lib.reduce_sum_rows(_ptr(x), _ptr(out), rows, cols)

    def reduce_max_rows(self, x: torch.Tensor, out: torch.Tensor) -> None:
        rows, cols = x.shape
        self.lib.reduce_max_rows(_ptr(x), _ptr(out), rows, cols)

    def reduce_sum_max_rows(self, x: torch.Tensor, sum_out: torch.Tensor, max_out: torch.Tensor) -> None:
        rows, cols = x.shape
        self.lib.reduce_sum_max_rows(_ptr(x), _ptr(sum_out), _ptr(max_out), rows, cols)

    # -- 3. transpose ----------------------------------------------------------

    def transpose(self, x: torch.Tensor, y: torch.Tensor, variant: str) -> None:
        rows, cols = x.shape
        function = {
            "naive": self.lib.transpose_naive,
            "tiled_padded": self.lib.transpose_tiled_padded,
            "tiled_unpadded": self.lib.transpose_tiled_unpadded,
        }[variant]
        function(_ptr(x), _ptr(y), rows, cols)

    # -- 4. scan ---------------------------------------------------------------

    def scan(self, x: torch.Tensor, y: torch.Tensor, variant: str) -> None:
        rows, cols = x.shape
        function = {"hillis": self.lib.scan_hillis, "blelloch": self.lib.scan_blelloch}[variant]
        function(_ptr(x), _ptr(y), rows, cols)

    # -- 5. SiLU + Gate --------------------------------------------------------

    def silu_gate_separated(self, gate: torch.Tensor, up: torch.Tensor, tmp: torch.Tensor,
                            y: torch.Tensor) -> None:
        """分离路径：两次 Kernel 启动（silu → mul）。"""
        n = gate.numel()
        self.lib.silu_only(_ptr(gate), _ptr(tmp), n)
        self.lib.mul_only(_ptr(tmp), _ptr(up), _ptr(y), n)

    def silu_gate_fused(self, gate: torch.Tensor, up: torch.Tensor, y: torch.Tensor) -> None:
        """融合路径：一次 Kernel 启动。"""
        n = gate.numel()
        self.lib.silu_gate_fused(_ptr(gate), _ptr(up), _ptr(y), n)

    def silu_gate_fused_v4(self, gate: torch.Tensor, up: torch.Tensor, y: torch.Tensor) -> bool:
        """float4 向量化融合；n 不是 4 的倍数时回退标量版并返回 False。"""
        n = gate.numel()
        status = self.lib.silu_gate_fused_v4(_ptr(gate), _ptr(up), _ptr(y), n)
        if status != 0:
            self.lib.silu_gate_fused(_ptr(gate), _ptr(up), _ptr(y), n)
            return False
        return True


def load_ops(backend: str, lib_name: str) -> OpsKernels:
    """从本目录加载对应平台的 libops_*.so，缺失时给出可执行动作的报错。"""
    directory = Path(__file__).resolve().parent
    lib_path = directory / lib_name
    if not lib_path.exists():
        raise FileNotFoundError(
            f"{lib_path} 不存在——请先在本目录执行 bash build.sh 构建（backend={backend}）"
        )
    return OpsKernels(lib_path)
