"""第 16 期共享实验骨架：算子×变体注册表、理论字节/FLOP 公式、输入装载。

compare_correctness.py 与 benchmark.py 共用，保证"跑的是同一批变体、
算的是同一套理论字节"——口径单源。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "implementations" / "cpu"))
sys.path.insert(0, str(ROOT / "implementations" / "cuda"))

from ops_reference import (  # noqa: E402
    reduce_max_rows_numpy,
    reduce_sum_rows_numpy,
    silu_gate_numpy,
    softmax_rows_numpy,
    scan_inclusive_rows_numpy,
    transpose_numpy,
)

ROW_COUNTS = [1, 128, 2048]
COL_COUNTS = [127, 1024, 4096]
SHAPES = [(rows, cols) for rows in ROW_COUNTS for cols in COL_COUNTS]

# 每算子的变体清单（GPU 变体由 Kernel 提供；CPU 侧以参考实现与 torch 组合式为变体）
GPU_VARIANTS: dict[str, list[str]] = {
    "softmax": ["kernel"],
    "reduce": ["sum", "max", "sum_max_fused"],
    "transpose": ["naive", "tiled_unpadded", "tiled_padded"],
    "scan": ["hillis", "blelloch"],
    "silu_gate": ["separated", "fused", "fused_v4"],
}
TORCH_EAGER_VARIANTS: dict[str, str | None] = {
    "softmax": "torch_softmax",
    "reduce": "torch_sum_max",
    "transpose": "torch_t_contiguous",
    "scan": "torch_cumsum",
    "silu_gate": "torch_silu_mul",
}
CPU_VARIANTS: dict[str, list[str]] = {
    "softmax": ["reference_numpy"],
    "reduce": ["reference_numpy"],
    "transpose": ["reference_numpy"],
    "scan": ["reference_numpy"],
    "silu_gate": ["reference_numpy"],
}

# 平台峰值带宽（GB/s），来源：大纲 §3.3 规格表（与第 01 期 benchmark.py 同源）
PEAK_BANDWIDTH_GB_S = {"cuda": 1008.0, "musa": 768.0, "cpu": None}


CPU_MODEL_VARIANT = {
    "softmax": "softmax_kernel",
    "reduce": "reduce_sum_max_fused",
    "transpose": "transpose_naive",
    "scan": "scan_hillis",
    "silu_gate": "silu_gate_fused",
}


def resolve_model_variant(op: str, variant: str) -> str:
    """把 GPU_VARIANTS 的短名（kernel/sum/naive/hillis/…）映射到理论字节模型名。"""
    if variant in ("softmax_kernel", "reduce_sum", "reduce_max", "reduce_sum_max_fused",
                   "silu_gate_separated", "silu_gate_fused", "silu_gate_fused_v4",
                   "torch_softmax", "torch_sum_max", "torch_t_contiguous",
                   "torch_cumsum", "torch_silu_mul", "reference_numpy"):
        return variant
    if variant.startswith("transpose_") or variant.startswith("scan_"):
        return variant
    if op == "softmax":
        return "softmax_kernel"
    if op == "reduce":
        return {"sum": "reduce_sum", "max": "reduce_max", "sum_max_fused": "reduce_sum_max_fused"}[variant]
    if op == "transpose":
        return f"transpose_{variant}"
    if op == "scan":
        return f"scan_{variant}"
    if op == "silu_gate":
        return f"silu_gate_{variant}"
    return variant


def theoretical_bytes(rows: int, cols: int, variant: str, itemsize: int = 4,
                      op: str | None = None) -> int:
    """每变体的全局显存理论字节数（含中间物化）。

    - softmax kernel：读 N + 写 N（max/sum 归约在片上完成）= 2N；
    - reduce sum/max：读 N + 写 rows；sum_max 融合：读 N + 写 2*rows；
    - transpose：读 N + 写 N = 2N；
    - scan（两种组织）：读 N + 写 N = 2N；
    - silu_gate separated：silu(读N+写N) + mul(读2N+写N) = 5N（物化中间 tmp）；
      fused/v4：读 2N + 写 N = 3N（无中间物化）。
    """
    n = rows * cols
    if variant == "softmax_kernel":
        return (2 * n) * itemsize
    if variant in ("reduce_sum", "reduce_max"):
        return (n + rows) * itemsize
    if variant == "reduce_sum_max_fused":
        return (n + 2 * rows) * itemsize
    if variant.startswith("transpose_") or variant.startswith("scan_"):
        return (2 * n) * itemsize
    if variant == "silu_gate_separated":
        return (5 * n) * itemsize
    if variant in ("silu_gate_fused", "silu_gate_fused_v4"):
        return (3 * n) * itemsize
    if variant == "reference_numpy" and op is not None:
        return theoretical_bytes(rows, cols, CPU_MODEL_VARIANT[op], itemsize)
    if variant in ("torch_softmax",):
        return (2 * n) * itemsize  # torch 单核 softmax 同为 2N 模型
    if variant == "torch_sum_max":
        return (2 * n + 2 * rows) * itemsize  # sum、max 各读一遍（eager 不融合）
    if variant == "torch_t_contiguous":
        return (2 * n) * itemsize
    if variant == "torch_cumsum":
        return (2 * n) * itemsize
    if variant == "torch_silu_mul":
        return (5 * n) * itemsize  # silu 写 tmp、mul 读回 → 5N（与手写分离同模型）
    raise ValueError(f"未知变体: {variant}")


def theoretical_flops(rows: int, cols: int, variant: str, op: str | None = None) -> int:
    """每变体的模型 FLOP 数（exp/比较各计 1 flop，量纲化用途，非精确指令计数）。"""
    n = rows * cols
    if variant == "softmax_kernel" or variant == "torch_softmax":
        return 3 * n  # 1 max 比较 + 1 exp + 1 除
    if variant == "reduce_sum" or variant == "reduce_sum_max_fused":
        return n
    if variant == "reduce_max":
        return n
    if variant.startswith("transpose_"):
        return 0
    if variant.startswith("scan_hillis"):
        # 每步全表加法：N_padded * ceil(log2(N_padded))，按 4096 上限内实际 padded 值
        padded = 1
        while padded < cols:
            padded <<= 1
        import math

        return n * max(1, int(math.log2(padded)))
    if variant.startswith("scan_blelloch"):
        return 2 * n  # up + down 各约 N 次加法
    if variant in ("silu_gate_separated", "silu_gate_fused", "silu_gate_fused_v4"):
        return 4 * n  # exp+add+div+mul
    if variant == "torch_silu_mul":
        return 4 * n
    if variant == "torch_t_contiguous":
        return 0  # 纯重排
    if variant == "torch_sum_max":
        return 2 * n
    if variant == "torch_cumsum":
        return n
    if variant == "reference_numpy" and op is not None:
        return theoretical_flops(rows, cols, CPU_MODEL_VARIANT[op])
    if variant == "reference_numpy":
        return 0  # CPU 参考不计 FLOP 模型（只作真值与延迟锚点）
    raise ValueError(f"未知变体: {variant}")


def launch_count(backend: str, op: str, variant: str) -> int:
    """每次调用的 Kernel Launch 数（launch 收益归因用）。"""
    if backend == "cpu":
        return 0
    if variant == "torch_silu_mul":
        return 2
    if resolve_model_variant(op, variant) == "silu_gate_separated":
        return 2
    return 1


def load_fixtures() -> tuple[dict[str, np.ndarray], dict]:
    fixtures_dir = ROOT / "fixtures"
    npz_path = fixtures_dir / "correctness.npz"
    manifest_path = fixtures_dir / "manifest.json"
    if not npz_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "fixtures 缺失：请先运行 python generate_fixtures.py（或 --verify-only 通过校验）"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, manifest


def load_binding(backend: str, lib_name: str):
    """按文件路径精确加载 implementations/<backend>/ops_binding.py（避免 sys.path
    顺序导致两个同名模块串台——4090 实测曾因 musa 目录先在 path 上而误加载）。"""
    import importlib.util

    module_path = ROOT / "implementations" / backend / "ops_binding.py"
    spec = importlib.util.spec_from_file_location(f"ops_binding_{backend}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_ops(backend, lib_name)


def allocate_gpu_buffers(device: torch.device, rows: int, cols: int) -> dict:
    """按 shape 一次性分配输入/输出设备缓冲（基准循环内只做 Kernel 启动）。"""
    n = rows * cols
    buffers = {
        "x": torch.empty(rows, cols, dtype=torch.float32, device=device),
        "up": torch.empty(rows, cols, dtype=torch.float32, device=device),
        "y_softmax": torch.empty(rows, cols, dtype=torch.float32, device=device),
        "y_scan": torch.empty(rows, cols, dtype=torch.float32, device=device),
        "y_transpose": torch.empty(cols, rows, dtype=torch.float32, device=device),
        "sum_out": torch.empty(rows, dtype=torch.float32, device=device),
        "max_out": torch.empty(rows, dtype=torch.float32, device=device),
        "tmp": torch.empty(rows, cols, dtype=torch.float32, device=device),
        "y_silu": torch.empty(rows, cols, dtype=torch.float32, device=device),
    }
    del n
    return buffers


def copy_inputs_to_device(arrays: dict[str, np.ndarray], rows: int, cols: int,
                          device: torch.device, buffers: dict) -> None:
    key = f"{rows}x{cols}"
    buffers["x"].copy_(torch.from_numpy(arrays[f"x_{key}"]))
    buffers["up"].copy_(torch.from_numpy(arrays[f"up_{key}"]))


def run_gpu_variant(binding, op: str, variant: str, buffers: dict, rows: int, cols: int) -> torch.Tensor:
    """执行一个 GPU 变体，返回主输出张量（reduce 返回 sum 与 max 拼接便于比较）。"""
    x, up = buffers["x"], buffers["up"]
    if op == "softmax":
        binding.softmax_rows(x, buffers["y_softmax"])
        return buffers["y_softmax"]
    if op == "reduce":
        if variant == "sum":
            binding.reduce_sum_rows(x, buffers["sum_out"])
            return buffers["sum_out"]
        if variant == "max":
            binding.reduce_max_rows(x, buffers["max_out"])
            return buffers["max_out"]
        if variant == "sum_max_fused":
            binding.reduce_sum_max_rows(x, buffers["sum_out"], buffers["max_out"])
            return torch.cat([buffers["sum_out"], buffers["max_out"]])
    if op == "transpose":
        mapping = {
            "naive": "naive", "tiled_unpadded": "tiled_unpadded", "tiled_padded": "tiled_padded"
        }
        binding.transpose(x, buffers["y_transpose"], mapping[variant])
        return buffers["y_transpose"]
    if op == "scan":
        binding.scan(x, buffers["y_scan"], variant)
        return buffers["y_scan"]
    if op == "silu_gate":
        if variant == "separated":
            binding.silu_gate_separated(x, up, buffers["tmp"], buffers["y_silu"])
        elif variant == "fused":
            binding.silu_gate_fused(x, up, buffers["y_silu"])
        elif variant == "fused_v4":
            binding.silu_gate_fused_v4(x, up, buffers["y_silu"])
        return buffers["y_silu"]
    raise ValueError(f"未知算子: {op}")


def run_torch_eager(op: str, buffers: dict) -> torch.Tensor:
    """torch eager 对照组（同一语义的框架表达式，含框架调度开销）。"""
    x, up = buffers["x"], buffers["up"]
    if op == "softmax":
        return torch.softmax(x, dim=-1)
    if op == "reduce":
        return torch.cat([x.sum(dim=1), x.max(dim=1).values])
    if op == "transpose":
        return x.t().contiguous()
    if op == "scan":
        return torch.cumsum(x, dim=1)
    if op == "silu_gate":
        return torch.nn.functional.silu(x) * up
    raise ValueError(f"未知算子: {op}")


def run_cpu_variant(op: str, arrays: dict[str, np.ndarray], rows: int, cols: int) -> np.ndarray:
    """CPU 变体：参考实现（真值同源）。"""
    key = f"{rows}x{cols}"
    x = arrays[f"x_{key}"]
    if op == "softmax":
        return softmax_rows_numpy(x)
    if op == "reduce":
        return np.concatenate([reduce_sum_rows_numpy(x), reduce_max_rows_numpy(x)])
    if op == "transpose":
        return transpose_numpy(x)
    if op == "scan":
        return scan_inclusive_rows_numpy(x)
    if op == "silu_gate":
        return silu_gate_numpy(x, arrays[f"up_{key}"])
    raise ValueError(f"未知算子: {op}")


def cpu_truth(op: str, arrays: dict[str, np.ndarray], rows: int, cols: int,
              variant: str | None = None) -> torch.Tensor:
    """从 fixtures 取该算子的 CPU 真值张量。

    reduce 的变体 truth：sum → 行和向量；max → 行最大向量；
    其余（sum_max_fused / torch_sum_max / 默认）→ [sum, max] 拼接（对齐输出顺序）。
    """
    key = f"{rows}x{cols}"
    if op == "softmax":
        return torch.from_numpy(arrays[f"y_softmax_{key}"])
    if op == "reduce":
        if variant == "sum":
            return torch.from_numpy(arrays[f"sum_{key}"])
        if variant == "max":
            return torch.from_numpy(arrays[f"max_{key}"])
        return torch.cat([torch.from_numpy(arrays[f"sum_{key}"]),
                          torch.from_numpy(arrays[f"max_{key}"])])
    if op == "transpose":
        return torch.from_numpy(arrays[f"y_transpose_{key}"])
    if op == "scan":
        return torch.from_numpy(arrays[f"y_scan_{key}"])
    if op == "silu_gate":
        return torch.from_numpy(arrays[f"y_silu_gate_{key}"])
    raise ValueError(f"未知算子: {op}")
