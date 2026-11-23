"""最小 safetensors 文件格式加载器（本地平台边界的兼容层）。

平台边界记录（KNOWN_ISSUES 同步）：本地 venv（torch 2.6.0+cpu）没有
safetensors 包，且协议禁止本地 pip install。本文件按 safetensors 官方
格式规范（8 字节小端 u64 头长度 + JSON 头 + 连续张量缓冲区）实现
load_file 的最小子集，仅供本地 CPU 实验使用；服务器上安装有官方
safetensors，驱动脚本仅在本机 import 失败时才把本目录加入 sys.path。

正确性兜底：模型权重的整体完整性由两层校验保证——
1. model.safetensors 文件 SHA256 与 chapter_01/models/MANIFEST.json 一致；
2. 用本加载器加载后计算的首 Token logits SHA256 与 ch01 真值 manifest
   的 first_token_logits_sha256 一致（smoke_test.py 断言）。
任何字节错位都会破坏第 2 层校验，因此不可能静默出错。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

# safetensors dtype -> (numpy dtype 或 torch 处理标记)
_NUMPY_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"),
    "U8": np.dtype("<u1"),
    "BOOL": np.dtype("<u1"),
}


def _to_torch(raw: np.ndarray, st_dtype: str) -> torch.Tensor:
    if st_dtype == "BF16":
        # numpy 无 bfloat16：uint16 原始视图 -> torch bfloat16 位视图。
        return torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16)
    if st_dtype == "F16":
        return torch.from_numpy(raw.view(np.uint16)).view(torch.float16)
    tensor = torch.from_numpy(raw.copy())
    if st_dtype == "BOOL":
        tensor = tensor.to(torch.bool)
    return tensor


def load_file(filename: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """safetensors load_file 的最小兼容实现（返回 {name: CPU tensor}）。"""
    path = Path(filename)
    with path.open("rb") as file:
        header_length = struct.unpack("<Q", file.read(8))[0]
        header_bytes = file.read(header_length)
        header = json.loads(header_bytes.decode("utf-8"))
        buffer = file.read()

    tensors: dict[str, torch.Tensor] = {}
    data_start = 8 + header_length
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype_name = spec["dtype"]
        begin, end = spec["data_offsets"]
        if dtype_name in ("BF16", "F16"):
            raw = np.frombuffer(buffer, dtype="<u2", count=(end - begin) // 2, offset=begin)
        else:
            numpy_dtype = _NUMPY_DTYPES[dtype_name]
            raw = np.frombuffer(buffer, dtype=numpy_dtype, count=(end - begin) // numpy_dtype.itemsize, offset=begin)
        tensor = _to_torch(raw, dtype_name).reshape(spec["shape"])
        if device != "cpu":
            tensor = tensor.to(device)
        tensors[name] = tensor
    return tensors
