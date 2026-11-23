#!/usr/bin/env python3
"""第 27 期 CUDA 扩展构建脚本（回填用骨架；与 ch26 build_ext.py 同口径）。

delantanet_chunked.cu 当前为设计冻结的骨架（K1/K2/K3 TODO(mark) 占位，
host 入口 TORCH_CHECK(false)）——本脚本在回填会话中执行：
    python3 build_ext.py            # torch cpp_extension 编译 .cu
编译产物（build/ 目录）不入库（协议 §11：编译产物不提交）。
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent


def main() -> int:
    print("[build_ext] 编译 deltanet_chunked.cu（K1/K2/K3 回填版）")
    module = load(
        name="deltanet_chunked_ch28",
        sources=[str(HERE / "deltanet_chunked.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
        build_directory=str(HERE / "build"),
    )
    print("[build_ext] 编译成功：", module)
    print("[build_ext] 正确性/实测由 cuda_test.py 承担（先跑 cuda_sim_check.py 桌面验证）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
