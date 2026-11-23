"""第 14 期实验 2（CPU 列）：Inductor 生成代码捕获。

运行方式（codegen_capture.sh）：以 TORCH_LOGS=output_code 启动本脚本，
stderr/stdout 重定向进 results/inductor-cpu/——生成代码是编译期日志，
必须经环境变量捕获（torch 2.6 无进程内稳定的 output_code 读取 API）。

捕获对象（均在 128×1024 + rows=1 各编译一次）：
- rmsnorm：RMSNorm 组合表达式 → 观察归一化链的融合形态；
- add_rmsnorm：分离段 → 观察编译器是否自动合成第 03 期手写融合的
  对应物（单 kernel 含 add+reduction+normalize）；
- add_rmsnorm_dynamic：dynamic=True 版本 → 观察符号 shape（syms）下的
  代码形态；
- add_rmsnorm_inplace：in-place 突变版 → 观察 functionalization 产物
  （突变被重写成什么、多几次拷贝）；
- qwen3：整段 → 观察跨算子图上的 kernel 分解（fuse 了什么、什么留给了
  extern kernel/mkldnn/mkllibblas）。
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", default="rmsnorm,add_rmsnorm,add_rmsnorm_dynamic,add_rmsnorm_inplace,qwen3")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    args = parser.parse_args()

    torch.set_num_threads(min(6, torch.get_num_threads()))
    rng = np.random.default_rng(20260415)
    r = torch.from_numpy(rng.standard_normal((args.rows, args.hidden), dtype=np.float32))
    h = torch.from_numpy(rng.standard_normal((args.rows, args.hidden), dtype=np.float32))
    w = torch.from_numpy(rng.standard_normal((args.hidden,), dtype=np.float32))

    import implementations.cpu.compiled_objects as objects
    from implementations.cpu.qwen3_segment import build_segment

    todo = [s.strip() for s in args.objects.split(",") if s.strip()]

    if "rmsnorm" in todo:
        print("\n########## CODEGEN CAPTURE: rmsnorm (inductor default) ##########", flush=True)
        torch._dynamo.reset()
        compiled = torch.compile(objects.rmsnorm_expr)
        out = compiled(r, w)
        print("RMSNORM-OK", tuple(out.shape), flush=True)

    if "add_rmsnorm" in todo:
        print("\n########## CODEGEN CAPTURE: add_rmsnorm (inductor default) ##########", flush=True)
        torch._dynamo.reset()
        compiled = torch.compile(objects.add_rmsnorm_separated)
        t, y = compiled(r, h, w)
        print("ADD_RMSNORM-OK", tuple(t.shape), tuple(y.shape), flush=True)

    if "add_rmsnorm_dynamic" in todo:
        print("\n########## CODEGEN CAPTURE: add_rmsnorm dynamic=True ##########", flush=True)
        torch._dynamo.reset()
        compiled = torch.compile(objects.add_rmsnorm_separated, dynamic=True)
        t, y = compiled(r, h, w)
        print("ADD_RMSNORM-DYN-OK", tuple(t.shape), flush=True)
        rows2 = 2 * args.rows
        r2 = torch.from_numpy(rng.standard_normal((rows2, args.hidden), dtype=np.float32))
        h2 = torch.from_numpy(rng.standard_normal((rows2, args.hidden), dtype=np.float32))
        t2, _ = compiled(r2, h2, w)
        print("ADD_RMSNORM-DYN-REUSE-OK", tuple(t2.shape), flush=True)

    if "add_rmsnorm_inplace" in todo:
        print("\n########## CODEGEN CAPTURE: add_rmsnorm inplace (functionalization) ##########", flush=True)
        torch._dynamo.reset()
        compiled = torch.compile(objects.add_rmsnorm_inplace_style)
        r_out, y = compiled(r.clone(), h, w)
        print("ADD_RMSNORM-INPLACE-OK", tuple(r_out.shape), tuple(y.shape), flush=True)

    if "qwen3" in todo:
        print("\n########## CODEGEN CAPTURE: qwen3 segment (inductor default) ##########", flush=True)
        torch._dynamo.reset()
        segment = build_segment()
        input_ids = torch.from_numpy(
            rng.integers(0, segment.config.vocab_size, size=(1, args.rows), dtype=np.int64)
        )
        compiled = torch.compile(segment)
        out = compiled(input_ids)
        print("QWEN3-OK", tuple(out.shape), flush=True)


if __name__ == "__main__":
    main()
