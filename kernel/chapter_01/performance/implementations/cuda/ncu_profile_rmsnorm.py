#!/usr/bin/env python3
"""ncu 的被测目标脚本：跑固定 shape 的 RMSNorm 组合表达式若干次。

Nsight Compute 对该进程做 kernel 级 profiling，证据用途：
- 证明 RMSNorm 组合表达式是 memory-bound（DRAM/Memory Throughput % 高、
  SM 忙碌率低、每 Kernel 实际 DRAM/L2 流量 vs 理论字节）；
- 统计组合表达式每迭代发射的 kernel 数与各自的访存行为。

kernel 顺序（每迭代 8 个，torch 2.12 eager 实测，可由 CSV launch 序号核对）：
  1 cast_in (x.float)  2 pow2  3 mean_reduce  4 add_eps
  5 rsqrt  6 broadcast_mul  7 cast_out (.to fp16)  8 mul_weight

用法：
    python3 ncu_profile_rmsnorm.py --rows 2048 --hidden 1024 --warmup 3 --iters 3
ncu 调用方式见 run_ncu.sh。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rmsnorm_reference import RMS_NORM_EPS, composite_rmsnorm_torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    arguments = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16 if arguments.dtype == "fp16" else torch.float32
    torch.cuda.init()
    print(f"[target] device={torch.cuda.get_device_name(0)} rows={arguments.rows} "
          f"hidden={arguments.hidden} dtype={arguments.dtype} warmup={arguments.warmup} iters={arguments.iters}",
          flush=True)

    weight = torch.randn(arguments.hidden, device=device, dtype=dtype)
    x = torch.randn(arguments.rows, arguments.hidden, device=device, dtype=dtype)

    for _ in range(arguments.warmup):
        composite_rmsnorm_torch(x, weight, RMS_NORM_EPS)
    torch.cuda.synchronize()
    print("[target] warmup done, measurement loop starts", flush=True)

    output = None
    for _ in range(arguments.iters):
        output = composite_rmsnorm_torch(x, weight, RMS_NORM_EPS)
    torch.cuda.synchronize()
    print(f"[target] done. output checksum={output.float().sum().item():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
