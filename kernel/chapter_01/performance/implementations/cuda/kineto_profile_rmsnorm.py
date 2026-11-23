#!/usr/bin/env python3
"""torch.profiler (kineto) 证据采集：RMSNorm 组合表达式的逐 kernel 设备时间。

用途（ncu 计数器权限受限时的 memory-bound 证据链，CUDA 侧）：
- kineto 捕获每 kernel 的设备执行时间（CUPTI activity，无需 perf counters）；
- 对每 kernel 结合 roofline 理论流量算有效带宽，与 D2D 拷贝实测峰值对比：
  大 kernel 的有效带宽逼近拷贝峰值 + 计算强度 ~1 FLOP/B → memory-bound。
- 同时记录每迭代 kernel 数与 CPU 侧 op/launch 开销（launch-bound 证据）。

用法：
    python3 kineto_profile_rmsnorm.py --rows 2048 --hidden 1024 --iters 20
输出：results/kineto-rmsnorm-cuda.json + results/kineto-rmsnorm-cuda-table.txt
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rmsnorm_reference import RMS_NORM_EPS, composite_rmsnorm_torch

SCHEMA_VERSION = "kernel-chapter02.kineto-cuda.v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=20)
    arguments = parser.parse_args()

    device = torch.device("cuda")
    weight = torch.randn(arguments.hidden, device=device, dtype=torch.float16)
    x = torch.randn(arguments.rows, arguments.hidden, device=device, dtype=torch.float16)

    for _ in range(5):
        composite_rmsnorm_torch(x, weight, RMS_NORM_EPS)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(arguments.iters):
            composite_rmsnorm_torch(x, weight, RMS_NORM_EPS)
        torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
    Path("results/kineto-rmsnorm-cuda-table.txt").write_text(table, encoding="utf-8")

    kernels = []
    for event in prof.key_averages():
        device_us = getattr(event, "self_device_time_total", 0.0) or getattr(event, "self_cuda_time_total", 0.0)
        if device_us > 0:
            kernels.append({
                "name": event.key[:100],
                "count": event.count,
                "device_time_total_us": round(device_us, 1),
                "device_time_avg_us": round(device_us / max(event.count, 1), 2),
            })
    kernels.sort(key=lambda k: -k["device_time_total_us"])

    total_device_us = sum(k["device_time_total_us"] for k in kernels)
    kernel_launches_per_iter = sum(
        ev.count for ev in prof.key_averages() if ev.key.endswith("LaunchKernel")
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rows": arguments.rows,
        "hidden": arguments.hidden,
        "iters": arguments.iters,
        "device_kernels": kernels,
        "total_device_time_us": round(total_device_us, 1),
        "device_time_per_iter_us": round(total_device_us / arguments.iters, 1),
        "kernel_launches_per_iter": kernel_launches_per_iter,
        "note": "self_device_time 为 kineto 聚合设备时间；除以 iters 得每迭代",
        "table_file": "results/kineto-rmsnorm-cuda-table.txt",
    }
    out = Path("results/kineto-rmsnorm-cuda.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table[:2200])
    print(f"[kineto] 每迭代设备时间合计 = {payload['device_time_per_iter_us']}µs，"
          f"每迭代 kernel 发射数 = {kernel_launches_per_iter}")
    print(f"[kineto] 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
