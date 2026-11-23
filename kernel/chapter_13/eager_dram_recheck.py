#!/usr/bin/env python3
"""KI-14-10 补测：手写 ch03 RMSNorm 在 MUSA 的 DRAM 区 eager-only 复测。

背景：benchmark 的 timed_op 走 Graph Replay 口径，S4000 上 Graph 口径对手写
kernel 大 shape 给出物理不可能数字（32768 行 0.093 ms = 2886 GB/s > 711 峰值
——M-02 族伪影）。本脚本只测 eager（OperatorTimer，事件对），tile=256，
结果独立落盘 results/dram-hw-eager-musa.json。

用法：python3 eager_dram_recheck.py --backend musa
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations"))
sys.path.insert(0, str(HERE / "implementations" / "reference_handwritten"))

PEAK = {"cuda": 918.8e9, "musa": 711.3e9}
L2_BYTES = {"cuda": 72 * 1024 * 1024, "musa": 24 * 1024 * 1024}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="musa")
    arguments = parser.parse_args()
    device = torch.device(arguments.backend)
    from reference_runners import HandwrittenRMSNorm  # noqa: PLC0415

    hw = HandwrittenRMSNorm(arguments.backend)  # 复用已构建扩展
    eps = 1e-6
    entries = []
    for dtype, rows_list, itemsize in ((torch.float32, [16384, 32768], 4),
                                       (torch.float16, [32768, 65536], 2)):
        rng_w = np.random.default_rng(20260301 + 1024)
        w = torch.from_numpy(rng_w.standard_normal(1024, dtype=np.float32)).to(
            device=device, dtype=dtype)
        for rows in rows_list:
            rng = np.random.default_rng(20260301 + rows * 100000 + 1024)
            x = torch.from_numpy(
                rng.standard_normal((rows, 1024), dtype=np.float32)).to(device=device,
                                                                        dtype=dtype)
            timer = OperatorTimerLocal(device, warmup=10, repeats=30)
            timing = timer.measure(lambda: hw.run(x, w, eps, 256))  # noqa: E731
            total = (rows * 1024 * 2 + 1024) * itemsize
            eff = total / (timing["p50_ms"] / 1000.0)
            entries.append({
                "impl": "handwritten-ch03", "dtype": "fp16" if dtype == torch.float16 else "fp32",
                "rows": rows, "tile": 256, "mode": "eager-only(KI-14-10)",
                "p50_ms": timing["p50_ms"], "p95_ms": timing["p95_ms"],
                "min_ms": timing["min_ms"], "repeats": timing["repeats"],
                "theoretical_bytes": total,
                "effective_bandwidth_gb_s": round(eff / 1e9, 2),
                "achievement_ratio": round(eff / PEAK[arguments.backend], 4),
                "working_set_bytes": rows * 1024 * itemsize,
                "dram_region": rows * 1024 * itemsize > L2_BYTES[arguments.backend],
            })
            print(f"[recheck/{arguments.backend}] {entries[-1]['dtype']} rows={rows} "
                  f"p50={timing['p50_ms']}ms eff={entries[-1]['effective_bandwidth_gb_s']}GB/s "
                  f"ach={entries[-1]['achievement_ratio']:.2%}")
    out = HERE / "results" / f"dram-hw-eager-{arguments.backend}.json"
    out.write_text(json.dumps({"schema_version": "kernel-chapter14.dramrecheck.v1",
                               "backend": arguments.backend, "entries": entries},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[recheck] 已写入 {out}")
    return 0


class OperatorTimerLocal:
    """eager-only 事件对计时（复制 experiment_utils.OperatorTimer 的 GPU 分支，去 Graph）。"""

    def __init__(self, device, warmup, repeats):
        self.device = device
        self.warmup = warmup
        self.repeats = repeats

    def measure(self, operation):
        module = getattr(torch, self.device.type)
        for _ in range(self.warmup):
            operation()
        module.synchronize()
        event_class = module.Event
        pairs = []
        for _ in range(self.repeats):
            start_event = event_class(enable_timing=True)
            end_event = event_class(enable_timing=True)
            start_event.record()
            operation()
            end_event.record()
            pairs.append((start_event, end_event))
        module.synchronize()
        samples = sorted(s.elapsed_time(e) for s, e in pairs)
        from statistics import mean
        return {"warmup": self.warmup, "repeats": self.repeats,
                "p50_ms": round(samples[len(samples) // 2], 4),
                "p95_ms": round(samples[min(len(samples) - 1,
                                            int(round(len(samples) * 0.95)) - 1)], 4),
                "min_ms": round(samples[0], 4), "mean_ms": round(mean(samples), 4)}


if __name__ == "__main__":
    raise SystemExit(main())
