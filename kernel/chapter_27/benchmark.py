#!/usr/bin/env python3
"""第 27 期实验 3：递推 vs chunked 延迟对照（CPU 列）。

口径：
  - 单序列 batch=1（合同声明）；FP32 主链（真值口径）；
  - OperatorTimer（ch01 口径）：CPU perf_counter，warmup=2、repeats=5
    （N=32768 的递推版单次 ~分钟级，只做 chunked；N=8192 递推单次 ~6s，
    warmup=1、repeats=3 并在结果中标注）；
  - 结构口径并记：串行步数（递推 T vs chunked (C−1)+2NC，见
    deltanet_chunked.serial_step_counts）与每步有效 FLOP；
  - 展示"可并行性"的算法意义：chunked 把 T 步 O(KV) 向量算子替换为
    (C−1)+NC 步批量矩阵算子——CPU 单线程下这是 op 数量级的差距；
    向量化收益 = 矩阵乘 vs 逐 token 小算子的 dispatch/内存层次差距。

产出 results/benchmark-results-cpu.json。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import deltanet_chunked as chunked  # noqa: E402
import deltanet_config as cfg  # noqa: E402
import deltanet_reference as ref  # noqa: E402
from experiment_utils import OperatorTimer, backend_sync, resolve_device, write_results_json  # noqa: E402

FLOP_PER_TOKEN = (  # 每 token 每 16 头的 MAC 数（递推一步的算术量）
    3 * cfg.NUM_KEY_HEADS * cfg.HEAD_K_DIM            # q/k/v 投影不在本测内——递推核本体：
)
STATE_FLOP_PER_TOKEN = (
    2 * cfg.NUM_VALUE_HEADS * cfg.HEAD_K_DIM * cfg.HEAD_V_DIM   # retrieval: Sᵀk
    + 2 * cfg.NUM_VALUE_HEADS * cfg.HEAD_K_DIM * cfg.HEAD_V_DIM  # write: k⊗u
    + 2 * cfg.NUM_VALUE_HEADS * cfg.HEAD_V_DIM * cfg.HEAD_K_DIM  # read: q·S
)


def bench(op, device, warmup, repeats, label):
    timer = OperatorTimer(device, warmup=warmup, repeats=repeats)
    r = timer.measure(op)
    print(f"[bench] {label}: p50={r['p50_ms']}ms p95={r['p95_ms']}ms min={r['min_ms']}ms "
          f"(warmup={warmup} repeats={repeats})")
    return r


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()
    device = resolve_device(args.backend)

    entries = []
    weights = ref.make_weights(cfg.SEED_BASE)
    for n in cfg.LATENCY_SEQ_LENS:
        x = ref.make_inputs(n)
        proj = ref.project_inputs(x, weights, "scalar")
        for b in cfg.CHUNK_SWEEP:
            counts = chunked.serial_step_counts(n, b)
            t_chunked = bench(
                lambda: chunked.chunked_gdn(proj["q"], proj["k"], proj["v"], proj["beta"],
                                            proj["g_log"], chunk_size=b),
                device, warmup=2, repeats=5, label=f"chunked N={n} B={b}")
            entries.append({
                "seq_len": n, "form": "chunked", "chunk_size": b, **t_chunked,
                "serial_steps": counts["chunked_serial_steps"],
                "chunk_count": counts["chunk_count"],
                "dtype": "float32", "batch": 1,
            })
        t_rec = bench(
            lambda: ref.recurrent_gdn(proj["q"], proj["k"], proj["v"], proj["beta"], proj["g_log"]),
            device, warmup=2, repeats=5, label=f"recurrent N={n}")
        entries.append({
            "seq_len": n, "form": "recurrent", "chunk_size": None, **t_rec,
            "serial_steps": n, "chunk_count": None, "dtype": "float32", "batch": 1,
        })
        rec_p50 = [e for e in entries if e["form"] == "recurrent" and e["seq_len"] == n][0]["p50_ms"]
        for b in cfg.CHUNK_SWEEP:
            ch_p50 = [e for e in entries if e["form"] == "chunked"
                      and e["seq_len"] == n and e["chunk_size"] == b][0]["p50_ms"]
            print(f"[ratio] N={n} B={b}: chunked/recurrent = {ch_p50/rec_p50:.3f}")

    # N=8192：递推 warmup=1 repeats=3（单次秒级），chunked 常规
    n = 8192
    x = ref.make_inputs(n)
    proj = ref.project_inputs(x, weights, "scalar")
    for b in (32, 64):
        t_chunked = bench(lambda b=b: chunked.chunked_gdn(proj["q"], proj["k"], proj["v"],
                                                          proj["beta"], proj["g_log"], chunk_size=b),
                          device, warmup=2, repeats=5, label=f"chunked N={n} B={b}")
        counts = chunked.serial_step_counts(n, b)
        entries.append({"seq_len": n, "form": "chunked", "chunk_size": b, **t_chunked,
                        "serial_steps": counts["chunked_serial_steps"],
                        "chunk_count": counts["chunk_count"], "dtype": "float32", "batch": 1})
    t_rec = bench(lambda: ref.recurrent_gdn(proj["q"], proj["k"], proj["v"],
                                            proj["beta"], proj["g_log"]),
                  device, warmup=1, repeats=3, label=f"recurrent N={n}（降重复）")
    entries.append({"seq_len": n, "form": "recurrent", "chunk_size": None, **t_rec,
                    "serial_steps": n, "chunk_count": None, "dtype": "float32", "batch": 1,
                    "timing_note": "warmup=1 repeats=3（单次秒级）"})

    # N=32768：只跑 chunked B=64（递推版分钟级，留结构估算）
    n = 32768
    x = ref.make_inputs(n)
    proj = ref.project_inputs(x, weights, "scalar")
    t_chunked = bench(lambda: chunked.chunked_gdn(proj["q"], proj["k"], proj["v"],
                                                  proj["beta"], proj["g_log"], chunk_size=64),
                      device, warmup=1, repeats=3, label=f"chunked N={n} B=64")
    counts = chunked.serial_step_counts(n, 64)
    entries.append({"seq_len": n, "form": "chunked", "chunk_size": 64, **t_chunked,
                    "serial_steps": counts["chunked_serial_steps"],
                    "chunk_count": counts["chunk_count"], "dtype": "float32", "batch": 1,
                    "timing_note": "warmup=1 repeats=3"})

    result = {
        "schema_version": "kernel-chapter28.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "device": str(device),
        "host_note": "本地 WSL2 12 核（CPU 参考实现计时，声明——串行 Python 步进"
                     "开销计入递推口径，恰为'无并行组织'的算法意义展示）",
        "dtype": "float32", "batch": 1, "torch_threads": torch.get_num_threads(),
        "state_flop_per_token_mac": STATE_FLOP_PER_TOKEN,
        "entries": entries,
    }
    write_results_json(HERE / "results" / f"benchmark-results-{args.backend}.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
