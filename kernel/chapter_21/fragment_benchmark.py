#!/usr/bin/env python3
"""实验 2：显存碎片率与可服务请求数（连续 vs 分页，CPU 记账模拟）。

负载（FRAG_SEED=20261122 固定）：32 请求，final_len ~ U[8, 2048]，initial_len
~ U[8, 64] ∧ ≤ final_len，到达 ~ 均匀排序，每步 1 token，生成完释放。
容量口径：全局 KV token 预算（两方案同一容量公平对比）。

产出（results/fragmentation-results.json）：
  1. 主场景：容量 = 需求的 {50%, 80%, 100%}，四方案
     （contig-reserve-global / contig-reserve-own / contig-exact-malloc / paged
     bs=128）的 served / 拒绝（容量 vs 碎片因果分离）/ 中途夭折 / 搬迁拷贝 /
     峰值占用 / 浪费 / 外部碎片率；
  2. block_size ∈ {16, 32, 64, 128} 扫描（paged，容量 80%）；
  3. 容量扫描（10 档 30%~110% × {contig-reserve-global, contig-exact-malloc,
     paged bs=128}）→ 可服务请求数曲线；
  4. METHOD 字段：记账方法全文（metrics 定义）随 JSON 落盘。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import paged_config as cfg  # noqa: E402
import roofline_paged as rp  # noqa: E402
from memory_accountant import (  # noqa: E402
    METHOD,
    block_size_sweep,
    capacity_sweep,
    make_workload,
    simulate,
)
from experiment_utils import write_results_json  # noqa: E402


def main() -> int:
    workload = make_workload()
    demand = sum(r["final"] for r in workload)
    max_len = max(r["final"] for r in workload)
    print(f"[frag] workload: {len(workload)} req, demand={demand} tokens, "
          f"max_len={max_len}, mean_len={demand/len(workload):.1f}")

    # ---- 1. 主场景（三档容量 × 四方案）----
    scenarios = []
    for frac in (0.5, 0.8, 1.0):
        cap = int(demand * frac)
        row = {"capacity_fraction": frac, "capacity_tokens": cap,
               "capacity_gb_kv_fp16": round(cap * rp.kv_bytes_per_token_model(2) / 1e9, 3)}
        for policy in METHOD["policy_names"]:
            r = simulate(policy, workload, cap, cfg.DEFAULT_BLOCK_SIZE)
            row[policy] = r
            print(f"[frag] cap={cap:>6} ({frac:.0%}) {policy:<22} "
                  f"served={r['served']:>2} rej={r['rejected_at_admission']:>2} "
                  f"(space={r['reject_no_space']},frag={r['reject_no_contiguous_span']}) "
                  f"abort={r['aborted_midlife']} copy={r['copy_tokens']:>6} "
                  f"peak_used={r['peak_used_tokens']:>6} "
                  f"waste@peak={r['internal_waste_at_peak']:>6} "
                  f"waste_mean={r['internal_waste_mean_active']:>8.1f} "
                  f"extfrag(mean/max)={r['ext_frag_mean']:.3f}/{r['ext_frag_max']:.3f}")
        scenarios.append(row)

    # ---- 2. block_size 扫描（paged，容量 80%）----
    cap80 = int(demand * 0.8)
    bs_rows = block_size_sweep(workload, cap80, cfg.FRAG_BLOCK_SIZE_SWEEP)
    for r in bs_rows:
        print(f"[frag] paged bs={r['block_size']:>3}: served={r['served']:>2} "
              f"abort={r['aborted_midlife']} waste@peak={r['internal_waste_at_peak']:>5} "
              f"waste_mean={r['internal_waste_mean_active']:>7.1f} "
              f"blocks={cap80 // r['block_size']}")
    # paged bs 极端对照：bs=1（零内部浪费）
    bs_rows.append(simulate("paged", workload, cap80, 1))

    # ---- 3. 容量扫描 ----
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    sweep = capacity_sweep(workload, cfg.DEFAULT_BLOCK_SIZE, fractions)
    for row in sweep:
        print(f"[frag] sweep cap={row['capacity_tokens']:>6} "
              f"({row['capacity_fraction']:.0%}): "
              f"served rg={row['served_contig-reserve-global']:>2} "
              f"malloc={row['served_contig-exact-malloc']:>2} "
              f"paged={row['served_paged']:>2}")

    payload = {
        "schema_version": "kernel-chapter22.fragmentation.v1",
        "method": METHOD,
        "kv_bytes_per_token_fp16": rp.kv_bytes_per_token_model(2),
        "workload": {
            "num_requests": len(workload),
            "final_len_distribution": f"U[{cfg.FRAG_LEN_MIN},{cfg.FRAG_LEN_MAX}] int",
            "initial_len_distribution": "U[8,64] int ∧ ≤ final",
            "arrivals": "独立均匀 [0,600) 后排序",
            "total_tokens_demanded": demand,
            "max_len": max_len,
            "requests": workload,
            "seed": cfg.FRAG_SEED,
        },
        "scenarios": scenarios,
        "block_size_sweep": bs_rows,
        "capacity_sweep": sweep,
    }
    write_results_json(HERE / "results" / "fragmentation-results.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
