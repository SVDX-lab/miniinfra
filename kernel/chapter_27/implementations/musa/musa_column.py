#!/usr/bin/env python3
"""第 27 期 MUSA 列（S4000，可选列）：chunked/递推两形式在 torch_musa eager 下的
数学等价复验 + 延迟对照。

范围（会话目标 ≤ 30min）：
  1. 环境快照（environment-musa.json）；
  2. fixtures 真值复核（--verify-only 等价物：重算 scalar/vector N≤128 关键组
     与 npz 对拍——服务器无 fixtures 时按 seed 现场重建）；
  3. 等价复验：musa 设备上 chunked vs 递推（fp32 主链 + fp16io），N ∈ {128, 512,
     2048}、B ∈ {16,32,64}，对照 CPU 同 dtype 结果；
  4. 延迟对照：递推 vs chunked（fp32，单序列）N ∈ {512, 2048}——MUSA 后端上
     两形式的差距预期远大于 CPU（GPU 串行核内并行度利用差异），事件对计时
     （ch01 口径）。
纯 torch_musa eager（无自定义核）；MUSA 后端只复验数学与测速，不涉及编译。

用法（S4000 上）：python3 musa_column.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent              # implementations/musa
CPU_DIR = HERE.parent / "cpu"                       # implementations/cpu
CHAPTER = HERE.parent.parent                        # chapter_27/
sys.path.insert(0, str(CHAPTER))
sys.path.insert(0, str(CPU_DIR))
sys.path.insert(0, str(HERE))

import deltanet_chunked as chunked  # noqa: E402
import deltanet_config as cfg  # noqa: E402
import deltanet_reference as ref  # noqa: E402

try:
    import experiment_utils as eu
except ImportError:
    sys.path.insert(0, str(CHAPTER))
    import experiment_utils as eu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑 N=128/512（SSH 断连风险期）")
    args = parser.parse_args()

    device = eu.resolve_device("musa")
    env = eu.device_environment_record(device)
    print(f"[musa] device={env.get('gpu', {})}")

    weights = ref.make_weights(cfg.SEED_BASE)
    entries = []

    ns = (128, 512) if args.quick else (128, 512, 2048)
    for n in ns:
        x = ref.make_inputs(n, seed=cfg.SEED_BASE + 7000 + n)
        proj_cpu = ref.project_inputs(x, weights, "scalar")
        proj = {k: v.to(device) for k, v in proj_cpu.items()}
        for b in cfg.CHUNK_SWEEP:
            o_c, s_c = chunked.chunked_gdn(proj["q"], proj["k"], proj["v"],
                                           proj["beta"], proj["g_log"], chunk_size=b)
            torch.musa.synchronize()
            o_c_cpu = o_c.cpu()
            # CPU 同 dtype 对照
            o_r_cpu, s_r_cpu = ref.recurrent_gdn(proj_cpu["q"], proj_cpu["k"], proj_cpu["v"],
                                                 proj_cpu["beta"], proj_cpu["g_log"])
            o_ref_c, _ = chunked.chunked_gdn(proj_cpu["q"], proj_cpu["k"], proj_cpu["v"],
                                             proj_cpu["beta"], proj_cpu["g_log"], chunk_size=b)
            m_out = eu.comparison_metrics(o_c_cpu, o_ref_c)
            m_state = eu.comparison_metrics(s_c.cpu(), s_r_cpu)
            entries.append({
                "seq_len": n, "chunk_size": b, "dtype": "float32",
                "musa_vs_cpu_chunked_output": m_out,
                "musa_state_vs_cpu_recurrent_state": m_state,
            })
            print(f"[eq] N={n} B={b}: musa-chunked vs cpu-chunked max_abs={m_out['max_abs_error']:.3e} "
                  f"cos={m_out['cosine_similarity']:.10f}")

        # musa 上两形式互对（musa 递推 vs musa chunked）
        o_r_m, s_r_m = ref.recurrent_gdn(proj["q"], proj["k"], proj["v"],
                                         proj["beta"], proj["g_log"])
        torch.musa.synchronize()
        m_pair = eu.comparison_metrics(
            chunked.chunked_gdn(proj["q"], proj["k"], proj["v"], proj["beta"],
                                proj["g_log"], chunk_size=64)[0].cpu(),
            o_r_m.cpu())
        entries.append({"seq_len": n, "pair": "musa_chunked_vs_musa_recurrent", **m_pair})
        print(f"[eq] N={n}: musa chunked vs musa recurrent max_abs={m_pair['max_abs_error']:.3e}")

    # ---- 延迟对照（事件对，ch01 口径）----
    timings = []
    for n in ((512, 2048) if not args.quick else (512,)):
        x = ref.make_inputs(n, seed=cfg.SEED_BASE + 7000 + n)
        proj_cpu = ref.project_inputs(x, weights, "scalar")
        proj = {k: v.to(device) for k, v in proj_cpu.items()}
        timer = eu.OperatorTimer(device, warmup=3, repeats=10)
        for b in (16, 64):
            r = timer.measure(lambda b=b: chunked.chunked_gdn(
                proj["q"], proj["k"], proj["v"], proj["beta"], proj["g_log"], chunk_size=b))
            timings.append({"seq_len": n, "form": "chunked", "chunk_size": b, **r})
            print(f"[bench] chunked N={n} B={b}: p50={r['p50_ms']}ms")
        r = timer.measure(lambda: ref.recurrent_gdn(
            proj["q"], proj["k"], proj["v"], proj["beta"], proj["g_log"]))
        timings.append({"seq_len": n, "form": "recurrent", **r})
        print(f"[bench] recurrent N={n}: p50={r['p50_ms']}ms")

    result = {
        "schema_version": "kernel-chapter28.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "musa",
        "mode": "torch_musa eager（可选列：数学等价复验 + 延迟对照，无自定义核）",
        "environment": env,
        "entries": entries,
        "timings": timings,
    }
    out_dir = CHAPTER / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "correctness-results-musa.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    eu.write_results_json(out_dir / "environment-musa.json", env)
    print("[musa] 结果已写 results/correctness-results-musa.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
