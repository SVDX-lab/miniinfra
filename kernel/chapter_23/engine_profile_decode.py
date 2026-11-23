# -*- coding: utf-8 -*-
"""实验 5+6：引擎一次 Decode 步的算子序列 + Profiler 复核微基准结论（MUSA 列）。

- enforce_eager 下用 torch.profiler（CPU + PrivateUse1）捕获 batch=1、prompt=20、
  max_tokens=8 的完整 generate：1 prefill + 7 decode
- 产出：设备 kernel 按名称聚合（时长/计数 Top30）、Chronological 前 300 设备事件
  （算子序列证据）、按算子类别的占比表（rms/rope/attention/gemm/cache/sampler）
- RMSNorm 占比 = rms 类 kernel 设备时长和 / 全部设备 kernel 时长和
  （对照 ep05 占比层稀释归因；M-04 口径注记：kineto 设备时长在 MUSA 上 eager 域参考）
产出 results/engine-profile-musa.json
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine24_common import (AccelMem, Timer, build_prompts, dump_json,  # noqa: E402
                             load_json, utcnow)

# 算子类别（对照 ch21-23 亲手实现 + ch05 融合 op 的定位表）
CATEGORIES = [
    ("rmsnorm", r"rms|Rms|RMS|norm_norm|fused_add"),
    ("rope", r"rotary|rope|Rope"),
    ("attention", r"attn|attention|flash|sdpa|softmax.*mask|paged"),
    ("gemm", r"gemm|matmul|mm\b|linear|Gemm|addmm"),
    ("kv_cache", r"cache|Cache|reshape_and_cache|kv_"),
    ("sampler", r"argmax|sampler|softmax|topk|logits"),
    ("quant", r"quant|Quant|dequant"),
    ("elementwise", r"elementwise|silu|gelu|mul|add_|copy_|cat|embedding"),
]


def classify(name: str) -> str:
    low = name.lower()
    for cat, pat in CATEGORIES:
        import re
        if re.search(pat, low):
            return cat
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--truth", default="fixtures/truth-ch01.json")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--out", default="results/engine-profile-musa.json")
    args = ap.parse_args()

    truth = load_json(args.truth)
    R = {
        "schema_version": "kernel-chapter24.profile.v1",
        "created_at_utc": utcnow(),
        "engine": "vllm-musa (enforce_eager=True)",
        "model_path": args.model,
        "prof_note": "M-04 口径：MUSA kineto 设备时长部分场景不可信；本表设备时长与计数并列记录，占比用时长口径并注明风险",
        "decode_steps": args.max_tokens - 1,
    }
    mem = AccelMem()

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="auto",
        enforce_eager=True,  # 算子序列需要 eager
        enable_prefix_caching=False,
        seed=1234,
    )
    prompts = build_prompts(truth["prompt_token_ids"], 20, 1)
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # warmup
    llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
    mem.reset_peak()

    try:
        from torch.profiler import ProfilerActivity, profile
        acts = [ProfilerActivity.CPU]
        if hasattr(ProfilerActivity, "PrivateUse1"):
            acts.append(ProfilerActivity.PrivateUse1)
        elif hasattr(ProfilerActivity, "MUSA"):
            acts.append(ProfilerActivity.MUSA)
        else:
            acts.append(ProfilerActivity.CUDA)  # torch_musa 别名兜底
        R["profiler_activities"] = [str(a) for a in acts]
    except Exception as e:  # noqa: BLE001
        R["profiler_activities"] = f"FALLBACK: {e}"
        acts = None

    with Timer() as t_all:
        if acts:
            with profile(activities=acts) as prof:
                outs = llm.generate(prompts, sp, use_tqdm=False)
        else:
            outs = llm.generate(prompts, sp, use_tqdm=False)
    R["profiled_generate_wall_s"] = round(t_all.elapsed, 3)
    gen = outs[0].outputs[0]
    R["generated_token_count"] = len(gen.token_ids)
    R["next_token_match_ch01_first"] = gen.token_ids[0] == truth["first_token_id"]

    if not acts:
        dump_json(R, args.out)
        print("profiler unavailable, wall-only dump written")
        return

    # ---- 事件聚合 ----
    dev_events = []
    for ev in prof.events():
        if getattr(ev, "device_type", None) is not None:
            # kineto: device_type 私有加速器/CUDA 家族口径
            dt = str(getattr(ev, "device_type", ""))
            if "PrivateUse1" in dt or "CUDA" in dt or "MUSA" in dt.upper():
                dev_events.append(ev)

    agg = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    total_dur = 0.0
    for ev in dev_events:
        d = float(getattr(ev, "device_time_total", 0) or 0)
        agg[ev.name]["count"] += 1
        agg[ev.name]["dur_us"] += d
        total_dur += d

    rows = []
    for name, s in agg.items():
        rows.append({
            "name": name,
            "count": s["count"],
            "dur_us": round(s["dur_us"], 1),
            "dur_share_pct": round(100 * s["dur_us"] / total_dur, 3) if total_dur else None,
            "category": classify(name),
        })
    rows.sort(key=lambda r: -r["dur_us"])
    R["total_device_kernel_count"] = sum(r["count"] for r in rows)
    R["total_device_dur_us"] = round(total_dur, 1)
    R["top30_by_time"] = rows[:30]
    rows_by_count = sorted(rows, key=lambda r: -r["count"])
    R["top30_by_count"] = rows_by_count[:30]

    # ---- 类别占比（微基准复核核心表）----
    cat_agg = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    for r in rows:
        c = cat_agg[r["category"]]
        c["count"] += r["count"]
        c["dur_us"] += r["dur_us"]
    cat_rows = []
    for cat, s in cat_agg.items():
        cat_rows.append({
            "category": cat,
            "count": s["count"],
            "dur_us": round(s["dur_us"], 1),
            "share_pct": round(100 * s["dur_us"] / total_dur, 2) if total_dur else None,
        })
    cat_rows.sort(key=lambda r: -r["dur_us"])
    R["category_share"] = cat_rows

    # rmsnorm 细分（区分 fused_add_rms_norm / rms_norm）
    R["rmsnorm_detail"] = [
        r for r in rows if r["category"] == "rmsnorm"
    ][:10]

    # ---- 时序算子序列（前 300 设备事件名；decode 序列证据）----
    # 用 chrome trace 解析（key_averages 无时序）
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        trace_path = tf.name
    try:
        prof.export_chrome_trace(trace_path)
        import json as _json
        with open(trace_path) as f:
            tr = _json.load(f)
        evs = [e for e in tr.get("traceEvents", [])
               if e.get("ph") == "X" and e.get("cat", "") in (
                   "kernel", "gpu_op", "Kernel", "gpu_kernel")]
        evs.sort(key=lambda e: e.get("ts", 0))
        R["chronological_device_events_first300"] = [
            {"name": e.get("name", "")[:120], "dur_us": e.get("dur", None)}
            for e in evs[:300]
        ]
        R["chronological_total_device_events"] = len(evs)
    except Exception as e:  # noqa: BLE001
        R["chronological_error"] = repr(e)
    finally:
        os.unlink(trace_path)

    # 每步均值（总 kernel 数 / decode 步 + prefill 1 步）
    steps = args.max_tokens  # prefill 1 + decode (N-1)
    R["per_step_avg"] = {
        "kernels_per_forward_total": round(R["total_device_kernel_count"] / steps, 1),
        "note": "total = 1 prefill + (max_tokens-1) decode；对照 ch10 手写 decode 1925 发射/步",
    }
    R["peak_mem"] = mem.snapshot()

    dump_json(R, args.out)
    print(f"profile written -> {args.out}")
    print("SUMMARY kernels=%d dur=%.1fms" % (
        R.get("total_device_kernel_count", -1), R.get("total_device_dur_us", 0) / 1000))
    for cr in cat_rows[:8]:
        print("  %-12s count=%-6d share=%.2f%%" % (cr["category"], cr["count"], cr["share_pct"] or 0))


if __name__ == "__main__":
    main()
