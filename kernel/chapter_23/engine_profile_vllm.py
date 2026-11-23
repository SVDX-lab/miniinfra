# -*- coding: utf-8 -*-
"""实验 5+6（修正版）：引擎 Decode 步算子序列 + Profiler 复核。

修正记录（KI-24-07）：V1 引擎模型在 EngineCore 子进程执行，父进程 torch.profiler
抓不到任何设备 kernel（首版 engine_profile_decode.py 0 事件）。改用 vLLM 内置
LLM.start_profile()/stop_profile()（worker 侧 torch profiler，chrome trace 落
VLLM_TORCH_PROFILER_DIR），从 trace 文件解析：
- 设备 kernel 按名称聚合（时长/计数 Top30）+ 类别占比（rms/rope/attention/gemm/cache…）
- RMSNorm 占比（对照 ep05 占比层稀释归因）
- 时序前 300 设备事件（decode 算子序列证据，对照 ch21-23）
- 两次捕获：batch=1（算子序列主证）+ batch=8（批串行归因）
产出 results/engine-profile-trace-<plat>.json + trace 文件本体。
"""
import argparse
import glob
import json as _json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine24_common import dump_json, load_json, utcnow  # noqa: E402

CATEGORIES = [
    ("rmsnorm", r"rms|fused_add|norm_norm"),
    ("rope", r"rotary|rope"),
    ("attention", r"attn|attention|flash|fmha|sdpa|paged|splat"),
    ("gemm", r"gemm|matmul|addmm|\bmm\b|linear|s16816|wig"),
    ("kv_cache", r"cache|cat_with|kv_"),
    ("sampler", r"argmax|sampl|softmax|topk|logits|penalt"),
    ("quant", r"quant|dequant"),
    ("elementwise", r"elementwise|silu|gelu|mul|add|copy_|cat|embedding|index"),
]


def classify(name: str) -> str:
    low = name.lower()
    for cat, pat in CATEGORIES:
        if re.search(pat, low):
            return cat
    return "other"


def parse_trace(path, R, tag):
    # vLLM 0.34 dump 的 trace 可能是 gzip（0x1f 0x8b 魔数）
    with open(path, "rb") as fb:
        magic = fb.read(2)
    if magic == b"\x1f\x8b":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            tr = _json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            tr = _json.load(f)
    evs = [e for e in tr.get("traceEvents", [])
           if e.get("ph") == "X" and e.get("cat", "").lower() in
           ("kernel", "gpu_op", "kernel_activity", "gpu_kernel")]
    if not evs:  # 兜底：部分版本 cat 是 "Kernel" 以外还有 "gpu_memcpy" 等
        evs = [e for e in tr.get("traceEvents", [])
               if e.get("ph") == "X" and "kernel" in e.get("cat", "").lower()]
    evs.sort(key=lambda e: e.get("ts", 0))
    agg = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    total = 0.0
    for e in evs:
        d = float(e.get("dur", 0) or 0)
        agg[e.get("name", "?")]["count"] += 1
        agg[e.get("name", "?")]["dur_us"] += d
        total += d
    rows = [{
        "name": n, "count": s["count"], "dur_us": round(s["dur_us"], 1),
        "dur_share_pct": round(100 * s["dur_us"] / total, 3) if total else None,
        "category": classify(n),
    } for n, s in agg.items()]
    rows.sort(key=lambda r: -r["dur_us"])
    cat_agg = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    for r in rows:
        c = cat_agg[r["category"]]
        c["count"] += r["count"]
        c["dur_us"] += r["dur_us"]
    cats = sorted(
        ({"category": k, "count": v["count"], "dur_us": round(v["dur_us"], 1),
          "share_pct": round(100 * v["dur_us"] / total, 2) if total else None}
         for k, v in cat_agg.items()),
        key=lambda r: -r["dur_us"])
    R[tag] = {
        "trace_file": os.path.basename(path),
        "device_kernel_events": len(evs),
        "total_device_dur_us": round(total, 1),
        "top30_by_time": rows[:30],
        "top30_by_count": sorted(rows, key=lambda r: -r["count"])[:30],
        "category_share": cats,
        "rmsnorm_detail": [r for r in rows if r["category"] == "rmsnorm"][:12],
        "chronological_first300": [
            {"name": e.get("name", "")[:120], "dur_us": e.get("dur")}
            for e in evs[:300]],
    }
    return len(evs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--truth", default="fixtures/truth-ch01.json")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--trace-dir", default="results/profile-trace")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    plat = os.environ.get("PLATFORM", "musa")
    if args.out is None:
        args.out = f"results/engine-profile-trace-{plat}.json"
    os.makedirs(args.trace_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(args.trace_dir, "*")))
    os.environ["VLLM_TORCH_PROFILER_DIR"] = os.path.abspath(args.trace_dir)

    truth = load_json(args.truth)
    R = {
        "schema_version": "kernel-chapter24.profiletrace.v1",
        "created_at_utc": utcnow(),
        "engine": f"vllm ({plat}) enforce_eager + LLM.start_profile/stop_profile",
        "model_path": args.model,
        "prof_note": ("V1 引擎模型在子进程——必须用 worker 侧 profiler（KI-24-07）；"
                      "trace 事件设备时长在 MUSA kineto 下部分场景不可信（M-04），时长与计数并列"),
        "max_tokens": args.max_tokens,
    }

    from vllm import LLM, SamplingParams

    llm_kw = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="auto",
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=1234,
    )
    # vLLM 0.34 起 start_profile 需显式 profiler_config（0.29 musa 版走
    # VLLM_TORCH_PROFILER_DIR 环境变量即可）；两个通道都配置，版本间兼容
    llm_kw["profiler_config"] = {
        "profiler": "torch",
        "torch_profiler_dir": os.path.abspath(args.trace_dir),
    }
    try:
        llm = LLM(**llm_kw)
    except TypeError:
        # 老版本不认识 profiler_config → 退回仅环境变量通道
        llm_kw.pop("profiler_config")
        llm = LLM(**llm_kw)
    p1 = [{"prompt_token_ids": truth["prompt_token_ids"]}]
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    llm.generate(p1, SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)  # warmup

    assert hasattr(llm, "start_profile"), "LLM.start_profile 不存在——引擎版本不支持"

    # 捕获 1：batch=1（decode 算子序列主证）；start/stop 失败不致命
    # （profiler_config 的 with_profile 调度在部分版本自动捕获）
    try:
        llm.start_profile()
    except Exception as e:  # noqa: BLE001
        R["start_profile_error_b1"] = repr(e)
    outs = llm.generate(p1, sp, use_tqdm=False)
    try:
        llm.stop_profile()
    except Exception as e:  # noqa: BLE001
        R["stop_profile_error_b1"] = repr(e)
    R["b1_generate"] = {
        "token_count": len(outs[0].outputs[0].token_ids),
        "first_token_match_ch01": outs[0].outputs[0].token_ids[0] == truth["first_token_id"],
    }

    # 捕获 2：batch=8（批串行归因：看引擎发射的是 1 份还是 8 份 kernel）
    p8 = [dict(p1[0]) for _ in range(8)]
    try:
        llm.start_profile()
    except Exception as e:  # noqa: BLE001
        R["start_profile_error_b8"] = repr(e)
    outs8 = llm.generate(p8, SamplingParams(temperature=0, max_tokens=args.max_tokens), use_tqdm=False)
    try:
        llm.stop_profile()
    except Exception as e:  # noqa: BLE001
        R["stop_profile_error_b8"] = repr(e)
    R["b8_generate"] = {"token_count_first": len(outs8[0].outputs[0].token_ids)}

    # 解析新产生的 trace 文件（按大小排序：b1 先跑文件小？以生成时间为准取 diff）
    new_files = sorted(set(glob.glob(os.path.join(args.trace_dir, "*"))) - before,
                       key=os.path.getmtime)
    R["trace_files_found"] = [os.path.basename(f) for f in new_files]
    if len(new_files) >= 2:
        # b1 = 早生成的，b8 = 晚生成的
        n1 = parse_trace(new_files[0], R, "b1")
        n2 = parse_trace(new_files[1], R, "b8")
        R["batch_serialization_evidence"] = {
            "b1_kernel_events": n1,
            "b8_kernel_events": n2,
            "b8_over_b1_ratio": round(n2 / n1, 3) if n1 else None,
            "interpretation": ("ratio≈8 → 每请求独立发射（串行/无批融合）；ratio≈1 → 真批处理",
                               ),
        }
    elif len(new_files) == 1:
        n1 = parse_trace(new_files[0], R, "b1")
        R["batch_serialization_evidence"] = {"b1_kernel_events": n1, "note": "b8 trace 未产生"}
    else:
        R["trace_parse_error"] = "no new trace files"

    dump_json(R, args.out)
    print(f"profile-trace written -> {args.out}")
    for tag in ("b1", "b8"):
        d = R.get(tag)
        if not d:
            continue
        print(f"[{tag}] events={d['device_kernel_events']} dur={d['total_device_dur_us']/1000:.1f}ms")
        for c in d["category_share"][:10]:
            print("   %-12s count=%-7d share=%.2f%%" % (c["category"], c["count"], c["share_pct"] or 0))


if __name__ == "__main__":
    main()
