#!/usr/bin/env python3
"""第 20 期统一计时口径基准（双形态 × seq 扫描 × dtype；实验 2/5 的数据源）。

测量对象：
  prefill kernel（tile + online softmax）    S ∈ {64, 512, 2048, 8192} × B {1,4} × {fp16, fp32}
  decode kernel（单 query chunked）          N ∈ {64, 512, 2048, 8192, 32768} × B {1,4} × {fp16, fp32}
  prefill 物化对照变体（三 kernel，S×S 满阵） S ∈ {512, 2048} × B {1,4} × fp16
  厂商库分母（torch SDPA eager）             同 prefill/decode 扫描（fp16）
  CPU 列                                     参考实现 fp32 计时（同一矩阵，S ≤ 2048）
  --backend musa                             委派 implementations/musa/attention_bench bench

口径：
  - 计时器 = ch01 OperatorTimer（事件对逐样本、末尾一次同步），P50/P95；
  - 理论字节/FLOPs/AI/bound/工作集分区由 roofline_attention 单源计算随结果写入；
  - 实测有效带宽 = 理论字节(unique 与 issued 两个口径)/P50；
  - 实测有效算力 = FLOPs/P50；
  - AI 曲线（实验 2）：prefill 理想 AI = (S+1)/3 随 seq 线性升、跨脊点；
    decode AI ≈ 2.0（unique）/1.0（issued）恒定 —— 实测达成率对照验证 bound。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import attention_config as cfg  # noqa: E402
import roofline_attention as ra  # noqa: E402
from attention_reference import (  # noqa: E402
    gqa_decode_reference,
    gqa_prefill_reference,
    naive_loops_decode,
    naive_loops_prefill,
)
from experiment_utils import (  # noqa: E402
    OperatorTimer,
    device_environment_record,
    resolve_device,
    write_results_json,
)

SEED_BASE = 20261121
PREFILL_LENS = cfg.BENCH_PREFILL_SEQ_LENS       # [64, 512, 2048, 8192]
DECODE_LENS = cfg.BENCH_DECODE_KV_LENS          # [64, 512, 2048, 8192, 32768]
BATCHES = cfg.BENCH_BATCHES                     # [1, 4]
MAT_LENS = cfg.MATERIALIZED_SEQ_LENS            # [512, 2048]


def bench_plan(seq: int) -> tuple[int, int]:
    """（warmup, repeats）随形状缩放：大形状限重复次数控会话时长。"""
    if seq <= 512:
        return 3, 30
    if seq <= 2048:
        return 2, 15
    if seq <= 8192:
        return 1, 5
    return 1, 3


# ---------------------------------------------------------------------------
# 输入生成（运行时，seed 规则同 fixtures tag 编码）
# ---------------------------------------------------------------------------

def gen_inputs(device: torch.device, dtype: torch.dtype, b: int, s: int,
               decode: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import numpy as np

    tag_map = {"q": 0, "k": 1, "v": 2, "dq": 3, "dk": 4, "dv": 5}

    def gen(shape, tag):
        seed = SEED_BASE + tag_map[tag] * 1000003 + sum(shape) * 7 + len(shape)
        rng = np.random.default_rng(seed)
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(device=device, dtype=dtype)

    if decode:
        q = gen((b, cfg.HQ, cfg.D), "dq")
        k = gen((b, cfg.HKV, s, cfg.D), "dk")
        v = gen((b, cfg.HKV, s, cfg.D), "dv")
    else:
        q = gen((b, cfg.HQ, s, cfg.D), "q")
        k = gen((b, cfg.HKV, s, cfg.D), "k")
        v = gen((b, cfg.HKV, s, cfg.D), "v")
    return q, k, v


def attach_theory(entry: dict, platform: str) -> dict:
    """把 roofline 单源理论档案 + 实测达成率并入结果条目。"""
    dtype = entry["dtype"]
    if entry["form"] == "prefill":
        profile = ra.prefill_profile(entry["batch"], entry["seq"], dtype, platform)
    else:
        profile = ra.decode_profile(entry["batch"], entry["kv_len"], dtype, platform)
    if "p50_ms" in entry and entry["p50_ms"] > 0:
        t_s = entry["p50_ms"] / 1000.0
        profile["achieved_gbs_unique"] = round(profile["bytes_unique"] / t_s / 1e9, 1)
        profile["achieved_gbs_issued"] = round(profile["bytes_issued"] / t_s / 1e9, 1)
        profile["achieved_tflops"] = round(profile["flops"] / t_s / 1e12, 3)
        copy_peak = cfg.PLATFORM_PEAKS[platform]["copy_gb_s_measured"]
        fp_peak = (cfg.PLATFORM_PEAKS[platform]["fp16_gemm_tf_measured"]
                   if dtype == "fp16" else cfg.PLATFORM_PEAKS[platform]["fp32_gemm_tf_measured"])
        profile["gbs_vs_copy_peak_pct"] = round(profile["achieved_gbs_unique"] / copy_peak * 100.0, 1)
        profile["tflops_vs_peak_pct"] = round(profile["achieved_tflops"] / fp_peak * 100.0, 1)
        profile["bound_verdict"] = (
            "memory-dominated (实测带宽达成率 ≥ 算力达成率)"
            if profile["gbs_vs_copy_peak_pct"] >= profile["tflops_vs_peak_pct"]
            else "compute-dominated (实测算力达成率 > 带宽达成率)")
    return profile


# ---------------------------------------------------------------------------
# CUDA 列
# ---------------------------------------------------------------------------

def bench_cuda(platform: str, prefill_lens=None, decode_lens=None,
               batches=None, dtypes=None, skip_sdpa=False, run_kernels=True) -> dict:
    device = resolve_device("cuda")
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    module = build_ext.get_module()
    prefill_lens = prefill_lens or PREFILL_LENS
    decode_lens = decode_lens or DECODE_LENS
    batches = batches or BATCHES
    dtypes = dtypes or [("fp16", torch.float16), ("fp32", torch.float32)]
    results: dict = {"backend": "cuda", "entries": [], "sdpa": [],
                     "warp_device": int(module.warp_size_device())}

    fp16_peak = cfg.PLATFORM_PEAKS[platform]["fp16_gemm_tf_measured"]
    fp32_peak = cfg.PLATFORM_PEAKS[platform]["fp32_gemm_tf_measured"]
    copy_peak = cfg.PLATFORM_PEAKS[platform]["copy_gb_s_measured"]
    results["denominators"] = {"copy_gb_s": copy_peak, "fp16_tf": fp16_peak, "fp32_tf": fp32_peak}

    if run_kernels:
        for dtype_name, dtype in dtypes:
            for s in prefill_lens:
                warm, rep = bench_plan(s)
                for b in batches:
                    q, k, v = gen_inputs(device, dtype, b, s, decode=False)
                    entry = {"form": "prefill", "batch": b, "seq": s, "dtype": dtype_name}
                    timing = OperatorTimer(device, warm, rep).measure(
                        lambda: module.gqa_prefill_attention(q, k, v, cfg.SOFTMAX_SCALE))
                    entry.update(timing)
                    entry["theory"] = attach_theory(entry, platform)
                    results["entries"].append(entry)
                    print(f"[bench/cuda] prefill {dtype_name} S={s} B={b}: "
                          f"p50={entry['p50_ms']}ms ai={entry['theory']['ai_unique']} "
                          f"{entry['theory']['bound_verdict']} "
                          f"bw%={entry['theory']['gbs_vs_copy_peak_pct']} "
                          f"tf%={entry['theory']['tflops_vs_peak_pct']}")
            for n in decode_lens:
                warm, rep = bench_plan(n)
                for b in batches:
                    q, k, v = gen_inputs(device, dtype, b, n, decode=True)
                    entry = {"form": "decode", "batch": b, "kv_len": n, "dtype": dtype_name}
                    timing = OperatorTimer(device, warm, rep).measure(
                        lambda: module.gqa_decode_attention(q, k, v, cfg.SOFTMAX_SCALE))
                    entry.update(timing)
                    entry["theory"] = attach_theory(entry, platform)
                    results["entries"].append(entry)
                    print(f"[bench/cuda] decode {dtype_name} N={n} B={b}: "
                          f"p50={entry['p50_ms']}ms "
                          f"bw%={entry['theory']['gbs_vs_copy_peak_pct']} "
                          f"tf%={entry['theory']['tflops_vs_peak_pct']} "
                          f"{entry['theory']['bound_verdict']}")

        # 物化对照变体（fp16，S ∈ {512, 2048}）
        for s in MAT_LENS:
            warm, rep = bench_plan(s)
            for b in batches:
                q, k, v = gen_inputs(device, torch.float16, b, s, decode=False)
                entry = {"form": "prefill", "variant": "materialized", "batch": b,
                         "seq": s, "dtype": "fp16"}
                timing = OperatorTimer(device, warm, rep).measure(
                    lambda: module.gqa_prefill_matmul(q, k, v, cfg.SOFTMAX_SCALE))
                entry.update(timing)
                prof = ra.prefill_profile(b, s, "fp16", platform)
                prof_mat = dict(prof)
                prof_mat["bytes_matmul_unique"] = ra.prefill_bytes_matmul_unique(b, s, 2)
                prof_mat["bytes_matmul_issued"] = ra.prefill_bytes_matmul(b, s, 2)
                prof_mat["extra_bytes_vs_online"] = (prof_mat["bytes_matmul_unique"]
                                                     - prof["bytes_unique"])
                if entry["p50_ms"] > 0:
                    prof_mat["achieved_gbs_matmul_unique"] = round(
                        prof_mat["bytes_matmul_unique"] / (entry["p50_ms"] / 1000.0) / 1e9, 1)
                entry["theory"] = prof_mat
                results["entries"].append(entry)
                print(f"[bench/cuda] prefill-materialized fp16 S={s} B={b}: p50={entry['p50_ms']}ms")

    # 厂商库分母：torch SDPA eager（fp16）。enable_gqa 首选；torch 2.12+cu130
    # 实测对 GQA 形状报 "Number of heads in key and value must divide the number
    # of heads in query"（平台边界）→ 回退手动 repeat_interleave 展开（pre-GQA
    # 标准路径，展开成本计入厂商分母，条目标注 mode）。
    def sdpa_call(q, k, v, causal):
        try:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal, enable_gqa=True)
            return out, "enable_gqa"
        except Exception:  # noqa: BLE001
            group = q.size(-3) // k.size(-3)
            kk = k.repeat_interleave(group, dim=-3)
            vv = v.repeat_interleave(group, dim=-3)
            out = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, is_causal=causal)
            return out, "kv_expand_fallback"

    if not skip_sdpa:
        for s in prefill_lens:
            warm, rep = bench_plan(s)
            for b in batches:
                q, k, v = gen_inputs(device, torch.float16, b, s, decode=False)
                entry = {"form": "prefill", "variant": "sdpa_eager", "batch": b, "seq": s,
                         "dtype": "fp16"}
                try:
                    _, mode = sdpa_call(q, k, v, True)
                    entry["mode"] = mode
                    entry.update(OperatorTimer(device, warm, rep).measure(
                        lambda: sdpa_call(q, k, v, True)[0]))
                except Exception as error:  # noqa: BLE001
                    entry["error"] = f"{type(error).__name__}: {error}"
                results["sdpa"].append(entry)
                print(f"[bench/cuda] SDPA prefill fp16 S={s} B={b}: "
                      f"p50={entry.get('p50_ms')}ms mode={entry.get('mode')}")
        for n in decode_lens:
            warm, rep = bench_plan(n)
            for b in batches:
                q, k, v = gen_inputs(device, torch.float16, b, n, decode=True)
                entry = {"form": "decode", "variant": "sdpa_eager", "batch": b, "kv_len": n,
                         "dtype": "fp16"}
                try:
                    _, mode = sdpa_call(q, k, v, False)
                    entry["mode"] = mode
                    entry.update(OperatorTimer(device, warm, rep).measure(
                        lambda: sdpa_call(q, k, v, False)[0]))
                except Exception as error:  # noqa: BLE001
                    entry["error"] = f"{type(error).__name__}: {error}"
                results["sdpa"].append(entry)
                print(f"[bench/cuda] SDPA decode fp16 N={n} B={b}: "
                      f"p50={entry.get('p50_ms')}ms mode={entry.get('mode')}")
    return results


def bench_cpu() -> dict:
    device = resolve_device("cpu")
    results: dict = {"backend": "cpu", "entries": []}
    for s in [64, 512, 2048]:
        warm, rep = bench_plan(s)
        rep = min(rep, 10) if s >= 2048 else rep
        for b in BATCHES:
            q, k, v = gen_inputs(device, torch.float32, b, s, decode=False)
            entry = {"form": "prefill", "batch": b, "seq": s, "dtype": "fp32"}
            entry.update(OperatorTimer(device, warm, rep).measure(
                lambda: gqa_prefill_reference(q, k, v, cfg.SOFTMAX_SCALE)))
            entry["theory"] = attach_theory(entry, "cuda")  # 曲线用，分母解读限 cuda 列
            results["entries"].append(entry)
            print(f"[bench/cpu] prefill fp32 S={s} B={b}: p50={entry['p50_ms']}ms")
    for n in [64, 512, 2048]:
        warm, rep = bench_plan(n)
        for b in BATCHES:
            q, k, v = gen_inputs(device, torch.float32, b, n, decode=True)
            entry = {"form": "decode", "batch": b, "kv_len": n, "dtype": "fp32"}
            entry.update(OperatorTimer(device, warm, rep).measure(
                lambda: gqa_decode_reference(q, k, v, cfg.SOFTMAX_SCALE)))
            entry["theory"] = attach_theory(entry, "cuda")
            results["entries"].append(entry)
            print(f"[bench/cpu] decode fp32 N={n} B={b}: p50={entry['p50_ms']}ms")
    # 字面循环（数量级锚点，ch18 风格）：S=64/N=64 B=1
    import numpy as np
    rng = np.random.default_rng(SEED_BASE)
    ql = rng.standard_normal((1, cfg.HQ, 64, cfg.D)).tolist()
    kl = rng.standard_normal((1, cfg.HKV, 64, cfg.D)).tolist()
    vl = rng.standard_normal((1, cfg.HKV, 64, cfg.D)).tolist()
    import time

    start = time.perf_counter()
    naive_loops_prefill(ql, kl, vl, cfg.SOFTMAX_SCALE)
    loops_prefill_ms = (time.perf_counter() - start) * 1000.0
    ql2 = rng.standard_normal((1, cfg.HQ, cfg.D)).tolist()
    kl2 = rng.standard_normal((1, cfg.HKV, 64, cfg.D)).tolist()
    vl2 = rng.standard_normal((1, cfg.HKV, 64, cfg.D)).tolist()
    start = time.perf_counter()
    naive_loops_decode(ql2, kl2, vl2, cfg.SOFTMAX_SCALE)
    loops_decode_ms = (time.perf_counter() - start) * 1000.0
    results["naive_loops"] = {
        "prefill_S64_B1_ms": round(loops_prefill_ms, 2),
        "decode_N64_B1_ms": round(loops_decode_ms, 2),
        "note": "纯 Python 逐元素循环（数量级锚点）",
    }
    print(f"[bench/cpu] naive loops: prefill S=64 = {loops_prefill_ms:.1f}ms, "
          f"decode N=64 = {loops_decode_ms:.1f}ms")
    return results


# ---------------------------------------------------------------------------
# MUSA 列（委派 standalone 二进制 + python 侧理论档案合并）
# ---------------------------------------------------------------------------

def bench_musa(platform: str) -> dict:
    bench_bin = HERE / "implementations" / "musa" / "attention_bench"
    if not bench_bin.exists():
        raise RuntimeError("attention_bench 未构建（先跑 implementations/musa/build.sh）")
    proc = subprocess.run([str(bench_bin), "bench"], capture_output=True,
                          text=True, timeout=3600, cwd=str(bench_bin.parent))
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        raise RuntimeError(f".mu bench rc={proc.returncode}: {proc.stderr[-2000:]}")
    mu = json.loads((bench_bin.parent / "musa-benchmark.json").read_text())
    results: dict = {"backend": "musa", "entries": [], "sdpa": [],
                     "warp_device_builtin": mu.get("warp_device_builtin")}
    for raw in mu["entries"]:
        entry = {"form": "prefill" if raw["form"].startswith("prefill") else "decode",
                 "variant": "materialized" if "materialized" in raw["form"] else "kernel",
                 "batch": raw["batch"], "seq": raw.get("seq"), "kv_len": raw.get("seq"),
                 "dtype": raw["dtype"],
                 "p50_ms": raw["p50_ms"], "p95_ms": raw["p95_ms"],
                 "mean_ms": raw["mean_ms"], "min_ms": raw["min_ms"],
                 "warmup": raw["warmup"], "repeats": raw["repeats"]}
        if entry["variant"] == "materialized":
            prof = ra.prefill_profile(entry["batch"], entry["seq"], "fp16", platform)
            prof["bytes_matmul_unique"] = ra.prefill_bytes_matmul_unique(entry["batch"], entry["seq"], 2)
            if entry["p50_ms"] > 0:
                prof["achieved_gbs_matmul_unique"] = round(
                    prof["bytes_matmul_unique"] / (entry["p50_ms"] / 1000.0) / 1e9, 1)
            entry["theory"] = prof
        else:
            entry["theory"] = attach_theory(
                {"form": entry["form"], "batch": entry["batch"], "seq": entry["seq"],
                 "kv_len": entry["kv_len"], "dtype": entry["dtype"],
                 "p50_ms": entry["p50_ms"]}, platform)
        results["entries"].append(entry)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--prefill-lens", default=None,
                        help="逗号分隔扫描（nsys profile 缩小进程用，如 2048）")
    parser.add_argument("--decode-lens", default=None)
    parser.add_argument("--batches", default=None, help="如 1")
    parser.add_argument("--dtypes", default=None, help="如 fp16")
    parser.add_argument("--skip-sdpa", action="store_true")
    parser.add_argument("--sdpa-only", action="store_true",
                        help="只跑 SDPA 厂商库分母（结果写 benchmark-results-{b}-sdpa.json）")
    arguments = parser.parse_args()
    if arguments.backend == "cpu":
        results = bench_cpu()
        results["environment"] = device_environment_record(resolve_device("cpu"))
        suffix = "cpu"
    elif arguments.backend == "cuda":
        prefill_lens = [int(x) for x in arguments.prefill_lens.split(",")] if arguments.prefill_lens else None
        decode_lens = [int(x) for x in arguments.decode_lens.split(",")] if arguments.decode_lens else None
        batches = [int(x) for x in arguments.batches.split(",")] if arguments.batches else None
        dtypes = [(x, torch.float16 if x == "fp16" else torch.float32)
                  for x in arguments.dtypes.split(",")] if arguments.dtypes else None
        results = bench_cuda("cuda", prefill_lens, decode_lens, batches, dtypes,
                             skip_sdpa=arguments.skip_sdpa,
                             run_kernels=not arguments.sdpa_only)
        if arguments.sdpa_only:
            results = {"backend": "cuda", "sdpa": results["sdpa"]}
            suffix = "cuda-sdpa"
            out_path = Path(arguments.out) if arguments.out else \
                HERE / "results" / "benchmark-results-cuda-sdpa.json"
            results["schema_version"] = "kernel-chapter21.benchmark.v1"
            write_results_json(out_path, results)
            return 0
        results["environment"] = device_environment_record(resolve_device("cuda"))
        suffix = "cuda"
    else:
        platform = "musa"
        results = bench_musa(platform)
        # MUSA 环境 JSON 由 .mu attrs 产物 + python torch_musa 记录合并
        try:
            results["environment"] = device_environment_record(resolve_device("musa"))
        except RuntimeError as error:
            results["environment_error"] = str(error)
        suffix = "musa"
    results["schema_version"] = "kernel-chapter21.benchmark.v1"
    results["scan"] = {"prefill_lens": PREFILL_LENS, "decode_lens": DECODE_LENS,
                       "batches": BATCHES, "materialized_lens": MAT_LENS}
    out_path = Path(arguments.out) if arguments.out else \
        HERE / "results" / f"benchmark-results-{suffix}.json"
    write_results_json(out_path, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
