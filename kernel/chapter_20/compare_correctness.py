#!/usr/bin/env python3
"""第 20 期正确性比较（复用 ch01 口径：max/mean abs、rel、cos、NaN/Inf、误差位置）。

--backend cpu  : 参考自洽链 + naive loops 交叉 + FP64 锚点 + 确定性 + GQA 抽查
--backend cuda : kernel {fp16, fp32} vs fixtures CPU FP32 真值（容差分层）+
                 物化对照变体 + 长上下文 GPU eager 对照 + 确定性（3 次位级）
--backend musa : 委派 implementations/musa/attention_bench correctness
                 （二进制内 CPU FP32 真值矩阵），合并 JSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import attention_config as cfg  # noqa: E402
from attention_reference import (  # noqa: E402
    gqa_decode_reference,
    gqa_decode_truth64,
    gqa_mapping_spotcheck,
    gqa_prefill_reference,
    gqa_prefill_truth64,
    naive_loops_decode,
    naive_loops_prefill,
)
from experiment_utils import (  # noqa: E402
    backend_sync,
    comparison_metrics,
    device_environment_record,
    resolve_device,
    write_results_json,
)

SEQ_LENS = cfg.CORRECTNESS_SEQ_LENS
BATCHES = cfg.CORRECTNESS_BATCHES
# 容差分层（输出幅度 ~0.3-0.5）：fp32 io 紧、fp16 io 输出舍入主导
TOL = {"fp32": 5e-4, "fp16": 2e-2}


def load_fixtures() -> dict[str, np.ndarray]:
    data = np.load(HERE / "fixtures" / "correctness.npz")
    return {k: data[k] for k in data.files}


def eval_case(name: str, out: torch.Tensor, truth: torch.Tensor, tier: str) -> dict:
    metrics = comparison_metrics(out, truth)
    metrics["case"] = name
    metrics["tier"] = tier
    metrics["tolerance"] = TOL[tier]
    metrics["pass"] = bool(metrics["max_abs_error"] <= TOL[tier]
                           and metrics["nan_count"] == 0 and metrics["inf_count"] == 0)
    return metrics


# ---------------------------------------------------------------------------
# CPU 列
# ---------------------------------------------------------------------------

def correctness_cpu(results: dict) -> None:
    fixtures = load_fixtures()
    cases: list[dict] = []

    for s in SEQ_LENS:
        for b in BATCHES:
            q = torch.from_numpy(fixtures[f"p_s{s}_b{b}_q"])
            k = torch.from_numpy(fixtures[f"p_s{s}_b{b}_k"])
            v = torch.from_numpy(fixtures[f"p_s{s}_b{b}_v"])
            truth32 = torch.from_numpy(fixtures[f"p_s{s}_b{b}_truth32"])
            # 自洽：参考实现复跑 vs 存档真值（位级应一致）
            out = gqa_prefill_reference(q, k, v, cfg.SOFTMAX_SCALE)
            cases.append(eval_case(f"prefill/selfconsistent S={s} B={b}", out, truth32, "fp32"))
            # FP64 锚点：重算 FP64 vs 存档 truth64（归档完整性）+ FP32 vs FP64（舍入量级）
            if f"p_s{s}_b{b}_truth64" in fixtures:
                out64 = gqa_prefill_truth64(q.double(), k.double(), v.double(), cfg.SOFTMAX_SCALE)
                archived64 = torch.from_numpy(fixtures[f"p_s{s}_b{b}_truth64"])
                cases.append(eval_case(f"prefill/anchor64_archive S={s} B={b}",
                                       out64.float(), archived64, "fp32"))
                cases.append(eval_case(f"prefill/fp32_vs_truth64 S={s} B={b}",
                                       out, out64.float(), "fp32"))
        # naive loops 交叉（S=64 B=1，HQ=16 全头逐元素循环）
        q = torch.from_numpy(fixtures["p_s64_b1_q"]).double()
        k = torch.from_numpy(fixtures["p_s64_b1_k"]).double()
        v = torch.from_numpy(fixtures["p_s64_b1_v"]).double()
        o_loops = torch.tensor(naive_loops_prefill(q.tolist(), k.tolist(), v.tolist(),
                                                   cfg.SOFTMAX_SCALE), dtype=torch.float64)
        o_ref = gqa_prefill_reference(q, k, v, cfg.SOFTMAX_SCALE)
        cases.append(eval_case("prefill/naive_loops S=64 B=1",
                               o_loops.float(), o_ref.float(), "fp32"))

    for n in SEQ_LENS:
        for b in BATCHES:
            q = torch.from_numpy(fixtures[f"d_n{n}_b{b}_q"])
            k = torch.from_numpy(fixtures[f"d_n{n}_b{b}_k"])
            v = torch.from_numpy(fixtures[f"d_n{n}_b{b}_v"])
            truth32 = torch.from_numpy(fixtures[f"d_n{n}_b{b}_truth32"])
            out = gqa_decode_reference(q, k, v, cfg.SOFTMAX_SCALE)
            cases.append(eval_case(f"decode/selfconsistent N={n} B={b}", out, truth32, "fp32"))
            if f"d_n{n}_b{b}_truth64" in fixtures:
                out64 = gqa_decode_truth64(q.double(), k.double(), v.double(), cfg.SOFTMAX_SCALE)
                archived64 = torch.from_numpy(fixtures[f"d_n{n}_b{b}_truth64"])
                cases.append(eval_case(f"decode/anchor64_archive N={n} B={b}",
                                       out64.float(), archived64, "fp32"))
                cases.append(eval_case(f"decode/fp32_vs_truth64 N={n} B={b}",
                                       out, out64.float(), "fp32"))
        q = torch.from_numpy(fixtures["d_n64_b1_q"]).double()
        k = torch.from_numpy(fixtures["d_n64_b1_k"]).double()
        v = torch.from_numpy(fixtures["d_n64_b1_v"]).double()
        o_loops = torch.tensor(naive_loops_decode(q.tolist(), k.tolist(), v.tolist(),
                                                  cfg.SOFTMAX_SCALE), dtype=torch.float64)
        o_ref = gqa_decode_reference(q, k, v, cfg.SOFTMAX_SCALE)
        cases.append(eval_case("decode/naive_loops N=64 B=1",
                               o_loops.float(), o_ref.float(), "fp32"))

    # GQA 映射抽查（正反向）
    spot = gqa_mapping_spotcheck(
        torch.from_numpy(fixtures["gqa_q"]), torch.from_numpy(fixtures["gqa_k"]),
        torch.from_numpy(fixtures["gqa_v"]), cfg.SOFTMAX_SCALE)
    spot["pass"] = spot["forward_ok"] and spot["wrong_maps_rejected"]

    # 确定性：参考实现连续 3 次位级一致
    q = torch.from_numpy(fixtures["p_s512_b1_q"])
    k = torch.from_numpy(fixtures["p_s512_b1_k"])
    v = torch.from_numpy(fixtures["p_s512_b1_v"])
    runs = [gqa_prefill_reference(q, k, v, cfg.SOFTMAX_SCALE) for _ in range(3)]
    deterministic = bool(torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2]))

    failures = sum(1 for c in cases if not c["pass"])
    results.update({
        "cases": cases, "gqa_spotcheck": spot, "deterministic_cpu": deterministic,
        "failures": failures,
        "summary": {
            "total": len(cases), "pass": len(cases) - failures,
            "worst_fp32": max((c["max_abs_error"] for c in cases), default=0.0),
        },
    })
    print(f"[correctness/cpu] {len(cases) - failures}/{len(cases)} PASS，"
          f"deterministic={deterministic}，GQA spot pass={spot['pass']}")


# ---------------------------------------------------------------------------
# CUDA 列
# ---------------------------------------------------------------------------

def correctness_cuda(results: dict) -> None:
    fixtures = load_fixtures()
    device = resolve_device("cuda")
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    module = build_ext.get_module()
    cases: list[dict] = []

    def to_dev(a: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(a).to(device=device, dtype=dtype)

    for s in SEQ_LENS:
        for b in BATCHES:
            for dtype_name, dtype in (("fp16", torch.float16), ("fp32", torch.float32)):
                q = to_dev(fixtures[f"p_s{s}_b{b}_q"], dtype)
                k = to_dev(fixtures[f"p_s{s}_b{b}_k"], dtype)
                v = to_dev(fixtures[f"p_s{s}_b{b}_v"], dtype)
                truth = torch.from_numpy(fixtures[f"p_s{s}_b{b}_truth32"]).float()
                out = module.gqa_prefill_attention(q, k, v, cfg.SOFTMAX_SCALE).float().cpu()
                cases.append(eval_case(f"prefill_kernel/{dtype_name} S={s} B={b}",
                                       out, truth, dtype_name))
                # 物化对照变体（fp16 主口径）
                if dtype_name == "fp16":
                    outm = module.gqa_prefill_matmul(q, k, v, cfg.SOFTMAX_SCALE).float().cpu()
                    cases.append(eval_case(f"prefill_matmul/{dtype_name} S={s} B={b}",
                                           outm, truth, dtype_name))
            print(f"[correctness/cuda] prefill S={s} B={b} 完成")

    for n in SEQ_LENS:
        for b in BATCHES:
            for dtype_name, dtype in (("fp16", torch.float16), ("fp32", torch.float32)):
                q = to_dev(fixtures[f"d_n{n}_b{b}_q"], dtype)
                k = to_dev(fixtures[f"d_n{n}_b{b}_k"], dtype)
                v = to_dev(fixtures[f"d_n{n}_b{b}_v"], dtype)
                truth = torch.from_numpy(fixtures[f"d_n{n}_b{b}_truth32"]).float()
                out = module.gqa_decode_attention(q, k, v, cfg.SOFTMAX_SCALE).float().cpu()
                cases.append(eval_case(f"decode_kernel/{dtype_name} N={n} B={b}",
                                       out, truth, dtype_name))
        print(f"[correctness/cuda] decode N={n} 完成")

    # GQA 映射（核）：gqa fixtures 走 kernel，vs CPU 参考
    q = to_dev(fixtures["gqa_q"], torch.float16)
    k = to_dev(fixtures["gqa_k"], torch.float16)
    v = to_dev(fixtures["gqa_v"], torch.float16)
    ref = gqa_prefill_reference(torch.from_numpy(fixtures["gqa_q"]),
                                torch.from_numpy(fixtures["gqa_k"]),
                                torch.from_numpy(fixtures["gqa_v"]), cfg.SOFTMAX_SCALE)
    out = module.gqa_prefill_attention(q, k, v, cfg.SOFTMAX_SCALE).float().cpu()
    cases.append(eval_case("prefill_kernel/gqa_mapping_fp16", out, ref, "fp16"))

    # 长上下文（GPU eager fp32 参考，fixtures 无此形状——运行时生成，seed 规则同 fixtures）
    gen = torch.Generator().manual_seed(20261121 + 999)
    q4096 = torch.randn(1, cfg.HQ, 4096, cfg.D, generator=gen).to(device=device, dtype=torch.float32)
    k4096 = torch.randn(1, cfg.HKV, 4096, cfg.D, generator=gen).to(device=device, dtype=torch.float32)
    v4096 = torch.randn(1, cfg.HKV, 4096, cfg.D, generator=gen).to(device=device, dtype=torch.float32)
    ref4096 = gqa_prefill_reference(q4096, k4096, v4096, cfg.SOFTMAX_SCALE).float().cpu()
    out4096 = module.gqa_prefill_attention(
        q4096.half(), k4096.half(), v4096.half(), cfg.SOFTMAX_SCALE).float().cpu()
    cases.append(eval_case("prefill_kernel/long_context_S4096_fp16", out4096, ref4096, "fp16"))
    n32768 = 32768
    q32k = torch.randn(1, cfg.HQ, cfg.D, generator=gen).to(device=device, dtype=torch.float32)
    k32k = torch.randn(1, cfg.HKV, n32768, cfg.D, generator=gen).to(device=device, dtype=torch.float32)
    v32k = torch.randn(1, cfg.HKV, n32768, cfg.D, generator=gen).to(device=device, dtype=torch.float32)
    ref32k = gqa_decode_reference(q32k, k32k, v32k, cfg.SOFTMAX_SCALE).float().cpu()
    out32k = module.gqa_decode_attention(
        q32k.half(), k32k.half(), v32k.half(), cfg.SOFTMAX_SCALE).float().cpu()
    cases.append(eval_case("decode_kernel/long_context_N32768_fp16", out32k, ref32k, "fp16"))

    # 确定性：两形态各 3 次位级
    q = to_dev(fixtures["p_s512_b1_q"], torch.float16)
    k = to_dev(fixtures["p_s512_b1_k"], torch.float16)
    v = to_dev(fixtures["p_s512_b1_v"], torch.float16)
    runs = [module.gqa_prefill_attention(q, k, v, cfg.SOFTMAX_SCALE) for _ in range(3)]
    backend_sync(device)
    det_prefill = bool(torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2]))
    qd = to_dev(fixtures["d_n512_b1_q"], torch.float16)
    kd = to_dev(fixtures["d_n512_b1_k"], torch.float16)
    vd = to_dev(fixtures["d_n512_b1_v"], torch.float16)
    runds = [module.gqa_decode_attention(qd, kd, vd, cfg.SOFTMAX_SCALE) for _ in range(3)]
    backend_sync(device)
    det_decode = bool(torch.equal(runds[0], runds[1]) and torch.equal(runds[1], runds[2]))

    failures = sum(1 for c in cases if not c["pass"])
    results.update({
        "cases": cases,
        "device_warp_size": int(module.warp_size_device()),
        "deterministic": {"prefill": det_prefill, "decode": det_decode},
        "failures": failures,
        "summary": {
            "total": len(cases), "pass": len(cases) - failures,
            "worst_fp32": max((c["max_abs_error"] for c in cases if c["tier"] == "fp32"), default=0.0),
            "worst_fp16": max((c["max_abs_error"] for c in cases if c["tier"] == "fp16"), default=0.0),
        },
    })
    print(f"[correctness/cuda] {len(cases) - failures}/{len(cases)} PASS，"
          f"fp32 worst={results['summary']['worst_fp32']:.2e} "
          f"fp16 worst={results['summary']['worst_fp16']:.2e}，"
          f"deterministic p/d={det_prefill}/{det_decode}，warp={results['device_warp_size']}")


# ---------------------------------------------------------------------------
# MUSA 列（委派 standalone 二进制）
# ---------------------------------------------------------------------------

def correctness_musa(results: dict) -> None:
    bench_bin = HERE / "implementations" / "musa" / "attention_bench"
    if not bench_bin.exists():
        raise RuntimeError("attention_bench 未构建（先跑 implementations/musa/build.sh）")
    proc = subprocess.run([str(bench_bin), "correctness"], capture_output=True,
                          text=True, timeout=3600, cwd=str(bench_bin.parent))
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        raise RuntimeError(f".mu correctness rc={proc.returncode}: {proc.stderr[-2000:]}")
    mu = json.loads((bench_bin.parent / "musa-correctness.json").read_text())
    env_path = HERE / "environment-musa.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else None

    # 本地 demo 校验和 vs 二进制 checksum（同算法同 seed 的分布一致性抽查：
    # S=64 B=1 q 的 float64 顺序和——由 fixtures manifest 记录）
    manifest = json.loads((HERE / "fixtures" / "manifest.json").read_text())
    demo_expect = manifest["mu_generator"]["checksum_demo_q_s64_b1_float64_sequential"]
    demo_case = next((c for c in mu["cases"]
                      if c["form"] == "prefill" and c["seq"] == 64 and c["batch"] == 1), None)
    checksum_ok = None
    if demo_case is not None:
        # 阈值 1e-8：.mu 校验和打印粒度 %.9f（首版 %.6f 时限 ~1.4e-9 相对）
        checksum_ok = abs(demo_case["checksum_q"] - demo_expect) / abs(demo_expect) < 1e-8

    failures = mu["failures"] + (0 if checksum_ok else 1)
    results.update({
        "mode": "delegated to implementations/musa/attention_bench",
        "mu_cases": mu["cases"], "mu_failures": mu["failures"],
        "warp_device_builtin": mu.get("warp_device_builtin"),
        "checksum_demo_ok": checksum_ok,
        "checksum_demo_expect": demo_expect,
        "failures": failures,
        "environment": env,
        "summary": {
            "total": len(mu["cases"]),
            "pass": len(mu["cases"]) - mu["failures"],
            "worst_fp32": max((c["max_abs"] for c in mu["cases"] if c["dtype"] == "fp32"), default=0.0),
            "worst_fp16": max((c["max_abs"] for c in mu["cases"] if c["dtype"] == "fp16"), default=0.0),
        },
    })
    print(f"[correctness/musa] {len(mu['cases']) - mu['failures']}/{len(mu['cases'])} PASS，"
          f"warp={mu.get('warp_device_builtin')}，checksum_demo={checksum_ok}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    arguments = parser.parse_args()
    results: dict = {
        "schema_version": "kernel-chapter21.correctness.v1",
        "backend": arguments.backend,
        "config": {"hq": cfg.HQ, "hkv": cfg.HKV, "d": cfg.D,
                   "scale": cfg.SOFTMAX_SCALE, "seq_lens": SEQ_LENS, "batches": BATCHES},
        "tolerance_tiers": TOL,
    }
    if arguments.backend == "cpu":
        correctness_cpu(results)
        results["environment"] = device_environment_record(resolve_device("cpu"))
        out_path = HERE / "results" / "correctness-results-cpu.json"
    elif arguments.backend == "cuda":
        correctness_cuda(results)
        results["environment"] = device_environment_record(resolve_device("cuda"))
        out_path = HERE / "results" / "correctness-results-cuda.json"
    else:
        correctness_musa(results)
        out_path = HERE / "results" / "correctness-results-musa.json"

    write_results_json(out_path, results)
    if results["failures"]:
        print(f"[correctness/{arguments.backend}] 存在 {results['failures']} 个失败")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
