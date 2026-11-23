#!/usr/bin/env python3
"""第 27 期 CUDA 列回填驱动：构建 + 正确性对拍 + 实测 + 分相位基准数据。

正确性（对照 CPU 同 dtype 现场 chunked + fixtures 锚点口径）：
  - N ∈ {1,3,33,127,128,2048} × B ∈ {16,32,64}（scalar 门控 fp32 主线）；
  - initial_state ≠ 0 情形（N=127）；
  - fp16io 路径（输入 fp16 存储 → host cast fp32 → kernel）；
  - 容差：GPU fp32 vs CPU fp32（不同归约序）rel ≤ 5e-5 验收
    （绝对值随 |o| 记录；fixtures FP32 锚 2e-4 为上限护栏）。
实测（事件对，ch01 口径）：N ∈ {512,2048,8192,32768} × B ∈ {16,32,64}
  scheme A（smem state）；scheme B（global state）对 B=64 两个 N 点。
K3 两方案对比是本期回填的验证点（设计文档 §1）。

用法（4090 上）：python3 cuda_test.py [--skip-bench]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parent.parent
sys.path.insert(0, str(CHAPTER))
sys.path.insert(0, str(HERE.parent / "cpu"))

import deltanet_chunked as chunked  # noqa: E402
import deltanet_config as cfg  # noqa: E402
import deltanet_reference as ref  # noqa: E402


def load_ext():
    from torch.utils.cpp_extension import load
    return load(
        name="deltanet_chunked_ch28",
        sources=[str(HERE / "deltanet_chunked.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
        build_directory=str(HERE / "build"),
    )


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    ext = load_ext()
    print("[cuda] extension loaded")

    weights = ref.make_weights(cfg.SEED_BASE)
    cases = []
    n_fail = 0
    for n in cfg.CORRECTNESS_SEQ_LENS:
        gen = torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + n)
        x = torch.randn(n, cfg.D_MODEL, generator=gen, dtype=torch.float32)
        proj = ref.project_inputs(x, weights, "scalar")
        q_cpu, k_cpu, v_cpu = proj["q"], proj["k"], proj["v"]
        beta_cpu, g_cpu = proj["beta"], proj["g_log"]
        o_ref, s_ref = chunked.chunked_gdn(q_cpu, k_cpu, v_cpu, beta_cpu, g_cpu,
                                           chunk_size=64)
        for b in cfg.CHUNK_SWEEP:
            o_ref_b, s_ref_b = chunked.chunked_gdn(q_cpu, k_cpu, v_cpu, beta_cpu,
                                                   g_cpu, chunk_size=b)
            for scheme in (0, 1):
                qc = q_cpu.cuda(); kc = k_cpu.cuda(); vc = v_cpu.cuda()
                bc = beta_cpu.cuda(); gc = g_cpu.cuda()
                outs = ext.gdn_chunked_forward(qc, kc, vc, bc, gc, b, cfg.SCALE,
                                               None, scheme)
                o_g, s_g = outs[0], outs[1]
                torch.cuda.synchronize()
                ro = rel_err(o_g.cpu(), o_ref_b)
                rs = rel_err(s_g.cpu(), s_ref_b)
                ok = ro <= 5e-5 and rs <= 5e-5
                n_fail += (not ok)
                cases.append({"seq_len": n, "chunk_size": b, "scheme": scheme,
                              "dtype": "fp32", "rel_err_output": ro,
                              "rel_err_state": rs, "pass": ok})
                if not ok or (n in (1, 33, 2048) and scheme == 0):
                    print(f"[eq] N={n} B={b} scheme={scheme}: rel_o={ro:.3e} rel_s={rs:.3e} "
                          f"{'OK' if ok else 'FAIL'}")

        # fp16io 单点（N=2048 B=64 scheme 0）
        if n == 2048:
            q16, k16, v16 = q_cpu.half().float(), k_cpu.half().float(), v_cpu.half().float()
            b16, g16 = beta_cpu.half().float(), g_cpu.half().float()
            o_ref16, _ = chunked.chunked_gdn(q16, k16, v16, b16, g16, chunk_size=64)
            o_g = ext.gdn_chunked_forward(q16.cuda(), k16.cuda(), v16.cuda(),
                                          b16.cuda(), g16.cuda(), 64, cfg.SCALE,
                                          None, 0)[0]
            torch.cuda.synchronize()
            ro = rel_err(o_g.cpu(), o_ref16)
            ok = ro <= 5e-5
            n_fail += (not ok)
            cases.append({"seq_len": n, "chunk_size": 64, "scheme": 0,
                          "dtype": "fp16io", "rel_err_output": ro, "pass": ok})
            print(f"[eq] fp16io N=2048 B=64: rel_o={ro:.3e} {'OK' if ok else 'FAIL'}")

    # initial_state ≠ 0
    gen = torch.Generator().manual_seed(cfg.SEED_BASE + 9000 + 127)
    s0 = torch.randn(cfg.NUM_VALUE_HEADS, cfg.HEAD_K_DIM, cfg.HEAD_V_DIM,
                     generator=gen, dtype=torch.float32) * 0.1
    gen = torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + 127)
    x = torch.randn(127, cfg.D_MODEL, generator=gen, dtype=torch.float32)
    proj = ref.project_inputs(x, weights, "scalar")
    o_ref, s_ref = chunked.chunked_gdn(proj["q"], proj["k"], proj["v"], proj["beta"],
                                       proj["g_log"], chunk_size=64, initial_state=s0)
    outs = ext.gdn_chunked_forward(proj["q"].cuda(), proj["k"].cuda(), proj["v"].cuda(),
                                   proj["beta"].cuda(), proj["g_log"].cuda(), 64,
                                   cfg.SCALE, s0.cuda(), 0)
    o_g, s_g = outs[0], outs[1]
    torch.cuda.synchronize()
    ro = rel_err(o_g.cpu(), o_ref)
    rs = rel_err(s_g.cpu(), s_ref)
    ok = ro <= 5e-5 and rs <= 5e-5
    n_fail += (not ok)
    cases.append({"seq_len": 127, "chunk_size": 64, "scheme": 0, "dtype": "fp32",
                  "initial_state": "nonzero", "rel_err_output": ro,
                  "rel_err_state": rs, "pass": ok})
    print(f"[eq] S0≠0 N=127 B=64: rel_o={ro:.3e} rel_s={rs:.3e} {'OK' if ok else 'FAIL'}")

    print(f"[correctness] {len(cases)} 组，失败 {n_fail}")

    env = {
        "schema_version": "kernel-chapter28.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "correctness": {"cases": cases, "fail": n_fail,
                        "tolerance": "rel ≤ 5e-5（GPU vs CPU fp32 归约序差）"},
    }
    (CHAPTER / "results" / "correctness-results-cuda.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.skip_bench:
        return 0 if n_fail == 0 else 1

    # ---- 实测 ----
    entries = []

    def bench(n, b, scheme, warmup, repeats):
        gen = torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + n)
        x = torch.randn(n, cfg.D_MODEL, generator=gen, dtype=torch.float32)
        proj = ref.project_inputs(x, weights, "scalar")
        qc = proj["q"].cuda(); kc = proj["k"].cuda(); vc = proj["v"].cuda()
        bc = proj["beta"].cuda(); gc = proj["g_log"].cuda()
        for _ in range(warmup):
            ext.gdn_chunked_forward(qc, kc, vc, bc, gc, b, cfg.SCALE, None, scheme)
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            ext.gdn_chunked_forward(qc, kc, vc, bc, gc, b, cfg.SCALE, None, scheme)
            e_ev.record()
            torch.cuda.synchronize()
            samples.append(s_ev.elapsed_time(e_ev))
        samples.sort()
        p50 = samples[len(samples) // 2]
        entries.append({"seq_len": n, "chunk_size": b, "scheme": scheme,
                        "p50_ms": round(p50, 4), "min_ms": round(samples[0], 4),
                        "warmup": warmup, "repeats": repeats,
                        "samples_ms": [round(s, 4) for s in samples]})
        print(f"[bench] N={n} B={b} scheme={scheme}: p50={p50:.3f}ms")

    for n in cfg.MEMORY_SEQ_LENS:
        for b in cfg.CHUNK_SWEEP:
            bench(n, b, 0, 3 if n <= 2048 else 1, 10 if n <= 2048 else 5)
    # K3 两方案对比：scheme B（global state）@B=64
    for n in (2048, 32768):
        bench(n, 64, 1, 2, 10)

    # 递推 eager GPU 对照（两形式 GPU 形态，MUSA 2.8-2.95× 的 CUDA 版）
    def bench_recurrent(n, warmup, repeats):
        gen = torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + n)
        x = torch.randn(n, cfg.D_MODEL, generator=gen, dtype=torch.float32)
        proj = ref.project_inputs(x, weights, "scalar")
        qg = proj["q"].cuda(); kg = proj["k"].cuda(); vg = proj["v"].cuda()
        bg = proj["beta"].cuda(); gg = proj["g_log"].cuda()

        def recurrent_eager():
            state = torch.zeros(cfg.NUM_VALUE_HEADS, cfg.HEAD_K_DIM, cfg.HEAD_V_DIM,
                                device="cuda")
            gb = gg.unsqueeze(-1)
            o = torch.empty(n, cfg.NUM_VALUE_HEADS, cfg.HEAD_V_DIM, device="cuda")
            qs = qg * cfg.SCALE
            for t in range(n):
                decayed = state * torch.exp(gb[t]).unsqueeze(-1)
                u = (vg[t] - torch.einsum("hkv,hk->hv", decayed, kg[t])) * bg[t].unsqueeze(-1)
                state = decayed + torch.einsum("hk,hv->hkv", kg[t], u)
                o[t] = torch.einsum("hkv,hk->hv", state, qs[t])
            return o, state

        for _ in range(warmup):
            recurrent_eager()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            recurrent_eager()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
        samples.sort()
        entries.append({"seq_len": n, "form": "recurrent_eager_gpu",
                        "p50_ms": round(samples[len(samples) // 2], 4),
                        "warmup": warmup, "repeats": repeats})
        print(f"[bench] recurrent eager N={n}: p50={samples[len(samples)//2]:.3f}ms")

    bench_recurrent(512, 2, 5)
    bench_recurrent(2048, 2, 5)

    result = {
        "schema_version": "kernel-chapter28.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "timing_note": "事件对口径（ch01）；N>2048 降 warmup=1 repeats=5",
        "entries": entries,
    }
    (CHAPTER / "results" / "benchmark-results-cuda.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[results] 已写 correctness-results-cuda.json / benchmark-results-cuda.json")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
