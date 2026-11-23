#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 30 期 实验 2：开源生态行"可用即验证"冒烟（4090 / CUDA）。

逐项：flash-attn（若已装）/ liger-kernel / sageattention / flashinfer / eplb / flaggems。
每项独立 try/except + pip 只收 wheel 超时守卫，失败记录原文（失败本身=生态边界证据）。
用法：python3 ecosystem_smoke.py [--only liger,sage,...]
输出：results/ecosystem-smoke-cuda.json
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

SCHEMA = "kernel-chapter31.ecosmoke.v1"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
SEED = 3131


def utcnow():
    return datetime.now(timezone.utc).strftime("%FT%TZ")


def pip_install(pkg, timeout=240, binary_only=True):
    cmd = [sys.executable, "-m", "pip", "install"]
    if binary_only:
        cmd += ["--only-binary=:all:"]
    cmd += ["--no-cache-dir", "--no-deps", pkg]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        tail = lambda s: "\n".join((s or "").splitlines()[-12:])
        return {"rc": p.returncode, "stdout_tail": tail(p.stdout), "stderr_tail": tail(p.stderr),
                "timeout_s": timeout, "binary_only": binary_only}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "stdout_tail": "", "stderr_tail": "TIMEOUT %ss" % timeout,
                "binary_only": binary_only}


def ver(modname):
    m = importlib.import_module(modname)
    return getattr(m, "__version__", "?")


# ---------------------------------------------------------------------------
def smoke_flash_attn(rec):
    import torch
    import flash_attn
    from flash_attn import flash_attn_func
    rec["version"] = getattr(flash_attn, "__version__", "?")
    B, S, H, D = 4, 8192, 16, 128
    g = torch.Generator(device="cuda").manual_seed(SEED)
    q, k, v = (torch.randn(B, S, H, D, generator=g, device="cuda", dtype=torch.float16) for _ in range(3))
    for _ in range(3):
        o = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
        s0.record(); o = flash_attn_func(q, k, v, causal=True); e0.record()
        torch.cuda.synchronize(); ts.append(s0.elapsed_time(e0))
    ts.sort(); p50 = ts[len(ts) // 2]
    flops = 2 * B * H * S * S * D  # causal 半三角 = 2*... (4*.../2)
    rec["smoke"] = {"ok": True, "shape": "B4 S8192 H16 D128 fp16 causal", "p50_ms": round(p50, 4),
                    "tflops_p50": round(flops / (p50 * 1e-3) / 1e12, 1),
                    "out_shape": list(o.shape), "nan_inf": int(torch.isnan(o).sum() + torch.isinf(o).sum())}


def smoke_liger(rec):
    import torch
    from liger_kernel.transformers import LigerRMSNorm
    rec["version"] = ver("liger_kernel")
    Hn, rows = 1024, 4096  # Qwen3-0.6B hidden_size 口径（ch02 合同真实模型 H）
    g = torch.Generator(device="cuda").manual_seed(SEED)
    x = torch.randn(rows, Hn, generator=g, device="cuda", dtype=torch.float16) * 2.0 + 0.1

    layer = LigerRMSNorm(Hn).to("cuda").half()
    y = layer(x + 0.0)
    # RMSNorm 有两个常见变体：Llama 式 x*rsqrt(mean+eps)*weight（weight 初始化为 1）
    # 与 Gemma 式 x*rsqrt(mean+eps)*(1+weight)（weight 初始化为 0）。
    # 参考对两种候选公式都算 err，记录安装版本实现的是哪一种（第二跑修正：0.8.5 为 Llama 式）
    norm = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).half()
    w = layer.weight.detach()
    err_llama = (y.float() - (norm * w).float()).abs().max()
    err_gemma = (y.float() - (norm * (1.0 + w)).float()).abs().max()
    variant = "llama(x*w)" if err_llama <= err_gemma else "gemma(x*(1+w))"
    err = min(err_llama, err_gemma)
    # 计时：liger vs eager 组合表达式（ch02 口径单算子对照；torch F.rms_norm 原生融合 op）
    import torch.nn.functional as F
    w_eff = w.contiguous()
    def eager(t):
        return F.rms_norm(t, [Hn], weight=w_eff, eps=1e-6)
    for _ in range(5):
        layer(x); eager(x)
    torch.cuda.synchronize()
    tl, te = [], []
    for _ in range(30):
        s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
        s0.record(); layer(x); e0.record(); torch.cuda.synchronize(); tl.append(s0.elapsed_time(e0))
        s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
        s0.record(); eager(x); e0.record(); torch.cuda.synchronize(); te.append(s0.elapsed_time(e0))
    tl.sort(); te.sort()
    rec["smoke"] = {"ok": True, "shape": "4096x1024 fp16", "formula_variant": variant,
                    "max_abs_err": float(err), "err_vs_llama_form": float(err_llama),
                    "err_vs_gemma_form": float(err_gemma),
                    "liger_p50_ms": round(tl[15], 4), "torch_eager_p50_ms": round(te[15], 4),
                    "speedup": round(te[15] / max(tl[15], 1e-9), 3)}


def smoke_sage(rec):
    import torch
    import sageattention
    rec["version"] = getattr(sageattention, "__version__", ver("sageattention"))
    from sageattention import sageattn
    B, S, H, D = 4, 8192, 16, 128
    g = torch.Generator(device="cuda").manual_seed(SEED)
    q, k, v = (torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.float16) for _ in range(3))
    o = sageattn(q, k, v, is_causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    err = (o.float() - ref.float()).abs()
    flops = 2 * B * H * S * S * D
    for _ in range(3):
        sageattn(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
        s0.record(); sageattn(q, k, v, is_causal=True); e0.record()
        torch.cuda.synchronize(); ts.append(s0.elapsed_time(e0))
    ts.sort(); p50 = ts[len(ts) // 2]
    rec["smoke"] = {"ok": True, "shape": "B4 S8192 H16 D128 fp16 causal (INT8 量化路径)",
                    "p50_ms": round(p50, 4), "tflops_p50": round(flops / (p50 * 1e-3) / 1e12, 1),
                    "max_abs_err_vs_sdpa": float(err.max()),
                    "cosine_vs_sdpa": float(torch.nn.functional.cosine_similarity(
                        o.float().flatten(), ref.float().flatten(), dim=0))}


def smoke_flashinfer(rec):
    import torch
    import flashinfer
    rec["version"] = getattr(flashinfer, "__version__", "?")
    # single_prefill：ch21 锚点同量级形状（flashinfer API 为 (num_tokens, H, D) 非 B 维）
    S, H, D = 8192, 16, 128
    g = torch.Generator(device="cuda").manual_seed(SEED)
    q = torch.randn(S, H, D, generator=g, device="cuda", dtype=torch.float16)
    k = torch.randn(S, H, D, generator=g, device="cuda", dtype=torch.float16)
    v = torch.randn(S, H, D, generator=g, device="cuda", dtype=torch.float16)
    if hasattr(flashinfer, "single_prefill_with_kv_cache"):
        fn = flashinfer.single_prefill_with_kv_cache
    else:
        from flashinfer import prefill
        fn = prefill.single_prefill_with_kv_cache
    o = fn(q, k, v, causal=True)
    flops = 2 * S * S * H * D
    for _ in range(3):
        fn(q, k, v, causal=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
        s0.record(); fn(q, k, v, causal=True); e0.record()
        torch.cuda.synchronize(); ts.append(s0.elapsed_time(e0))
    ts.sort(); p50 = ts[len(ts) // 2]
    rec["smoke"] = {"ok": True, "shape": "S8192 H16 D128 fp16 causal", "p50_ms": round(p50, 4),
                    "tflops_p50": round(flops / (p50 * 1e-3) / 1e12, 1),
                    "out_shape": list(o.shape)}


def smoke_eplb(rec):
    import torch
    import eplb
    rec["version"] = getattr(eplb, "__version__", "?")
    # 官方 README 最小调用：权重热统计 -> rebalance（CPU/GPU 皆可，这里 cuda）
    weight = torch.tensor(
        [[3, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
         [2, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0],
         [0, 2, 6, 2, 0, 0, 4, 2, 0, 2, 0, 2],
         [0, 0, 0, 0, 6, 3, 3, 3, 6, 0, 3, 6]], dtype=torch.float32, device="cuda")
    num_layers, num_replicas, num_groups, num_nodes, num_gpus = 4, 12, 2, 1, 2
    result = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)
    rec["smoke"] = {"ok": True, "weight_shape": list(weight.shape),
                    "phys_to_logical_shape": list(result[0].shape) if isinstance(result, tuple) and hasattr(result[0], "shape") else str(type(result))}


def smoke_flaggems(rec):
    import torch
    import flaggems  # noqa: F401 import 即注册后端
    rec["version"] = getattr(flaggems, "__version__", "?")
    g = torch.Generator(device="cuda").manual_seed(SEED)
    a = torch.randn(1024, 1024, generator=g, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, 1024, generator=g, device="cuda", dtype=torch.float32)
    c = a @ b
    ref = a.cpu() @ b.cpu()
    err = (c.cpu() - ref).abs().max()
    rec["smoke"] = {"ok": True, "shape": "1024x1024 fp32 matmul", "max_abs_err": float(err)}


ITEMS = {
    # key: (check/import module, pip package, smoke fn, pure_python)
    "flash_attn": ("flash_attn", "flash-attn", smoke_flash_attn, False),
    "liger": ("liger_kernel", "liger-kernel", smoke_liger, False),
    "sage": ("sageattention", "sageattention", smoke_sage, False),
    "flashinfer": ("flashinfer", "flashinfer-python", smoke_flashinfer, False),
    "eplb": ("eplb", "eplb", smoke_eplb, True),
    "flaggems": ("flaggems", "flaggems", smoke_flaggems, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="liger,sage,flashinfer,eplb,flaggems,flash_attn")
    args = ap.parse_args()
    names = [n.strip() for n in args.only.split(",") if n.strip() in ITEMS]
    log = {"schema_version": SCHEMA, "ts": utcnow(), "episode": 31,
           "note": "信息时点 2026-11-22；pip 只收预编译 wheel（纯 Python 包 wheel 失败后允许 sdist 回退，wheel-only 失败原文保留=生态边界证据）",
           "items": {}}
    import torch
    log["env"] = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                  "capability": list(torch.cuda.get_device_capability(0))}
    for n in names:
        modname, pkg, fn, pure_py = ITEMS[n]
        t0 = time.time()
        rec = {"item": n, "pip_package": pkg}
        # 先看是否已装
        rc = 1
        try:
            importlib.import_module(modname)
            rc = 0
        except Exception as e:  # noqa: BLE001
            rec["import_error_before_pip"] = repr(e)[:300]
        if rc != 0:
            rec["pip"] = pip_install(pkg, binary_only=True)
            rec["pip"]["seconds"] = round(time.time() - t0, 1)
            if rec["pip"]["rc"] != 0 and pure_py:
                # 纯 Python 包回退 sdist（无编译风险）；wheel-only 失败原文保留=边界证据
                rec["pip_sdist_fallback"] = pip_install(pkg, binary_only=False, timeout=300)
                rec["pip_sdist_fallback"]["seconds"] = round(time.time() - t0, 1)
                if rec["pip_sdist_fallback"]["rc"] == 0:
                    rec["pip"] = dict(rec["pip_sdist_fallback"], note="installed via sdist fallback")
        else:
            rec["pip"] = {"rc": 0, "note": "already installed"}
        try:
            fn(rec)
        except Exception as e:  # noqa: BLE001
            rec["smoke"] = {"ok": False, "error": repr(e)[:700]}
        rec["seconds"] = round(time.time() - t0, 1)
        log["items"][n] = rec
        print("[ECO] %-12s smoke_ok=%s" % (n, rec.get("smoke", {}).get("ok")))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "ecosystem-smoke-cuda.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print("WROTE", os.path.join(RESULTS, "ecosystem-smoke-cuda.json"))


if __name__ == "__main__":
    main()
