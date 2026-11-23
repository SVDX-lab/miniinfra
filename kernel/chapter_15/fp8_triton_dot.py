#!/usr/bin/env python3
"""第 15 期 FP8 B 批：Triton FP8 dot 尝试（4090 / Triton 3.7.1 / sm_89）。

两个变体逐个 try/except，报错全文留档（平台边界证据）：
- B1 权重-only：W 存储 FP8 E4M3，Kernel 内 cast→FP16 × per-channel scale →
  tl.dot(fp16, fp16, acc=fp32)——与自研 int8_fused kernel 同语义（对照口径）；
- B2 纯 FP8 MMA：激活同转 FP8，tl.dot(fp8e4nv, fp8e4nv, acc=fp32)——sm_89
  FP8 Tensor Core 直通路径（W8A8 语义，测 Triton 是否发射 FP8 MMA）。

每变体：正确性（vs torch FP32 反量化参考）+ OperatorTimer 事件对 P50；
输出 results/fp8-triton-cuda.json。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fp8-triton-cuda.json"

M, K, N = 256, 1024, 1024
BM, BN, BK = 64, 64, 64
SEED = 20261122


def main() -> int:
    try:
        import triton
        import triton.language as tl
    except Exception as exc:
        OUT.write_text(json.dumps({"triton_import": f"FAIL {exc}"}, indent=1) + "\n")
        print(f"[fp8-triton] triton import FAIL: {exc}")
        return 1

    payload: dict = {
        "schema_version": "kernel-chapter16.fp8triton.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "triton": triton.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "shape": {"m": M, "k": K, "n": N, "block": [BM, BN, BK]},
        "variants": {},
    }
    print(f"[fp8-triton] triton={triton.__version__} torch={torch.__version__} "
          f"dev={payload['device']} cc={payload['capability']}")

    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    a16 = (torch.randn(M, K, device=dev) * 0.5).to(torch.float16)
    w32 = (torch.randn(K, N, device=dev) * 0.05)
    scale = (w32.abs().amax(dim=0) / 127.0).float()
    q8 = torch.clip(torch.round(w32 / scale[None, :]), -127, 127).to(torch.float8_e4m3fn)
    y_ref = (a16.float() @ (q8.float() * scale[None, :]))
    print(f"[fp8-triton] fixtures ok: q8 dtype={q8.dtype} scale[{scale.min():.4f},{scale.max():.4f}]")

    def timed(fn, warmup=10, repeats=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
               for _ in range(repeats)]
        for s, e in evs:
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        ts = sorted(s.elapsed_time(e) for s, e in evs)
        return ts[len(ts) // 2]

    # ------------------------------------------------------------------
    # B1：权重-only FP8 存储 + Kernel 内 cast→FP16 + per-channel scale
    # ------------------------------------------------------------------
    @triton.jit
    def w8a16_fp8_kernel(a_ptr, w_ptr, scale_ptr, c_ptr, M, N, K,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
        w_ptrs = w_ptr + rk[:, None] * N + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
            w8 = tl.load(w_ptr + (k + rk)[:, None] * N + rn[None, :])
            scale = tl.load(scale_ptr + rn)
            w = w8.to(tl.float16) * scale.to(tl.float16)[None, :]
            acc += tl.dot(a, w)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))

    try:
        c = torch.empty(M, N, device=dev, dtype=torch.float16)
        grid = (M // BM, N // BN)
        w8a16_fp8_kernel[grid](a16, q8, scale, c, M, N, K, BM=BM, BN=BN, BK=BK)
        torch.cuda.synchronize()
        err = (c.float() - y_ref).abs().max().item()
        t = timed(lambda: w8a16_fp8_kernel[grid](a16, q8, scale, c, M, N, K, BM=BM, BN=BN, BK=BK))
        flops = 2.0 * M * K * N
        payload["variants"]["B1_weight_only_fp8store"] = {
            "ok": True,
            "max_abs_vs_dequant_ref": round(err, 6),
            "p50_ms": round(t, 4),
            "tflops": round(flops / (t / 1000.0) / 1e12, 3),
        }
        print(f"[fp8-triton] B1 weight-only OK: max_abs={err:.4f} p50={t:.4f}ms "
              f"({flops / (t / 1000.0) / 1e12:.2f} TF)")
    except Exception as exc:
        payload["variants"]["B1_weight_only_fp8store"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[fp8-triton] B1 weight-only FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # B2：纯 FP8 dot（激活同转 FP8；fp8e4nv × fp8e4nv → acc fp32，sm_89 FP8 MMA）
    # ------------------------------------------------------------------
    @triton.jit
    def fp8_dot_kernel(a_ptr, w_ptr, c_ptr, M, N, K,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a8 = tl.load(a_ptr + rm[:, None] * K + (k + tl.arange(0, BK))[None, :])
            w8 = tl.load(w_ptr + (k + tl.arange(0, BK))[:, None] * N + rn[None, :])
            acc += tl.dot(a8, w8)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))

    try:
        a8 = torch.clip(torch.round(a16.float() * 2.0), -448, 448).to(torch.float8_e4m3fn)
        c2 = torch.empty(M, N, device=dev, dtype=torch.float16)
        grid = (M // BM, N // BN)
        fp8_dot_kernel[grid](a8, q8, c2, M, N, K, BM=BM, BN=BN, BK=BK)
        torch.cuda.synchronize()
        y2_ref = a8.float() @ q8.float()  # 纯 FP8 算术参考（两者都是 FP8 值的精确表示）
        err = (c2.float() - y2_ref).abs().max().item()
        t = timed(lambda: fp8_dot_kernel[grid](a8, q8, c2, M, N, K, BM=BM, BN=BN, BK=BK))
        flops = 2.0 * M * K * N
        payload["variants"]["B2_pure_fp8_dot"] = {
            "ok": True,
            "max_abs_vs_fp8_arithmetic_ref": round(err, 6),
            "p50_ms": round(t, 4),
            "tflops": round(flops / (t / 1000.0) / 1e12, 3),
        }
        print(f"[fp8-triton] B2 pure fp8 dot OK: max_abs={err:.4f} p50={t:.4f}ms "
              f"({flops / (t / 1000.0) / 1e12:.2f} TF)")
    except Exception as exc:
        payload["variants"]["B2_pure_fp8_dot"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[fp8-triton] B2 pure fp8 dot FAIL: {type(exc).__name__}: {exc}")

    # 对照：B1 同 shape 的 fp16 权重 GEMM（torch 引擎）
    w16 = (q8.float() * scale[None, :]).to(torch.float16)
    t16 = timed(lambda: a16 @ w16)
    payload["reference_fp16_engine"] = {
        "p50_ms": round(t16, 4),
        "tflops": round(2.0 * M * K * N / (t16 / 1000.0) / 1e12, 3),
    }
    print(f"[fp8-triton] fp16 engine ref: p50={t16:.4f}ms")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[fp8-triton] → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
