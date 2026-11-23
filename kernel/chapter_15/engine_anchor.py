#!/usr/bin/env python3
"""第 15 期引擎侧锚点与平台能力探测（torch / torch_musa）。

内容（对应大纲 §14 准入第 9 条：INT8 数据类型与运算路径确认；§9 引擎参考）：
1. INT8 数据类型与基本路径确认：randint 生成 int8（randn 不支持 int8 属正常）→
   to(device) → 转 FP16 → per-channel scale 乘法；
2. 引擎 INT8 运算路径探测：torch.matmul(int8)（预期不支持）、torch._int_mm
   （INT8 引擎路径，逐平台记录可用性）、torch.quantize_per_channel(qint8)；
3. FP16 引擎 GEMM 锚点（同 shape 同口径 OperatorTimer，作"自研 vs 引擎"标尺）；
4. （仅 cuda 后端）FP8 e4m3 转换与 torch._scaled_mm 探测（4090 修复后执行）。

输出：results/engine-anchor-{backend}.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiment_utils import (  # noqa: E402
    SCHEMA_VERSION,
    OperatorTimer,
    backend_sync,
    device_environment_record,
    resolve_device,
    write_results_json,
)

DECODE_ANCHOR_SHAPES = [(m, k, n) for m in (1, 4, 8) for k in (1024, 4096) for n in (1024, 4096)]
PREFILL_ANCHOR_SHAPES = [(256, 1024, 1024), (1024, 1024, 1024)]


def probe_int8_paths(device) -> dict:
    """§14 准入第 9 条：INT8 数据类型与运算路径逐项实测。"""
    import torch

    probe: dict = {"checks": []}

    def record(name: str, fn):
        try:
            detail = fn()
            probe["checks"].append({"name": name, "ok": True, "detail": detail})
            print(f"[probe] {name}: OK {detail}")
        except Exception as exc:
            probe["checks"].append(
                {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"[probe] {name}: FAIL {type(exc).__name__}: {exc}")

    def make_int8():
        w = torch.randint(-127, 128, (256, 512), dtype=torch.int8, device=device)
        return (
            f"randint int8 shape={tuple(w.shape)} dtype={w.dtype} "
            f"min={w.min().item()} max={w.max().item()}"
        )

    def to_fp16():
        t = torch.randint(-127, 128, (64, 64), dtype=torch.int8).to(device).to(torch.float16)
        return f"converted dtype={t.dtype} shape={tuple(t.shape)}"

    def scale_mult():
        q = torch.randint(-127, 128, (128, 256), dtype=torch.int8).to(device)
        scale = torch.rand(256, dtype=torch.float32, device=device) * 0.01 + 0.001
        deq = q.to(torch.float16) * scale.to(torch.float16)[None, :]
        return f"per-channel scale mult ok dtype={deq.dtype} mean_abs={deq.abs().mean().item():.5f}"

    def matmul_int8():
        a = torch.randint(-127, 128, (8, 128), dtype=torch.int8, device=device)
        b = torch.randint(-127, 128, (128, 256), dtype=torch.int8, device=device)
        c = a @ b
        backend_sync(device)
        return f"int8 matmul ok dtype={c.dtype}"

    def int_mm():
        a = torch.randint(-127, 128, (8, 128), dtype=torch.int8, device=device)
        b = torch.randint(-127, 128, (256, 128), dtype=torch.int8, device=device).t()
        c = torch._int_mm(a, b)
        backend_sync(device)
        return f"torch._int_mm ok dtype={c.dtype} shape={tuple(c.shape)}"

    def qint8():
        w = torch.randn(64, 32, device=device)
        qw = torch.quantize_per_channel(
            w,
            scales=torch.abs(w).amax(dim=1) / 127,
            zero_points=torch.zeros(64, dtype=torch.long, device=device),
            axis=0,
            dtype=torch.qint8,
        )
        return f"quantize_per_channel qint8 ok scales={tuple(qw.q_per_channel_scales().shape)}"

    def fp8_convert():
        x = torch.randn(32, 32, device=device)
        x8 = x.to(torch.float8_e4m3fn)
        return f"fp8 e4m3 convert ok dtype={x8.dtype}"

    def scaled_mm():
        a = torch.randn(16, 32, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        b = torch.randn(32, 16, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        c = torch._scaled_mm(a, b)
        backend_sync(device)
        return f"torch._scaled_mm ok dtype={c.dtype}"

    record("int8_create_randint", make_int8)
    record("int8_to_fp16_convert", to_fp16)
    record("int8_scale_multiply_fp16", scale_mult)
    record("engine_matmul_int8_native", matmul_int8)
    record("engine_torch_int_mm", int_mm)
    record("torch_quantize_per_channel_qint8", qint8)
    record("fp8_e4m3_convert", fp8_convert)
    if device.type == "cuda":
        record("engine_torch_scaled_mm_fp8", scaled_mm)
    return probe


def anchor_fp16_matmul(device) -> list[dict]:
    """FP16 引擎 GEMM 锚点（同 shape，OperatorTimer 事件对口径）。"""
    import torch

    entries = []
    timer = OperatorTimer(device, warmup=20, repeats=50)  # MUSA 首轮 JIT 预热（ch02 记录）
    for (m, k, n) in DECODE_ANCHOR_SHAPES + PREFILL_ANCHOR_SHAPES:
        a = (torch.randn(m, k, device=device) * 0.5).to(torch.float16)
        w = (torch.randn(k, n, device=device) * 0.05).to(torch.float16)
        timing = timer.measure(lambda: a @ w)
        flops = 2.0 * m * k * n
        entries.append(
            {
                "section": "engine_fp16_matmul_anchor",
                "m": m,
                "k": k,
                "n": n,
                "bucket": "decode" if m <= 8 else "prefill",
                "dtype": "fp16",
                **timing,
                "tflops_p50": round(flops / (timing["p50_ms"] / 1000.0) / 1e12, 4),
            }
        )
        print(
            f"[anchor] fp16 {m}x{k}x{n}: p50={timing['p50_ms']:.4f}ms "
            f"tflops={entries[-1]['tflops_p50']:.2f}"
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    parser.add_argument("--out", default="")
    arguments = parser.parse_args()

    import time

    device = resolve_device(arguments.backend)
    env = device_environment_record(device)
    print(f"[anchor] backend={arguments.backend} device={env.get('gpu', {}).get('name')}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": env,
        "int8_path_probe": probe_int8_paths(device),
        "fp16_engine_anchor": anchor_fp16_matmul(device),
    }

    out_path = (
        Path(arguments.out)
        if arguments.out
        else ROOT / "results" / f"engine-anchor-{arguments.backend}.json"
    )
    write_results_json(out_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
