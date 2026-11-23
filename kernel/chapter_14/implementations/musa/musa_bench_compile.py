"""第 14 期 MUSA 列：若 torch.compile 可用（含 monkeypatch 路径），与 ch04 手写锚点对照。

对照锚点（ch04 EXPERIMENT_LOG §3.5，直接引用不重测）：
  手写融合 fp16 @2048×1024: 0.0613 ms；分离 fp16 @2048: 0.1067 ms（1.74×）
  手写融合 fp32 @2048×1024: 0.0677 ms；分离 fp32 @2048: 0.1305 ms（1.93×）
  M-12：MUSA launch 事件对地板 ~28µs。

可用性判定来自 musa-compile-probe.json（v2：usable + workaround 字段）；
workaround=monkeypatch 时本脚本先打同款补丁再编译（并显式标注）。
不可用则输出 UNSUPPORTED 并退出 0（平台边界，不算失败）。
"""

from __future__ import annotations

import argparse
import json
import types
from datetime import datetime, timezone
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"

CH04_ANCHORS = {
    "handwritten_fused_fp16_rows2048_ms": 0.0613,
    "handwritten_sep_fp16_rows2048_ms": 0.1067,
    "handwritten_fused_fp32_rows2048_ms": 0.0677,
    "handwritten_sep_fp32_rows2048_ms": 0.1305,
    "copy_peak_gbps": 711.3,
    "launch_floor_us": 28.0,
}


def apply_probe_workaround() -> str | None:
    probe_path = RESULTS / "musa-compile-probe.json"
    if not probe_path.exists():
        return None
    probe = json.loads(probe_path.read_text())
    if probe.get("usable") is None:
        return None
    if str(probe.get("workaround", "")).startswith("torch_musa.streams"):
        import torch_musa

        torch_musa.streams = types.SimpleNamespace(Stream=torch.musa.Stream)
        return probe["usable"]
    return probe["usable"]


def add_rmsnorm(r, h, w, eps: float = 1e-6):
    t = r + h
    ft = t.float()
    var = ft.pow(2).mean(-1, keepdim=True)
    y = w * (ft * torch.rsqrt(var + eps)).to(t.dtype)
    return t, y


def time_musa(fn, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    pairs = []
    for _ in range(repeats):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        pairs.append((start, end))
    torch.musa.synchronize()
    samples = sorted(s.elapsed_time(e) for s, e in pairs)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "p50_ms": round(samples[len(samples) // 2], 4),
        "p95_ms": round(samples[min(len(samples) - 1, int(round(len(samples) * 0.95)) - 1)], 4),
        "min_ms": round(samples[0], 4),
        "samples_ms": [round(v, 4) for v in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--shapes", default="128,2048,32768")
    args = parser.parse_args()

    import torch_musa  # noqa: F401

    payload: dict = {
        "schema_version": "kernel-chapter15.musa-compile-bench.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "anchors": CH04_ANCHORS,
        "entries": [],
    }

    usable_backend = apply_probe_workaround()
    payload["usable_backend"] = usable_backend
    payload["workaround_applied"] = usable_backend is not None and "monkeypatch" in str(
        (RESULTS / "musa-compile-probe.json").read_text()[:2000] if (RESULTS / "musa-compile-probe.json").exists() else ""
    ) or (usable_backend or "").startswith("streams-monkeypatch")
    payload["workaround_note"] = (
        "streams-monkeypatch = 实验性补丁（torch_musa.streams = SimpleNamespace("
        "Stream=torch.musa.Stream)），仅用于探测 guard 缺口的唯一性，非生产建议"
        if (usable_backend or "").startswith("streams-monkeypatch")
        else None
    )

    if args.check_only:
        print("usable:", usable_backend)
        return

    if usable_backend is None:
        payload["verdict"] = (
            "UNSUPPORTED：torch_musa 上无任何 compile backend 能完成编译+首调"
            "（guard 构造期 torch.cuda.streams 别名缺口，报错全文见 "
            "musa-compile-probe.json）；compile 前后延迟对照不成立（平台边界）"
        )
        out = RESULTS / "musa-compile-bench.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(payload["verdict"])
        return

    for rows in [int(v) for v in args.shapes.split(",")]:
        for dtype in (torch.float16, torch.float32):
            gen = torch.Generator(device="cpu").manual_seed(20260415 + rows)
            r = torch.randn(rows, 1024, generator=gen).to(device="musa", dtype=dtype)
            h = torch.randn(rows, 1024, generator=gen).to(device="musa", dtype=dtype)
            w = torch.randn(1024, generator=gen).to(device="musa", dtype=dtype)
            torch._dynamo.reset()
            compiled = torch.compile(add_rmsnorm)
            t_c, y_c = compiled(r, h, w)
            torch.musa.synchronize()
            # 正确性对拍（eager musa 口径）
            t_e, y_e = add_rmsnorm(r, h, w)
            torch.musa.synchronize()
            max_abs_y = float((y_c - y_e).abs().max().item())
            t_bitwise = bool(torch.equal(t_c, t_e))
            eager_latency = time_musa(lambda: add_rmsnorm(r, h, w), 3, 10)
            compiled_latency = time_musa(lambda: compiled(r, h, w), 3, 10)
            key = str(dtype).replace("torch.", "")
            anchor_fused = (
                CH04_ANCHORS["handwritten_fused_fp16_rows2048_ms"] if dtype == torch.float16
                else CH04_ANCHORS["handwritten_fused_fp32_rows2048_ms"]
            )
            entry = {
                "rows": rows,
                "dtype": key,
                "t_bitwise_vs_eager": t_bitwise,
                "max_abs_y_vs_eager": max_abs_y,
                "eager": eager_latency,
                "compiled": compiled_latency,
                "speedup_p50": round(eager_latency["p50_ms"] / compiled_latency["p50_ms"], 4),
                "vs_ch04_handwritten_fused_rows2048": {
                    "anchor_ms": anchor_fused,
                    "compiled_over_anchor": round(
                        compiled_latency["p50_ms"] / anchor_fused, 4
                    ),
                },
            }
            payload["entries"].append(entry)
            print(
                f"[bench] {rows}x1024 {key}: eager {eager_latency['p50_ms']} ms, "
                f"compiled {compiled_latency['p50_ms']} ms (x{entry['speedup_p50']}, "
                f"t_bitwise={t_bitwise}, max_abs_y={max_abs_y:.2e}, "
                f"vs_ch04_fused={entry['vs_ch04_handwritten_fused_rows2048']['compiled_over_anchor']})",
                flush=True,
            )

    out = RESULTS / "musa-compile-bench.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[bench] written {out}")


if __name__ == "__main__":
    main()
