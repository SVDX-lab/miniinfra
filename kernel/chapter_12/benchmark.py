#!/usr/bin/env python3
"""第 12 期实验 4：遥测输出对照——per-op 计时与 kernel launch 数的可观察性。

口径（与第 01–04 期一致）：
  - 微基准：OperatorTimer（GPU 事件对 w2 r10 / CPU perf_counter），
    融合 op vs 原生组合表达式链，rows ∈ {1, 20, 128}（20 = truth prompt 长度）；
  - 模型级：prefill 一次 + Greedy 8 步的 per-step wall（backend.sync 边界）；
  - 遥测：profile_one_forward 的 per-op 表 + count_kernel_launches
    （CUDA = kernel 事件 + cudaLaunchKernel；MUSA = musaLaunchKernel CPU 侧
    事件、设备时长全 0（M-04/M-30 运行时层体现）；CPU = N/A 平台边界）。

输出：results/benchmark-results-<backend>.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import (  # noqa: E402
    OperatorTimer,
    backend_sync,
    device_environment_record,
    write_results_json,
)
from implementations.runtime import Qwen3Runtime, create_backend  # noqa: E402
from implementations.runtime.qwen3_runtime import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    bootstrap_compat,
    load_truth,
)

ROW_COUNTS = [1, 20, 128]
HIDDEN = 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--dtype", default=None, choices=[None, "float32", "float16"])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    bootstrap_compat()
    if arguments.backend == "musa":
        import torch_musa  # noqa: F401

    model_dir = Path(arguments.model_dir) if arguments.model_dir else DEFAULT_MODEL_DIR
    dtype = getattr(torch, arguments.dtype) if arguments.dtype else None

    backend = create_backend(arguments.backend)
    backend.ensure_available()
    backend.setup()
    device = backend.resolve_device()
    if dtype is None:
        dtype = torch.float32 if arguments.backend == "cpu" else torch.float16

    payload: dict = {
        "schema_version": "kernel-chapter13.benchmark.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": arguments.backend,
        "dtype": str(dtype).replace("torch.", ""),
        "environment": device_environment_record(device),
        "capability_report": backend.query_capabilities(deep=True).to_dict(),
        "timer_semantics": "gpu:event_pairs_w2_r10" if device.type != "cpu"
        else "cpu:perf_counter_w2_r10",
    }

    # ---- 层 1：融合 op vs 组合表达式链（微基准）----
    router = backend.fused_ops_protocol()
    micro_rows = []
    for rows in ROW_COUNTS:
        generator = torch.Generator().manual_seed(20261113 + rows)
        r = torch.randn(rows, HIDDEN, generator=generator).to(device=device, dtype=dtype)
        h = torch.randn(rows, HIDDEN, generator=generator).to(device=device, dtype=dtype)
        w = torch.randn(HIDDEN, generator=generator).to(device=device, dtype=dtype)

        def composite_chain():
            t = r + h
            float_states = t.float()
            variance = float_states.pow(2).mean(dim=-1, keepdim=True)
            return w * (float_states * torch.rsqrt(variance + 1e-6)).to(t.dtype)

        def fused_call():
            hh = h.clone()
            rr = r.clone()
            router.fused_add_rmsnorm(hh, rr, w, 1e-6)

        timer = OperatorTimer(device, warmup=2, repeats=10)
        composite_timing = timer.measure(composite_chain)
        fused_timing = timer.measure(fused_call)
        ratio = round(composite_timing["p50_ms"] / fused_timing["p50_ms"], 3) \
            if fused_timing["p50_ms"] > 0 else None
        micro_rows.append({
            "rows": rows,
            "composite_p50_ms": composite_timing["p50_ms"],
            "fused_op_p50_ms": fused_timing["p50_ms"],
            "speedup_fused_vs_composite": ratio,
            "composite_full": composite_timing,
            "fused_full": fused_timing,
        })
        print(f"[micro rows={rows}] composite {composite_timing['p50_ms']}ms "
              f"vs fused {fused_timing['p50_ms']}ms = {ratio}x")
    payload["layer1_micro"] = micro_rows

    # ---- 模型级遥测：prefill + Greedy 8 步 + per-op 表 + launch 数 ----
    truth = load_truth()
    runtime = Qwen3Runtime(backend, model_dir, dtype=dtype, fused=True)
    runtime.load()
    prompt_ids = runtime.encode_chat_prompt(truth["prompt"])
    input_ids = runtime.input_ids_tensor(prompt_ids)

    # prefill 一次（w2 r5）。
    timer = OperatorTimer(device, warmup=2, repeats=5)
    prefill_timing = timer.measure(lambda: runtime.model(input_ids))
    payload["prefill"] = prefill_timing
    print(f"[prefill] p50={prefill_timing['p50_ms']}ms")

    # Greedy 解码 N 步（per-step wall，backend.sync 边界）。
    generate_result = runtime.generate(input_ids, max_new_tokens=arguments.decode_steps,
                                       eos_token_id=truth["eos_token_id"])
    payload["decode_8step"] = {
        "steps": generate_result.step_times_ms,
        "mean_step_ms": generate_result.to_dict()["mean_step_ms"],
        "first_step_ms": generate_result.step_times_ms[0] if generate_result.step_times_ms else None,
        "op_call_counts": generate_result.op_call_counts,
    }
    print(f"[decode] steps={payload['decode_8step']['steps']}")

    # per-op 表 + 发射数（遥测可观察性核心对照）。
    profile_stats = runtime.profile_one_forward(input_ids, top_n=12)
    payload["telemetry_profile"] = profile_stats
    print(f"[telemetry] launch_stats available={profile_stats['launch_stats'].get('available')} "
          f"launches/iter={profile_stats['launch_stats'].get('kernel_launches_per_iter')} "
          f"runtime_events={profile_stats['launch_stats'].get('runtime_launch_events')} "
          f"device_duration_claim={profile_stats['device_kernel_duration_observable']}")

    # native vs fused 单次前向（层 2 轻量对照）。
    native_runtime = Qwen3Runtime(backend, model_dir, dtype=dtype, fused=False)
    native_runtime.load()
    native_timing = timer.measure(lambda: native_runtime.model(input_ids))
    fused_timing = timer.measure(lambda: runtime.model(input_ids))
    payload["forward_native_vs_fused"] = {
        "native_p50_ms": native_timing["p50_ms"],
        "fused_p50_ms": fused_timing["p50_ms"],
        "ratio_native_over_fused": round(native_timing["p50_ms"] / fused_timing["p50_ms"], 4)
        if fused_timing["p50_ms"] > 0 else None,
    }
    print(f"[forward] native {native_timing['p50_ms']}ms vs fused {fused_timing['p50_ms']}ms")

    out_path = Path(arguments.out) if arguments.out else \
        HERE / "results" / f"benchmark-results-{arguments.backend}.json"
    write_results_json(out_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
