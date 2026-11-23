#!/usr/bin/env python3
"""第 04 期三层测量（对照第 01/02/03 期 baseline，解释微基准收益被稀释的过程）。

层 1（单算子微基准）：融合 op（注册路径）vs 原生组合表达式链 vs ctypes 直传
    （同一 kernel 的第 03 期通道，分离 dispatch 开销）；
层 2（单层 Transformer）：真实权重 + 探针形状（seq=20）的原生层 vs 融合层；
层 3（完整推理）：首 Token Prefill + 32 Token Greedy Decode（每步重算整段，
    与第 01 期口径一致），原生 vs 融合；报告融合 op 时间占比与端到端收益上限。

口径：OperatorTimer 事件对（GPU）/ perf_counter（CPU），Warm-up/重复次数
入结果；基准 dtype 与模型 dtype 一致（GPU fp16 / CPU fp32）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE / "implementations")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import (  # noqa: E402
    OperatorTimer,
    backend_sync,
    count_kernel_launches,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from implementations.backend_setup import setup_backend  # noqa: E402
from implementations.cpu.fused_add_rmsnorm_ref import RMS_NORM_EPS  # noqa: E402
import qwen3_model_fused as model_lib  # noqa: E402

HIDDEN_SIZE = 1024
ROW_COUNTS = [1, 4, 20, 128, 512, 2048]  # 20 = 探针序列长度（真值 prompt token 数）
PEAK_BANDWIDTH = {"cuda": 918.8e9, "musa": 711.3e9, "cpu": None}  # ep02 实测 D2D 峰值
MODEL_DEFAULT = HERE.parent / "chapter_01" / "models" / "qwen3-0.6b"
TRUTH_DEFAULT = HERE.parent / "chapter_01" / "truth"
OP_CALLS_PER_FORWARD = 56  # 28 层 × 2（layer0 只 1 次层内 op + 终局 op，见模型文件头注释）


def seeded_pair(rows: int, hidden: int, device: torch.device, dtype: torch.dtype):
    seed = 20260501 + rows * 100000 + hidden
    generator = torch.Generator().manual_seed(seed)
    r = torch.randn(rows, hidden, generator=generator).to(device=device, dtype=dtype)
    h = torch.randn(rows, hidden, generator=generator).to(device=device, dtype=dtype)
    w = torch.randn(hidden, generator=generator).to(device=device, dtype=dtype)
    return r, h, w


def composite_chain_operation(r: torch.Tensor, h: torch.Tensor, w: torch.Tensor,
                              eps: float):
    """原生组合表达式（模型中原样）：t = r + h; y = RMSNorm 模块数学（out-of-place）。"""
    t = r + h
    float_states = t.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return w * normalized.to(t.dtype)


def layer1_microbench(backend: str, device: torch.device, router, op_module) -> dict:
    dtype = torch.float32 if backend == "cpu" else torch.float16
    results = {}
    for rows in ROW_COUNTS:
        r, h, w = seeded_pair(rows, HIDDEN_SIZE, device, dtype)
        repeats = 200 if rows <= 128 else (100 if rows == 512 else 50)
        timer = OperatorTimer(device, warmup=10, repeats=repeats)

        def fused_op_call():
            router.fused_add_rmsnorm(h, r, w, RMS_NORM_EPS)

        composite_timing = timer.measure(
            lambda: composite_chain_operation(r, h, w, RMS_NORM_EPS))
        fused_timing = timer.measure(fused_op_call)

        entry = {"rows": rows, "hidden_size": HIDDEN_SIZE, "dtype": str(dtype),
                 "composite_chain": composite_timing, "fused_op_registered": fused_timing}
        if backend != "cpu":
            # ctypes 直传通道（第 03 期绑定，同一 kernel 的非注册路径）：
            # dispatch 开销对照 + 注册层是否引入额外成本。
            import importlib

            binding = importlib.import_module("ops_binding")
            lib_name = "libmini_fused_cuda.so" if backend == "cuda" else "libmini_fused_musa.so"
            kernels = binding.load_kernels(backend, lib_name)
            warp_size = kernels.probe_warp_size(device)
            r2, h2, w2 = seeded_pair(rows, HIDDEN_SIZE, device, dtype)
            y2 = torch.empty_like(r2)
            ctypes_timing = timer.measure(
                lambda: kernels.fused_add_rmsnorm(r2, h2, w2, y2, RMS_NORM_EPS, 256, warp_size))
            entry["ctypes_fused_ep04_channel"] = ctypes_timing
        speedup = composite_timing["p50_ms"] / fused_timing["p50_ms"]
        entry["speedup_fused_vs_composite_p50"] = round(speedup, 3)
        results[f"rows_{rows}"] = entry
        print(f"[L1/{backend}] rows={rows:5d} {dtype}: composite={composite_timing['p50_ms']:.4f}ms "
              f"fused_op={fused_timing['p50_ms']:.4f}ms 加速比={speedup:.2f}×"
              + (f" ctypes={entry['ctypes_fused_ep04_channel']['p50_ms']:.4f}ms" if backend != "cpu" else ""))

    # 发射数证据：组合表达式链 vs 注册 op（dispatch 是否真的进入单一 Kernel）
    r, h, w = seeded_pair(20, HIDDEN_SIZE, device, dtype)
    results["launch_counts_rows20"] = {
        "composite_chain": count_kernel_launches(
            lambda: composite_chain_operation(r, h, w, RMS_NORM_EPS), 10, device),
        "fused_op_registered": count_kernel_launches(
            lambda: router.fused_add_rmsnorm(h, r, w, RMS_NORM_EPS), 10, device),
    }
    return results


def layer2_single_layer(backend: str, device: torch.device, router, model,
                        input_ids) -> dict:
    model_core = model.model
    layer = model_core.layers[0]
    eps = float(model.config.rms_norm_eps)
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)
    cosine, sine = model_core.rotary_emb(position_ids, model_core.embed_tokens(input_ids).dtype)
    mask = model_core.make_attention_mask(
        None, seq, model_core.embed_tokens(input_ids).dtype, device)

    def native_layer():
        embed = model_core.embed_tokens(input_ids)
        normed = layer.input_layernorm(embed)
        attn_out = layer.self_attn(normed, cosine, sine, mask)
        hidden = embed + attn_out
        normed2 = layer.post_attention_layernorm(hidden)
        mlp_out = layer.mlp(normed2)
        return hidden + mlp_out

    def fused_layer():
        embed = model_core.embed_tokens(input_ids)
        residual = embed
        normed = layer.input_layernorm(residual)
        attn_out = layer.self_attn(normed, cosine, sine, mask)
        router.fused_add_rmsnorm(attn_out, residual,
                                 layer.post_attention_layernorm.weight, eps)
        mlp_out = layer.mlp(attn_out)
        return residual + mlp_out

    timer = OperatorTimer(device, warmup=5, repeats=50)
    with torch.no_grad():
        native_timing = timer.measure(native_layer)
        fused_timing = timer.measure(fused_layer)
    speedup = native_timing["p50_ms"] / fused_timing["p50_ms"]
    print(f"[L2/{backend}] layer0(seq={seq}) native={native_timing['p50_ms']:.4f}ms "
          f"fused={fused_timing['p50_ms']:.4f}ms 加速比={speedup:.3f}×")
    return {"seq": seq, "native_layer": native_timing, "fused_layer": fused_timing,
            "speedup_p50": round(speedup, 3),
            "note": "单层含 1 次融合 op 调用（post_attention 边界）；input_layernorm 为原生模块"}


def _greedy_timed(model, input_ids, max_new_tokens: int, eos_token_id, device):
    """逐步计时的 Greedy 解码（与第 01 期同口径：每步重算整段序列、torch.no_grad）。"""
    generated = input_ids
    step_ms = []
    finished = False
    with torch.no_grad():
        for step in range(max_new_tokens):
            backend_sync(device)
            start = time.perf_counter()
            logits = model(generated)
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token_id), dim=1)
            backend_sync(device)
            step_ms.append((time.perf_counter() - start) * 1000.0)
            if eos_token_id is not None and bool((next_token_id == eos_token_id).all()):
                finished = True
                break
    return generated, step_ms, finished


def layer3_full_inference(backend: str, device: torch.device, router, model,
                          input_ids, max_new_tokens: int, eos_token_id) -> dict:
    timer = OperatorTimer(device, warmup=3, repeats=20)
    results = {}
    tokens = input_ids.shape[1]

    if backend != "cpu":
        if backend == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.musa.reset_peak_memory_stats()

    for mode in ("native", "fused"):
        model.model.fused_ops = router if mode == "fused" else None
        with torch.no_grad():
            prefill = timer.measure(lambda: model(input_ids))
        generated, step_ms, finished = _greedy_timed(
            model, input_ids, max_new_tokens, eos_token_id, device)
        steps = sorted(step_ms)
        entry = {
            "prefill": prefill,
            "decode_steps_ms": [round(v, 4) for v in step_ms],
            "decode_total_ms": round(sum(step_ms), 4),
            "decode_avg_ms": round(sum(step_ms) / len(step_ms), 4),
            "decode_p50_ms": round(steps[len(steps) // 2], 4),
            "decode_steps_count": len(step_ms),
            "finished_by_eos": finished,
        }
        if backend != "cpu":
            if backend == "cuda":
                entry["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            else:
                entry["peak_memory_allocated_bytes"] = torch.musa.max_memory_allocated()
        results[mode] = entry
        print(f"[L3/{backend}] {mode}: prefill={prefill['p50_ms']:.3f}ms "
              f"decode_avg={entry['decode_avg_ms']:.3f}ms × {len(step_ms)} 步 "
              f"(eos={finished})")
    model.model.fused_ops = None

    native = results["native"]
    fused = results["fused"]
    results["end_to_end_gain"] = {
        "prefill_gain_ratio": round(native["prefill"]["p50_ms"] / fused["prefill"]["p50_ms"], 4),
        "decode_gain_ratio": round(native["decode_total_ms"] / fused["decode_total_ms"], 4),
        "decode_time_saved_ms": round(native["decode_total_ms"] - fused["decode_total_ms"], 4),
        "prefill_time_saved_ms": round(native["prefill"]["p50_ms"] - fused["prefill"]["p50_ms"], 4),
    }
    return results


def dilution_analysis(backend: str, layer1: dict, layer3: dict, seq: int,
                      micro_dtype: str) -> dict:
    """收益稀释归因：微基准收益 × 调用占比 → 端到端理论上限 vs 实测。"""
    fused_rows_seq = layer1[f"rows_{seq}"]["fused_op_registered"]["p50_ms"]
    composite_rows_seq = layer1[f"rows_{seq}"]["composite_chain"]["p50_ms"]
    per_forward_op_time_fused = fused_rows_seq * OP_CALLS_PER_FORWARD
    per_forward_op_time_composite = composite_rows_seq * OP_CALLS_PER_FORWARD
    native_prefill = layer3["native"]["prefill"]["p50_ms"]
    fused_prefill = layer3["fused"]["prefill"]["p50_ms"]
    theoretical_max_prefill_gain_ms = per_forward_op_time_composite - per_forward_op_time_fused
    measured_prefill_gain_ms = native_prefill - fused_prefill
    return {
        "seq_rows": seq,
        "op_calls_per_forward": OP_CALLS_PER_FORWARD,
        "microbench_dtype": micro_dtype,
        "op_time_share_of_fused_prefill": round(
            per_forward_op_time_fused / fused_prefill, 4),
        "op_time_share_of_native_prefill": round(
            per_forward_op_time_composite / native_prefill, 4),
        "theoretical_prefill_gain_ms_microbench_summed": round(
            theoretical_max_prefill_gain_ms, 4),
        "measured_prefill_gain_ms": round(measured_prefill_gain_ms, 4),
        "realization_vs_microbench_sum": round(
            measured_prefill_gain_ms / theoretical_max_prefill_gain_ms, 4)
            if theoretical_max_prefill_gain_ms > 0 else None,
        "note": ("上限 = 56 × (组合链 P50 - 融合 op P50)@rows=seq；"
                 "实测差含注意力/MLP 等其他算子的缓存与调度扰动，比值即稀释程度"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--model-dir", default=str(MODEL_DEFAULT))
    parser.add_argument("--truth-dir", default=str(TRUTH_DEFAULT))
    parser.add_argument("--skip-model", action="store_true",
                        help="只跑层 1（无权重环境）")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    setup = setup_backend(arguments.backend)
    router = setup["router"]
    op_module = setup["op_module"]

    model_dtype = torch.float32 if arguments.backend == "cpu" else torch.float16
    results = {
        "schema_version": "kernel-chapter05.benchmark.v1",
        "backend": arguments.backend,
        "model_dtype": str(model_dtype),
        "registration_evidence": setup["evidence"],
        "op_calls_per_forward": OP_CALLS_PER_FORWARD,
        "peak_bandwidth_source": "ep02 实测 D2D 峰值（分母单源）",
    }

    results["layer1_microbench"] = layer1_microbench(
        arguments.backend, device, router, op_module)

    seq = 20
    if not arguments.skip_model:
        model_directory = model_lib.resolve_model_directory(arguments.model_dir)
        model = model_lib.load_handwritten_model(
            model_directory, device, dtype=model_dtype).eval()
        truth = json.loads(
            (Path(arguments.truth_dir) / "truth-manifest.json").read_text(encoding="utf-8"))
        input_ids = torch.tensor([truth["prompt_token_ids"]], dtype=torch.long, device=device)

        results["layer2_single_layer"] = layer2_single_layer(
            arguments.backend, device, router, model, input_ids)
        results["layer3_full_inference"] = layer3_full_inference(
            arguments.backend, device, router, model, input_ids,
            arguments.max_new_tokens, truth.get("eos_token_id"))
        results["dilution_analysis_prefill"] = dilution_analysis(
            arguments.backend, results["layer1_microbench"],
            results["layer3_full_inference"], seq, str(model_dtype))
        decode_seq_max = seq + arguments.max_new_tokens - 1
        results["dilution_analysis_decode_note"] = (
            f"decode 步 seq 从 {seq} 增长到 {decode_seq_max}，"
            f"op 占比应按 rows∈[{seq},{decode_seq_max}] 的 L1 区间理解")
    else:
        results["model_section"] = "skipped（--skip-model）"

    results["environment"] = device_environment_record(device)
    output = HERE / "results" / f"benchmark-results-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
