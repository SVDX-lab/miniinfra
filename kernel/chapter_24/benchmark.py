#!/usr/bin/env python3
"""第 24 期统一计时口径基准（decode 形态：单 query，KV 长度扫描）。

测量对象与变体分解：
  full decode step（q_a → q_b → kv_a → [核] → [输出吸收/o 汇聚] → o_proj）：
    1. unabsorbed       ：torch eager 未吸收链（K/V 物化 bmm）
    2. absorbed_torch   ：torch eager 吸收链（同代数改写、无自定义核）
    3. absorbed_kernel  ：torch 投影 + 自定义 CUDA 核 + torch 输出吸收
       （1 vs 2 = 代数改写收益；2 vs 3 = 核融合收益）
  attention 核 only（N 相关部分）：
    4. unabsorbed core  ：K bmm + score bmm + softmax + V bmm + wsum bmm
    5. absorbed core    ：自定义核（split+merge 两 kernel）/ CPU 用吸收 bmm 链
  逐 stage 拆解（CUDA full step，逐算子计时，用于对照理论字节的逐项下限）

口径：
  - 计时器与 P50/P95 口径复用 chapter_01 experiment_utils（复制改造，schema v25）；
  - CUDA 用事件对逐次记录、末尾一次同步（小算子不被发射延迟淹没）；
  - 输入运行时生成（seed 规则同 fixtures：SEED_BASE+1000+N），He 缩放权重
    （生成后常驻显存，两种形式共用同一份）；
  - 理论字节/FLOPs/延迟下限/工作集分区（L2 72MB vs DRAM）由 roofline_mla 单源
    计算，随结果一起写入；
  - 有效带宽 = 理论字节数 / P50（对照 ch02 实测拷贝峰值 918.8 GB/s 与规格
    1008 GB/s 两个分母）；达成率 = 有效带宽 / 分母。
  - KV 长度扫描 N ∈ {1, 128, 2048, 32768}（大纲要求）。

用法：
    python benchmark.py --backend cpu
    python benchmark.py --backend cuda
    python benchmark.py --backend cuda --profile-mode core --kv-len 32768 --repeats 30
                        （nsys 被 profile 的最小进程模式）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    OperatorTimer,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from mla_reference import (  # noqa: E402
    apply_rope,
    mla_core_absorbed_bmm,
    mla_core_unabsorbed_bmm,
    mla_decode_absorbed_bmm,
    mla_decode_unabsorbed_bmm,
    q_projection_stage,
    rotate_q_rope,
)
from roofline_mla import (  # noqa: E402
    PLATFORM_PEAKS,
    absorbed_core_bytes,
    unabsorbed_core_bytes,
    working_set_note,
)
import mla_config as cfg  # noqa: E402

SEED_BASE = 20260325
BENCH_KV_LENS = [1, 128, 2048, 32768]
WEIGHT_NAMES = ("W_QA", "W_QB", "W_KVA", "W_UK", "W_UV", "W_O")


def load_cuda_module():
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    return build_ext.get_module()


def make_inputs(device: torch.device, dtype: torch.dtype, n: int,
                weights32: dict[str, torch.Tensor]):
    """运行时生成 decode 输入（seed 规则同 fixtures；权重用 FP32 版按 dtype 转换）。"""
    gen = torch.Generator().manual_seed(SEED_BASE + 1000 + n)
    x = torch.randn(1, cfg.D_MODEL, generator=gen).to(device=device, dtype=dtype)
    c_kv = torch.randn(n, cfg.KV_LORA_RANK, generator=gen).to(device=device, dtype=dtype)
    k_rope_raw = torch.randn(n, cfg.QK_ROPE_HEAD_DIM, generator=gen).to(device=device, dtype=dtype)
    freqs = cfg.ROPE_THETA ** (-torch.arange(0, cfg.QK_ROPE_HEAD_DIM, 2,
                                             dtype=torch.float32) / cfg.QK_ROPE_HEAD_DIM)
    freqs = freqs.to(device=device)
    k_rope = apply_rope(k_rope_raw, torch.arange(n, dtype=torch.float32, device=device), freqs)
    weights = {name: weights32[name].to(dtype) for name in WEIGHT_NAMES}
    pos_q = n
    return x, c_kv, k_rope, weights, freqs, pos_q


# ---------------------------------------------------------------------------
# full decode step 变体（每个都是单次可计时 callable）
# ---------------------------------------------------------------------------

def full_step_unabsorbed(x, c_kv, k_rope, weights, freqs, pos_q, scale):
    return mla_decode_unabsorbed_bmm(x, weights, c_kv, k_rope, pos_q, freqs, scale)["out"]


def full_step_absorbed_torch(x, c_kv, k_rope, weights, freqs, pos_q, scale):
    return mla_decode_absorbed_bmm(x, weights, c_kv, k_rope, pos_q, freqs, scale)["out"]


def make_full_step_absorbed_kernel(module):
    def step(x, c_kv, k_rope, weights, freqs, pos_q, scale):
        q_nope, q_rope_raw = q_projection_stage(x, weights)
        q_rope = rotate_q_rope(q_rope_raw, pos_q, freqs)
        q_abs = torch.matmul(q_nope.unsqueeze(1), weights["W_UK"].transpose(1, 2)).squeeze(1)
        s = module.mla_absorbed_attention(q_abs, q_rope, c_kv, k_rope, scale, 0)
        out_h = torch.matmul(s.unsqueeze(1), weights["W_UV"]).squeeze(1)
        return out_h.reshape(1, cfg.O_IN) @ weights["W_O"]
    return step


def core_unabsorbed_only(q_nope, q_rope, c_kv, k_rope, weights, scale):
    """core-only 计时体：q 侧已在外面算好（口径=§3.3 核级理论字节，只含 N 相关部分）。"""
    return mla_core_unabsorbed_bmm(q_nope, q_rope, c_kv, k_rope, weights, scale)["out_h"]


def make_core_absorbed_kernel(module):
    def core(q_abs, q_rope, c_kv, k_rope, scale):
        return module.mla_absorbed_attention(q_abs, q_rope, c_kv, k_rope, scale, 0)
    return core


def core_absorbed_torch_only(q_abs, q_rope, c_kv, k_rope, weights, scale):
    return mla_core_absorbed_bmm_qabs(q_abs, q_rope, c_kv, k_rope, weights, scale)["out_h"]


def mla_core_absorbed_bmm_qabs(q_abs, q_rope, c_kv, k_rope, weights, scale):
    """吸收核的 matmul 链（q_abs 已吸收好，core-only 口径）。"""
    scores_nope = torch.matmul(q_abs.unsqueeze(1), c_kv.t().unsqueeze(0))     # [H,1,N]
    scores_rope = q_rope @ k_rope.t()                                         # [H,N]
    scores = (scores_nope + scores_rope.unsqueeze(1)) * scale
    attn = torch.softmax(scores, dim=-1)                                      # [H,1,N]
    s = torch.matmul(attn, c_kv)                                              # [H,1,512]
    out_h = torch.matmul(s, weights["W_UV"]).squeeze(1)                       # [H,128]
    return {"out_h": out_h, "scores": scores.squeeze(1), "attn": attn.squeeze(1),
            "s": s.squeeze(1)}


# ---------------------------------------------------------------------------
# CUDA 逐 stage 拆解
# ---------------------------------------------------------------------------

def cuda_stage_breakdown(module, device, dtype, weights32, n, warmup, repeats) -> dict:
    """吸收形式（带自定义核）与未吸收形式逐 stage 计时（FP16）。"""
    x, c_kv, k_rope, weights, freqs, pos_q = make_inputs(device, dtype, n, weights32)
    q_nope, q_rope_raw = q_projection_stage(x, weights)
    q_rope = rotate_q_rope(q_rope_raw, pos_q, freqs)
    q_a = x @ weights["W_QA"]
    q_abs = torch.matmul(q_nope.unsqueeze(1), weights["W_UK"].transpose(1, 2)).squeeze(1)
    k_nope = torch.matmul(c_kv, weights["W_UK"])
    v = torch.matmul(c_kv, weights["W_UV"])
    scores = (torch.matmul(q_nope.unsqueeze(1), k_nope.transpose(1, 2))
              + (q_rope @ k_rope.t()).unsqueeze(1)) * cfg.SOFTMAX_SCALE
    attn = torch.softmax(scores, dim=-1)
    s = torch.matmul(attn, c_kv).squeeze(1)
    out_h_kernel = torch.matmul(s.unsqueeze(1), weights["W_UV"]).squeeze(1)

    stages_absorbed = {
        "q_a_proj": lambda: x @ weights["W_QA"],
        "q_b_proj": lambda: q_a @ weights["W_QB"],
        "kv_a_proj(新token)": lambda: x @ weights["W_KVA"],
        "吸收W_UK bmm": lambda: torch.matmul(q_nope.unsqueeze(1),
                                             weights["W_UK"].transpose(1, 2)).squeeze(1),
        "核(split+merge)": lambda: module.mla_absorbed_attention(
            q_abs, q_rope, c_kv, k_rope, cfg.SOFTMAX_SCALE, 0),
        "吸收W_UV bmm": lambda: torch.matmul(s.unsqueeze(1), weights["W_UV"]).squeeze(1),
        "o_proj": lambda: out_h_kernel.reshape(1, cfg.O_IN) @ weights["W_O"],
    }
    stages_unabsorbed = {
        "q_a_proj": lambda: x @ weights["W_QA"],
        "q_b_proj": lambda: q_a @ weights["W_QB"],
        "kv_a_proj(新token)": lambda: x @ weights["W_KVA"],
        "K投影 bmm": lambda: torch.matmul(c_kv, weights["W_UK"]),
        "score=Q·K^T bmm": lambda: torch.matmul(q_nope.unsqueeze(1), k_nope.transpose(1, 2)),
        "softmax": lambda: torch.softmax(scores, dim=-1),
        "V投影 bmm": lambda: torch.matmul(c_kv, weights["W_UV"]),
        "加权求和 attn·V bmm": lambda: torch.matmul(attn, v),
        "o_proj": lambda: (torch.matmul(attn, v).squeeze(1).reshape(1, cfg.O_IN)
                           @ weights["W_O"]),
    }
    timer = OperatorTimer(device, warmup, repeats)
    breakdown = {"absorbed": {}, "unabsorbed": {}}
    for name, op in stages_absorbed.items():
        breakdown["absorbed"][name] = timer.measure(op)
    wsum_out = torch.matmul(attn, v).squeeze(1).reshape(1, cfg.O_IN)
    for name, op in stages_unabsorbed.items():
        if name == "o_proj":
            breakdown["unabsorbed"][name] = timer.measure(lambda: wsum_out @ weights["W_O"])
        else:
            breakdown["unabsorbed"][name] = timer.measure(op)
    return breakdown


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def bench_backend(backend: str, kv_lens: list[int], warmup: int, repeats: int,
                  profile_mode: str | None, profile_kv_len: int | None) -> dict:
    device = resolve_device(backend)
    dtype = torch.float16 if backend == "cuda" else torch.float32
    # 权重固定用 CPU seed 生成一份 FP32（与 fixtures 同 seed 规则），再转 dtype
    gen_weights = {}
    for idx, name in enumerate(sorted(cfg.WEIGHT_SHAPES)):
        shape = cfg.WEIGHT_SHAPES[name]
        g = torch.Generator().manual_seed(SEED_BASE + idx + 1)
        gen_weights[name] = torch.randn(*shape, generator=g) / (shape[0] ** 0.5)
    weights32 = {name: gen_weights[name].to(device) for name in gen_weights}

    module = load_cuda_module() if backend == "cuda" else None
    results: dict = {
        "schema_version": "kernel-chapter25.benchmark.v1",
        "backend": backend,
        "dtype": str(dtype).replace("torch.", ""),
        "kv_lens": kv_lens,
        "timing": {"warmup": warmup, "repeats": repeats},
        "softmax_scale": cfg.SOFTMAX_SCALE,
        "platform_peaks": PLATFORM_PEAKS,
        "per_kv_len": [],
        "theory": [],
    }

    lens = [profile_kv_len] if profile_kv_len else kv_lens
    for n in lens:
        x, c_kv, k_rope, weights, freqs, pos_q = make_inputs(device, dtype, n, weights32)
        scale = cfg.SOFTMAX_SCALE
        timer = OperatorTimer(device, warmup, repeats)

        entry = {"kv_len": n}
        if profile_mode in (None, "full"):
            entry["unabsorbed"] = timer.measure(
                lambda: full_step_unabsorbed(x, c_kv, k_rope, weights, freqs, pos_q, scale))
            entry["absorbed_torch"] = timer.measure(
                lambda: full_step_absorbed_torch(x, c_kv, k_rope, weights, freqs, pos_q, scale))
            if backend == "cuda":
                step_kernel = make_full_step_absorbed_kernel(module)
                entry["absorbed_kernel"] = timer.measure(
                    lambda: step_kernel(x, c_kv, k_rope, weights, freqs, pos_q, scale))
        if profile_mode in (None, "core"):
            # core-only 口径：q 侧（投影+rope+吸收 bmm）在计时体之外完成，
            # 只计 N 相关部分（与 roofline_mla.core_*_bytes 同口径）
            q_nope, q_rope_raw = q_projection_stage(x, weights)
            q_rope = rotate_q_rope(q_rope_raw, pos_q, freqs)
            q_abs = torch.matmul(q_nope.unsqueeze(1),
                                 weights["W_UK"].transpose(1, 2)).squeeze(1)
            entry["core_unabsorbed"] = timer.measure(
                lambda: core_unabsorbed_only(q_nope, q_rope, c_kv, k_rope, weights, scale))
            if backend == "cuda":
                core_kernel = make_core_absorbed_kernel(module)
                entry["core_absorbed_kernel"] = timer.measure(
                    lambda: core_kernel(q_abs, q_rope, c_kv, k_rope, scale))
                # splits 敏感性小扫描（heuristic=0 / 1 / 16 / 64 封顶）
                sweep = {}
                for s_req in (1, 0, 16, 64):
                    sweep[f"splits={s_req}"] = timer.measure(
                        lambda s_req=s_req: module.mla_absorbed_attention(
                            q_abs, q_rope, c_kv, k_rope, scale, s_req))
                entry["kernel_splits_sweep"] = sweep
            else:
                entry["core_absorbed_torch"] = timer.measure(
                    lambda: core_absorbed_torch_only(q_abs, q_rope, c_kv, k_rope, weights, scale))
        results["per_kv_len"].append(entry)

        # 理论口径
        ub = unabsorbed_core_bytes(n)
        ab = absorbed_core_bytes(n)
        ws = working_set_note(n)
        theory = {
            "kv_len": n,
            "core_unabsorbed_bytes": ub,
            "core_absorbed_bytes": ab,
            "working_set": ws,
            "core_lower_bound_ms_cuda_peak": {
                "unabsorbed": ub["total_bytes"] / (PLATFORM_PEAKS["cuda"]["copy_gb_s_measured"] * 1e9) * 1000.0,
                "absorbed": ab["total_bytes"] / (PLATFORM_PEAKS["cuda"]["copy_gb_s_measured"] * 1e9) * 1000.0,
            },
        }
        # 实测有效带宽（核级）
        if "core_unabsorbed" in entry:
            p50 = entry["core_unabsorbed"]["p50_ms"]
            theory["achieved_gbs_core_unabsorbed"] = ub["total_bytes"] / (p50 / 1000.0) / 1e9
        if "core_absorbed_kernel" in entry:
            p50 = entry["core_absorbed_kernel"]["p50_ms"]
            theory["achieved_gbs_core_absorbed_kernel"] = ab["total_bytes"] / (p50 / 1000.0) / 1e9
        elif "core_absorbed_torch" in entry:
            p50 = entry["core_absorbed_torch"]["p50_ms"]
            theory["achieved_gbs_core_absorbed_torch"] = ab["total_bytes"] / (p50 / 1000.0) / 1e9
        results["theory"].append(theory)

        if backend == "cuda" and profile_mode in (None, "full"):
            results.setdefault("stage_breakdown", {})[f"n{n}"] = cuda_stage_breakdown(
                module, device, dtype, weights32, n, warmup, repeats)

        def fmt(value):
            return round(value, 4) if value is not None else None

        p50_unabs = entry.get("unabsorbed", {}).get("p50_ms")
        p50_abst = entry.get("absorbed_torch", {}).get("p50_ms")
        p50_absk = entry.get("absorbed_kernel", {}).get("p50_ms")
        p50_core_u = entry.get("core_unabsorbed", {}).get("p50_ms")
        p50_core_k = entry.get("core_absorbed_kernel", {}).get("p50_ms")
        p50_core_t = entry.get("core_absorbed_torch", {}).get("p50_ms")
        print(f"[bench/{backend}] N={n}: full unabs={fmt(p50_unabs)} "
              f"absorbed_torch={fmt(p50_abst)} absorbed_kernel={fmt(p50_absk)} | "
              f"core unabs={fmt(p50_core_u)} absorbed_torch={fmt(p50_core_t)} "
              f"kernel={fmt(p50_core_k)}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-mode", choices=["full", "core"], default=None,
                        help="nsys 被测最小进程模式：只跑指定链条")
    parser.add_argument("--kv-len", type=int, default=None, help="配合 --profile-mode")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    if arguments.backend == "cpu" and arguments.kv_len == 32768:
        arguments.repeats = min(arguments.repeats, 5)  # CPU FP32 大 N 限重复次数

    results = bench_backend(arguments.backend, BENCH_KV_LENS, arguments.warmup,
                            arguments.repeats, arguments.profile_mode, arguments.kv_len)
    results["environment"] = device_environment_record(resolve_device(arguments.backend))
    suffix = f"-{arguments.profile_mode}-n{arguments.kv_len}" if arguments.profile_mode else ""
    output = Path(arguments.out) if arguments.out else \
        HERE / "results" / f"benchmark-results-{arguments.backend}{suffix}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
