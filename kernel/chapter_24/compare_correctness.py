#!/usr/bin/env python3
"""第 24 期正确性对照（复用第 01 期比较口径：max/mean/相对误差/余弦/NaN/Inf/最大误差位置）。

对照链（按大纲 §9 新结构规则：吸收形式与未吸收形式在同一组合成权重下
数学等价，用两种形式的输出对照验证代数改写；参考实现以 CPU/PyTorch eager 为准）：

CPU 后端：
  A. fixtures 逐 N（{1,3,33,128,2048}）FP32：
     A1 未吸收 FP32 vs FP64 真值（实现差）
     A2 吸收 FP32 vs FP64 真值
     A3 吸收 FP32 vs 未吸收 FP32（§9 等价对，主判据）
     A5 einsum 教学形式 vs bmm/matmul 基准形式（证明两条链同义）
     A4 FP64 吸收 vs FP64 未吸收（位级等价判定，N=128）
  B. RoPE 吸收尝试实验（N=128 与 2048，FP32 + N=128 附加 FP64 对）：
     B1 完全吸收尝试（t_ref=pos_q / t_ref=缓存中部）的逐位置 score 误差
     B2 端到端损伤（attempt rope 分数代入完整 attention vs 真值输出）
     B3 合法的 q 侧折叠（R_{p_q} 折进 W_QB rope 列）：q_rope 与完整输出误差
     B4 latent→rope 线性映射最小二乘（held-out R²）
  C. 确定性：两形式各连续 3 次位级比对

CUDA 后端（4090）：
  D. fixtures 逐 N：FP32 eager 两形式 vs CPU FP64 真值；FP16 eager 两形式 vs CPU FP64 真值
  E. CUDA 吸收核 vs torch 吸收核（FP16 io，s 与最终 out 两级）
  F. 带核的完整吸收 decode 步（torch 投影 FP16 + 自定义核 + torch 输出吸收）vs CPU FP64 真值
  G. 长上下文 N=32768（运行时生成，seed 规则同 fixtures；GPU FP32 eager 参考）
  H. 确定性：核与 eager 链各连续 3 次位级比对

用法：
    python compare_correctness.py --backend cpu
    python compare_correctness.py --backend cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    comparison_metrics,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from mla_reference import (  # noqa: E402
    apply_rope,
    latent_to_rope_linear_map,
    mla_core_absorbed_bmm,
    mla_decode_absorbed,
    mla_decode_absorbed_bmm,
    mla_decode_unabsorbed,
    mla_decode_unabsorbed_bmm,
    merged_q_projection_weight,
    q_projection_stage,
    rope_absorb_attempt,
    rope_fold_q_side_weight,
    rope_freqs,
    rope_segment_scores_correct,
    rotate_q_rope,
)
import mla_config as cfg  # noqa: E402

FIXTURES_DEFAULT = HERE / "fixtures" / "correctness.npz"
ROPE_EXP_LENS = [128, 2048]
WEIGHT_NAMES = ("W_QA", "W_QB", "W_KVA", "W_UK", "W_UV", "W_O")


def load_fixtures(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    with path.with_name("manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return arrays, manifest


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.detach().cpu().contiguous().numpy().tobytes() == \
        b.detach().cpu().contiguous().numpy().tobytes()


def load_cuda_module():
    """构建入口单源：implementations/cuda/build_ext.py 的 get_module()。"""
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    return build_ext.get_module()


def fixture_weights(arrays: dict, device: torch.device, dtype: torch.dtype) -> dict:
    return {name: torch.from_numpy(arrays[f"weight_{name}"]).to(device=device, dtype=dtype)
            for name in WEIGHT_NAMES}


# ---------------------------------------------------------------------------
# A. fixtures 等价链
# ---------------------------------------------------------------------------

def equivalence_chain(device: torch.device, arrays: dict, kv_lens: list[int],
                      dtype: torch.dtype) -> dict:
    """A 组：逐 N 的误差链（dtype=FP32 为 CPU 锚点口径；FP16 为 CUDA 公共路径，
    真值始终是 CPU FP64 链，独立于本地 dtype）。"""
    freqs = rope_freqs().to(device=device)
    weights = fixture_weights(arrays, device, dtype)
    groups = []
    for n in kv_lens:
        prefix = f"n{n}_"
        x = torch.from_numpy(arrays["x"]).to(device=device, dtype=dtype)
        c_kv = torch.from_numpy(arrays[prefix + "c_kv"]).to(device=device, dtype=dtype)
        k_rope_rot = torch.from_numpy(arrays[prefix + "k_rope_rot"]).to(device=device, dtype=dtype)
        pos_q = n
        truth64 = torch.from_numpy(arrays[prefix + "out_truth_fp64"]).cpu()

        unabs = mla_decode_unabsorbed(x, weights, c_kv, k_rope_rot, pos_q, freqs)
        absorbed = mla_decode_absorbed(x, weights, c_kv, k_rope_rot, pos_q, freqs)

        group = {
            "kv_len": n,
            "dtype": str(dtype).replace("torch.", ""),
            "unabs_vs_truth_fp64": comparison_metrics(unabs["out"], truth64),
            "abs_vs_truth_fp64": comparison_metrics(absorbed["out"], truth64),
            "abs_vs_unabs": comparison_metrics(absorbed["out"], unabs["out"]),
            "scores_abs_vs_unabs": comparison_metrics(absorbed["scores"], unabs["scores"]),
            "attn_abs_vs_unabs": comparison_metrics(absorbed["attn"], unabs["attn"]),
        }
        if dtype == torch.float32:
            unabs_bmm = mla_decode_unabsorbed_bmm(x, weights, c_kv, k_rope_rot, pos_q, freqs)
            absorbed_bmm = mla_decode_absorbed_bmm(x, weights, c_kv, k_rope_rot, pos_q, freqs)
            group["einsum_vs_bmm_unabs"] = comparison_metrics(unabs["out"], unabs_bmm["out"])
            group["einsum_vs_bmm_abs"] = comparison_metrics(absorbed["out"], absorbed_bmm["out"])
        groups.append(group)
        print(f"[equivalence/{device.type}/{group['dtype']}] N={n}: "
              f"abs-vs-unabs max={group['abs_vs_unabs']['max_abs_error']:.3e} "
              f"cos={group['abs_vs_unabs']['cosine_similarity']:.8f} | "
              f"abs-vs-truth64 max={group['abs_vs_truth_fp64']['max_abs_error']:.3e}")

    # A4：FP64 位级等价（仅 CPU）
    fp64_pair = None
    if device.type == "cpu":
        n = 128
        prefix = f"n{n}_"
        x64 = torch.from_numpy(arrays["x"]).double()
        c64 = torch.from_numpy(arrays[prefix + "c_kv"]).double()
        k64 = torch.from_numpy(arrays[prefix + "k_rope_rot"]).double()
        w64 = {name: torch.from_numpy(arrays[f"weight_{name}"]).double() for name in WEIGHT_NAMES}
        u64 = mla_decode_unabsorbed(x64, w64, c64, k64, n, rope_freqs().double())
        a64 = mla_decode_absorbed(x64, w64, c64, k64, n, rope_freqs().double())
        fp64_pair = {
            "kv_len": n,
            "out_bitwise_equal": bitwise_equal(u64["out"], a64["out"]),
            "scores_bitwise_equal": bitwise_equal(u64["scores"], a64["scores"]),
            "out_max_abs_error_raw": float((a64["out"] - u64["out"]).abs().max()),
            "scores_max_abs_error_raw": float((a64["scores"] - u64["scores"]).abs().max()),
            "out_metrics": comparison_metrics(a64["out"], u64["out"]),
            "scores_metrics": comparison_metrics(a64["scores"], u64["scores"]),
            "note": "FP64 下两种形式是同一数学对象的两条不同浮点求和路径"
                    "（未吸收先物化 K/V 再点积；吸收先折 W 再点积 latent），"
                    "位级差异是预期现象；代数等价的数值判据是"
                    "误差降到 FP64 舍入量级（~1e-13 相对）而非位级相同",
        }
        print(f"[fp64-pair] N=128: out 位级一致={fp64_pair['out_bitwise_equal']} "
              f"scores 位级一致={fp64_pair['scores_bitwise_equal']} "
              f"out max={fp64_pair['out_metrics']['max_abs_error']:.3e} "
              f"scores max={fp64_pair['scores_metrics']['max_abs_error']:.3e}")
    return {"groups": groups, "fp64_pair": fp64_pair}


# ---------------------------------------------------------------------------
# B. RoPE 吸收尝试实验（FP32；N=128 附加 FP64 对）
# ---------------------------------------------------------------------------

def rope_absorption_experiment(device: torch.device, arrays: dict) -> dict:
    freqs = rope_freqs().to(device=device)
    weights = fixture_weights(arrays, device, torch.float32)
    results: dict = {"per_len": []}
    for n in ROPE_EXP_LENS:
        prefix = f"n{n}_"
        x = torch.from_numpy(arrays["x"]).to(device=device)
        c_kv = torch.from_numpy(arrays[prefix + "c_kv"]).to(device=device)
        k_rope_raw = torch.from_numpy(arrays[prefix + "k_rope_raw"]).to(device=device)
        k_rope_rot = torch.from_numpy(arrays[prefix + "k_rope_rot"]).to(device=device)
        positions = torch.arange(n, dtype=torch.float32, device=device)
        pos_q = n
        truth = mla_decode_unabsorbed(x, weights, c_kv, k_rope_rot, pos_q, freqs)
        q_nope, q_rope_raw = q_projection_stage(x, weights)

        # B1：分数级误差（t_ref 两个取点）
        scores_correct = rope_segment_scores_correct(q_rope_raw, k_rope_raw, pos_q, positions, freqs)
        per_ref: dict = {}
        for label, t_ref in (("pos_q_N", float(pos_q)), ("mid_N_half", float(n // 2))):
            scores_attempt = rope_absorb_attempt(q_rope_raw, k_rope_raw, pos_q, t_ref, freqs)
            err = (scores_correct - scores_attempt).abs()  # [H,N]
            entry = {
                "t_ref": t_ref,
                "max_error_all": float(err.max()),
                "mean_error_all": float(err.mean()),
                "max_error_last_col_nearest_query": float(err[:, -1].max()),
                "max_error_first_col": float(err[:, 0].max()),
            }
            if label == "mid_N_half":
                entry["max_error_at_t_ref_col"] = float(err[:, n // 2].max())
                entry["note"] = "t_ref 列误差应为机器精度量级（该列代数上精确），其余列随旋转角差增大"
            per_ref[label] = entry

        # B2：端到端损伤（t_ref=缓存中部——对 attempt 最"有利"的取点）
        attempt_mid = rope_absorb_attempt(q_rope_raw, k_rope_raw, pos_q, float(n // 2), freqs)
        k_nope = torch.einsum("nd,hdo->hno", c_kv, weights["W_UK"])
        scores_nope = torch.einsum("hd,hno->hn", q_nope, k_nope)
        scores = (scores_nope + attempt_mid) * cfg.SOFTMAX_SCALE
        attn = torch.softmax(scores, dim=-1)
        v = torch.einsum("nd,hdo->hno", c_kv, weights["W_UV"])
        out_h = torch.einsum("hn,hno->ho", attn, v)
        out_attempt = out_h.reshape(1, cfg.O_IN) @ weights["W_O"]
        end_to_end = comparison_metrics(out_attempt, truth["out"])

        # B3：合法 q 侧折叠（R_{p_q} 折进 W_QB rope 列）
        w_qb_folded = rope_fold_q_side_weight(weights, pos_q, freqs)
        q_a = x @ weights["W_QA"]
        q_rope_folded = (q_a @ w_qb_folded).view(cfg.N_HEADS, cfg.Q_HEAD_DIM)[:, cfg.QK_NOPE_HEAD_DIM:]
        q_rope_runtime_rot = rotate_q_rope(q_rope_raw, pos_q, freqs)
        q_side_fold = comparison_metrics(q_rope_folded, q_rope_runtime_rot)

        results["per_len"].append({
            "kv_len": n,
            "scores_correct_max_abs": float(scores_correct.abs().max()),
            "scores_error_by_t_ref": per_ref,
            "end_to_end_attempt_mid": end_to_end,
            "query_side_fold_q_rope": q_side_fold,
        })
        r = results["per_len"][-1]
        print(f"[rope-attempt/{device.type}] N={n}: "
              f"attempt(mid) max={per_ref['mid_N_half']['max_error_all']:.3e} "
              f"(t_ref列 {per_ref['mid_N_half']['max_error_at_t_ref_col']:.3e}) "
              f"端到端 max={end_to_end['max_abs_error']:.3e} cos={end_to_end['cosine_similarity']:.6f} | "
              f"q 侧折叠 max={q_side_fold['max_abs_error']:.3e}")

    # B1-FP64：N=128 的 FP64 对（证明 t_ref 列误差可到 1e-11，其余列是真实代数误差）
    n = 128
    prefix = f"n{n}_"
    c64 = torch.from_numpy(arrays[prefix + "c_kv"]).to(device=device).double()
    k64raw = torch.from_numpy(arrays[prefix + "k_rope_raw"]).to(device=device).double()
    w64 = {name: torch.from_numpy(arrays[f"weight_{name}"]).to(device=device).double()
           for name in ("W_QA", "W_QB")}
    q_rope64 = q_projection_stage(torch.from_numpy(arrays["x"]).to(device=device).double(), w64)[1]
    freqs64 = rope_freqs().to(device=device).double()
    pos64 = torch.arange(n, dtype=torch.float64, device=device)
    sc64 = rope_segment_scores_correct(q_rope64, k64raw, n, pos64, freqs64)
    sa64 = rope_absorb_attempt(q_rope64, k64raw, n, float(n // 2), freqs64)
    e64 = (sc64 - sa64).abs()
    results["fp64_scores_n128"] = {
        "max_error_at_t_ref_col": float(e64[:, n // 2].max()),
        "max_error_first_col": float(e64[:, 0].max()),
        "max_error_all": float(e64.max()),
        "note": "t_ref 列 ~1e-11（FP64 机器精度，代数上精确）；其余列保持 O(|score|×旋转角差)",
    }

    # B4：latent→rope 线性映射（N=2048 超定，train/test 分半）
    c2048 = torch.from_numpy(arrays["n2048_c_kv"]).to(device=device)
    k2048raw = torch.from_numpy(arrays["n2048_k_rope_raw"]).to(device=device)
    results["latent_to_rope_lstsq"] = latent_to_rope_linear_map(c2048, k2048raw)
    print(f"[rope-lstsq] held_out R²="
          f"{results['latent_to_rope_lstsq']['held_out_r_squared']:.4f}")
    return results


# ---------------------------------------------------------------------------
# C/H. 确定性
# ---------------------------------------------------------------------------

def determinism_chain(device: torch.device, arrays: dict) -> dict:
    n = 128
    prefix = f"n{n}_"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    freqs = rope_freqs().to(device=device)
    x = torch.from_numpy(arrays["x"]).to(device=device, dtype=dtype)
    c_kv = torch.from_numpy(arrays[prefix + "c_kv"]).to(device=device, dtype=dtype)
    k_rope = torch.from_numpy(arrays[prefix + "k_rope_rot"]).to(device=device, dtype=dtype)
    weights = fixture_weights(arrays, device, dtype)
    runs_unabs = [mla_decode_unabsorbed_bmm(x, weights, c_kv, k_rope, n, freqs)["out"]
                  for _ in range(3)]
    runs_abs = [mla_decode_absorbed_bmm(x, weights, c_kv, k_rope, n, freqs)["out"]
                for _ in range(3)]
    kernel_bitwise = None
    if device.type == "cuda":
        module = load_cuda_module()
        q_nope, q_rope_raw = q_projection_stage(x, weights)
        q_rope = rotate_q_rope(q_rope_raw, n, freqs)
        q_abs = torch.matmul(q_nope.unsqueeze(1), weights["W_UK"].transpose(1, 2)).squeeze(1)
        kernel_runs = [module.mla_absorbed_attention(q_abs, q_rope, c_kv, k_rope,
                                                     cfg.SOFTMAX_SCALE, 0)
                       for _ in range(3)]
        kernel_bitwise = bitwise_equal(kernel_runs[0], kernel_runs[1]) and \
            bitwise_equal(kernel_runs[0], kernel_runs[2])
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "kv_len": n,
        "unabsorbed_bitwise_identical": bitwise_equal(runs_unabs[0], runs_unabs[1]) and
            bitwise_equal(runs_unabs[0], runs_unabs[2]),
        "absorbed_bitwise_identical": bitwise_equal(runs_abs[0], runs_abs[1]) and
            bitwise_equal(runs_abs[0], runs_abs[2]),
        "cuda_kernel_bitwise_identical": kernel_bitwise,
    }


# ---------------------------------------------------------------------------
# E/F/G：CUDA 专属链
# ---------------------------------------------------------------------------

def cuda_kernel_chain(device: torch.device, arrays: dict, manifest: dict) -> dict:
    module = load_cuda_module()
    results: dict = {"groups": []}

    # E：核级对照（FP16）：torch 吸收核 vs 自定义核（s 与最终 out 两级）
    for n in (1, 3, 33, 128, 2048):
        prefix = f"n{n}_"
        x16 = torch.from_numpy(arrays["x"]).to(device=device, dtype=torch.float16)
        c16 = torch.from_numpy(arrays[prefix + "c_kv"]).to(device=device, dtype=torch.float16)
        k16 = torch.from_numpy(arrays[prefix + "k_rope_rot"]).to(device=device, dtype=torch.float16)
        w16 = fixture_weights(arrays, device, torch.float16)
        freqs16 = rope_freqs().to(device=device)
        q_nope, q_rope_raw = q_projection_stage(x16, w16)
        q_rope = rotate_q_rope(q_rope_raw, n, freqs16)
        core_torch = mla_core_absorbed_bmm(q_nope, q_rope, c16, k16, w16, cfg.SOFTMAX_SCALE)
        s_kernel = module.mla_absorbed_attention(core_torch["q_abs"], q_rope, c16, k16,
                                                 cfg.SOFTMAX_SCALE, 0)
        out_h_kernel = torch.matmul(s_kernel.unsqueeze(1), w16["W_UV"]).squeeze(1)
        out_kernel = out_h_kernel.reshape(1, cfg.O_IN) @ w16["W_O"]
        out_torch = core_torch["out_h"].reshape(1, cfg.O_IN) @ w16["W_O"]
        group = {
            "kv_len": n,
            "s_kernel_vs_torch": comparison_metrics(s_kernel, core_torch["s"]),
            "out_kernel_vs_torch": comparison_metrics(out_kernel, out_torch),
            "out_kernel_vs_truth_fp64": comparison_metrics(
                out_kernel.float().cpu(),
                torch.from_numpy(arrays[prefix + "out_truth_fp64"])),
        }
        results["groups"].append(group)
        print(f"[kernel/{device.type}] N={n}: s max={group['s_kernel_vs_torch']['max_abs_error']:.3e} "
              f"out max={group['out_kernel_vs_torch']['max_abs_error']:.3e} "
              f"vs-truth64 max={group['out_kernel_vs_truth_fp64']['max_abs_error']:.3e}")

    # F：带核的完整吸收 decode 步 vs CPU FP64 真值（N=128）
    n = 128
    prefix = f"n{n}_"
    x16 = torch.from_numpy(arrays["x"]).to(device=device, dtype=torch.float16)
    c16 = torch.from_numpy(arrays[prefix + "c_kv"]).to(device=device, dtype=torch.float16)
    k16 = torch.from_numpy(arrays[prefix + "k_rope_rot"]).to(device=device, dtype=torch.float16)
    w16 = fixture_weights(arrays, device, torch.float16)
    freqs16 = rope_freqs().to(device=device)
    q_nope, q_rope_raw = q_projection_stage(x16, w16)
    q_rope = rotate_q_rope(q_rope_raw, n, freqs16)
    q_abs = torch.matmul(q_nope.unsqueeze(1), w16["W_UK"].transpose(1, 2)).squeeze(1)
    s_kernel = module.mla_absorbed_attention(q_abs, q_rope, c16, k16, cfg.SOFTMAX_SCALE, 0)
    out_h = torch.matmul(s_kernel.unsqueeze(1), w16["W_UV"]).squeeze(1)
    out_full = out_h.reshape(1, cfg.O_IN) @ w16["W_O"]
    results["full_step_with_kernel_n128"] = comparison_metrics(
        out_full.float().cpu(), torch.from_numpy(arrays[prefix + "out_truth_fp64"]))
    print(f"[full-step-kernel] N=128 vs FP64 真值: "
          f"max={results['full_step_with_kernel_n128']['max_abs_error']:.3e} "
          f"cos={results['full_step_with_kernel_n128']['cosine_similarity']:.8f}")

    # 合并投影变体（吸收矩阵折进 Q 投影）核级一致性（N=128，FP16）
    w_abs = merged_q_projection_weight(
        {name: torch.from_numpy(arrays[f"weight_{name}"]).float() for name in ("W_QA", "W_QB", "W_UK")}
    ).to(device=device, dtype=torch.float16)
    q_a16 = x16 @ w16["W_QA"]
    q_abs_merged16 = (q_a16 @ w_abs).view(cfg.N_HEADS, cfg.KV_LORA_RANK)
    results["merged_projection_vs_two_step_fp16_n128"] = comparison_metrics(q_abs_merged16, q_abs)
    print(f"[merged-projection] fp16 N=128: "
          f"max={results['merged_projection_vs_two_step_fp16_n128']['max_abs_error']:.3e}")

    # G：长上下文 N=32768（运行时生成，seed 规则同 fixtures；GPU FP32 eager 参考）
    n = 32768
    gen = torch.Generator(device="cuda").manual_seed(manifest["seed_base"] + 1000 + n)
    w32 = fixture_weights(arrays, device, torch.float32)
    x32 = torch.from_numpy(arrays["x"]).to(device=device)
    c_long = torch.randn(n, cfg.KV_LORA_RANK, generator=gen, device=device, dtype=torch.float32)
    k_long_raw = torch.randn(n, cfg.QK_ROPE_HEAD_DIM, generator=gen, device=device, dtype=torch.float32)
    freqs32 = rope_freqs().to(device=device)
    k_long = apply_rope(k_long_raw, torch.arange(n, dtype=torch.float32, device=device), freqs32)
    pos_q = n
    unabs32 = mla_decode_unabsorbed_bmm(x32, w32, c_long, k_long, pos_q, freqs32)
    abs32 = mla_decode_absorbed_bmm(x32, w32, c_long, k_long, pos_q, freqs32)
    results["long_n32768_fp32_pair"] = comparison_metrics(abs32["out"], unabs32["out"])

    q_nope16, q_rope_raw16 = q_projection_stage(x16, w16)
    q_rope16 = rotate_q_rope(q_rope_raw16, pos_q, freqs16)
    core16 = mla_core_absorbed_bmm(q_nope16, q_rope16, c_long.half(), k_long.half(),
                                   w16, cfg.SOFTMAX_SCALE)
    s_kernel_long = module.mla_absorbed_attention(core16["q_abs"], q_rope16,
                                                  c_long.half(), k_long.half(),
                                                  cfg.SOFTMAX_SCALE, 0)
    results["long_n32768_kernel_vs_torch_fp16"] = comparison_metrics(s_kernel_long, core16["s"])
    print(f"[long-N] N=32768: FP32 pair max={results['long_n32768_fp32_pair']['max_abs_error']:.3e} "
          f"cos={results['long_n32768_fp32_pair']['cosine_similarity']:.9f} | "
          f"kernel-vs-torch fp16 s max={results['long_n32768_kernel_vs_torch_fp16']['max_abs_error']:.3e}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--fixtures", default=str(FIXTURES_DEFAULT))
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    arrays, manifest = load_fixtures(Path(arguments.fixtures))

    results: dict = {
        "schema_version": "kernel-chapter25.correctness.v1",
        "backend": arguments.backend,
        "config": manifest["config"],
        "config_source": manifest["config_source"],
        "fixture_kv_lens": manifest["kv_lens"],
    }
    if arguments.backend == "cpu":
        results["equivalence_fp32"] = equivalence_chain(device, arrays, manifest["kv_lens"],
                                                        torch.float32)
        results["rope_absorption"] = rope_absorption_experiment(device, arrays)
    else:
        # CUDA：先 FP32 eager 对照（隔离 dtype 误差），再 FP16 公共路径
        results["equivalence_fp32"] = equivalence_chain(device, arrays, manifest["kv_lens"],
                                                        torch.float32)
        results["equivalence_fp16"] = equivalence_chain(device, arrays, manifest["kv_lens"],
                                                        torch.float16)
        results["cuda_kernel"] = cuda_kernel_chain(device, arrays, manifest)
    results["determinism"] = determinism_chain(device, arrays)
    results["environment"] = device_environment_record(device)

    output = HERE / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
