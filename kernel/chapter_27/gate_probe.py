#!/usr/bin/env python3
"""第 27 期实验 4：门控（gating）对状态更新的影响注记。

三个子实验（全部递推形式，FP32；合成权重同 fixtures 口径）：
  (a) 状态范数曲线：恒定衰减 g ∈ {−0.05,−0.2,−0.5}（α={0.951,0.819,0.607}）
      vs FLA 式数据依赖门控（A_log/dt_bias 初始化 + softplus 门）在 T=4096 的
      每 token 每头 ‖S_t‖_F。理论上限：‖S_∞‖ ≈ β̄‖v‖/sqrt(1−ᾱ²)-ish，
      有效记忆窗口 ≈ 1/(1−ᾱ) token；
  (b) 检索深度探针（ep26 planted relevance 注入法的线性注意力版）：
      t=0 注入 (k_p, v_p)，k_p 为单位基向量 e_r（非退化检索结构）；之后
      t ∈ (0, D] 持续写入随机 (k_t,v_t) 干扰；在 t=D 用 q=e_r 读出，
      读出保真 = o_D·v_p / ‖v_p‖²。扫距离 D ∈ {64,256,1024,2048,4095}；
      结论形态：恒定衰减的保真 ≈ e^{g·D}（解析可预测），数据依赖门控
      按"当前 token 是否该记住"选择性保持——测二者曲线分离；
  (c) per-head 门控率分布：数据依赖门控下每头平均 α 的分布（heads 特化：
      有的头长记忆、有的头快速遗忘——GDN 论文的核心主张），与恒定衰减
      的单率对照。

产出 results/gate-probe.json。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import deltanet_config as cfg  # noqa: E402
import deltanet_reference as ref  # noqa: E402
from experiment_utils import write_results_json  # noqa: E402


def run_recurrent_norms(q, k, v, beta, g_log):
    _, s, norms = ref.recurrent_gdn(q, k, v, beta, g_log, return_state_norms=True)
    return s, norms


def retrieval_probe(g_log_mode: str, distance: int, seed: int = 0) -> dict:
    """注入-干扰-态差读出（状态空间保持度量）。

    同一 key 流跑两遍：注入版（v_0 = v_p）与非注入版（v_0 = 0 ⇒ t=0 无写入）。
    ΔS_D = S_inj(D) − S_no(D) 隔离出注入痕迹（含 delta 规则的后续非线性差）。
    retention = ⟨ΔS_D, k_p⊗v_p⟩ / ‖k_p⊗v_p‖²：注入的秩-1 痕迹在 D 步后
    剩余的系数——恒定衰减解析锚 ≈ α^D（干扰写对 e_r 的重叠 ~1/√K，擦除慢）。
    输出空间：读出差 Δo_D = o_inj(D) − o_no(D)，report cos(Δo_D, v_p)。
    """
    h, kd, vd = 1, cfg.HEAD_K_DIM, cfg.HEAD_V_DIM
    gen = torch.Generator().manual_seed(cfg.SEED_BASE + 20000 + seed)
    total = distance + 1
    q = torch.randn(total, h, kd, generator=gen)
    k = torch.randn(total, h, kd, generator=gen)
    v = torch.randn(total, h, vd, generator=gen)
    beta = torch.rand(total, h, generator=gen) * 0.5 + 0.3

    r = int(torch.randint(0, kd, (1,), generator=gen))
    k[0, 0] = 0
    k[0, 0, r] = 1.0                      # k_p = e_r（注入键，单位基向量）
    v_p = torch.randn(vd, generator=gen)
    v_p = v_p / v_p.norm()
    v_inj = v.clone()
    v_inj[0, 0] = v_p
    v_no = v.clone()
    v_no[0, 0] = 0.0                      # 非注入：t=0 写入为零

    if g_log_mode.startswith("const:"):
        g_value = float(g_log_mode.split(":")[1])
        g_log = torch.full((total, h), g_value)
    else:
        g_log = -torch.rand(total, h, generator=gen) * 0.15
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    o_i, s_i = ref.recurrent_gdn(q, k, v_inj, beta, g_log)
    o_n, s_n = ref.recurrent_gdn(q, k, v_no, beta, g_log)
    delta_s = s_i - s_n
    trace = torch.einsum("k,v->kv", k[0, 0], v_p)         # k_p⊗v_p
    retention = float((delta_s * trace).sum() / (trace * trace).sum())
    delta_o = o_i[distance, 0] - o_n[distance, 0]
    cos_out = float(torch.nn.functional.cosine_similarity(delta_o, v_p, dim=0))
    return {"retention": retention, "output_diff_cos_v_p": cos_out,
            "delta_s_norm": float(delta_s.norm()),
            "analytic_alpha_pow_d": float(torch.exp(
                torch.tensor(float(g_log_mode.split(":")[1]) * distance)))
            if g_log_mode.startswith("const:") else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", help="递推形式 CPU 列")
    args = parser.parse_args()

    weights = ref.make_weights(cfg.SEED_BASE)
    t_probe = cfg.GATE_PROBE_SEQ
    x = ref.make_inputs(t_probe)
    proj_scalar = ref.project_inputs(x, weights, "scalar")
    beta = proj_scalar["beta"]

    # ---- (a) 状态范数曲线 ----
    norm_curves = {}
    for g_value in cfg.CONST_DECAY_LOG_G:
        g_const = torch.full_like(proj_scalar["g_log"], g_value)
        _, norms = run_recurrent_norms(proj_scalar["q"], proj_scalar["k"],
                                       proj_scalar["v"], beta, g_const)
        picks = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095]
        norm_curves[f"const_g={g_value}"] = {
            "alpha": float(torch.exp(torch.tensor(g_value))),
            "horizon_1_over_1_minus_alpha": 1.0 / (1.0 - float(torch.exp(torch.tensor(g_value)))),
            "norms_at_checkpoints": {str(p): float(norms[p].mean()) for p in picks},
            "norm_mean_last_512": float(norms[-512:].mean()),
        }
        print(f"[a] const g={g_value} (α={torch.exp(torch.tensor(g_value)):.3f}): "
              f"‖S‖@t=63={norms[63].mean():.3f} @t=511={norms[511].mean():.3f} "
              f"@t=4095={norms[4095].mean():.3f}")
    _, norms_learned = run_recurrent_norms(proj_scalar["q"], proj_scalar["k"],
                                           proj_scalar["v"], beta, proj_scalar["g_log"])
    norm_curves["learned_fla_style"] = {
        "norms_at_checkpoints": {str(p): float(norms_learned[p].mean()) for p in
                                 (0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095)},
        "norm_mean_last_512": float(norms_learned[-512:].mean()),
        "per_token_alpha_mean": float(proj_scalar["g_log"].exp().mean()),
        "per_token_alpha_p05": float(proj_scalar["g_log"].exp().quantile(0.05)),
        "per_token_alpha_p95": float(proj_scalar["g_log"].exp().quantile(0.95)),
    }
    print(f"[a] learned: ‖S‖@t=63={norms_learned[63].mean():.3f} @t=511={norms_learned[511].mean():.3f} "
          f"@t=4095={norms_learned[4095].mean():.3f} ᾱ={proj_scalar['g_log'].exp().mean():.4f}")

    # ---- (b) 检索深度探针（状态空间保持度量）----
    distances = cfg.RETRIEVE_DISTANCES
    probe = {}
    for mode in [f"const:{g}" for g in cfg.CONST_DECAY_LOG_G] + ["const:0.0", "learned"]:
        probe[mode] = {}
        for d in distances:
            probe[mode][str(d)] = retrieval_probe(mode, d)
        print(f"[b] {mode}: " + " ".join(
            f"D={d}:{probe[mode][str(d)]['retention']:.4f}" for d in distances))

    # 解析对照：恒定衰减的理论保真 ≈ Π_{t=1..D} e^g·(1−β_t·retrieval...)——
    # 简化解析下界 e^{g·D}·Πβ；记录 e^{g·D} 供对照
    analytic = {f"const:{g}": {str(d): float(torch.exp(torch.tensor(g * d)))
                               for d in distances} for g in cfg.CONST_DECAY_LOG_G}

    # ---- (c) per-head 门控率分布（learned, T=4096）----
    alpha_per_head = proj_scalar["g_log"].exp().mean(dim=0).squeeze()  # [H]
    sorted_alpha, _ = alpha_per_head.sort()
    head_dist = {
        "alpha_per_head_mean": [float(a) for a in alpha_per_head],
        "alpha_sorted_min": float(sorted_alpha[0]),
        "alpha_sorted_p25": float(sorted_alpha[cfg.NUM_VALUE_HEADS // 4]),
        "alpha_median": float(sorted_alpha[cfg.NUM_VALUE_HEADS // 2]),
        "alpha_sorted_p75": float(sorted_alpha[3 * cfg.NUM_VALUE_HEADS // 4]),
        "alpha_sorted_max": float(sorted_alpha[-1]),
        "implied_horizon_per_head_1_over_1_minus_alpha": [
            1.0 / (1.0 - float(a)) for a in alpha_per_head],
        "const_baseline_horizon": {str(g): 1.0 / (1.0 - float(torch.exp(torch.tensor(g))))
                                   for g in cfg.CONST_DECAY_LOG_G},
    }
    print(f"[c] learned per-head ᾱ: min={head_dist['alpha_sorted_min']:.4f} "
          f"median={head_dist['alpha_median']:.4f} max={head_dist['alpha_sorted_max']:.4f} "
          f"（恒定基线 horizon {head_dist['const_baseline_horizon']}）")

    result = {
        "schema_version": "kernel-chapter28.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "seq_len": t_probe,
        "state_norm_curves": norm_curves,
        "retrieval_depth_probe": {
            "method": "t=0 注入 (k=e_r, v_p) vs v_0=0 非注入对照；ΔS_D 隔离注入痕迹；"
                      "retention = ⟨ΔS_D, k_p⊗v_p⟩/‖k_p⊗v_p‖²（探针专用 H=1 口径）",
            "by_mode_and_distance": probe,
            "analytic_exp_gd_for_const": analytic,
        },
        "per_head_gate_distribution": head_dist,
        "notes": {
            "delta_rule_stability": "β‖k‖²≤β<1（qk l2norm 后）是收缩条件；无 l2norm 的"
                                    "合成 k 使精确算术下也发散（见 EXPERIMENT_LOG §5）",
            "norm_saturation": "delta 写入的误差修正性质使 ‖S‖ 饱和（非单调增长）——"
                               "与线性注意力无衰减版（DeltaNet）和纯衰减 RetNet 都不同",
        },
    }
    write_results_json(HERE / "results" / "gate-probe.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
