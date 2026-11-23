#!/usr/bin/env python3
"""第 28 期实验 4：MoE 激活参数与计算量估算（附录 B 题 11 校核，纸面+代码双过）。

流程（先校核后外推）：
  1. 用 4 个公开 config（Qwen3-30B-A3B / DeepSeek-V2-Lite / Mixtral-8x7B /
     DeepSeek-V3）逐项核算总参数与每 token 激活参数，与官方公开总参/激活数
     （30.5/3.3、15.7/2.4、46.7/12.9、671/37 B）对表——验证估算流程本身；
  2. 对大纲给定的 V4-Pro（1.6T 总参/49B 激活，公开报道口径、无官方 config）
     做反推：扫描 (d_model, d_ff, E, k, L, shared) 组合，找激活比≈49/1600=3.06%
     的可行解空间，并以 DeepSeek-V3/K2 类结构为参照给出最可能组合（显式假设）；
  3. 每 token 专家 GEMM FLOPs 推导：2 × k × 3 × d_model × d_ff（MAC×2），
     与 dense 等价 FFN（intermediate = k×d_ff）对照。
产出 results/param-estimate.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import moe_config as cfg  # noqa: E402


def mha_params_per_layer(d: int, q_heads: int, kv_heads: int, head_dim: int) -> int:
    """MHA/GQA 注意力参数：Q 投影 + KV 投影 + O 投影（无 bias，Qwen3 口径）。"""
    q = d * (q_heads * head_dim)
    kv = d * (2 * kv_heads * head_dim)
    o = (q_heads * head_dim) * d
    return q + kv + o


def mla_params_per_layer(c: cfg.PublicMoEConfig) -> int:
    """MLA 注意力参数（DeepSeek 口径）：q_a/q_b + kv_a(MZA 合并) + kv_b + o_proj。"""
    d = c.d_model
    qk_nope = c.qk_nope_head_dim
    rope = c.qk_rope_head_dim
    q_heads = c.q_heads
    if c.q_lora_rank:
        q = d * c.q_lora_rank + c.q_lora_rank * q_heads * (qk_nope + rope)
    else:  # V2-Lite: q_lora_rank=None → 直接 q 投影（HF 实现为 d→heads*(nope+rope)）
        q = d * q_heads * (qk_nope + rope)
    kv_a = d * (c.kv_lora_rank + rope)
    kv_b = c.kv_lora_rank * q_heads * (qk_nope + c.v_head_dim)
    o = q_heads * c.v_head_dim * d
    return q + kv_a + kv_b + o


def count_model(c: cfg.PublicMoEConfig) -> dict:
    """按公开 config 核算总参数与每 token 激活参数（不含 embed 前 tie 处理另计）。"""
    d, d_ff = c.d_model, c.d_ff_moe
    ep = cfg.expert_params(d, d_ff)
    attn = mla_params_per_layer(c) if c.attn_kind == "mla" else mha_params_per_layer(
        d, c.q_heads, c.kv_heads, c.head_dim)
    L_total = c.n_layers
    L_moe = c.moe_layers
    L_dense = L_total - L_moe

    expert_total = L_moe * c.n_experts * ep
    expert_activated = L_moe * c.topk * ep
    shared_total = L_moe * c.n_shared * ep
    shared_activated = shared_total  # 共享专家每 token 全激活
    attn_total = L_total * attn
    dense_ffn_total = L_dense * (3 * d * c.dense_ff) if L_dense else 0
    embed = c.vocab * d * (1 if c.tie_word_embeddings else 2)  # embed + lm_head
    norms = L_total * 2 * d + d * 2 + 1_000  # 每层 2 norm + 终 norm + 多头 q/k norm≈小量

    total = expert_total + shared_total + attn_total + dense_ffn_total + embed + norms
    activated = expert_activated + shared_activated + attn_total + (
        L_dense * (3 * d * c.dense_ff)) + embed + norms
    flops_per_token_expert = 2 * c.topk * ep * L_moe  # MAC→FLOP ×2
    flops_per_token_total = 2 * (activated)  # 2 FLOP/参数/token 惯例（附录 B 题 5）
    return {
        "name": c.name,
        "expert_params_each": ep,
        "n_experts": c.n_experts, "topk": c.topk, "n_shared": c.n_shared,
        "moe_layers": L_moe, "dense_layers": L_dense,
        "attn_per_layer": attn, "attn_kind": c.attn_kind,
        "expert_total_B": round(expert_total / 1e9, 2),
        "expert_activated_B": round(expert_activated / 1e9, 2),
        "shared_B": round(shared_total / 1e9, 2),
        "attn_total_B": round(attn_total / 1e9, 2),
        "dense_ffn_B": round(dense_ffn_total / 1e9, 2),
        "embed_B": round(embed / 1e9, 2),
        "total_B_est": round(total / 1e9, 2),
        "activated_B_est": round(activated / 1e9, 2),
        "activated_ratio_est_pct": round(activated / total * 100, 2),
        "published_total_B": c.published_total_b,
        "published_active_B": c.published_active_b,
        "total_err_pct": round((total / 1e9 - c.published_total_b) / c.published_total_b * 100, 1)
        if c.published_total_b else None,
        "active_err_pct": round((activated / 1e9 - c.published_active_b) / c.published_active_b * 100, 1)
        if c.published_active_b else None,
        "expert_flops_per_token_GFLOP": round(flops_per_token_expert / 1e9, 2),
        "flops_per_token_GFLOP_2xact": round(flops_per_token_total / 1e9, 2),
        "source": c.source,
    }


def v4pro_reverse_fit() -> dict:
    """V4-Pro（1.6T 总参/49B 激活）反推：激活比 3.06% 的可行 (d,d_ff,E,k,L) 空间。

    显式假设：V3 类结构（MLA 注意力、无 tie embed、激活口径含 attention+embed）。
    激活比公式（忽略 norm/embed 二阶项时）：
        ratio ≈ [k×ep×L_moe + shared×ep×L_moe + attn×L + embed] /
                [E×ep×L_moe + shared×ep×L_moe + attn×L + embed]
    """
    target_ratio = 49.0 / 1600.0
    candidates = []
    # 参照现有公开结构外推的候选（d, d_ff, E, k, shared, L_moe, L_dense, q_heads, vocab）
    grid = [
        (7168, 2048, 384, 8, 1, 60, 1, 64, 163840),   # Kimi-K2 原样（1T/32B 参照锚点）
        (7168, 2048, 616, 8, 1, 60, 1, 64, 163840),   # K2 专家扩容到 1.6T
        (7168, 2048, 384, 8, 1, 96, 2, 64, 163840),   # K2 加深到 1.6T
        (7168, 2048, 512, 8, 1, 71, 3, 64, 163840),   # 双命中候选
        (7168, 2048, 512, 10, 1, 71, 3, 64, 163840),  # 双命中候选（k=10）
        (7168, 2048, 512, 8, 2, 71, 3, 64, 163840),   # 双命中候选（shared=2）
        (7168, 2048, 256, 8, 1, 58, 3, 128, 129280),  # DeepSeek-V3 原样（对照）
        (8192, 2048, 384, 8, 1, 80, 4, 128, 129280),
        (8192, 2048, 448, 8, 1, 72, 4, 128, 129280),
        (8192, 2304, 384, 8, 1, 80, 4, 128, 129280),
        (6144, 1536, 512, 8, 1, 96, 6, 96, 151936),
    ]
    for (d, d_ff, n_e, k, n_sh, L_moe, L_dense, qh, vocab) in grid:
        ep = cfg.expert_params(d, d_ff)
        q_lora = 1536 if d == 7168 else int(round(d * 1536 / 7168 / 64) * 64)
        c = cfg.PublicMoEConfig(
            name=f"V4Pro-candidate d={d} E={n_e} L={L_moe}", d_model=d, d_ff_moe=d_ff,
            n_experts=n_e, topk=k, n_layers=L_moe + L_dense, n_shared=n_sh,
            n_moe_layers=L_moe,
            attn_kind="mla", q_heads=qh, kv_heads=qh, head_dim=128,
            q_lora_rank=q_lora, kv_lora_rank=512, qk_nope_head_dim=128,
            qk_rope_head_dim=64, v_head_dim=128,
            vocab=vocab, tie_word_embeddings=False)
        r = count_model(c)
        candidates.append({
            "d_model": d, "d_ff": d_ff, "n_experts": n_e, "topk": k,
            "shared": n_sh, "moe_layers": L_moe, "q_heads": qh,
            "total_B_est": r["total_B_est"],
            "total_vs_1600": round(r["total_B_est"] / 1600.0, 3),
            "activated_B_est": r["activated_B_est"],
            "activated_vs_49": round(r["activated_B_est"] / 49.0, 3),
            "ratio_pct": r["activated_ratio_est_pct"],
            "ratio_vs_target": round(r["activated_ratio_est_pct"] / (target_ratio * 100), 3),
        })
    best = min(candidates, key=lambda c: abs(c["total_vs_1600"] - 1.0) + abs(c["activated_vs_49"] - 1.0))
    return {
        "target": {"total_B": 1600.0, "activated_B": 49.0, "ratio_pct": round(target_ratio * 100, 3)},
        "formula": "激活 ≈ 共享部分 + top-k × 单专家参数（附录 B 题 11）；"
                   "ratio = [k×ep×L_moe + sh×ep×L_moe + attn×L + embed] / "
                   "[E×ep×L_moe + sh×ep×L_moe + attn×L + embed]",
        "assumptions": [
            "V4-Pro 无官方 config（大纲给定公开报道口径 1.6T/49B）——只校核估算流程的"
            "自洽性，不主张还原其真实结构",
            "反推空间限定在 DeepSeek-V3/Kimi-K2 类结构邻域（MLA 注意力、V3/K2 维度家族）",
            "K2（1T/32B, ratio≈3.2%）与 V4-Pro（1.6T/49B, ratio=3.06%）同类——先用 K2 "
            "原样候选验证流程，再外推 1.6T",
            "注意力参数按 q_lora=1536（K2/V3 同值）；embed 按 vocab 无 tie 双份",
        ],
        "candidates": candidates,
        "best_dual_match": best,
        "conclusion": "见 EXPERIMENT_LOG §4：总参 1.6T 与激活 49B 双约束的可行解邻域为 "
                      "E∈[512,616]×L_moe∈[60,71]（E×L_moe×ep≈1.6T 主导）配 k=8~10、shared=1~2；"
                      "流程已由 5 个公开 config 校核（Qwen3/V2-Lite/Mixtral/V3/K2）",
    }


def paper_derivation() -> dict:
    """纸面推导（公式串，进 EXPERIMENT_LOG 与教材用）。"""
    d, d_ff, k, E, L = 2048, 1408, 6, 64, 26
    ep = 3 * d * d_ff
    return {
        "per_expert_params": "3 × d_model × d_ff = 3×2048×1408 = 8,654,336 (8.65M)",
        "per_token_activated_routed": f"k × ep = 6 × 8.65M = {k * ep:,}",
        "per_token_activated_with_shared": f"(k + shared) × ep = (6+2) × 8.65M = {(k + 2) * ep:,}",
        "layer_pool": f"E × ep = 64 × 8.65M = {E * ep:,} (553.9M/层)",
        "model_routed_total_V2Lite": f"L_moe × E × ep = 26 × 553.9M = {L * E * ep / 1e9:.2f}B",
        "activated_fraction_of_routed_pool": f"k/E = 6/64 = {k / E:.4f}（每层；整模型同）",
        "expert_gemm_flops_per_token": f"2 × k × 3 × d × d_ff × L_moe = "
                                       f"{2 * k * ep * L / 1e9:.2f} GFLOP/token（V2-Lite 口径）",
        "dense_equivalent_ffn": "intermediate = k×d_ff 的单体 FFN 与专家段 FLOPs 相等"
                                "（实验 2 的 dense_equiv 分母）",
        "flops_ratio_claim": "MoE 相对 dense（同总参容量 E×d_ff）省算力 ≈ k/E 倍"
                             "（每 token 只算 k 个专家）——实测见实验 2/5",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    table = [count_model(c) for c in cfg.PUBLIC_CONFIGS]
    for r in table:
        print(f"[param] {r['name']:24s} total_est={r['total_B_est']:8.2f}B "
              f"(pub {r['published_total_B']}B, err {r['total_err_pct']:+.1f}%) | "
              f"activated_est={r['activated_B_est']:7.2f}B "
              f"(pub {r['published_active_B']}B, err {r['active_err_pct']:+.1f}%) | "
              f"ratio={r['activated_ratio_est_pct']:.2f}% | "
              f"expert FLOPs/token={r['expert_flops_per_token_GFLOP']:.2f} GFLOP")
    fit = v4pro_reverse_fit()
    print(f"[param] V4-Pro 反推 target ratio={fit['target']['ratio_pct']:.3f}% "
          f"best_dual_match={fit['best_dual_match']}")
    paper = paper_derivation()
    payload = {
        "schema_version": "kernel-chapter29.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "appendix_b_question": "题 11（MoE 激活估算）：激活 ≈ 共享部分 + top-k × 单专家参数；"
                               "以 V4-Pro 1.6T/49B 公开报道口径校核估算流程",
        "public_config_checks": table,
        "v4pro_reverse_fit": fit,
        "paper_derivation": paper,
    }
    out = HERE / "results" / "param-estimate.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[param] 写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
