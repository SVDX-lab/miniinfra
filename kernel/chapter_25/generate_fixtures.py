#!/usr/bin/env python3
"""生成第 25 期 DSA fixtures（固定 seed 的合成数据、给定索引集与真值输出）。

内容（全部进 correctness.npz，跨机器共用同一份）：
- 每 correctness N ∈ {1,3,33,127,128,2048}：
    cache：c_kv [N,512]（planted relevance）、k_rope_raw/rot [N,64]、
           k_i [N,128]（索引 K 缓存 = W_KL 投影 + relevance 注入）、positions；
    relevance：ids [S]、tiers [S]（0=strong 1=medium 2=weak）；
    索引器真值：index_scores_fp32 [N]（主链）、index_scores_fp64 [N]（FP64 锚点）；
    dense 真值：dense_out_fp32 [1,2048]（einsum 吸收核）、dense_out_fp64（FP64 锚点）、
                dense_attn_head0 [N]（探针）；
    给定索引集真值（§9 主对象）：三组"已知"构造集（等距 33 点 + 边界 0/N-1、
    全集恒等、单点 {0}）与三组随机集（seed 固定）——每组存
    set_sparse_einsum_fp32_out / set_sparse_fp64_out / set_subset_dense_fp32_out
    （同一索引集下：稀疏 gather 核 vs 子集 dense 核 vs FP64 锚点）；
    端到端：dsa_step_k{K}_out（N≥512 用 k=512，否则 k=N）、dsa_step_indices、
    dsa_step_vs_dense 质量统计（captured/oracle/recall）。
- 权重与 x 不进 npz（纯 RNG 可重建，纯 torch.Generator 固定 seed）；
  服务器上 verify-only 重算整条流水并与 npz 真值做数值复核（跨 torch 版本
  GEMM 舍入不同，用容差比较而非 SHA——manifest 记录口径）。

seed 规则（与 benchmark 运行时生成同规则，见 dsa_reference 模块 docstring）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys_path = str(HERE / "implementations" / "cpu")
import sys  # noqa: E402

sys.path.insert(0, sys_path)
sys.path.insert(0, str(HERE))

import dsa_config as cfg  # noqa: E402
from dsa_reference import (  # noqa: E402
    SEED_BASE,
    absorbed_query,
    apply_rope,
    calibrate_plants,
    dense_attention_absorbed_bmm,
    dense_attention_absorbed_einsum,
    dense_decode_step,
    dsa_decode_step,
    generate_cache,
    index_quality,
    index_query_stage,
    index_scores,
    index_weights_stage,
    make_weights,
    plant_directions,
    q_projection_stage,
    rope_freqs,
    sparse_gather_attention_bmm,
    sparse_gather_attention_einsum,
)

KV_LENS = list(cfg.KV_LENS)
END_TO_END_K = 512


def sha256_of_tensor(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def constructed_sets(n: int) -> dict[str, torch.Tensor]:
    """已知索引集（§9：构造已知索引集 + 随机索引集两组）：
    strided：等距 33 点（含首尾边界——覆盖非整块/首尾 gather 行为）；
    identity：全集 arange(N)（k=N 恒等：稀疏核 == dense 核）；
    single：{0}（k=1 最小非空集）。"""
    step = max(1, n // 33)
    strided = torch.arange(0, n, step)
    if strided[-1].item() != n - 1:
        strided = torch.cat([strided, torch.tensor([n - 1])])
    strided = torch.unique(strided)
    return {
        "strided": strided,
        "identity": torch.arange(n),
        "single": torch.tensor([0]),
    }


def random_sets(n: int, count: int = 3, size: int = 33) -> list[torch.Tensor]:
    sets = []
    for i in range(count):
        gen = torch.Generator().manual_seed(SEED_BASE + 4000 + n * 10 + i)
        sets.append(torch.randperm(n, generator=gen)[:size])
    return sets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fixtures")
    parser.add_argument("--verify-only", action="store_true",
                        help="重算整条流水并与 npz 真值数值复核（容差 2e-4），不写文件")
    arguments = parser.parse_args()
    out_dir = Path(arguments.out_dir)

    started = time.perf_counter()
    weights = make_weights()
    x = torch.randn(1, cfg.D_MODEL, generator=torch.Generator().manual_seed(SEED_BASE + 1))
    freqs = rope_freqs()
    dirs = plant_directions(x, weights)
    w64 = {k: v.double() for k, v in weights.items()}

    arrays: dict[str, torch.Tensor] = {}
    manifest: dict = {
        "schema_version": "kernel-chapter26.fixtures.v1",
        "seed_base": SEED_BASE,
        "seed_rule": "权重=SEED_BASE+sorted(WEIGHT_SHAPES)序号+1；x=SEED_BASE+1；"
                     "cache(N)=SEED_BASE+1000+N；relevance(N)=SEED_BASE+2000+N 的 "
                     "randperm 前 S(n)；校准采样=SEED_BASE+3000+N；随机索引集="
                     "SEED_BASE+4000+N*10+i",
        "rng": "torch.randn (torch.Generator CPU) × He 缩放 (fan_in=收缩维输入维度；"
               "2-D 用 shape[0]，3-D per-head 用 shape[1]——ep25 用 shape[0] 对 3-D "
               "偏热的差异已在 make_weights docstring 声明)",
        "torch_version": torch.__version__,
        "verify_protocol": "npz 是跨机器共享真值；--verify-only 在目标机重算全流水与 "
                           "npz 真值做数值复核（max abs < 2e-4，覆盖 GEMM 舍入差），"
                           "而非 SHA——planted 数据经 GEMM/校准，跨 torch 版本位级"
                           "不可复现，属预期",
        "portable_note": "权重/x/freqs 纯 RNG 可重建；cache 含 planted（校准涉及 "
                         "GEMM），以 npz 分发值为准",
        "config": {
            "d_model": cfg.D_MODEL, "n_heads": cfg.N_HEADS,
            "q_lora_rank": cfg.Q_LORA_RANK, "kv_lora_rank": cfg.KV_LORA_RANK,
            "qk_nope_head_dim": cfg.QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": cfg.QK_ROPE_HEAD_DIM, "v_head_dim": cfg.V_HEAD_DIM,
            "index_n_heads": cfg.INDEX_HEADS, "index_head_dim": cfg.INDEX_HEAD_DIM,
            "index_topk_default": cfg.INDEX_TOPK_DEFAULT,
            "softmax_scale": cfg.SOFTMAX_SCALE, "index_scale": cfg.INDEX_SCALE,
            "index_weight_scale": cfg.INDEX_WEIGHT_SCALE, "rope_theta": cfg.ROPE_THETA,
        },
        "config_source": cfg.CONFIG_SOURCE,
        "kv_lens": KV_LENS,
        "relevance_setting": {
            "support_rule": "S(n)=max(256, round(0.025*n))，三档 6.25%/18.75%/75%",
            "z_tiers_main": cfg.Z_TIERS, "z_tiers_index": cfg.Z_TIERS_INDEX,
            "rationale": "背景 mass≈2%、三档≈33/31/35%，误差—k 曲线在 k≈S(n) 处膝点"
                         "（EXPÉRIMENT_LOG §4 推导）".replace("É", "E"),
        },
        "truth_implementation": "dense/sparse 吸收核 einsum 形式（FP32）+ FP64 锚点；"
                                "给定索引集另存 bmm 子集 dense 对照",
        "weights_shapes": {k: list(v) for k, v in cfg.WEIGHT_SHAPES.items()},
        "quality_stats": {},
        "calib_stats": {},
        "tensors": {},
    }

    for n in KV_LENS:
        prefix = f"n{n}_"
        calib = calibrate_plants(n, weights, x, dirs)
        cache = generate_cache(n, weights, x, freqs, calib)
        c_kv = cache["c_kv"]
        k_rope = cache["k_rope_rot"]
        k_i = cache["k_i"]
        pos_q = n
        ids, tiers = cache["relevance_ids"], cache["relevance_tiers"]

        arrays[prefix + "c_kv"] = c_kv
        arrays[prefix + "k_rope_raw"] = cache["k_rope_raw"]
        arrays[prefix + "k_rope_rot"] = k_rope
        arrays[prefix + "k_i"] = k_i
        arrays[prefix + "positions"] = cache["positions"]
        arrays[prefix + "relevance_ids"] = ids
        arrays[prefix + "relevance_tiers"] = tiers

        # 索引器真值（FP32 主链 + FP64 锚点）
        q_i = index_query_stage(x @ weights["W_QA"], weights)
        w_used = index_weights_stage(x, weights)
        s_i32 = index_scores(q_i, w_used, k_i)
        s_i64 = index_scores(q_i.double(), w_used.double(), k_i.double())
        arrays[prefix + "index_scores_fp32"] = s_i32
        arrays[prefix + "index_scores_fp64"] = s_i64

        # dense 真值
        q_nope, q_rope_raw = q_projection_stage(x, weights)
        q_rope = apply_rope(q_rope_raw, torch.tensor([pos_q], dtype=torch.float32), freqs)
        q_abs = absorbed_query(q_nope, weights)
        dense32 = dense_attention_absorbed_einsum(q_abs, q_rope, c_kv, k_rope, weights)
        dense64 = dense_attention_absorbed_einsum(
            q_abs.double(), q_rope.double(), c_kv.double(), k_rope.double(), w64)
        arrays[prefix + "dense_out_fp32"] = dense32["out_h"].reshape(1, cfg.O_IN) @ weights["W_O"]
        arrays[prefix + "dense_out_fp64"] = (dense64["out_h"].reshape(1, cfg.O_IN) @ w64["W_O"])
        arrays[prefix + "dense_s_fp64"] = dense64["s"]
        arrays[prefix + "dense_attn_head0"] = dense32["attn"][0]

        # 给定索引集真值（§9 主对象）
        set_defs = {}
        for name, idx in constructed_sets(n).items():
            set_defs[name] = idx
        for i, ridx in enumerate(random_sets(n)):
            set_defs[f"random{i}"] = ridx
        for name, idx in set_defs.items():
            sp32 = sparse_gather_attention_einsum(q_abs, q_rope, c_kv, k_rope, idx, weights)
            sp64 = sparse_gather_attention_einsum(
                q_abs.double(), q_rope.double(), c_kv.double(), k_rope.double(), idx, w64)
            sub_dense = dense_attention_absorbed_bmm(
                q_abs, q_rope, c_kv.index_select(0, idx), k_rope.index_select(0, idx), weights)
            tag = f"{prefix}set_{name}_"
            arrays[tag + "indices"] = idx
            arrays[tag + "sparse_einsum_fp32_out"] = sp32["out_h"].reshape(1, cfg.O_IN) @ weights["W_O"]
            arrays[tag + "sparse_fp64_out"] = sp64["out_h"].reshape(1, cfg.O_IN) @ w64["W_O"]
            arrays[tag + "sparse_fp64_s"] = sp64["s"]
            arrays[tag + "subset_dense_fp32_out"] = sub_dense["out_h"].reshape(1, cfg.O_IN) @ weights["W_O"]

        # 端到端 DSA step（k=512 或 N）
        k_e2e = min(END_TO_END_K, n)
        dsa = dsa_decode_step(x, weights, c_kv, k_rope, k_i, k_e2e, pos_q, freqs)
        dense_step = dense_decode_step(x, weights, c_kv, k_rope, pos_q, freqs)
        qual = index_quality(dense_step["scores"], dsa["indices"], ids)
        arrays[prefix + "dsa_step_indices"] = dsa["indices"]
        arrays[prefix + "dsa_step_out_fp32"] = dsa["out"]
        arrays[prefix + "dense_step_out_fp32"] = dense_step["out"]
        manifest["quality_stats"][f"n{n}"] = {
            "k_e2e": k_e2e, **qual,
            "sparse_vs_dense_out_max_abs": float((dsa["out"] - dense_step["out"]).abs().max()),
        }
        manifest["calib_stats"][f"n{n}"] = {
            "sigma_s_raw": calib["sigma_s_raw"], "sigma_index": calib["sigma_index"],
            "mean_latent_projection": calib["mean_latent_projection"],
            "gamma_main": {str(z): round(g, 3) for z, g in calib["gamma_main"].items()},
            "gamma_index": {str(z): round(g, 4) for z, g in calib["gamma_index"].items()},
            "support": cfg.relevance_support(n), "tier_sizes": cfg.tier_sizes(n),
        }
        print(f"[fixtures] N={n}: support={cfg.relevance_support(n)} "
              f"captured={qual['captured_mass_mean']:.3f} recall={qual['relevance_recall']:.2f} "
              f"e2e_err={manifest['quality_stats'][f'n{n}']['sparse_vs_dense_out_max_abs']:.4f}")

    if arguments.verify_only:
        import numpy as np

        manifest_path = out_dir / "manifest.json"
        npz_path = out_dir / "correctness.npz"
        if not manifest_path.is_file() or not npz_path.is_file():
            print(f"[verify] 找不到 {manifest_path} 或 {npz_path}，无法核对")
            return 1
        with np.load(npz_path) as data:
            stored = {key: data[key] for key in data.files}
        bad, checked = [], 0
        for name, tensor in arrays.items():
            if name not in stored:
                bad.append(f"{name}: npz 缺失")
                continue
            checked += 1
            err = float((tensor.float() - torch.from_numpy(stored[name]).float()).abs().max())
            if err > 2e-4:
                bad.append(f"{name}: 数值复核失败 max={err:.3e}")
        if bad:
            print("[verify] 失败：")
            for line in bad[:20]:
                print("  -", line)
            return 1
        print(f"[verify] 通过：{checked} 张量重算与 npz 真值 max abs < 2e-4 "
              f"（本机 torch {torch.__version__}；GEMM 舍入跨版本差异容差内）")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    npz_payload = {name: t.numpy() for name, t in arrays.items()}
    np.savez(out_dir / "correctness.npz", **npz_payload)
    for name, t in arrays.items():
        manifest["tensors"][name] = {
            "shape": list(t.shape), "dtype": str(t.dtype), "sha256": sha256_of_tensor(t),
        }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    total_bytes = sum(t.numel() * t.element_size() for t in arrays.values())
    print(f"[fixtures] 写出 {out_dir}/correctness.npz（{len(arrays)} 张量, "
          f"{total_bytes / 1024 / 1024:.1f} MB, 生成+写盘 {time.perf_counter() - started:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
