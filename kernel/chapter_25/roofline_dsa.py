"""第 25 期 DSA 理论口径：计算量/访存分解、break-even 公式、附录 B 题 10 校核。

口径单源约定（继承 chapter_01/performance roofline.py / chapter_24 roofline_mla.py 方法学）：
  - MAC 数按 GEMM/点积语义逐项列出（乘加各计 1），FLOPs = 2×MAC；
  - 理论字节 = 每步必须跨越 DRAM 的 io 字节（fp16 io；fp32 累加在片上）；
  - 延迟下限 = 理论字节 / 平台实测拷贝带宽；算力下限 = FLOPs / 实测峰值算力；
  - 平台分母（chapter_01/performance/results/peak-measurement-*.json 实测单源，与 ep25 同）：
      4090: D2D 918.8 GB/s、FP16 GEMM 171.63 TF；L2 = 72 MB（ep03 C-API 实测）
  - L2 驻留警示：每个扫描点标注工作集在 L2/DRAM 哪个区（ep03/ep25 教训）。

每 Token 计算量（decode 单 query、单层、FP16、主 Attention = ep25 吸收核）：
  主 Attention 每选中 token：
    score  = q_abs_h·c_kv(512) + q_rope_h·k_rope(64)   → 16 头 × 576 = 9216 MAC
    wsum   = attn_h · c_kv(512)                         → 16 头 × 512 = 8192 MAC
    合计 17408 MAC/token（与 N 无关、与选中集大小成正比）
  索引器每候选 token（每 query 都要对全部 N 个候选算一遍）：
    score  = Σ_h w_h·ReLU(q^I_h·k^I_t)                  → 64 头 × 128 = 8192 MAC
  索引器 K 投影（写侧、每 token 一次，与 query 数无关）：
    k^I_t = c_kv(t) @ W_KL                              → 512×128 = 65536 MAC
    （官方从 hidden 投影且独立缓存；合成设定见 dsa_config——我们的 W_KL 是
     latent→128 代理，重算成本的乐观下界）

两个设计点（"索引开销何时反噬"的两个开关，EXPERIMENT_LOG §6 展开）：
  Design B（官方对齐）：索引 K 独立缓存——每 query 索引器只需读 k^I 缓存
    （N×256B fp16）+ 算分（N×8192 MAC）。
  Design A（反例）：不缓存、每 query 从 latent 重算 k^I——额外 N×65536 MAC +
    N×1024B latent 重读。代入 N=128K：流量 168MB > dense 151MB、算力 4.25×
    dense——两个维度全部反噬，这就是官方必须为索引器单独建 K cache 的原因。

KV cache 每 Token 字节（§10.4 结构化指标）：
  MLA：latent 512 + rope 64 = 576 个 fp16 = 1152 B
  DSA：+ 索引 K 128 个 fp16 = 256 B（官方生产 kernel 为 fp8 = 128 B + scale）
  → DSA-on-MLA 每层每 Token 1408 B（fp16 索引 K）/ 1280 B（fp8），+22%/+11%。
"""

from __future__ import annotations

from dsa_config import (
    CALC_K,
    CALC_N,
    INDEX_HEADS,
    INDEX_HEAD_DIM,
    KV_LORA_RANK,
    N_HEADS,
    N_SWEEP,
    NUM_HIDDEN_LAYERS,
    QK_ROPE_HEAD_DIM,
)

# V3.2 层数（config.json num_hidden_layers=61）用于模型级外推
V32_LAYERS = 61

ITEMSIZE_FP16 = 2
ITEMSIZE_FP32 = 4

# chapter_01/performance/results/peak-measurement-*.json（2026-11-20 实测，口径单源）
PLATFORM_PEAKS = {
    "cuda": {
        "gpu_name": "NVIDIA GeForce RTX 4090 24GB (SM 8.9 Ada)",
        "copy_gb_s_measured": 918.8,
        "copy_gb_s_spec": 1008.0,
        "fp16_tf_measured": 171.63,
        "fp32_tf_measured": 55.6,
        "l2_bytes": 72 * 1024 * 1024,  # ep03 实测 72 MB
        "source": "chapter_01/performance/results/peak-measurement-cuda.json",
    },
    "musa": {
        "gpu_name": "MTT S4000 48GB",
        "copy_gb_s_measured": 711.28,
        "copy_gb_s_spec": 768.0,
        "fp16_tf_measured": 64.55,  # ch02 标注存疑
        "fp32_tf_measured": 12.53,
        "l2_bytes": None,
        "source": "chapter_01/performance/results/peak-measurement-musa.json",
    },
}

# ---- 每 Token MAC/字节常量（decode 单 query 单层）----
MAIN_SCORE_MACS_PER_TOKEN = N_HEADS * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)   # 16×576 = 9216
MAIN_WSUM_MACS_PER_TOKEN = N_HEADS * KV_LORA_RANK                         # 16×512 = 8192
MAIN_MACS_PER_TOKEN = MAIN_SCORE_MACS_PER_TOKEN + MAIN_WSUM_MACS_PER_TOKEN  # 17408
INDEX_SCORE_MACS_PER_TOKEN = INDEX_HEADS * INDEX_HEAD_DIM                 # 64×128 = 8192
INDEX_K_PROJ_MACS_PER_TOKEN = KV_LORA_RANK * INDEX_HEAD_DIM               # 512×128 = 65536
MAIN_BYTES_PER_TOKEN = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * ITEMSIZE_FP16  # 1152
INDEX_K_CACHE_BYTES_PER_TOKEN = INDEX_HEAD_DIM * ITEMSIZE_FP16            # 256（fp8=128）
INDEX_SCORE_WRITE_BYTES_PER_TOKEN = ITEMSIZE_FP32                         # 4（fp32 分数）
INDEX_READ_BYTES_PER_TOKEN = INDEX_K_CACHE_BYTES_PER_TOKEN + INDEX_SCORE_WRITE_BYTES_PER_TOKEN


def per_query_account(n: int, k: int, n_heads_main: int = N_HEADS) -> dict:
    """decode 单 query 的计算量/流量分解（FP16 io；MAC 与 FLOP 双计）。

    Design B（索引 K 缓存）与 Design A（从 latent 重算 k_I 的代理）并列；
    dense = 全量吸收核（ep25 同款）。主 Attention 头数可参数化（V3.2 128 头
    外推用），索引器固定 64×128（公开值）。"""
    main_per_tok = n_heads_main * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) + n_heads_main * KV_LORA_RANK
    dense_macs = n * main_per_tok
    dense_bytes = n * MAIN_BYTES_PER_TOKEN
    idx_score_macs = n * INDEX_SCORE_MACS_PER_TOKEN
    idx_score_bytes = n * INDEX_READ_BYTES_PER_TOKEN
    topk_bytes = n * ITEMSIZE_FP32 + k * ITEMSIZE_FP32        # 读分数 + 写索引(int32)
    sparse_macs = k * main_per_tok
    sparse_bytes = k * MAIN_BYTES_PER_TOKEN + k * ITEMSIZE_FP32  # 读索引
    b_macs = idx_score_macs + sparse_macs
    b_bytes = idx_score_bytes + topk_bytes + sparse_bytes
    a_extra_macs = n * INDEX_K_PROJ_MACS_PER_TOKEN
    a_extra_bytes = n * (KV_LORA_RANK * ITEMSIZE_FP16)        # 重算读 latent
    a_macs = b_macs + a_extra_macs
    a_bytes = b_bytes + a_extra_bytes
    out = {
        "n": n, "k": k, "n_heads_main": n_heads_main,
        "macs_per_token": {"main_score": n_heads_main * (KV_LORA_RANK + QK_ROPE_HEAD_DIM),
                           "main_wsum": n_heads_main * KV_LORA_RANK,
                           "main_total": main_per_tok,
                           "index_score": INDEX_SCORE_MACS_PER_TOKEN,
                           "index_k_proj(write-once)": INDEX_K_PROJ_MACS_PER_TOKEN},
        "dense": {"macs": dense_macs, "flops": 2 * dense_macs, "bytes": dense_bytes},
        "dsa_design_b_index_k_cached": {
            "index_score_macs": idx_score_macs, "sparse_main_macs": sparse_macs,
            "macs": b_macs, "flops": 2 * b_macs, "bytes": b_bytes,
            "index_share_of_macs": idx_score_macs / b_macs,
            "index_share_of_bytes": idx_score_bytes / b_bytes,
        },
        "dsa_design_a_recompute_from_latent": {
            "index_k_proj_macs": a_extra_macs, "index_score_macs": idx_score_macs,
            "sparse_main_macs": sparse_macs, "macs": a_macs, "flops": 2 * a_macs,
            "bytes": a_bytes,
        },
        "ratios": {
            "dsa_b_vs_dense_macs": b_macs / dense_macs,
            "dsa_b_vs_dense_bytes": b_bytes / dense_bytes,
            "dsa_a_vs_dense_macs": a_macs / dense_macs,
            "dsa_a_vs_dense_bytes": a_bytes / dense_bytes,
            "dense_vs_dsa_b_speedup_macs": dense_macs / b_macs,
            "dense_vs_dsa_b_speedup_bytes": dense_bytes / b_bytes,
        },
    }
    return out


def break_even(n_heads_main: int = N_HEADS, index_cache_bytes: int = INDEX_K_CACHE_BYTES_PER_TOKEN,
               index_cache_dtype: str = "fp16") -> dict:
    """DSA 收益为正的 k 上界（decode 单 query，忽略写侧摊销项）：

    算力：N·index_score ≤ (N−k)·main_per_tok → k/N ≤ 1 − index/main
    流量：N·index_read ≤ (N−k)·main_bytes → k/N ≤ 1 − index_read/main_bytes
    （主 Attention 头数增大 → main_per_tok 增大 → break-even 放宽）"""
    main_per_tok = n_heads_main * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) + n_heads_main * KV_LORA_RANK
    main_bytes = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * ITEMSIZE_FP16
    idx_read = index_cache_bytes + INDEX_SCORE_WRITE_BYTES_PER_TOKEN
    k_compute = 1.0 - INDEX_SCORE_MACS_PER_TOKEN / main_per_tok
    k_bandwidth = 1.0 - idx_read / main_bytes
    return {
        "n_heads_main": n_heads_main,
        "index_cache_dtype": index_cache_dtype,
        "index_read_bytes_per_token": idx_read,
        "main_macs_per_token": main_per_tok,
        "k_over_n_compute_break_even": k_compute,
        "k_over_n_bandwidth_break_even": k_bandwidth,
        "formula": "k/N ≤ 1 − index_score/main_macs（算力）；k/N ≤ 1 − index_read/main_bytes（流量）",
    }


def appendix_b_q10_check() -> dict:
    """附录 B 题 10（v4 文档行 1270；任务书行 1082 写"题 9"，编号错位已在
    EXPERIMENT_LOG 记录，ep25 同款错位——v4 §13 的引用都比附录 B 小 1）校核：
    dense O(N²×d) vs DSA 索引 O(N×d_index) + 稀疏 O(N×k×d)，N=128K、k=2048。

    序列级口径（题 10 公式的读法）：N 个 query 各看 ~N 候选/各选 k：
      dense MACs ≈ N²/2 × main_per_tok（因果三角）
      DSA   MACs ≈ N²/2 × index_score + N·k × main_per_tok
    per-query 口径（decode 实测形态）：见 per_query_account。"""
    n, k = CALC_N, CALC_K
    main_per_tok = MAIN_MACS_PER_TOKEN
    dense_seq = n * n // 2 * main_per_tok
    dsa_seq = n * n // 2 * INDEX_SCORE_MACS_PER_TOKEN + n * k * main_per_tok
    per_query = per_query_account(n, k)
    return {
        "N": n, "k": k,
        "formula_in_appendix": "dense O(N²×d)；DSA = 索引 O(N×d_index) + 稀疏 O(N×k×d)",
        "per_query_decode": per_query,
        "sequence_level": {
            "dense_macs": dense_seq, "dense_flops": 2 * dense_seq,
            "dsa_macs": dsa_seq, "dsa_flops": 2 * dsa_seq,
            "dsa_vs_dense": dsa_seq / dense_seq,
            "dsa_index_share": (n * n // 2 * INDEX_SCORE_MACS_PER_TOKEN) / dsa_seq,
        },
        "layers_v32": V32_LAYERS,
        "note": "题 10 公式里'稀疏主计算 O(N×k×d)'是序列级读法（N 个 query 各 k 个"
                "选中 token）；per-query decode 是 O(k×d)。两口径比值一致（~0.49），"
                "与实验实测互相校核。",
    }


def kv_bytes_per_token(dtype_bytes: int = ITEMSIZE_FP16) -> dict:
    """每层每 Token KV cache 字节：MHA/GQA/MLA/DSA 同 head_dim=128 口径。"""
    head_dim = 128
    mha16 = 2 * N_HEADS * head_dim * dtype_bytes
    gqa8 = 2 * 8 * head_dim * dtype_bytes
    mla = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * dtype_bytes
    dsa_fp16 = mla + INDEX_HEAD_DIM * dtype_bytes
    dsa_fp8 = mla + INDEX_HEAD_DIM * 1
    return {
        "dtype_bytes": dtype_bytes,
        "MHA_16heads": mha16,
        "GQA_8kv_heads": gqa8,
        "MLA_latent512_rope64": mla,
        "DSA_MLA_plus_indexK_fp16": dsa_fp16,
        "DSA_MLA_plus_indexK_fp8": dsa_fp8,
        "ratio_DSA_fp16_over_MLA": dsa_fp16 / mla,
        "ratio_DSA_fp8_over_MLA": dsa_fp8 / mla,
        "ratio_MLA_over_MHA16": mla / mha16,
    }


def kv_bytes_model_level(dtype_bytes: int = ITEMSIZE_FP16) -> dict:
    per_tok = kv_bytes_per_token(dtype_bytes)
    out = {"per_layer_per_token": per_tok}
    for name in ("MHA_16heads", "MLA_latent512_rope64", "DSA_MLA_plus_indexK_fp16"):
        out[f"{name}_27layers_KB_per_token"] = per_tok[name] * NUM_HIDDEN_LAYERS / 1024
        out[f"{name}_61layers_KB_per_token"] = per_tok[name] * V32_LAYERS / 1024
    for ctx in (131072,):
        out[f"DSA_fp16_61layers_ctx{ctx}_GB"] = per_tok["DSA_MLA_plus_indexK_fp16"] * V32_LAYERS * ctx / 1024 ** 3
        out[f"MLA_61layers_ctx{ctx}_GB"] = per_tok["MLA_latent512_rope64"] * V32_LAYERS * ctx / 1024 ** 3
    return out


def working_set_note(n: int, k: int) -> dict:
    """每个扫描点的工作集 vs 4090 L2=72MB（ep03 教训：标注测量区）。

    dense 主核工作集 = latent+rope cache（读一遍）；索引器 score 核工作集 =
    k_I 缓存 + 分数；稀疏核工作集 = 选中行 gather（k×1152B）+ 索引数组。"""
    l2 = PLATFORM_PEAKS["cuda"]["l2_bytes"]
    ws_dense = n * MAIN_BYTES_PER_TOKEN
    ws_index = n * INDEX_K_CACHE_BYTES_PER_TOKEN + n * INDEX_SCORE_WRITE_BYTES_PER_TOKEN
    ws_sparse = k * MAIN_BYTES_PER_TOKEN + k * ITEMSIZE_FP32
    return {
        "n": n, "k": k, "l2_bytes": l2,
        "dense_core_working_set_bytes": ws_dense,
        "dense_regime": "L2-resident" if ws_dense < l2 else "DRAM",
        "indexer_working_set_bytes": ws_index,
        "indexer_regime": "L2-resident" if ws_index < l2 else "DRAM",
        "sparse_core_working_set_bytes": ws_sparse,
        "sparse_regime": "L2-resident" if ws_sparse < l2 else "DRAM",
    }


def latency_lower_bounds(n: int, k: int) -> dict:
    acc = per_query_account(n, k)
    bw = PLATFORM_PEAKS["cuda"]["copy_gb_s_measured"] * 1e9
    tf = PLATFORM_PEAKS["cuda"]["fp16_tf_measured"] * 1e12
    return {
        "n": n, "k": k,
        "dense_bytes_lower_ms": acc["dense"]["bytes"] / bw * 1000.0,
        "dense_flops_lower_ms": acc["dense"]["flops"] / tf * 1000.0,
        "dsa_b_bytes_lower_ms": acc["dsa_design_b_index_k_cached"]["bytes"] / bw * 1000.0,
        "dsa_b_flops_lower_ms": acc["dsa_design_b_index_k_cached"]["flops"] / tf * 1000.0,
        "dsa_a_bytes_lower_ms": acc["dsa_design_a_recompute_from_latent"]["bytes"] / bw * 1000.0,
    }


def sweep_matrix() -> list[dict]:
    """实验 2/3 的 N×k 理论对照矩阵（写进 benchmark 结果 JSON 的 theory 节）。"""
    rows = []
    for n in N_SWEEP:
        for k in (256, 512, 1024, 2048, 4096):
            if k > n:
                continue
            acc = per_query_account(n, k)
            ws = working_set_note(n, k)
            lb = latency_lower_bounds(n, k)
            rows.append({"n": n, "k": k,
                         "dsa_b_macs": acc["dsa_design_b_index_k_cached"]["macs"],
                         "dense_macs": acc["dense"]["macs"],
                         "dsa_b_bytes": acc["dsa_design_b_index_k_cached"]["bytes"],
                         "dense_bytes": acc["dense"]["bytes"],
                         "index_share_of_macs": acc["dsa_design_b_index_k_cached"]["index_share_of_macs"],
                         "mac_ratio_vs_dense": acc["ratios"]["dsa_b_vs_dense_macs"],
                         "byte_ratio_vs_dense": acc["ratios"]["dsa_b_vs_dense_bytes"],
                         "working_set": ws, "lower_bounds_ms": lb})
    return rows


if __name__ == "__main__":
    import json

    print("=== 附录 B 题 10 校核（N=128K, k=2048）===")
    print(json.dumps(appendix_b_q10_check(), indent=1, ensure_ascii=False))
    print("=== break-even ===")
    for heads, cache_dt, cb in ((16, "fp16", 256), (16, "fp8", 128), (128, "fp16", 256)):
        print(json.dumps(break_even(heads, cb, cache_dt), ensure_ascii=False))
    print("=== KV 字节 ===")
    print(json.dumps(kv_bytes_per_token(), indent=1))
    print(json.dumps(kv_bytes_model_level(), indent=1))
    print("=== 工作集分区（N 扫描）===")
    for n in N_SWEEP:
        ws = working_set_note(n, min(2048, n))
        print(f"N={n}: dense={ws['dense_regime']} ({ws['dense_core_working_set_bytes']/1e6:.1f}MB) "
              f"indexer={ws['indexer_regime']} ({ws['indexer_working_set_bytes']/1e6:.1f}MB) "
              f"sparse(k=2048)={ws['sparse_regime']} ({ws['sparse_core_working_set_bytes']/1e6:.1f}MB)")
