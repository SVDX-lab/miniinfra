"""第 24 期 MLA 理论口径：GEMM 形状表、理论访存、理论 FLOPs、每 Token KV 字节。

口径单源约定（继承 chapter_01/performance roofline.py 的方法学，公式按 GEMM 语义重写）：
  - GEMM 理论字节 = (读 A[M,K] + 读 B[K,N] + 读 C[M,N](若累加) + 写 C[M,N]) × itemsize；
    decode 单 query 场景 C 初值为 0 且只写一遍：bytes = (M*K + K*N + M*N) × itemsize。
    bmm（batched GEMM）按 batch 数线性展开。
  - GEMM 理论 FLOPs = 2 × M × N × K（乘加各计 1）× batch。
  - 延迟下限 = 理论字节 / 平台实测峰值带宽；算力下限 = 理论 FLOPs / 平台实测峰值算力。
  - 平台分母（chapter_01/performance/results/peak-measurement-*.json 实测单源，本期不改口径）：
      4090: D2D 拷贝 918.8 GB/s，FP16 GEMM 171.63 TF，FP32 GEMM 55.6 TF
      S4000: D2D 拷贝 711.28 GB/s，FP16 GEMM 64.55 TF（ch02 已标注存疑）
    4090 官方规格带宽 1008 GB/s：达成率同时给出两个分母（实测分母为主口径，
    规格分母括注），与 ch02/ch03 报告一致。
  - L2 驻留警示（ep03 实测教训）：4090 L2 = 72 MB。KV 长度扫描中工作集小于
    L2 的测量点测到的是 L2 带宽而非 DRAM 带宽；本模块同时输出每个 N 的
    attention 核工作集字节数，用于把扫描点标注到 "L2 区 / DRAM 区"。

每 Token KV 字节推导（§10.4 结构化指标 + 附录 B"MLA KV 压缩"条校核对象，
大纲行 1269；任务书写的"题 8"按行号定位即该条，当前 v4 文档编号为题 9，
编号差异已在 EXPERIMENT_LOG 显式记录）：
  MHA : 2 × n_heads × head_dim × bytes      （K 与 V 各存一份全头 KV）
  GQA : 2 × kv_heads × head_dim × bytes     （同 head_dim，kv_heads < n_heads）
  MQA : 2 × 1 × head_dim × bytes            （单 KV 头）
  MLA : (kv_lora_rank + rope_dim) × bytes   （latent + 共享 rope 头，与头数无关）
"""

from __future__ import annotations

from mla_config import (
    D_MODEL,
    KV_A_OUT,
    KV_LORA_RANK,
    N_HEADS,
    O_IN,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    Q_LORA_RANK,
    V_HEAD_DIM,
)

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
        "fp16_tf_measured": 64.55,  # ch02 标注：FP16 matmul 首轮 JIT，数值存疑
        "fp32_tf_measured": 12.53,
        "l2_bytes": None,
        "source": "chapter_01/performance/results/peak-measurement-musa.json",
    },
}


def gemm_bytes(
    m: int,
    k: int,
    n: int,
    itemsize: int = ITEMSIZE_FP16,
    batch: int = 1,
    shared_a: bool = False,
    shared_b: bool = False,
) -> int:
    """decode 单 query GEMM 理论字节：读 A + 读 B + 写 C（C 无需读回）。

    shared_a / shared_b：batch 维共享同一 A 或 B（stride-0 bmm operand，如
    K/V 投影 bmm 与吸收形式 score/加权 bmm 里所有头共用 c_kv[N,512]）——
    DRAM 下限按共享 operand 读一遍计（批间重读可被 L2 吸收，不计 DRAM）。
    io dtype 口径：GEMM 输入输出 itemsize（fp16 路径全 fp16 io），fp32 累加在
    片上完成、不计 DRAM。
    """
    a_bytes = m * k * itemsize if shared_a else batch * m * k * itemsize
    b_bytes = k * n * itemsize if shared_b else batch * k * n * itemsize
    return a_bytes + b_bytes + batch * m * n * itemsize


def gemm_flops(m: int, k: int, n: int, batch: int = 1) -> int:
    return 2 * m * n * k * batch


def gemm_row(
    name: str,
    m: int,
    k: int,
    n: int,
    itemsize: int = ITEMSIZE_FP16,
    batch: int = 1,
    n_dependent: bool = False,
    shared_a: bool = False,
    shared_b: bool = False,
) -> dict:
    return {
        "name": name,
        "batch": batch,
        "m": m,
        "k": k,
        "n": n,
        "macs": m * n * k * batch,
        "flops": gemm_flops(m, k, n, batch),
        "bytes": gemm_bytes(m, k, n, itemsize, batch, shared_a, shared_b),
        "n_dependent": n_dependent,
        "shared_a": shared_a,
        "shared_b": shared_b,
    }


def unabsorbed_gemm_table(kv_len: int, itemsize: int = ITEMSIZE_FP16) -> list[dict]:
    """未吸收形式 decode（单 query，KV 长 N）的 GEMM 序列形状表。

    per-head 小 GEMM 用 batch=16 的 bmm 表达（真实实现同构：einsum→bmm）。
    N 相关项：K 投影 bmm、score 点积、V 投影 bmm、加权求和 bmm。
    """
    h, dk, dr, dv = N_HEADS, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, V_HEAD_DIM
    c, qlr = KV_LORA_RANK, Q_LORA_RANK
    rows = [
        gemm_row("q_a_proj", 1, D_MODEL, qlr, itemsize),
        gemm_row("q_b_proj", 1, qlr, h * (dk + dr), itemsize),
        gemm_row("kv_a_proj(新token)", 1, D_MODEL, KV_A_OUT, itemsize),
        gemm_row("K投影 bmm（per-head 小GEMM）", kv_len, c, dk, itemsize, batch=h, n_dependent=True, shared_a=True),
        gemm_row("score=Q·K^T bmm", 1, dk + dr, kv_len, itemsize, batch=h, n_dependent=True),
        gemm_row("V投影 bmm（per-head 小GEMM）", kv_len, c, dv, itemsize, batch=h, n_dependent=True, shared_a=True),
        gemm_row("加权求和 attn·V bmm", 1, kv_len, dv, itemsize, batch=h, n_dependent=True),
        gemm_row("o_proj", 1, O_IN, D_MODEL, itemsize),
    ]
    return rows


def absorbed_gemm_table(kv_len: int, itemsize: int = ITEMSIZE_FP16) -> list[dict]:
    """吸收形式 decode（单 query，KV 长 N）的 GEMM 序列形状表。

    吸收的两处改写（把 per-head 小 GEMM 从"每 Token 一遍"搬成"每 Query 一遍"）：
      1) W_UK 吸收进 Q 侧：score_nope = q_nope·(c·W_UK) = (q_nope·W_UK^T)·c
         → 新增每 Query 一次 bmm [16×(1×128)×(128×512)]，KV 侧 N 相关投影消失；
      2) W_UV 吸收进输出侧：out = Σ_t a_t (c_t·W_UV) = (Σ_t a_t c_t)·W_UV
         → V 投影 bmm（N 相关）换成"加权 latent 求和"（N 相关但只读 latent，
         不物化 V）+ 每 Query 一次 bmm [16×(1×512)×(512×128)]。
    """
    h, dk, dr, dv = N_HEADS, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, V_HEAD_DIM
    c, qlr = KV_LORA_RANK, Q_LORA_RANK
    rows = [
        gemm_row("q_a_proj", 1, D_MODEL, qlr, itemsize),
        gemm_row("q_b_proj", 1, qlr, h * (dk + dr), itemsize),
        gemm_row("kv_a_proj(新token)", 1, D_MODEL, KV_A_OUT, itemsize),
        gemm_row("吸收W_UK bmm：q_nope·W_UK^T", 1, dk, c, itemsize, batch=h),
        gemm_row("score=Q_abs·C bmm", 1, c + dr, kv_len, itemsize, batch=h, n_dependent=True, shared_b=True),
        gemm_row("加权latent求和 attn·C bmm", 1, kv_len, c, itemsize, batch=h, n_dependent=True, shared_b=True),
        gemm_row("吸收W_UV bmm：s·W_UV", 1, c, dv, itemsize, batch=h),
        gemm_row("o_proj", 1, O_IN, D_MODEL, itemsize),
    ]
    return rows


def table_totals(rows: list[dict]) -> dict:
    return {
        "macs": sum(r["macs"] for r in rows),
        "flops": sum(r["flops"] for r in rows),
        "bytes": sum(r["bytes"] for r in rows),
        "n_dependent_macs": sum(r["macs"] for r in rows if r["n_dependent"]),
        "n_independent_macs": sum(r["macs"] for r in rows if not r["n_dependent"]),
    }


# ---- 每层每 Token KV cache 字节（§10.4 结构化指标口径）----

def kv_bytes_per_token(dtype_bytes: int = ITEMSIZE_FP16) -> dict:
    """同 head_dim=128 口径下 MHA/GQA/MQA/MLA 每层每 Token KV cache 字节。

    MLA 的 KV 与 query 头数无关（每 Token 一份 latent + 一份共享 rope 头，
    所有头复用）——这是"MQA 式 rope + 低秩 latent"结构带来的压缩。
    """
    head_dim = 128
    mha = 2 * N_HEADS * head_dim * dtype_bytes
    gqa8 = 2 * 8 * head_dim * dtype_bytes    # Qwen3-0.6B 同款 8 KV 头（附录 B 题 2 口径）
    gqa4 = 2 * 4 * head_dim * dtype_bytes
    mqa = 2 * 1 * head_dim * dtype_bytes
    mla = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * dtype_bytes
    return {
        "dtype_bytes": dtype_bytes,
        "head_dim": head_dim,
        "MHA_16heads": mha,
        "GQA_8kv_heads": gqa8,
        "GQA_4kv_heads": gqa4,
        "MQA_1kv_head": mqa,
        "MLA_latent512_rope64": mla,
        "ratio_MLA_over_MHA": mla / mha,
        "ratio_MLA_over_GQA8": mla / gqa8,
        "ratio_MLA_over_MQA": mla / mqa,
    }


def kv_bytes_model_level(dtype_bytes: int = ITEMSIZE_FP16) -> dict:
    """模型级口径：× num_hidden_layers(27) × 上下文 Token 数。"""
    per_tok = kv_bytes_per_token(dtype_bytes)
    layers = 27
    out = {"per_layer_per_token": per_tok}
    for name in ("MHA_16heads", "GQA_8kv_heads", "MLA_latent512_rope64"):
        out[f"{name}_27layers_KB_per_token"] = per_tok[name] * layers / 1024
    for ctx in (2048, 32768, 131072):
        out[f"MHA_16heads_27layers_ctx{ctx}_MB"] = per_tok["MHA_16heads"] * layers * ctx / 1024 / 1024
        out[f"MLA_27layers_ctx{ctx}_MB"] = per_tok["MLA_latent512_rope64"] * layers * ctx / 1024 / 1024
    return out


def v3_extrapolation(dtype_bytes: int = ITEMSIZE_FP16) -> dict:
    """外推到 V3/DeepSeek-V2(full) 128-head 规模（合同"扩展系数另注"条目）。

    MLA 每层每 Token KV 字节与头数无关（仍是 latent 512 + rope 64），
    MHA 按头数线性放大 → 压缩比随头数线性放大。附录 B 题 9 的
    "数十 KB / 压缩到约 1/100" 即 V3 128 头口径。
    """
    head_dim, v3_heads = 128, 128
    mla = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * dtype_bytes
    mha_v3 = 2 * v3_heads * head_dim * dtype_bytes
    return {
        "v3_heads": v3_heads,
        "MLA_per_layer_per_token_bytes": mla,
        "MHA_128heads_per_layer_per_token_bytes": mha_v3,
        "ratio": mha_v3 / mla,
        "note": "MLA KV 字节与头数无关；V2-Lite(16头) 压缩比 7.11x，V3(128头) 56.9x",
    }


# ---- attention 核（N 相关部分）理论流量：两种形式对照 ----

def unabsorbed_core_bytes(kv_len: int, itemsize: int = ITEMSIZE_FP16) -> dict:
    """未吸收 attention 核链（K bmm + score bmm + softmax + V bmm + wsum bmm）
    的理论 DRAM 流量下限（io dtype 口径：fp16 路径全 fp16 io，fp32 累加在片上；
    共享 A 的 bmm 的 A 按 DRAM 读一遍计，见 gemm_bytes docstring）。

    逐项：
      K bmm:    读 c_kv[N,512]（16 头共享，读一遍）+ 读 W_UK 2.1MB + 写 K[N,16,128]
      score bmm:读 K[N,16,128] + 写 S[16,N]（fp16）
      softmax:  读 S + 写 A（[16,N] fp16 ×2）
      V bmm:    读 c_kv[N,512]（第二遍）+ 读 W_UV 2.1MB + 写 V[N,16,128]
      wsum bmm: 读 V[N,16,128] + 读 A + 写 O[16,128]
    """
    h, dk, dv, c, dr = N_HEADS, QK_NOPE_HEAD_DIM, V_HEAD_DIM, KV_LORA_RANK, QK_ROPE_HEAD_DIM
    w_uk = h * c * dk * itemsize
    w_uv = h * c * dv * itemsize
    kv_latent = kv_len * (c + dr) * itemsize      # latent + rope cache，每次 bmm 读一遍
    k_mat = kv_len * h * dk * itemsize            # 物化 K
    v_mat = kv_len * h * dv * itemsize            # 物化 V
    scores = kv_len * h * itemsize                # [16,N] fp16
    attn = kv_len * h * itemsize                  # [16,N] fp16
    out_mat = h * dv * itemsize
    reads = (
        kv_latent + w_uk           # K bmm（c_kv 读一遍）
        + k_mat                    # score bmm 读 K
        + scores                   # softmax 读
        + kv_latent + w_uv         # V bmm（c_kv 再读一遍）
        + v_mat + attn             # wsum 读 V 和 attn
    )
    writes = k_mat + scores + attn + v_mat + out_mat
    return {
        "reads_bytes": reads,
        "writes_bytes": writes,
        "total_bytes": reads + writes,
        "per_token_bytes": (reads + writes) / kv_len,
        "weight_bytes_each_step": w_uk + w_uv,
        "materialized_KV_bytes": k_mat + v_mat,
        "note": "c_kv 按读两遍计（K、V bmm 各一遍）；S/A 按 fp16 io；fp32 累加在片上",
    }


def absorbed_core_bytes(kv_len: int, itemsize: int = ITEMSIZE_FP16) -> dict:
    """吸收形式 attention 核（单个融合核：读 latent 直接出加权 latent s）。

    读：c_kv[N,512] 一遍 + k_rope[N,64] 一遍 + q_abs/q_rope（常数小）+ W_UV 4MB；
    写：s[16,512]。
    """
    h, c, dr, dv = N_HEADS, KV_LORA_RANK, QK_ROPE_HEAD_DIM, V_HEAD_DIM
    kv_latent = kv_len * (c + dr) * itemsize
    q_vec = h * (c + dr) * itemsize
    w_uv = h * c * dv * itemsize
    s_out = h * c * itemsize
    reads = kv_latent + q_vec + w_uv
    writes = s_out
    return {
        "reads_bytes": reads,
        "writes_bytes": writes,
        "total_bytes": reads + writes,
        "per_token_bytes": (reads + writes) / kv_len,
        "weight_bytes_each_step": w_uv,
        "materialized_KV_bytes": 0,
        "note": "W_UK 已吸收进 Q（每 Query 常数开销，不计入核）；无 K/V 物化",
    }


def working_set_note(kv_len: int, itemsize: int = ITEMSIZE_FP16) -> dict:
    """每个扫描点的工作集字节数 vs 4090 L2=72MB（ep03 教训：标注测量区）。"""
    l2 = PLATFORM_PEAKS["cuda"]["l2_bytes"]
    latent = kv_len * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * itemsize
    k_mat = kv_len * N_HEADS * QK_NOPE_HEAD_DIM * itemsize
    v_mat = kv_len * N_HEADS * V_HEAD_DIM * itemsize
    ws_unabs = latent + k_mat + v_mat          # 未吸收核物化 K/V 后的总工作集
    ws_abs = latent                            # 吸收核只读 latent
    return {
        "kv_len": kv_len,
        "unabsorbed_core_working_set_bytes": ws_unabs,
        "absorbed_core_working_set_bytes": ws_abs,
        "l2_bytes": l2,
        "unabsorbed_regime": "L2-resident" if ws_unabs < l2 else "DRAM",
        "absorbed_regime": "L2-resident" if ws_abs < l2 else "DRAM",
    }


def latency_lower_bound(total_bytes: int, backend: str) -> float:
    """毫秒 = 理论字节 / 实测拷贝带宽（ch02 分母）。"""
    bw = PLATFORM_PEAKS[backend]["copy_gb_s_measured"] * 1e9
    return total_bytes / bw * 1000.0


if __name__ == "__main__":
    import json

    for n in (1, 128, 2048, 32768):
        ua = table_totals(unabsorbed_gemm_table(n))
        ab = table_totals(absorbed_gemm_table(n))
        print(f"--- N={n} ---")
        print("  unabsorbed: flops=%.3f G bytes=%.2f MB" % (ua["flops"] / 1e9, ua["bytes"] / 1e6))
        print("  absorbed  : flops=%.3f G bytes=%.2f MB" % (ab["flops"] / 1e9, ab["bytes"] / 1e6))
        ub = unabsorbed_core_bytes(n)
        abd = absorbed_core_bytes(n)
        print("  core unabsorbed: %.2f MB (%.0f B/token) lower=%.3f ms" % (
            ub["total_bytes"] / 1e6, ub["per_token_bytes"], latency_lower_bound(ub["total_bytes"], "cuda")))
        print("  core absorbed  : %.2f MB (%.0f B/token) lower=%.3f ms" % (
            abd["total_bytes"] / 1e6, abd["per_token_bytes"], latency_lower_bound(abd["total_bytes"], "cuda")))
        ws = working_set_note(n)
        print("  regime: unabs=%s abs=%s" % (ws["unabsorbed_regime"], ws["absorbed_regime"]))
    print(json.dumps(kv_bytes_per_token(), indent=1))
    print(json.dumps(v3_extrapolation(), indent=1))
