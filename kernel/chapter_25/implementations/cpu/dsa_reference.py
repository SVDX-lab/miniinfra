"""第 25 期 DSA 参考实现（CPU / torch eager，真值锚点）+ 合成数据生成。

三段结构（大纲"Lightning Indexer + top-k + 稀疏 gather Attention"）：

1) Lightning Indexer（HF transformers modeling_deepseek_v32.py Indexer 类同构，
   维度出处见 dsa_config.py）：
       q^I_h = q_a @ W_QI_h                 （官方 wq_b 输入 = q_lora_rank 路径输出）
       w     = (x @ W_IW) × n_heads^-0.5     （官方 weights_proj × n_heads**-0.5）
       k^I_t = c_kv(t) @ W_KL                （合成设定：官方 wk 输入是 hidden state
                                              且 k 经 LayerNorm+RoPE 后独立缓存；
                                              本期用 latent→128 合成投影代替）
       I_t   = Σ_h w_h · ReLU(scale_I · q^I_h · k^I_t)     （fp32 计算，官方同式）
   scale_I = 1/sqrt(128)（官方 softmax_scale = index_head_dim**-0.5），scale>0
   使 ReLU 前后乘法可交换，只改分数绝对量级、不改 top-k 排名。
2) top-k 选择：token 级 torch.topk（官方同：k = min(index_topk, seq_len)，
   返回 value 降序）。空集 k=0 合法（返回空索引）；k>N 截断到 N。
3) 稀疏 gather Attention（主计算 = ep25 的 MLA 吸收核在索引子集上执行）：
       score_h(t) = (q_abs_h · c_kv(t) + q_rope_h · k_rope(t)) × scale,  t ∈ topk 集
       attn = softmax(score), s_h = Σ_t attn·c_kv(t), out_h = s_h @ W_UV_h
   给定同一索引集时与"对同一子集跑 dense Attention"必须等价（大纲 §9 DSA 条）。

planted relevance（dsa_config 声明的合成设定 3）：
   随机权重不含"索引排名 ↔ 主 Attention 相关性"（那是训练出来的）。本模块在
   合成数据里显式注入三档受控相关性：
     - 主 Attention 侧：relevance token 的 latent 加 γ·v̂，v̂ = normalize(Σ_h q̂_h^abs)
       ——v̂ 与每个头的 q_abs[h] 都有 ~2.8 的投影（16 个近正交单位向量之和），
       使所有头的分数同步抬升；γ 按档位 z∈{12.0, 10.5, 9.5} 自校准
       （z = 分数抬升 / 分数噪声 std，闭合式：主分数对 γ 线性）。
     - 索引器侧：relevance token 的 k^I 加 γ_idx·d̂，d̂ = normalize(Σ_{w>0} w_h q̂^I_h)
       ——只让 w_h>0 的头参与（ReLU 会截掉负权重头的贡献），γ_idx 按档位
       z∈{11.0, 10.0, 9.0} 对 ReLU 非线性二分自校准。
     - 支撑集大小 S(n) = max(256, round(0.025·n))，三档 6.25%/18.75%/75%。
   目标分布形态（推导见 EXPERIMENT_LOG §4）：背景 mass ~2-3%，三档各 ~25-40%，
   使误差—k 曲线在 k≈S(n) 处出现膝点；z 取值已核对 fp32 exp 范围安全。

seed 规则（generate_fixtures 与 benchmark 运行时生成共用同一规则）：
   权重 = SEED_BASE + sorted(WEIGHT_SHAPES) 序号 + 1；x = SEED_BASE + 1；
   cache(N) = SEED_BASE + 1000 + N（c_kv 与 k_rope_raw 同一生成器顺序取出）；
   relevance(N) = SEED_BASE + 2000 + N 的 randperm 前 S 个；
   校准采样 = SEED_BASE + 3000 + N。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # chapter_25/

from dsa_config import (
    INDEX_HEADS,
    INDEX_HEAD_DIM,
    INDEX_SCALE,
    INDEX_WEIGHT_SCALE,
    KV_LORA_RANK,
    N_HEADS,
    O_IN,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    Q_LORA_RANK,
    ROPE_THETA,
    SOFTMAX_SCALE,
    V_HEAD_DIM,
    Z_TIERS,
    Z_TIERS_INDEX,
    WEIGHT_SHAPES,
    relevance_support,
    tier_sizes,
)

ROPE_HALF = QK_ROPE_HEAD_DIM // 2  # 32

SEED_BASE = 20260326


# ---------------------------------------------------------------------------
# RoPE 工具（与 ep25 逐行同款：Llama/HF 半分约定）
# ---------------------------------------------------------------------------

def rope_freqs(theta: float = ROPE_THETA) -> torch.Tensor:
    return theta ** (-torch.arange(0, QK_ROPE_HEAD_DIM, 2, dtype=torch.float32) / QK_ROPE_HEAD_DIM)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    angles = positions.to(freqs.dtype).unsqueeze(-1) * freqs.unsqueeze(0)  # [P,32]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    if positions.numel() == 1:
        cos = cos.reshape(*([1] * (x.dim() - 1)), ROPE_HALF)
        sin = sin.reshape(*([1] * (x.dim() - 1)), ROPE_HALF)
    x1, x2 = x[..., :ROPE_HALF], x[..., ROPE_HALF:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# 权重与投影阶段
# ---------------------------------------------------------------------------

WEIGHT_KEYS = tuple(WEIGHT_SHAPES.keys())


def make_weights(seed_base: int = SEED_BASE) -> dict[str, torch.Tensor]:
    """合成权重：randn × He 缩放（std = 1/sqrt(fan_in)，fan_in = 收缩维输入维度）。

    ep25 教训①：N(0,1) 权重会使激活逐层放大、softmax 退化 one-hot；He 缩放使
    激活 ~O(1)、score×scale ~O(1)。
    与 ep25 的差异声明：ep25 对全部权重用 shape[0] 作 fan_in——对 3-D per-head
    权重 W_UK/W_UV [16,512,128] 那意味着 fan_in=16（头数），而实际收缩维是
    512（latent），导致 ep25 的 q_abs 范数偏热 5.6×。该偏差对 ep25 的等价性
    结论无影响（两种形式共用同一权重），但 ep26 的 planted 分数分布设计要求
    softmax 单位下噪声 std ≈ 1，故本期改用正确 fan_in：2-D [in,out] 用
    shape[0]，3-D [heads,in,out] 用 shape[1]。
    """
    weights = {}
    for idx, name in enumerate(sorted(WEIGHT_KEYS)):
        shape = WEIGHT_SHAPES[name]
        fan_in = shape[0] if len(shape) == 2 else shape[1]
        gen = torch.Generator().manual_seed(seed_base + idx + 1)
        weights[name] = torch.randn(*shape, generator=gen, dtype=torch.float32) / (fan_in ** 0.5)
    return weights


def q_projection_stage(x: torch.Tensor, w: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Q 路径（与 ep25 同款）：x → q_a [1536] → q_b → (q_nope [16,128], q_rope_raw [16,64])。

    官方 V3.2 索引器的 wq_b 输入正是 q_a（q_resid）路径输出——q_a 在本期被
    主 Attention 与索引器共用（同一份投影，无额外 Q 侧成本）。"""
    q_a = x @ w["W_QA"]
    q_all = (q_a @ w["W_QB"]).view(N_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    return q_all[:, :QK_NOPE_HEAD_DIM], q_all[:, QK_NOPE_HEAD_DIM:]


def absorbed_query(q_nope: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """q_abs [H,512]：W_UK 吸收进 Q 侧（ep25 吸收改写 1），latent 空间 query。"""
    return torch.einsum("hd,hcd->hc", q_nope, w["W_UK"])


def index_query_stage(q_a: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """索引 Q [64,128]：q_a @ W_QI 再按头切分（官方 wq_b 同构；本期省略
    官方的 LayerNorm 与 RoPE 段——合成设定 1，dsa_config 已声明）。"""
    return (q_a @ w["W_QI"]).view(INDEX_HEADS, INDEX_HEAD_DIM)


def index_weights_stage(x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """索引头权重 w [64]：（x @ W_IW) × n_heads^-0.5（官方 forward 同式）。"""
    return (x @ w["W_IW"]).reshape(INDEX_HEADS) * INDEX_WEIGHT_SCALE


def index_k_projection(c_kv: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """索引 K [N,128]：c_kv @ W_KL（合成设定 2：官方从 hidden state 投影并
    独立缓存；本期用 latent→128 合成投影生成同一形状的索引 K 缓存）。"""
    return c_kv @ w["W_KL"]


# ---------------------------------------------------------------------------
# 1) Lightning Indexer 分数
# ---------------------------------------------------------------------------

def index_scores(
    q_i: torch.Tensor,      # [64,128]
    w_used: torch.Tensor,   # [64]（已乘 n_heads^-0.5）
    k_i: torch.Tensor,      # [N,128]
    scale: float = INDEX_SCALE,
) -> torch.Tensor:
    """I_t = Σ_h w_h · ReLU(scale · q^I_h · k^I_t) → [N]（输入 dtype 决定精度：
    fp32 主链 / fp64 锚点；官方实现 score 段强制 fp32）。"""
    dots = torch.einsum("hd,nd->hn", q_i, k_i) * scale      # [64,N]
    return torch.einsum("h,hn->n", w_used, torch.relu(dots))


# ---------------------------------------------------------------------------
# 2) top-k 选择（官方：token 级 topk，k = min(index_topk, seq_len)，int32）
# ---------------------------------------------------------------------------

def select_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """返回 [min(k,N)] int64 索引（value 降序）。

    并列处理声明：浮点连续分布下严格并列是零概率事件；若发生，torch.topk 的
    顺序未定义而 CUDA 阈值法按索引升序破并列——正确性比较以索引集合相等为准
    （注意力对集合序不敏感，和的浮点顺序差异在容差内）。k=0 合法（空集），
    k>N 截断为 N（官方 min(index_topk, seq_len) 同式）。"""
    n = scores.shape[-1]
    return torch.topk(scores, min(k, n), largest=True).indices


# ---------------------------------------------------------------------------
# 3) 稀疏 gather Attention 与 dense Attention（ep25 吸收核的两种执行方式）
# ---------------------------------------------------------------------------

def _gather(kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return kv.index_select(0, indices)


def sparse_gather_attention_einsum(
    q_abs: torch.Tensor,    # [H,512]
    q_rope: torch.Tensor,   # [H,64]（已按 query 位置旋转）
    c_kv: torch.Tensor,     # [N,512]
    k_rope: torch.Tensor,   # [N,64]
    indices: torch.Tensor,  # [k'] int64（可乱序、可含重复？——合同：无重复）
    w: dict[str, torch.Tensor],
    scale: float = SOFTMAX_SCALE,
) -> dict[str, torch.Tensor]:
    """稀疏 gather Attention（einsum 教学形式）：gather 子集 → 打分 → softmax →
    加权 latent → W_UV 输出。k'=0（空集）→ softmax 对空维 → s=0、out=0
    （torch 语义，行为在 smoke/合同显式记录，§9 空集条目）。"""
    dtype = q_abs.dtype
    if indices.numel() == 0:
        h = q_abs.shape[0]
        z = torch.zeros(h, KV_LORA_RANK, dtype=dtype)
        zo = torch.zeros(h, V_HEAD_DIM, dtype=dtype)
        return {"s": z, "out_h": zo, "attn": torch.zeros(h, 0, dtype=dtype),
                "scores": torch.zeros(h, 0, dtype=dtype)}
    c_sub = _gather(c_kv, indices).to(dtype)
    k_sub = _gather(k_rope, indices).to(dtype)
    scores_nope = torch.einsum("hc,nc->hn", q_abs.to(dtype), c_sub)   # [H,k']
    scores_rope = torch.einsum("hd,nd->hn", q_rope.to(dtype), k_sub)  # [H,k']
    scores = (scores_nope + scores_rope) * scale
    attn = torch.softmax(scores, dim=-1)
    s = torch.einsum("hn,nc->hc", attn, c_sub)                        # [H,512]
    out_h = torch.einsum("hc,hcd->hd", s, w["W_UV"].to(dtype))        # [H,128]
    return {"s": s, "out_h": out_h, "attn": attn, "scores": scores}


def dense_attention_absorbed_einsum(
    q_abs: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    w: dict[str, torch.Tensor],
    scale: float = SOFTMAX_SCALE,
) -> dict[str, torch.Tensor]:
    """dense Attention（全部 N 个 token；einsum 形式，真值锚点链）。"""
    dtype = q_abs.dtype
    scores_nope = torch.einsum("hc,nc->hn", q_abs.to(dtype), c_kv.to(dtype))
    scores_rope = torch.einsum("hd,nd->hn", q_rope.to(dtype), k_rope.to(dtype))
    scores = (scores_nope + scores_rope) * scale
    attn = torch.softmax(scores, dim=-1)
    s = torch.einsum("hn,nc->hc", attn, c_kv.to(dtype))
    out_h = torch.einsum("hc,hcd->hd", s, w["W_UV"].to(dtype))
    return {"s": s, "out_h": out_h, "attn": attn, "scores": scores}


def sparse_gather_attention_bmm(
    q_abs: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    w: dict[str, torch.Tensor],
    scale: float = SOFTMAX_SCALE,
) -> dict[str, torch.Tensor]:
    """稀疏 gather Attention（bmm/matmul 基准形式——benchmark 与 CUDA 对照链）。

    与 CUDA 核的算法对应：gather（核内随机行读）→ score bmm → online softmax
    （核内；此处 dense softmax 数学等价）→ 加权 latent bmm。"""
    dtype = q_abs.dtype
    if indices.numel() == 0:
        h = q_abs.shape[0]
        return {"s": torch.zeros(h, KV_LORA_RANK, dtype=dtype),
                "out_h": torch.zeros(h, V_HEAD_DIM, dtype=dtype),
                "attn": torch.zeros(h, 0, dtype=dtype),
                "scores": torch.zeros(h, 0, dtype=dtype)}
    c_sub = _gather(c_kv, indices).to(dtype)
    k_sub = _gather(k_rope, indices).to(dtype)
    scores = (torch.matmul(q_abs.to(dtype), c_sub.transpose(0, 1))
              + q_rope.to(dtype) @ k_sub.transpose(0, 1)) * scale   # [H,k']
    attn = torch.softmax(scores, dim=-1)
    s = torch.matmul(attn, c_sub)                                   # [H,512]
    out_h = torch.matmul(s.unsqueeze(1), w["W_UV"].to(dtype)).squeeze(1)  # [H,128]
    return {"s": s, "out_h": out_h, "attn": attn, "scores": scores}


def dense_attention_absorbed_bmm(
    q_abs: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    w: dict[str, torch.Tensor],
    scale: float = SOFTMAX_SCALE,
) -> dict[str, torch.Tensor]:
    """dense Attention（bmm 形式；与 ep25 mla_core_absorbed_bmm 同构，
    k=N 的恒等子集）。"""
    dtype = q_abs.dtype
    scores = (torch.matmul(q_abs.to(dtype), c_kv.to(dtype).t())
              + q_rope.to(dtype) @ k_rope.to(dtype).t()) * scale    # [H,N]
    attn = torch.softmax(scores, dim=-1)
    s = torch.matmul(attn, c_kv.to(dtype))                          # [H,512]
    out_h = torch.matmul(s.unsqueeze(1), w["W_UV"].to(dtype)).squeeze(1)  # [H,128]
    return {"s": s, "out_h": out_h, "attn": attn, "scores": scores}


# ---------------------------------------------------------------------------
# 完整 decode 步（DSA 三段 vs dense 全量）
# ---------------------------------------------------------------------------

def dsa_decode_step(
    x: torch.Tensor,          # [1,2048]
    w: dict[str, torch.Tensor],
    c_kv: torch.Tensor,       # [N,512]
    k_rope: torch.Tensor,     # [N,64]（已逐位置旋转）
    k_i: torch.Tensor,        # [N,128] 索引 K 缓存
    k: int,
    pos_q: int,
    freqs: torch.Tensor,
    scale: float = SOFTMAX_SCALE,
    core: str = "bmm",
) -> dict[str, torch.Tensor]:
    """完整 DSA decode 步：Q 投影 → 索引器 → top-k → 稀疏 gather Attention →
    W_UV 吸收输出已在核内 → o_proj。返回带全部中间量的 dict（探针）。"""
    q_nope, q_rope_raw = q_projection_stage(x, w)
    q_rope = apply_rope(q_rope_raw, torch.tensor([pos_q], dtype=torch.float32), freqs)
    q_abs = absorbed_query(q_nope, w)
    q_a = x @ w["W_QA"]
    q_i = index_query_stage(q_a, w)
    w_used = index_weights_stage(x, w)
    scores_i = index_scores(q_i, w_used, k_i)                       # [N] fp32
    indices = select_topk(scores_i, k)                              # [k'] int64
    core_fn = sparse_gather_attention_bmm if core == "bmm" else sparse_gather_attention_einsum
    att = core_fn(q_abs, q_rope, c_kv, k_rope, indices, w, scale)
    out = att["out_h"].reshape(1, O_IN) @ w["W_O"]                  # [1,2048]
    return {"out": out, "indices": indices, "scores_i": scores_i,
            "q_abs": q_abs, "q_rope": q_rope, **att}


def dense_decode_step(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    pos_q: int,
    freqs: torch.Tensor,
    scale: float = SOFTMAX_SCALE,
    core: str = "bmm",
) -> dict[str, torch.Tensor]:
    """dense 全量 decode 步（无索引器；同一份投影与核）。"""
    q_nope, q_rope_raw = q_projection_stage(x, w)
    q_rope = apply_rope(q_rope_raw, torch.tensor([pos_q], dtype=torch.float32), freqs)
    q_abs = absorbed_query(q_nope, w)
    core_fn = dense_attention_absorbed_bmm if core == "bmm" else dense_attention_absorbed_einsum
    att = core_fn(q_abs, q_rope, c_kv, k_rope, w, scale)
    out = att["out_h"].reshape(1, O_IN) @ w["W_O"]
    return {"out": out, "q_abs": q_abs, "q_rope": q_rope, **att}


def given_set_decode_step(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    pos_q: int,
    freqs: torch.Tensor,
    scale: float = SOFTMAX_SCALE,
    core: str = "einsum",
) -> dict[str, torch.Tensor]:
    """给定索引集的 decode 步（§9 正确性主对象：跳过索引器与 top-k）。"""
    q_nope, q_rope_raw = q_projection_stage(x, w)
    q_rope = apply_rope(q_rope_raw, torch.tensor([pos_q], dtype=torch.float32), freqs)
    q_abs = absorbed_query(q_nope, w)
    core_fn = sparse_gather_attention_einsum if core == "einsum" else sparse_gather_attention_bmm
    att = core_fn(q_abs, q_rope, c_kv, k_rope, indices, w, scale)
    out = att["out_h"].reshape(1, O_IN) @ w["W_O"]
    return {"out": out, "q_abs": q_abs, "q_rope": q_rope, **att}


# ---------------------------------------------------------------------------
# planted relevance 合成数据
# ---------------------------------------------------------------------------

def plant_directions(x: torch.Tensor, w: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """planted 相关性的两个注入方向（一次计算，与 N 无关）：

    v_latent [512]：等化方向 v* = Q^T (Q Q^T)^{-1} 1（Q = q_abs [16,512]，FP64 解）。
      性质：q_abs[h]·v* = 1 对所有 h 精确成立（最小范数解）——单一 latent 方向
      给全部 16 头"完全相同"的分数抬升 γ·1。这消除了头间投影离散（各头 query
      经共享 x 相关，朴素 Σq̂ 方向的头间投影会散开 ±33%，质量分层被高 z 头
      主导——开发期实测教训）。γ_tier = z_tier·σ_s 即每头的精确 z。
    d_index [128]：normalize(Σ_{w_h>0} w_h q̂^I_h)——只用正权重头（ReLU 截负），
      索引分数是头加权和的单标量，无头间结构问题。"""
    q_nope, _ = q_projection_stage(x, w)
    q_abs = absorbed_query(q_nope, w).double()                      # [16,512]
    gram = q_abs @ q_abs.transpose(0, 1)                            # [16,16]
    v_eq = q_abs.transpose(0, 1) @ torch.linalg.solve(gram, torch.ones(q_abs.shape[0], dtype=torch.float64))
    proj = (q_abs @ v_eq)
    assert (proj - 1.0).abs().max() < 1e-6, f"等化方向投影偏差 {float((proj - 1.0).abs().max()):.2e}"
    v_latent = v_eq.float()
    q_a = x @ w["W_QA"]
    q_i = index_query_stage(q_a, w)
    w_used = index_weights_stage(x, w)
    pos = w_used > 0
    d_index = (q_i[pos] * w_used[pos].unsqueeze(1)).sum(dim=0)
    d_index = d_index / d_index.norm()
    return {"v_latent": v_latent, "d_index": d_index,
            "v_latent_norm": float(v_latent.norm()),
            "mean_latent_projection": float(proj.mean())}


def calibrate_plants(
    n: int,
    w: dict[str, torch.Tensor],
    x: torch.Tensor,
    dirs: dict[str, torch.Tensor],
    seed_base: int = SEED_BASE,
    sample_size: int = 2048,
) -> dict[str, Any]:
    """按 z 目标自校准 γ（每 N 一次；采样无 planted 的 base token 测噪声）。

    主 Attention 侧闭合式：score 对 γ 线性（shift = γ·(q_abs[h]·v̂)），
      γ_tier = z_tier · σ_s / mean_h(q_abs[h]·v̂)，
      σ_s = std(原始分数)（latent 点积 + rope 点积的合成噪声，采样估计）。
    索引器侧 ReLU 非线性 → 对 γ_idx 二分求"平均分数抬升 = z·σ_I"。
    采样与 N 无关（噪声分布相同，N 只进 seed）；固定 2048 避免 N<2 时 std=NaN。"""
    m = sample_size
    gen = torch.Generator().manual_seed(seed_base + 3000 + n)
    c_s = torch.randn(m, KV_LORA_RANK, generator=gen)
    kr_s = torch.randn(m, QK_ROPE_HEAD_DIM, generator=gen)
    q_nope, q_rope_raw = q_projection_stage(x, w)
    q_abs = absorbed_query(q_nope, w)
    # 噪声 std（原始分数单位）：rope 段用未旋转样本（旋转保持点积分布）
    dots = q_abs @ c_s.t() + q_rope_raw @ kr_s.t()                  # [16,m]
    sigma_s = float(dots.std())
    mean_proj = dirs["mean_latent_projection"]
    gamma_main = {z: z * sigma_s / mean_proj for z in Z_TIERS}

    q_i = index_query_stage(x @ w["W_QA"], w)
    w_used = index_weights_stage(x, w)
    k_i_s = index_k_projection(c_s, w)
    base_i = index_scores(q_i, w_used, k_i_s)
    sigma_i = float(base_i.std())
    target = {z: float(base_i.mean()) + z * sigma_i for z in Z_TIERS_INDEX}
    d = dirs["d_index"]

    def planted_mean(gamma: float) -> float:
        k_planted = k_i_s + gamma * d.unsqueeze(0)
        return float(index_scores(q_i, w_used, k_planted).mean())

    gamma_index = {}
    for z, tgt in target.items():
        lo, hi = 0.0, 1.0
        while planted_mean(hi) < tgt:
            hi *= 2.0
            if hi > 1e6:
                break
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if planted_mean(mid) < tgt:
                lo = mid
            else:
                hi = mid
        gamma_index[z] = 0.5 * (lo + hi)
    return {
        "sigma_s_raw": sigma_s,
        "sigma_index": sigma_i,
        "mean_latent_projection": mean_proj,
        "gamma_main": gamma_main,
        "gamma_index": gamma_index,
        "sample_size": m,
        "v_latent": dirs["v_latent"],
        "d_index": dirs["d_index"],
    }


def relevance_ids(n: int, seed_base: int = SEED_BASE) -> torch.Tensor:
    """relevance 支撑集：randperm(n) 前 S(n) 个（确定性，跨机器可重建）。

    档位：前 |strong| 个 strong，接下来 medium，其余 weak（tier_sizes）。"""
    gen = torch.Generator().manual_seed(seed_base + 2000 + n)
    perm = torch.randperm(n, generator=gen)
    strong, medium, _ = tier_sizes(n)
    count = min(relevance_support(n), n)
    ids = perm[:count]
    tiers = torch.empty(count, dtype=torch.int8)   # 0=strong 1=medium 2=weak
    ns, nm = min(strong, count), min(strong + medium, count)
    tiers[:ns] = 0
    tiers[ns:nm] = 1
    tiers[nm:] = 2
    return ids, tiers


def generate_cache(
    n: int,
    w: dict[str, torch.Tensor],
    x: torch.Tensor,
    freqs: torch.Tensor,
    calib: dict[str, float] | None = None,
    seed_base: int = SEED_BASE,
) -> dict[str, torch.Tensor]:
    """生成 N 个 token 的合成 cache（fixtures 与 benchmark 运行时同规则）：

    c_kv = randn(N,512)；relevance token 按档位加 γ_main(z)·v̂
    k_rope_raw = randn(N,64)，k_rope_rot = apply_rope(raw, positions)
    k_I = c_kv @ W_KL；relevance token 按档位加 γ_index(z)·d̂（索引空间注入）"""
    if calib is None:
        calib = calibrate_plants(n, w, x, plant_directions(x, w), seed_base)
    gen = torch.Generator().manual_seed(seed_base + 1000 + n)
    c_kv = torch.randn(n, KV_LORA_RANK, generator=gen)
    k_rope_raw = torch.randn(n, QK_ROPE_HEAD_DIM, generator=gen)
    ids, tiers = relevance_ids(n, seed_base)
    v = calib["v_latent"]
    d = calib["d_index"]
    # 档位按位置配对：Z_TIERS[t]（主分数）↔ Z_TIERS_INDEX[t]（索引分数），
    # tiers 编码 0=strong 1=medium 2=weak。
    for t, z_main in enumerate(Z_TIERS):
        mask = tiers == t
        if mask.any():
            c_kv[ids[mask]] += calib["gamma_main"][z_main] * v
    k_i = index_k_projection(c_kv, w)
    for t, z_idx in enumerate(Z_TIERS_INDEX):
        mask = tiers == t
        if mask.any():
            k_i[ids[mask]] += calib["gamma_index"][z_idx] * d
    positions = torch.arange(n, dtype=torch.float32)
    k_rope_rot = apply_rope(k_rope_raw, positions, freqs)
    return {
        "c_kv": c_kv, "k_rope_raw": k_rope_raw, "k_rope_rot": k_rope_rot,
        "k_i": k_i, "positions": positions,
        "relevance_ids": ids, "relevance_tiers": tiers,
        "calib": {kk: vv for kk, vv in calib.items() if kk not in ("v_latent", "d_index")},
    }


# ---------------------------------------------------------------------------
# 检索质量统计（三元曲线的上下文指标）
# ---------------------------------------------------------------------------

def index_quality(
    scores_full: torch.Tensor,   # [H,N] dense 主分数（fp32/fp64）
    indices: torch.Tensor,       # [k'] top-k 选中（int64）
    relevance: torch.Tensor,     # [S] relevance ids
) -> dict[str, float]:
    """检索质量：attention mass 捕获率（选中集上的平均每头质量）、
    shared-set oracle（按总质量排名的 top-k 上界）、relevance 召回率。"""
    attn = torch.softmax(scores_full, dim=-1)                     # [H,N]
    k_prime = indices.numel()
    captured = float(attn[:, indices].sum(dim=1).mean()) if k_prime else 0.0
    if k_prime:
        mass_total = attn.sum(dim=0)                              # [N]
        oracle = torch.topk(mass_total, min(k_prime, mass_total.numel())).indices
        oracle_mass = float(attn[:, oracle].sum(dim=1).mean())
        per_head = attn[:, indices].sum(dim=1)
        min_head_mass = float(per_head.min())
    else:
        oracle_mass = 0.0
        min_head_mass = 0.0
    rel_set = set(relevance.tolist())
    sel_set = set(indices.tolist())
    recall = len(rel_set & sel_set) / max(len(rel_set), 1)
    return {
        "k_selected": int(k_prime),
        "captured_mass_mean": captured,
        "captured_mass_min_head": min_head_mass,
        "oracle_mass_shared_topk": oracle_mass,
        "relevance_recall": recall,
        "relevance_size": len(rel_set),
    }


def describe() -> str:
    return "dsa_reference: indexer + topk + sparse gather attention (absorbed core) + planted relevance"


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import dsa_config as cfg  # noqa: E402

    w = make_weights()
    x = torch.randn(1, cfg.D_MODEL, generator=torch.Generator().manual_seed(SEED_BASE + 1))
    freqs = rope_freqs()
    n = 4096
    t0 = time.perf_counter()
    dirs = plant_directions(x, w)
    calib = calibrate_plants(n, w, x, dirs)
    cache = generate_cache(n, w, x, freqs, calib)
    print(f"[gen] N={n} 用时 {time.perf_counter() - t0:.1f}s")
    print(f"[gen] gamma_main={ {z: round(v, 2) for z, v in calib['gamma_main'].items()} }")
    print(f"[gen] gamma_index={ {z: round(v, 2) for z, v in calib['gamma_index'].items()} }")
    q = dsa_decode_step(x, w, cache["c_kv"], cache["k_rope_rot"], cache["k_i"], 512, n, freqs)
    d = dense_decode_step(x, w, cache["c_kv"], cache["k_rope_rot"], n, freqs)
    qual = index_quality(d["scores"], q["indices"], cache["relevance_ids"])
    err = (q["out"] - d["out"]).abs().max().item()
    print(f"[step] k=512: sparse-vs-dense out max={err:.4f} "
          f"captured={qual['captured_mass_mean']:.4f} oracle={qual['oracle_mass_shared_topk']:.4f} "
          f"recall={qual['relevance_recall']:.3f}")
