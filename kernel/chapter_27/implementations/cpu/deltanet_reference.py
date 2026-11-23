"""第 27 期 CPU 参考实现（真值锚点）：Gated DeltaNet 逐 Token 递推形式。

算术定义（与 FLA fla/ops/gated_delta_rule/naive.py
naive_recurrent_gated_delta_rule 逐行同构，2026-11-21 抓取；状态布局 [H,K,V]）：

    S_t = exp(g_t) ⊙ S_{t-1}                      # 衰减（标量 g 按头、向量 g 按 K 行）
    u_t = β_t · (v_t − S_t^decayed ᵀ k_t)          # delta 规则：用"衰减后状态重构 v"
    S_t = S_t^decayed + k_t u_tᵀ                  # 秩-1 写入
    o_t = scale · q_tᵀ S_t                        # 读出（scale = 1/sqrt(K)，FLA 默认）

与任务书形式 S_t = α_t·S_{t-1} + β_t·k_t·(v_t − S_{t-1}ᵀk_t)ᵀ 的对应：α_t = exp(g_t)，
FLA 顺序为先衰减后 delta（delta 作用在衰减后状态上），两者在 α=1 或无 delta 时
重合；本期以 FLA 顺序为准并在 smoke 中用 α=1（g=0）校验退化一致性。

精度策略（ep25/26 口径）：
  - 权重/输入 FP32 合成（固定 seed + He 缩放）；
  - FP32 主链：递推在 FP32 下逐步计算（真值锚点）；
  - FP64 代数锚点：同一份 FP32 权重/输入 cast 到 float64 再递推；
  - FP16 路径：权重/输入 cast 到 float16 后递推（记录容差，不作真值）。

门控合成口径（出处见 deltanet_config.py 头注）：
  g_log_t,h = −exp(A_log_h) · softplus(g_raw_t,h + dt_bias_h)，g_log ≤ 0 恒成立
  （softplus ≥ 0 且 A > 0），保证衰减矩阵 ≤ 1、chunk 内 cumsum 单调不增——
  这是 chunked 形式数值安全（fp16 不上溢）的结构性前提。
  scalar 变体：g_raw = x @ W_GATE（[T,16]）；
  vector 变体：g_raw = (x @ W_GATE_DIM_A @ W_GATE_DIM_B)（[T,16,128] 低秩，
  秩 R=64，KDA 式 per-K-通道衰减；具体秩未从公开材料确证——合成设定）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # chapter_27/

import deltanet_config as cfg  # noqa: E402

WEIGHT_KEYS = (
    "W_Q", "W_K", "W_V",           # [2048, 2048] = 16 头 × 128
    "W_BETA",                      # [2048, 16]  b_proj（sigmoid 前的 logits）
    "W_GATE",                      # [2048, 16]  a_proj（标量门控）
    "W_GATE_DIM_A",                # [2048, 64]  per-dim 门控低秩 A
    "W_GATE_DIM_B",                # [64, 2048]  per-dim 门控低秩 B（16×128 展平）
)
WEIGHT_SHAPES = {
    "W_Q": (cfg.D_MODEL, cfg.NUM_KEY_HEADS * cfg.HEAD_K_DIM),
    "W_K": (cfg.D_MODEL, cfg.NUM_KEY_HEADS * cfg.HEAD_K_DIM),
    "W_V": (cfg.D_MODEL, cfg.NUM_VALUE_HEADS * cfg.HEAD_V_DIM),
    "W_BETA": (cfg.D_MODEL, cfg.NUM_VALUE_HEADS),
    "W_GATE": (cfg.D_MODEL, cfg.NUM_VALUE_HEADS),
    "W_GATE_DIM_A": (cfg.D_MODEL, 64),
    "W_GATE_DIM_B": (64, cfg.NUM_VALUE_HEADS * cfg.HEAD_K_DIM),
}


def make_weights(seed_base: int = cfg.SEED_BASE) -> dict[str, torch.Tensor]:
    """合成权重：randn × He 缩放（std = 1/sqrt(fan_in)，2-D 用 shape[0]，ep26 同法）。

    ep25 教训：N(0,1) 权重使激活逐层放大/退化；He 缩放使各投影输出 ~O(1)，
    softplus(g_raw+dt_bias) ~ O(1)，每 token 衰减率落在有意义的动态范围。
    """
    weights = {}
    for idx, name in enumerate(sorted(WEIGHT_KEYS)):
        shape = WEIGHT_SHAPES[name]
        fan_in = shape[0]
        gen = torch.Generator().manual_seed(seed_base + idx + 1)
        weights[name] = torch.randn(*shape, generator=gen, dtype=torch.float32) / (fan_in ** 0.5)
    # FLA GatedDeltaNet 初始化：A = uniform(0,16) → A_log；dt ∈ [0.001,0.1]
    # → inverse softplus 得 dt_bias（fla/layers/gated_deltanet.py 逐行同构）
    gen_a = torch.Generator().manual_seed(seed_base + 101)
    a = torch.empty(cfg.NUM_VALUE_HEADS).uniform_(0, cfg.GATE_A_UNIFORM_HI, generator=gen_a)
    weights["A_LOG"] = torch.log(a)
    gen_dt = torch.Generator().manual_seed(seed_base + 102)
    dt = torch.exp(
        torch.rand(cfg.NUM_VALUE_HEADS, generator=gen_dt)
        * (torch.log(torch.tensor(cfg.GATE_DT_MAX)) - torch.log(torch.tensor(cfg.GATE_DT_MIN)))
        + torch.log(torch.tensor(cfg.GATE_DT_MIN))
    ).clamp(min=cfg.GATE_DT_FLOOR)
    weights["DT_BIAS"] = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
    # per-dim 门控的通道级 A_log：合成设定（KDA 公开材料未给出通道 A 分布细节）
    gen_av = torch.Generator().manual_seed(seed_base + 103)
    a_dim = torch.empty(cfg.NUM_VALUE_HEADS * cfg.HEAD_K_DIM).uniform_(
        0, cfg.GATE_A_UNIFORM_HI, generator=gen_av
    )
    weights["A_LOG_DIM"] = torch.log(a_dim).view(cfg.NUM_VALUE_HEADS, cfg.HEAD_K_DIM)
    weights["DT_BIAS_DIM"] = weights["DT_BIAS"].unsqueeze(-1).expand(
        cfg.NUM_VALUE_HEADS, cfg.HEAD_K_DIM
    ).contiguous()
    return weights


def project_inputs(
    x: torch.Tensor, weights: dict[str, torch.Tensor], gate_variant: str = "scalar",
    qk_l2norm: bool = True,
) -> dict[str, torch.Tensor]:
    """x [T,2048] → q/k/v [T,H,K|V]、beta [T,H]、g_log [T,H] 或 [T,H,K]。

    beta = sigmoid(b_proj(x))、g_log = −exp(A_log)·softplus(a_proj(x)+dt_bias)
    （FLA GatedDeltaNet 同构）。
    qk_l2norm=True（FLA 层默认 use_qk_l2norm_in_kernel=True，逐 (t,h) 沿 K
    归一）：这不是可省的装饰——delta 规则迭代矩阵的谱半径含 β‖k‖²，未归一
    的合成 k（‖k‖²≈K）使 β‖k‖²≫1，精确算术下也发散（本期实测，见
    EXPERIMENT_LOG §5）；归一后 β‖k‖²≤β<1 收缩。省略项（不影响递推核
    等价性）：short conv、输出门 g_proj/RMSNormGated。
    """
    t = x.shape[0]
    q = (x @ weights["W_Q"]).view(t, cfg.NUM_KEY_HEADS, cfg.HEAD_K_DIM)
    k = (x @ weights["W_K"]).view(t, cfg.NUM_KEY_HEADS, cfg.HEAD_K_DIM)
    if qk_l2norm:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v = (x @ weights["W_V"]).view(t, cfg.NUM_VALUE_HEADS, cfg.HEAD_V_DIM)
    beta = torch.sigmoid(x @ weights["W_BETA"])
    a_raw = x @ weights["W_GATE"]
    if gate_variant == "scalar":
        g_log = -torch.exp(weights["A_LOG"]).unsqueeze(0) * torch.nn.functional.softplus(
            a_raw + weights["DT_BIAS"].unsqueeze(0)
        )
    elif gate_variant == "vector":
        g_raw = (x @ weights["W_GATE_DIM_A"]) @ weights["W_GATE_DIM_B"]
        g_raw = g_raw.view(t, cfg.NUM_VALUE_HEADS, cfg.HEAD_K_DIM)
        g_log = -torch.exp(weights["A_LOG_DIM"]).unsqueeze(0) * torch.nn.functional.softplus(
            g_raw + weights["DT_BIAS_DIM"].unsqueeze(0)
        )
    else:
        raise ValueError("gate_variant 必须是 scalar 或 vector: " + gate_variant)
    return {"q": q, "k": k, "v": v, "beta": beta, "g_log": g_log}


def recurrent_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_log: torch.Tensor,
    scale: float = cfg.SCALE,
    initial_state: torch.Tensor | None = None,
    return_state_norms: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐 Token 递推（输入 dtype 决定计算 dtype——FP32 主链 / FP64 锚点共用）。

    q/k [T,H,K]，v [T,H,V]，beta [T,H]，g_log [T,H] 或 [T,H,K]，state [H,K,V]。
    返回 (o [T,H,V], S_final [H,K,V])；return_state_norms 时另返回
    norms [T,H]（每 token 每头的 ‖S_t‖_F，门控行为实验用，避免存全状态）。
    """
    t, h, dim_k = k.shape
    dim_v = v.shape[-1]
    state = (
        torch.zeros(h, dim_k, dim_v, dtype=q.dtype, device=q.device) if initial_state is None
        else initial_state.to(device=q.device, dtype=q.dtype).clone()
    )
    q_scaled = q * scale
    g_broadcast = g_log.unsqueeze(-1) if g_log.dim() == 2 else g_log  # [T,H,1] / [T,H,K]
    outputs = torch.empty(t, h, dim_v, dtype=q.dtype, device=q.device)
    norms = torch.empty(t, h, dtype=q.dtype) if return_state_norms else None
    for step in range(t):
        decayed = state * torch.exp(g_broadcast[step]).unsqueeze(-1)   # [H,K,V]
        retrieval = torch.einsum("hkv,hk->hv", decayed, k[step])       # 衰减后状态重构 v
        u = (v[step] - retrieval) * beta[step].unsqueeze(-1)           # delta 误差×写入率
        state = decayed + torch.einsum("hk,hv->hkv", k[step], u)       # 秩-1 写入
        outputs[step] = torch.einsum("hkv,hk->hv", state, q_scaled[step])
        if return_state_norms:
            norms[step] = state.norm(dim=(-2, -1))
    if return_state_norms:
        return outputs, state, norms
    return outputs, state


def recurrent_full_layer(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    gate_variant: str = "scalar",
    initial_state: torch.Tensor | None = None,
    qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """整层入口：x [T,2048] → 投影 → 递推。返回 (o [T,H,V], S_final)。"""
    proj = project_inputs(x, weights, gate_variant, qk_l2norm=qk_l2norm)
    return recurrent_gdn(
        proj["q"], proj["k"], proj["v"], proj["beta"], proj["g_log"],
        initial_state=initial_state,
    )


def make_inputs(
    seq_len: int, seed: int = cfg.SEED_BASE
) -> torch.Tensor:
    """层输入 x [T,2048]：N(0,1) 合成激活（He 缩放投影下各路径输出 ~O(1)）。

    本期全实验单序列口径（batch=1，合同声明——chunked 的并行性按"块间/头间"
    讲，不加 batch 维）。planted 结构不进入主等价实验（等价性只依赖代数）；
    非退化数据要求由检索深度探针实验（gate_probe.py）承担——ep26 relevance
    注入法在那里使用。
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(seq_len, cfg.D_MODEL, generator=gen, dtype=torch.float32)


def to_dtype(tensors: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {name: value.to(dtype) for name, value in tensors.items()}


def state_bytes_per_layer(
    dtype_bytes: int = 2, h: int = cfg.NUM_VALUE_HEADS,
    k: int = cfg.HEAD_K_DIM, v: int = cfg.HEAD_V_DIM,
) -> int:
    """递推状态字节数：H×K×V×dtype_bytes（与序列长度无关——解耦性的核心数字）。"""
    return h * k * v * dtype_bytes
