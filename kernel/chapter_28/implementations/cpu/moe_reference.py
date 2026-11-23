"""第 28 期 CPU 参考实现：MoE top-k 路由 + 分组 GEMM（sort/gather/scatter）。

三条代码路径（同一语义 y = Σ_{e∈topk(t)} w_te · FFN_e(x_t)）：
1. naive_moe：逐 token 走全量专家再筛选——先算所有 E 个专家的输出，
   再按路由分数降序（slot 序）加权求和。真值锚点（只用于小 N）。
2. grouped_moe：sort/gather/按专家分桶 GEMM/scatter + 加权合并——
   (token,slot) 对按专家 ID 稳定排序 → gather x 行 → 逐专家桶 GEMM →
   index_add 合并（桶 = 专家 ID 升序，桶内 token 升序）。
3. dense_ffn_equivalent：等价容量的单体大 GEMM（intermediate = k×d_ff），
   与 MoE 专家段 FLOPs 相同——分组化代价的分母。

确定性（tie-break 规则显式化）：
- 路由选择 = softmax 后对分数做 descending + stable=True 的 argsort，
  分数相等时【专家索引小者胜出】（等价于 (score, index) 字典序）。
- 合并顺序固定：naive 路径按 slot（分数降序）求和；grouped 路径按
  专家桶升序 index_add。两条路径的浮点结合律差异在实验 1 显式记录（接 ch08）。

FP32/FP64 双锚点：所有函数 dtype 透明（由传入张量决定）；权重生成用
He 缩放（fan_in = 收缩维，ep26 修正法）。
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# 权重/输入生成（固定 seed + He 缩放，ep26 同法）
# ---------------------------------------------------------------------------


def _he(shape: list[int], fan_in: int, gen: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    """randn × 1/sqrt(fan_in)：softmax 单位下激活噪声 std≈1（ep25 教训①）。"""
    return torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype) / (fan_in ** 0.5)


def make_expert_weights(n_experts: int, d_model: int, d_ff: int, seed: int,
                        dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """专家权重：W_gate/W_up [E, d_ff, d]（fan_in=d）、W_down [E, d, d_ff]（fan_in=d_ff）。"""
    gen = torch.Generator().manual_seed(seed)
    return {
        "W_gate": _he([n_experts, d_ff, d_model], d_model, gen, dtype),
        "W_up": _he([n_experts, d_ff, d_model], d_model, gen, dtype),
        "W_down": _he([n_experts, d_model, d_ff], d_ff, gen, dtype),
    }


def make_router(n_experts: int, d_model: int, seed: int,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """路由矩阵 [E, d]（fan_in=d）。"""
    gen = torch.Generator().manual_seed(seed + 7)
    return _he([n_experts, d_model], d_model, gen, dtype)


def make_inputs(n_tokens: int, d_model: int, seed: int,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + 13)
    return torch.randn(n_tokens, d_model, generator=gen, dtype=torch.float32).to(dtype)


# ---------------------------------------------------------------------------
# 路由：gate logits → softmax → top-k（stable tie-break）→ 权重归一化
# ---------------------------------------------------------------------------


def gate_logits(x: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
    """[N, d] @ [E, d]^T → [N, E] 路由分数。维度不匹配显式报错。"""
    if x.shape[-1] != router.shape[1]:
        raise ValueError(
            f"x 最后一维 {x.shape[-1]} 与 router fan_in {router.shape[1]} 不匹配")
    return x @ router.T


def route_topk(logits: torch.Tensor, k: int, norm: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax 后 top-k 选择 + 权重归一化。

    tie-break：descending + stable argsort ⇒ 分数相同时专家索引小者排前。
    norm=True（Qwen3 口径）：w /= Σw（top-k 内归一化，Σ=1）；
    norm=False（V2-Lite 口径声明）：直接用 softmax 截断值。
    返回 (indices [N, k] long, weights [N, k] 同 dtype)。
    """
    if k < 1 or k > logits.shape[1]:
        raise ValueError(f"topk={k} 超出专家数 {logits.shape[1]}")
    probs = torch.softmax(logits, dim=-1)
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    idx = order[:, :k]
    w = torch.take_along_dim(probs, idx, dim=-1)
    if norm:
        w = w / w.sum(dim=-1, keepdim=True)
    return idx, w


def route(x: torch.Tensor, router: torch.Tensor, k: int, norm: bool = True):
    logits = gate_logits(x, router)
    return route_topk(logits, k, norm)


# ---------------------------------------------------------------------------
# 单专家 FFN（SiLU·gate ⊙ up → down；与主流 MoE 实现同构）
# ---------------------------------------------------------------------------


def silu(t: torch.Tensor) -> torch.Tensor:
    return t / (1.0 + torch.exp(-t))


def expert_ffn(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor,
               W_down: torch.Tensor) -> torch.Tensor:
    """x [M, d] → [M, d_ff] = silu(x@Wg^T) ⊙ (x@Wu^T) → [M, d] = · @ Wd^T。"""
    h = silu(x @ W_gate.T) * (x @ W_up.T)
    return h @ W_down.T


# ---------------------------------------------------------------------------
# 路径 1：naive——逐 token 全量专家再筛选（真值锚点，小 N 专用）
# ---------------------------------------------------------------------------


def naive_moe(x: torch.Tensor, weights: dict[str, torch.Tensor], router: torch.Tensor,
              k: int, norm: bool = True) -> torch.Tensor:
    """所有 E 专家输出 [E, N, d] 后按 slot（分数降序）加权求和。"""
    idx, w = route(x, router, k, norm)
    return naive_moe_pairs(x, weights, idx, w)


def naive_moe_pairs(x: torch.Tensor, weights: dict[str, torch.Tensor],
                    idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """给定路由选择的全量专家锚点：E 个专家对同一批 x 各算一遍再筛选。"""
    n = x.shape[0]
    experts = weights["W_gate"].shape[0]
    outs = torch.stack(
        [expert_ffn(x, weights["W_gate"][e], weights["W_up"][e], weights["W_down"][e])
         for e in range(experts)],
        dim=0,
    )  # [E, N, d]
    gathered = outs[idx, torch.arange(n, device=x.device).unsqueeze(1), :]  # [N, k, d]
    return (gathered * w.unsqueeze(-1)).sum(dim=1)  # slot 序求和


# ---------------------------------------------------------------------------
# 路径 2：grouped——sort/gather/分桶 GEMM/scatter + 加权合并
# ---------------------------------------------------------------------------


def _pairs_with_weights(idx: torch.Tensor, w: torch.Tensor):
    """[N, k] 选择 + 权重展开为 (token, expert) 对 + 稳定排序。

    perm = argsort(pair_expert, stable=True)：桶 = 专家 ID 升序、
    桶内 token 升序（与原始 pair 展开顺序一致 → 确定性）。
    返回 (pair_token, pair_expert, pair_w)。
    """
    n, k = idx.shape
    flat_expert = idx.reshape(-1)
    flat_token = torch.arange(n, device=idx.device).repeat_interleave(k)
    flat_w = w.reshape(-1)
    perm = torch.argsort(flat_expert, stable=True)
    return flat_token[perm], flat_expert[perm], flat_w[perm]


def grouped_moe(x: torch.Tensor, weights: dict[str, torch.Tensor], router: torch.Tensor,
                k: int, norm: bool = True, return_buckets: bool = False):
    """分组 GEMM 前向。return_buckets=True 时附带桶结构（实验 3 用）。"""
    idx, w = route(x, router, k, norm)
    return grouped_moe_pairs(x, weights, idx, w, return_buckets=return_buckets)


def grouped_moe_pairs(x: torch.Tensor, weights: dict[str, torch.Tensor],
                      idx: torch.Tensor, w: torch.Tensor, return_buckets: bool = False):
    """给定路由选择的分组 GEMM（负载不均实验直接注入分布用）。"""
    n, d = x.shape
    experts = weights["W_gate"].shape[0]
    pair_token, pair_expert, pair_w = _pairs_with_weights(idx, w)

    pair_x = x[pair_token]  # gather
    counts = torch.bincount(pair_expert, minlength=experts)
    offsets = torch.cumsum(counts, dim=0)
    y = torch.zeros_like(x)
    depth_max = int(counts.max().item()) if pair_token.numel() else 0
    for e in range(experts):
        s = int(offsets[e - 1].item()) if e > 0 else 0
        t = int(offsets[e].item())
        if s == t:
            continue  # 空桶：该专家本批无 token
        xe = pair_x[s:t]
        ye = expert_ffn(xe, weights["W_gate"][e], weights["W_up"][e], weights["W_down"][e])
        y.index_add_(0, pair_token[s:t], ye * pair_w[s:t].unsqueeze(-1))
    if return_buckets:
        meta = {
            "counts": counts, "offsets": offsets, "depth_max": depth_max,
            "depth_mean": float(counts.sum().item()) / experts,
            "n_empty": int((counts == 0).sum().item()),
            "pair_token": pair_token, "pair_expert": pair_expert, "pair_w": pair_w,
        }
        return y, meta
    return y


def grouped_moe_profiled(x: torch.Tensor, weights: dict[str, torch.Tensor],
                         router: torch.Tensor, k: int, norm: bool = True):
    """带分段计时的 grouped 前向（单遍 wall-clock 分段，perf_counter，返回秒）。

    分段：gate_mm / topk_norm / sort_pairs / gather_x / expert_gate_up /
    silu_mul / expert_down / scatter_combine。
    routing = 前四段 + scatter_combine；expert = 三个专家段。
    """
    import time

    seg: dict[str, float] = {}
    t0 = time.perf_counter()
    logits = x @ router.T
    seg["gate_mm"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs = torch.softmax(logits, dim=-1)
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    idx = order[:, :k]
    w = torch.take_along_dim(probs, idx, dim=-1)
    if norm:
        w = w / w.sum(dim=-1, keepdim=True)
    seg["topk_norm"] = time.perf_counter() - t0

    return grouped_moe_profiled_pairs(x, weights, idx, w, seg=seg)


def grouped_moe_profiled_pairs(x: torch.Tensor, weights: dict[str, torch.Tensor],
                               idx: torch.Tensor, w: torch.Tensor,
                               seg: dict[str, float] | None = None):
    """给定路由选择的分段计时（grouped_moe_profiled 的注入式变体）。

    seg 已含 gate_mm/topk_norm 时继续累加其余段；否则从 sort_pairs 起测。
    返回 (y, seg)（seg 单位 = 秒）。
    """
    import time

    n, k = idx.shape
    d = x.shape[1]
    experts = weights["W_gate"].shape[0]
    if seg is None:
        seg = {"gate_mm": 0.0, "topk_norm": 0.0}
    seg.setdefault("sort_pairs", 0.0)
    seg.setdefault("gather_x", 0.0)
    seg.setdefault("expert_gate_up", 0.0)
    seg.setdefault("silu_mul", 0.0)
    seg.setdefault("expert_down", 0.0)
    seg.setdefault("scatter_combine", 0.0)

    t0 = time.perf_counter()
    flat_expert = idx.reshape(-1)
    flat_token = torch.arange(n, device=x.device).repeat_interleave(k)
    perm = torch.argsort(flat_expert, stable=True)
    pair_expert = flat_expert[perm]
    pair_token = flat_token[perm]
    pair_w = w.reshape(-1)[perm]
    counts = torch.bincount(pair_expert, minlength=experts)
    offsets = torch.cumsum(counts, dim=0)
    seg["sort_pairs"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    pair_x = x[pair_token]
    seg["gather_x"] += time.perf_counter() - t0

    h = torch.empty(pair_x.shape[0], weights["W_gate"].shape[1], dtype=x.dtype,
                    device=x.device)  # 显式 device（M-50：MUSA 对设备缺省零宽容）
    t0 = time.perf_counter()
    gate_out = torch.empty_like(h)
    for e in range(experts):
        s = int(offsets[e - 1].item()) if e > 0 else 0
        t = int(offsets[e].item())
        if s == t:
            continue
        torch.mm(pair_x[s:t], weights["W_gate"][e].T, out=gate_out[s:t])
        torch.mm(pair_x[s:t], weights["W_up"][e].T, out=h[s:t])
    seg["expert_gate_up"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    act = torch.nn.functional.silu(gate_out) * h
    seg["silu_mul"] += time.perf_counter() - t0

    down_out = torch.empty(pair_x.shape[0], d, dtype=x.dtype, device=x.device)
    t0 = time.perf_counter()
    for e in range(experts):
        s = int(offsets[e - 1].item()) if e > 0 else 0
        t = int(offsets[e].item())
        if s == t:
            continue
        torch.mm(act[s:t], weights["W_down"][e].T, out=down_out[s:t])
    seg["expert_down"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    y = torch.zeros_like(x)
    y.index_add_(0, pair_token, down_out * pair_w.unsqueeze(-1))
    seg["scatter_combine"] += time.perf_counter() - t0

    return y, seg


# ---------------------------------------------------------------------------
# 路径 3：等价容量单体大 GEMM（分组的分母）
# ---------------------------------------------------------------------------


def dense_ffn_equivalent(x: torch.Tensor, d_ff_total: int, seed: int,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """单体 FFN，intermediate = k×d_ff（与 MoE 专家段同 FLOPs）。

    权重现场生成（seed 固定；仅作延迟分母，不与 MoE 输出对数值）。
    """
    d = x.shape[1]
    gen = torch.Generator().manual_seed(seed)
    Wg = _he([d_ff_total, d], d, gen, x.dtype)
    Wu = _he([d_ff_total, d], d, gen, x.dtype)
    Wd = _he([d, d_ff_total], d_ff_total, gen, x.dtype)
    return (silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T


# ---------------------------------------------------------------------------
# FLOPs / 字节核算（整数）
# ---------------------------------------------------------------------------


def moe_flop_counts(n_tokens: int, n_experts: int, topk: int, d_model: int, d_ff: int):
    """分组路径的整数 MAC/FLOP 核算（2 FLOP/MAC 惯例）。"""
    pairs = n_tokens * topk
    mac_gate_up = pairs * 2 * d_model * d_ff  # gate + up 两投影
    mac_down = pairs * d_model * d_ff
    return {
        "pairs": pairs,
        "mac_gate_up": mac_gate_up,
        "mac_down": mac_down,
        "mac_total": mac_gate_up + mac_down,
        "flops_total": 2 * (mac_gate_up + mac_down),
        "routing_pairs_bytes_fp32": pairs * (4 * 3 + 8),  # expert/token/weight(float32)+perm
    }


def dense_flop_counts(n_tokens: int, d_model: int, d_ff_total: int):
    mac = n_tokens * 3 * d_model * d_ff_total
    return {"mac_total": mac, "flops_total": 2 * mac}


def expert_weight_bytes(n_experts: int, d_model: int, d_ff: int, dtype_bytes: int | float) -> int:
    return round(n_experts * 3 * d_model * d_ff * dtype_bytes)


def activated_weight_bytes(topk: int, d_model: int, d_ff: int, dtype_bytes: int | float) -> int:
    return round(topk * 3 * d_model * d_ff * dtype_bytes)


# ---------------------------------------------------------------------------
# 负载不均（planted Zipf）构造：直接构造每 token 的路由选择（ep26 注入法同思想）
# ---------------------------------------------------------------------------


def make_zipf_selection(n_tokens: int, n_experts: int, topk: int, seed: int,
                        hot_frac: float = 0.2, token_frac: float = 0.8,
                        alpha: float = 1.0, hot_full_k: bool = False):
    """构造倾斜分布的 (idx, w) + 统计。

    top-1 分布 = token_frac×Zipf(H 热专家, p∝1/(i+1)^α) + (1−token_frac)×均匀(E)；
    其余 k-1 专家：hot_full_k=False 时均匀不放回抽样（全池），True 时从热集内
    均匀补齐（若 k>H 则从全池补）。logits 按 rank 保持方式合成，再过 softmax——
    选择 = 构造意图，权重 = 真实 softmax 截断。
    返回 (idx, w, stats)。
    """
    n_hot = max(1, int(round(hot_frac * n_experts)))
    gen = torch.Generator().manual_seed(seed)
    hot_p = 1.0 / torch.arange(1, n_hot + 1, dtype=torch.float64) ** alpha
    hot_p /= hot_p.sum()

    idx = torch.zeros(n_tokens, topk, dtype=torch.long)
    logits = torch.zeros(n_tokens, n_experts, dtype=torch.float32)
    top1_hot = 0
    hot_access = 0
    total_access = 0
    for t in range(n_tokens):
        u = torch.rand(1, generator=gen).item()
        if u < token_frac:
            top1 = int(torch.multinomial(hot_p.float(), 1, generator=gen).item())
            top1_hot += 1
        else:
            top1 = int(torch.randint(n_experts, (1,), generator=gen).item())
        if hot_full_k and u < token_frac and n_hot >= topk:
            hot_set = list(range(n_hot))
            if top1 not in hot_set:
                sel = [top1]
                rest_pool = [e for e in hot_set if e != top1]
                rest = [rest_pool[j] for j in torch.randperm(len(rest_pool), generator=gen)[: topk - 1].tolist()]
                sel += rest
            else:
                rest_pool = [e for e in hot_set if e != top1]
                rest = [rest_pool[j] for j in torch.randperm(len(rest_pool), generator=gen)[: topk - 1].tolist()]
                sel = [top1] + rest
            hot_access += topk
        else:
            rest_pool = [e for e in range(n_experts) if e != top1]
            rest = [rest_pool[j] for j in torch.randperm(len(rest_pool), generator=gen)[: topk - 1].tolist()]
            sel = [top1] + rest
            hot_access += 1 if (top1 < n_hot) else 0
        total_access += topk
        for j, e in enumerate(sel):
            idx[t, j] = e
        row = torch.randn(n_experts, generator=gen)  # 未选中：N(0,1)
        sel_sorted = sorted(sel)
        for rank, e in enumerate(sel_sorted):
            if e == top1:
                row[e] = 8.0
            else:
                row[e] = 4.0 + 0.8 * (len(sel_sorted) - rank) + 0.1 * torch.rand(1, generator=gen).item()
        logits[t] = row
    w = torch.softmax(logits, dim=-1)
    w = torch.take_along_dim(w, idx, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    stats = {"n_hot": n_hot, "hot_share_measured": top1_hot / n_tokens,
             "hot_access_share": hot_access / total_access}
    return idx, w, stats


def describe() -> str:
    return ("moe_reference: top-k routing (softmax+stable topk+norm) + grouped GEMM "
            "(sort/gather/bucket/scatter) + naive all-expert anchor + dense equivalent; "
            "FP32/FP64 dual anchors; explicit tie-break (lower expert index wins)")
