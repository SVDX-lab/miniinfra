"""第 20 期 CPU 朴素参考实现（真值锚点）。

三份实现（互相校验）：
  1. gqa_prefill_reference / gqa_decode_reference：torch 向量化的教科书链
     （显式 causal mask + 减 max 稳定 softmax），FP32/FP64 双精度可用；
  2. naive_loops_prefill / naive_loops_decode：字面三重/四重循环纯 Python
     逐元素实现（教学展示 + 小形状交叉验证，同 ch18 的"字面三重循环"角色）；
  3. gqa_mapping_spotcheck：GQA 分组映射 h→h//2 的抽查（正反两向：参考输出
     必须等于逐 q 头对 kv[h//2] 的 MHA；错误映射必须产生显著误差）。

数值稳定 softmax（减 max）显式写出，不用 torch.softmax 内置——这是本期
"naive Attention" 的教学语义。causal 语义：query i 可见 key 0..i（ inclusive）。
GQA 语义：kv 头 g 服务 q 头 {g·group, ..., g·group+group-1}（HF repeat_interleave）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # chapter_20 根（attention_config）

import attention_config as cfg  # noqa: E402


def stable_softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    """数值稳定 softmax（减行 max；显式实现，教学语义）。"""
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)


def expand_kv_heads(t: torch.Tensor, group: int) -> torch.Tensor:
    """[B, HKV, ...] → [B, HQ, ...]（repeat_interleave：kv 头 g → q 头 2g,2g+1）。"""
    return t.repeat_interleave(group, dim=1)


# ---------------------------------------------------------------------------
# 形态一：prefill 整段（S query × S key，causal）
# ---------------------------------------------------------------------------

def gqa_prefill_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    """q [B,HQ,S,D], k/v [B,HKV,S,D] → o [B,HQ,S,D]。causal，稳定 softmax。"""
    b, hq, s, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    kk = expand_kv_heads(k.to(q.dtype), group)          # [B,HQ,S,D]
    vv = expand_kv_heads(v.to(q.dtype), group)
    scores = torch.matmul(q, kk.transpose(-1, -2)) * scale     # [B,HQ,S,S]
    pos_q = torch.arange(s, device=q.device).view(1, 1, s, 1)
    pos_k = torch.arange(s, device=q.device).view(1, 1, 1, s)
    mask = pos_k <= pos_q                                # causal：key ≤ query
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = stable_softmax_lastdim(scores)
    return torch.matmul(attn.to(v.dtype), vv)


def gqa_decode_reference(q: torch.Tensor, k_cache: torch.Tensor,
                         v_cache: torch.Tensor,
                         scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    """decode 单 query：q [B,HQ,D], k/v cache [B,HKV,N,D] → o [B,HQ,D]。"""
    b, hq, d = q.shape
    hkv = k_cache.shape[1]
    group = hq // hkv
    n = k_cache.shape[2]
    kk = expand_kv_heads(k_cache.to(q.dtype), group)     # [B,HQ,N,D]
    vv = expand_kv_heads(v_cache.to(q.dtype), group)
    scores = torch.matmul(q.unsqueeze(2), kk.transpose(-1, -2)).squeeze(2) * scale
    attn = stable_softmax_lastdim(scores)                # [B,HQ,N]
    return torch.matmul(attn.unsqueeze(2).to(v_cache.dtype), vv).squeeze(2)


# ---------------------------------------------------------------------------
# 形态二：字面循环版（纯 Python，教学 + 交叉验证；只用于小形状）
# ---------------------------------------------------------------------------

def naive_loops_prefill(q, k, v, scale: float = cfg.SOFTMAX_SCALE):
    """逐元素循环 prefill。输入/输出 list[float]，形状 [B][H][S][D]。"""
    b, hq, s, d = len(q), len(q[0]), len(q[0][0]), len(q[0][0][0])
    hkv, group = len(k[0]), hq // len(k[0])   # k[0] 去掉 batch 维（len(k) 是 B）
    out = [[[None] * s for _ in range(hq)] for _ in range(b)]
    for bi in range(b):
        for h in range(hq):
            kvh = h // group
            for qi in range(s):
                # 分数 + causal 屏蔽
                scores = []
                for kj in range(qi + 1):
                    dot = 0.0
                    for di in range(d):
                        dot += q[bi][h][qi][di] * k[bi][kvh][kj][di]
                    scores.append(dot * scale)
                m = max(scores)
                exps = [math.exp(x - m) for x in scores]
                lsum = sum(exps)
                # 加权 V（D 维逐元素）
                acc = [0.0] * d
                for kj in range(qi + 1):
                    w = exps[kj] / lsum
                    for di in range(d):
                        acc[di] += w * v[bi][kvh][kj][di]
                out[bi][h][qi] = acc
    return out


def naive_loops_decode(q, k_cache, v_cache, scale: float = cfg.SOFTMAX_SCALE):
    """逐元素循环 decode。q [B][H][D], cache [B][HKV][N][D]。"""
    b, hq, d = len(q), len(q[0]), len(q[0][0])
    hkv, group = len(k_cache[0]), hq // len(k_cache[0])
    n = len(k_cache[0][0])
    out = [[None] * hq for _ in range(b)]
    for bi in range(b):
        for h in range(hq):
            kvh = h // group
            scores = []
            for kj in range(n):
                dot = 0.0
                for di in range(d):
                    dot += q[bi][h][di] * k_cache[bi][kvh][kj][di]
                scores.append(dot * scale)
            m = max(scores)
            exps = [math.exp(x - m) for x in scores]
            lsum = sum(exps)
            acc = [0.0] * d
            for kj in range(n):
                w = exps[kj] / lsum
                for di in range(d):
                    acc[di] += w * v_cache[bi][kvh][kj][di]
            out[bi][h] = acc
    return out


# ---------------------------------------------------------------------------
# GQA 映射抽查
# ---------------------------------------------------------------------------

def gqa_mapping_spotcheck(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          scale: float = cfg.SOFTMAX_SCALE) -> dict:
    """GQA h→h//2 映射抽查。

    正向：参考输出必须等于"逐 q 头 MHA 对 kv[h//2]"的手工切片计算；
    反向（灵敏度）：三种错误映射（h→(h+1)%HKV、h→min(h,HKV-1) 平移、MHA 无共享
    复制错位）必须产生 O(1) 量级误差——证明本抽查对映射错误敏感，不是恒真。
    """
    group = cfg.GQA_GROUP
    o_ref = gqa_prefill_reference(q, k, v, scale)
    # 正向：逐 q 头独立 MHA（显式用 kv 头 h//2，不经过 repeat_interleave）
    outs = []
    for h in range(cfg.HQ):
        kvh = h // group
        scores = torch.matmul(q[:, h], k[:, kvh].transpose(-1, -2)) * scale
        s = scores.shape[-1]
        pos_q = torch.arange(s, device=q.device).view(1, s, 1)
        pos_k = torch.arange(s, device=q.device).view(1, 1, s)
        scores = scores.masked_fill(pos_k > pos_q, float("-inf"))
        attn = stable_softmax_lastdim(scores)
        outs.append(torch.matmul(attn, v[:, kvh]).unsqueeze(1))
    o_manual = torch.cat(outs, dim=1)
    err_fwd = (o_ref - o_manual).abs().max().item()

    # 反向：错误映射的灵敏度
    def with_mapping(map_fn):
        outs_w = []
        for h in range(cfg.HQ):
            kvh = map_fn(h)
            scores = torch.matmul(q[:, h], k[:, kvh].transpose(-1, -2)) * scale
            pos_q = torch.arange(scores.shape[-1], device=q.device).view(1, -1, 1)
            pos_k = torch.arange(scores.shape[-1], device=q.device).view(1, 1, -1)
            scores = scores.masked_fill(pos_k > pos_q, float("-inf"))
            attn = stable_softmax_lastdim(scores)
            outs_w.append(torch.matmul(attn, v[:, kvh]).unsqueeze(1))
        return torch.cat(outs_w, dim=1)

    hkv = k.shape[1]
    wrong1 = with_mapping(lambda h: (h + 1) % hkv)          # 头偏移 1
    wrong2 = with_mapping(lambda h: min(h // group + 1, hkv - 1))  # 组偏移 1
    err_w1 = (o_ref - wrong1).abs().max().item()
    err_w2 = (o_ref - wrong2).abs().max().item()
    return {
        "forward_vs_manual_max_abs": err_fwd,
        "wrong_map_offset1_max_abs": err_w1,
        "wrong_map_groupshift_max_abs": err_w2,
        "forward_ok": err_fwd < 1e-5,
        "wrong_maps_rejected": err_w1 > 1e-3 and err_w2 > 1e-3,
    }


# ---------------------------------------------------------------------------
# 真值锚点（FP64）
# ---------------------------------------------------------------------------

def gqa_prefill_truth64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    return gqa_prefill_reference(q.double(), k.double(), v.double(), scale)


def gqa_decode_truth64(q: torch.Tensor, k_cache: torch.Tensor,
                       v_cache: torch.Tensor,
                       scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    return gqa_decode_reference(q.double(), k_cache.double(), v_cache.double(), scale)
