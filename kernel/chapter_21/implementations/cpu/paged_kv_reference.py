"""第 21 期 CPU 参考实现（真值锚点）：连续 KV 与分页块表两种组织 + 块表语义。

四组角色（对照体系，大纲 §9 行 483 主对）：
  1. contiguous_decode_reference / contiguous_prefill_reference：
     连续布局 [B, HKV, N, D] 的教科书链（显式减 max 稳定 softmax + causal，
     GQA repeat_interleave 语义）——与 ch21 参考同式，本期自含（不跨章 import）。
  2. gather_from_pool：块表语义的"纯 gather"参考——按 paged_config.py 合同的
     token 级映射把分页池取回连续布局。scatter_to_pool 是其精确逆（fixtures
     构造用：逻辑 K/V 连续生成后按表散入池）。
  3. paged_decode_reference / paged_prefill_reference：
     "gather + 连续数学"的分解实现——gather 是精确拷贝，故与 1 在同一逻辑 KV
     下**位级一致**（fp32），这就是"值透明性"断言的 CPU 载体。
  4. validate_block_table：合同校验（越界/空洞/重复物理块/max_blocks 不足），
     显式抛错不静默（第 02 期合同风格）。

FP64 锚点：contiguous_decode_truth64（与 ch21 相同的双锚点口径）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # chapter_21 根（paged_config）

import paged_config as cfg  # noqa: E402


def stable_softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    """数值稳定 softmax（减行 max；显式实现）。"""
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)


def expand_kv_heads(t: torch.Tensor, group: int) -> torch.Tensor:
    """[B, HKV, ...] → [B, HQ, ...]（repeat_interleave：kv 头 g → q 头 2g,2g+1）。"""
    return t.repeat_interleave(group, dim=1)


# ---------------------------------------------------------------------------
# 块表构造与合同校验
# ---------------------------------------------------------------------------

def build_block_table(seq_lens: list[int], block_size: int, num_blocks: int,
                      mode: str, seed: int = 0) -> torch.Tensor:
    """构造 [B, max_blocks] int32 块表（合同见 paged_config.py 模块头）。

    mode:
      "identity" — 恒等块表：B=1 时 p == t（分页布局与连续布局严格同构，
        gather 是恒等拷贝）；B>1 时退化为"请求独占连续物理段"（段基址 =
        前面请求的块数和）——连续布局的批量形态，合同上与 identity 同族；
      "rand"     — 全局随机置换后按请求切片：请求间物理块保证不重叠
        （池合同：一块同一时刻只属一个请求），请求内块序随机（局部性对照组）；
      "per_request_identity" — 同 identity 的 B>1 形态（显式别名）。
    padding 表项 = cfg.TABLE_PAD（-1）。
    """
    b = len(seq_lens)
    max_blocks = max(cfg.blocks_for(n, block_size) for n in seq_lens)
    table = torch.full((b, max_blocks), cfg.TABLE_PAD, dtype=torch.int32)
    rng = torch.Generator().manual_seed(seed)
    total_nb = sum(cfg.blocks_for(n, block_size) for n in seq_lens)
    if total_nb > num_blocks:
        raise ValueError(f"请求集共需 {total_nb} 块 > 池容量 {num_blocks}")
    perm = torch.randperm(num_blocks, generator=rng)
    offset = 0
    for i, n in enumerate(seq_lens):
        nb = cfg.blocks_for(n, block_size)
        if mode in ("identity", "per_request_identity"):
            base = sum(cfg.blocks_for(m, block_size) for m in seq_lens[:i])
            table[i, :nb] = torch.arange(base, base + nb)
        elif mode == "rand":
            table[i, :nb] = perm[offset: offset + nb]
            offset += nb
        else:
            raise ValueError(f"未知块表模式: {mode}")
    return table


def validate_block_table(block_table: torch.Tensor, seq_lens: list[int],
                         block_size: int, num_blocks: int) -> None:
    """合同校验：越界 / 有效区间空洞 / 请求内外物理块重复 / 表长不足。显式抛错。"""
    b, max_blocks = block_table.shape
    all_valid = []
    for i, n in enumerate(seq_lens):
        nb = cfg.blocks_for(n, block_size)
        if nb > max_blocks:
            raise ValueError(f"请求 {i}: 需要 {nb} 块 > 块表宽 {max_blocks}")
        row = block_table[i, :nb]
        if bool((row < 0).any()):
            raise ValueError(f"请求 {i}: 有效区间内出现 padding/-1（块表损坏）")
        if bool((row >= num_blocks).any()):
            raise ValueError(f"请求 {i}: 表项越界 ≥ num_blocks={num_blocks}")
        if nb > 1 and torch.unique(row).numel() != nb:
            raise ValueError(f"请求 {i}: 物理块重复（aliasing）")
        all_valid.append(row)
    if all_valid:
        joined = torch.cat(all_valid)
        if torch.unique(joined).numel() != joined.numel():
            raise ValueError("跨请求物理块重叠（一块同时属多个请求——池合同违例）")


# ---------------------------------------------------------------------------
# gather / scatter（块表语义的纯映射参考）
# ---------------------------------------------------------------------------

def gather_from_pool(k_pool: torch.Tensor, v_pool: torch.Tensor,
                     block_table: torch.Tensor, seq_lens: list[int],
                     block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """分页池 → 连续布局 [B, HKV, max_len, D]（右侧 padding 区不定义，填 0）。

    严格按合同 token 级映射：K[b,:,t,:] = k_pool[table[b, t//bs], :, t%bs, :]。
    torch 高级索引即 gather；-1 padding 表项若被索引到即 gather 越界
    （此处按合同只索引 t < seq_len_b，不会触碰 padding）。
    """
    b = len(seq_lens)
    n_max = max(seq_lens)
    hk = k_pool.shape[1]
    k_out = torch.zeros(b, hk, n_max, k_pool.shape[3], dtype=k_pool.dtype)
    v_out = torch.zeros_like(k_out)
    for i, n in enumerate(seq_lens):
        nb = cfg.blocks_for(n, block_size)
        row = block_table[i, :nb].long()
        if bool((row < 0).any()):
            raise ValueError(f"gather: 请求 {i} 有效区间含 -1")
        t = torch.arange(n)
        tb, ot = t // block_size, t % block_size
        # 高级索引的广播维在前：k_pool[row[tb], :, ot, :] → [n, HKV, D]
        k_out[i, :, :n, :] = k_pool[row[tb], :, ot, :].permute(1, 0, 2)
        v_out[i, :, :n, :] = v_pool[row[tb], :, ot, :].permute(1, 0, 2)
    return k_out, v_out


def scatter_to_pool(k_contig: torch.Tensor, v_contig: torch.Tensor,
                    block_table: torch.Tensor, seq_lens: list[int],
                    block_size: int, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """连续布局 → 分页池（gather 的精确逆；fixtures 构造与 pool 校验用）。

    池中未被任何表项覆盖的位置填 0（并返回覆盖掩码，供"池污染"检查）。
    """
    b, hk, n_max, d = k_contig.shape
    k_pool = torch.zeros(num_blocks, hk, block_size, d, dtype=k_contig.dtype)
    v_pool = torch.zeros_like(k_pool)
    touched = torch.zeros(num_blocks, dtype=torch.bool)
    for i, n in enumerate(seq_lens):
        nb = cfg.blocks_for(n, block_size)
        row = block_table[i, :nb].long()
        if bool((row < 0).any()):
            raise ValueError(f"scatter: 请求 {i} 有效区间含 -1")
        t = torch.arange(n)
        tb, ot = t // block_size, t % block_size
        # 广播维在前（同 gather）：写入侧 permute 回 [n, HKV, D]
        k_pool[row[tb], :, ot, :] = k_contig[i, :, :n, :].permute(1, 0, 2)
        v_pool[row[tb], :, ot, :] = v_contig[i, :, :n, :].permute(1, 0, 2)
        touched[row] = True
    return k_pool, v_pool


# ---------------------------------------------------------------------------
# 连续布局参考（教科书链；ch21 同式自含）
# ---------------------------------------------------------------------------

def contiguous_decode_reference(q: torch.Tensor, k_cache: torch.Tensor,
                                v_cache: torch.Tensor, seq_lens: list[int],
                                scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    """decode 单 query（变长支持）：q [B,HQ,D]，cache [B,HKV,N,D]（N=max_len）。

    每请求只 attends [0, seq_len_b)；padding 区分数置 -inf。
    """
    b, hq, d = q.shape
    hkv, n = k_cache.shape[1], k_cache.shape[2]
    group = hq // hkv
    kk = expand_kv_heads(k_cache.to(q.dtype), group)
    vv = expand_kv_heads(v_cache.to(q.dtype), group)
    scores = torch.matmul(q.unsqueeze(2), kk.transpose(-1, -2)).squeeze(2) * scale
    pos = torch.arange(n).view(1, 1, -1).expand(b, hq, n)
    lens = torch.tensor(seq_lens).view(b, 1, 1)
    scores = scores.masked_fill(pos >= lens, float("-inf"))
    attn = stable_softmax_lastdim(scores)
    return torch.matmul(attn.unsqueeze(2).to(v_cache.dtype), vv).squeeze(2)


def contiguous_prefill_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    """prefill 整段 causal：q [B,HQ,S,D]，k/v [B,HKV,S,D] → o [B,HQ,S,D]。"""
    b, hq, s, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    kk = expand_kv_heads(k.to(q.dtype), group)
    vv = expand_kv_heads(v.to(q.dtype), group)
    scores = torch.matmul(q, kk.transpose(-1, -2)) * scale
    pos_q = torch.arange(s).view(1, 1, s, 1)
    pos_k = torch.arange(s).view(1, 1, 1, s)
    scores = scores.masked_fill(pos_k > pos_q, float("-inf"))
    attn = stable_softmax_lastdim(scores)
    return torch.matmul(attn.to(v.dtype), vv)


# ---------------------------------------------------------------------------
# 分页参考（gather + 连续数学；值透明性的 CPU 载体）
# ---------------------------------------------------------------------------

def paged_decode_reference(q: torch.Tensor, k_pool: torch.Tensor,
                           v_pool: torch.Tensor, block_table: torch.Tensor,
                           seq_lens: list[int], block_size: int,
                           scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    """decode 单 query，KV 按块表从池中取（gather 精确拷贝 → 与连续版位级一致）。"""
    kk, vv = gather_from_pool(k_pool, v_pool, block_table, seq_lens, block_size)
    return contiguous_decode_reference(q, kk, vv, seq_lens, scale)


def paged_prefill_reference(q: torch.Tensor, k_pool: torch.Tensor,
                            v_pool: torch.Tensor, block_table: torch.Tensor,
                            block_size: int,
                            scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    """prefill 整段，KV 在分页池（S = 每请求同长；causal 在连续逻辑位形上）。"""
    s = q.shape[2]
    seq_lens = [s] * q.shape[0]
    kk, vv = gather_from_pool(k_pool, v_pool, block_table, seq_lens, block_size)
    return contiguous_prefill_reference(q, kk, vv, scale)


# ---------------------------------------------------------------------------
# 真值锚点（FP64）与等价断言
# ---------------------------------------------------------------------------

def contiguous_decode_truth64(q: torch.Tensor, k_cache: torch.Tensor,
                              v_cache: torch.Tensor, seq_lens: list[int],
                              scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    return contiguous_decode_reference(q.double(), k_cache.double(),
                                       v_cache.double(), seq_lens, scale)


def contiguous_prefill_truth64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               scale: float = cfg.SOFTMAX_SCALE) -> torch.Tensor:
    return contiguous_prefill_reference(q.double(), k.double(), v.double(), scale)


def assert_value_transparency(k_contig: torch.Tensor, v_contig: torch.Tensor,
                              block_table: torch.Tensor, seq_lens: list[int],
                              block_size: int, num_blocks: int) -> dict:
    """值透明性断言：scatter→gather 往返必须逐元素还原（任意块表模式）。"""
    kp, vp = scatter_to_pool(k_contig, v_contig, block_table, seq_lens,
                             block_size, num_blocks)
    kg, vg = gather_from_pool(kp, vp, block_table, seq_lens, block_size)
    n_max = max(seq_lens)
    b = len(seq_lens)
    ok = True
    for i, n in enumerate(seq_lens):
        ok = ok and bool(torch.equal(kg[i, :, :n, :], k_contig[i, :, :n, :]))
        ok = ok and bool(torch.equal(vg[i, :, :n, :], v_contig[i, :, :n, :]))
    return {"roundtrip_bitexact": ok, "batch": b, "block_size": block_size,
            "max_len": n_max}


if __name__ == "__main__":
    torch.manual_seed(0)
    seq_lens = [511, 2048, 17]
    bs = 128
    nb_total = sum(cfg.blocks_for(n, bs) for n in seq_lens)
    k = torch.randn(len(seq_lens), cfg.HKV, max(seq_lens), cfg.D)
    v = torch.randn_like(k)
    for mode in ("identity", "per_request_identity", "rand"):
        table = build_block_table(seq_lens, bs, nb_total, mode, seed=hash(mode) % 2**31)
        validate_block_table(table, seq_lens, bs, nb_total)
        r = assert_value_transparency(k, v, table, seq_lens, bs, nb_total)
        print(f"[{mode}] roundtrip_bitexact={r['roundtrip_bitexact']} "
              f"table_shape={tuple(table.shape)}")
