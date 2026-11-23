"""第 21 期显存记账器（实验 2：连续 vs 分页的碎片率与可服务请求数）。

纯 CPU 记账模拟（不分配真实显存）：容量单位 = KV token 数（两方案同一容量
口径，公平对比；字节 = tokens × kv_bytes_per_token_model(itemsize)）。

三种分配策略（大纲"连续 vs 分页的显存碎片率"的三条曲线）：
  P1 contiguous-reserve-global  连续预留（pre-vLLM 静态方案）：admit 时按
      全局最大声明长度 [batch, max_ctx] 式预留连续区间——浪费 = 预留-实际。
  P1b contiguous-reserve-own    连续预留（per-request own final_len 变体）：
      预留请求自身的最终长度——比 P1 省，但仍要求整段连续。
  P2 contiguous-exact           malloc 式精确按需：按当前长度分配；增长时
      原地扩展（紧跟的空闲段被吞并），失败则整体搬迁（计搬运 token 数），
      再失败则请求中途夭折（external fragmentation 的直接受害者）。
  P3 paged                      块池按需：ceil(len/bs) 个任意位置的块；
      无连续性要求 → external fragmentation 恒 0（构造性保证），
      只剩尾部部分块的 internal waste。

记账方法（metrics 定义，随结果 JSON 的 "method" 字段落盘）：
  ext_fragmentation = 1 − largest_free_span / total_free
      （free=0 时记 0；仅对要求连续区间的方案有意义；P3 构造性为 0）
  internal_waste    = Σ_active (reserved − used)（P1/P1b: 预留−当前长度；
      P3: blocks×bs − 当前长度——尾部部分块；P2: 0）
  rejected_at_admission：admit 失败（cause ∈ {no_space, no_contiguous_span}）
      —— no_contiguous_span = 总空闲 ≥ 需求但最大连续段不够（碎片性拒绝，
      容量性拒绝不可能发生时它才可见，故两者分开计数）。
  aborted_midlife：P2 增长失败 / P3 块耗尽（请求中途夭折，比拒绝更糟）。
  copy_tokens：P2 搬迁的总 token 搬运量（分页免搬运的对照项）。
  served：完整跑完 final_len 的请求数（headline：可服务请求数）。
负载（FRAG_SEED 固定）：32 请求，final_len ~ U[8, 2048]，initial_len ~
  U[8, 64] ∧ ≤ final_len，到达时刻 ~ 独立均匀后排序，每步生成 1 token，
  final_len − initial_len 步后释放。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import paged_config as cfg  # noqa: E402

METHOD = {
    "unit": "KV tokens（字节 = tokens × kv_bytes_per_token_model）",
    "ext_fragmentation": "1 − largest_free_span/total_free；free=0 记 0；P3 构造性 0",
    "internal_waste": "P1/P1b: Σ(预留−当前长度)；P3: Σ(blocks×bs−当前长度)；P2: 0",
    "rejection_causes": ["no_space", "no_contiguous_span"],
    "aborted_midlife": "P2 增长失败 / P3 块耗尽（中途夭折）",
    "served": "完整生成到 final_len 的请求数",
    "policy_names": ["contig-reserve-global", "contig-reserve-own",
                     "contig-exact-malloc", "paged"],
}


# ---------------------------------------------------------------------------
# 请求负载
# ---------------------------------------------------------------------------

def make_workload(seed: int = cfg.FRAG_SEED,
                  num_requests: int = cfg.FRAG_NUM_REQUESTS,
                  len_min: int = cfg.FRAG_LEN_MIN,
                  len_max: int = cfg.FRAG_LEN_MAX,
                  arrive_window: int = 600) -> list[dict[str, int]]:
    rng = np.random.default_rng(seed)
    final_lens = rng.integers(len_min, len_max + 1, num_requests)
    init_lens = np.minimum(
        final_lens, rng.integers(len_min, 65, num_requests))
    arrives = np.sort(rng.integers(0, arrive_window, num_requests))
    return [
        {"id": i, "arrive": int(arrives[i]), "initial": int(init_lens[i]),
         "final": int(final_lens[i])}
        for i in range(num_requests)
    ]


# ---------------------------------------------------------------------------
# 连续区间管理（first-fit span list；P1/P1b/P2 共用）
# ---------------------------------------------------------------------------

class SpanArena:
    """连续 token 区间分配器：sorted free spans + first-fit + 释放合并。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.free_spans: list[list[int]] = [[0, capacity]]  # [start, end)

    def total_free(self) -> int:
        return sum(e - s for s, e in self.free_spans)

    def largest_span(self) -> int:
        return max((e - s for s, e in self.free_spans), default=0)

    def ext_fragmentation(self) -> float:
        free = self.total_free()
        if free == 0:
            return 0.0
        return 1.0 - self.largest_span() / free

    def alloc(self, size: int) -> tuple[int, str] | None:
        """first-fit；返回 (start, cause-if-had-to-fail)。cause 仅在失败时返回。"""
        if size == 0:
            return (-1, "")
        if size > self.total_free():
            return None
        for idx, (s, e) in enumerate(self.free_spans):
            if e - s >= size:
                start = s
                if e - s == size:
                    del self.free_spans[idx]
                else:
                    self.free_spans[idx] = [s + size, e]
                return (start, "")
        return None  # 总空闲够但最大连续段不够 → 碎片性拒绝

    def alloc_contiguous_at_end_extend(self, start: int, cur_end: int,
                                       size: int = 1) -> bool:
        """原地扩展：[cur_end, cur_end+size) 是否空闲且属同一段前沿。"""
        for idx, (s, e) in enumerate(self.free_spans):
            if s == cur_end and e - s >= size:
                if e - s == size:
                    del self.free_spans[idx]
                else:
                    self.free_spans[idx] = [s + size, e]
                return True
        return False

    def free_span(self, start: int, size: int) -> None:
        """释放并合并相邻段。"""
        spans = self.free_spans
        pos = 0
        while pos < len(spans) and spans[pos][0] < start:
            pos += 1
        spans.insert(pos, [start, start + size])
        # 合并
        merged: list[list[int]] = []
        for s in spans:
            if merged and merged[-1][1] == s[0]:
                merged[-1][1] = s[1]
            else:
                merged.append(list(s))
        self.free_spans = merged


class FreeBlocks:
    """物理块空闲表（P3）：FIFO/升序分配。mode="seq" 时升序（新池天然连续段），
    mode="rand" 时随机取（模拟池老化后的碎片化空闲表）。"""

    def __init__(self, num_blocks: int, mode: str = "seq", seed: int = 0):
        self.free = list(range(num_blocks))
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def take(self) -> int | None:
        if not self.free:
            return None
        if self.mode == "rand":
            return int(self.free.pop(int(self.rng.integers(0, len(self.free)))))
        return int(self.free.pop(0))

    def put(self, blk: int) -> None:
        self.free.append(blk)
        self.free.sort()

    def __len__(self) -> int:
        return len(self.free)


# ---------------------------------------------------------------------------
# 模拟器
# ---------------------------------------------------------------------------

def simulate(policy: str, requests: list[dict[str, int]], capacity_tokens: int,
             block_size: int = cfg.DEFAULT_BLOCK_SIZE,
             pool_free_mode: str = "seq") -> dict[str, Any]:
    """逐步模拟到达→逐 token 生成→释放；返回记账指标。

    policy ∈ METHOD["policy_names"]；capacity_tokens 为全局 KV token 预算。
    """
    kvh = cfg.HKV
    d = cfg.D
    max_len_decl = max(r["final"] for r in requests)
    arena = SpanArena(capacity_tokens)
    num_blocks = capacity_tokens // block_size
    granular_capacity = num_blocks * block_size
    blocks = FreeBlocks(num_blocks, pool_free_mode, seed=cfg.FRAG_SEED)

    active: dict[int, dict[str, Any]] = {}
    stats = {
        "policy": policy, "block_size": block_size if policy == "paged" else None,
        "capacity_tokens": capacity_tokens,
        "granular_capacity_tokens": granular_capacity,
        "served": 0, "rejected_at_admission": 0,
        "reject_no_space": 0, "reject_no_contiguous_span": 0,
        "aborted_midlife": 0, "copy_tokens": 0,
        "peak_active": 0, "peak_used_tokens": 0,
        "internal_waste_at_peak": 0, "ext_frag_mean": 0.0, "ext_frag_max": 0.0,
        "frag_samples": 0, "timeline": [],
    }
    finished = set()
    step = 0
    horizon = max(r["arrive"] + r["final"] for r in requests) + 1

    def admit(req: dict[str, int]) -> None:
        need = req["initial"]
        if policy in ("contig-reserve-global", "contig-reserve-own"):
            reserve = max_len_decl if policy == "contig-reserve-global" else req["final"]
            res = arena.alloc(reserve)
            if res is None:
                stats["rejected_at_admission"] += 1
                if reserve > arena.total_free():
                    stats["reject_no_space"] += 1
                else:
                    stats["reject_no_contiguous_span"] += 1
                return
            start, _ = res
            active[req["id"]] = {"start": start, "reserved": reserve,
                                 "blocks": None, "len": need,
                                 "table": None}
        elif policy == "contig-exact-malloc":
            res = arena.alloc(need)
            if res is None:
                stats["rejected_at_admission"] += 1
                if need > arena.total_free():
                    stats["reject_no_space"] += 1
                else:
                    stats["reject_no_contiguous_span"] += 1
                return
            start, _ = res
            active[req["id"]] = {"start": start, "reserved": need,
                                 "blocks": None, "len": need, "table": None}
        elif policy == "paged":
            need_blocks = (need + block_size - 1) // block_size
            if need_blocks > len(blocks):
                stats["rejected_at_admission"] += 1
                stats["reject_no_space"] += 1
                return
            table = [blocks.take() for _ in range(need_blocks)]
            active[req["id"]] = {"start": None, "reserved": need_blocks * block_size,
                                 "blocks": None, "len": need, "table": table}

    def append_token(rid: int) -> None:
        """请求 rid 增长 1 token；容量不足时按方案语义处理。"""
        st = active[rid]
        new_len = st["len"] + 1
        if policy in ("contig-reserve-global", "contig-reserve-own"):
            st["len"] = new_len                    # 预留内增长，永不失败
        elif policy == "contig-exact-malloc":
            cur_end = st["start"] + st["len"]
            if arena.alloc_contiguous_at_end_extend(st["start"], cur_end, 1):
                st["reserved"] += 1
                st["len"] = new_len
                return
            # 搬迁：找 ≥ new_len 的连续段
            res = arena.alloc(new_len)
            if res is None:
                cause = "reject_no_space" if new_len > arena.total_free() \
                    else "reject_no_contiguous_span"
                arena.free_span(st["start"], st["reserved"])
                del active[rid]
                stats["aborted_midlife"] += 1
                stats[cause] += 1
                return
            nstart, _ = res
            stats["copy_tokens"] += st["len"]      # 搬迁拷贝旧内容
            arena.free_span(st["start"], st["reserved"])
            st["start"], st["reserved"], st["len"] = nstart, new_len, new_len
        elif policy == "paged":
            old_blocks = (st["len"] + block_size - 1) // block_size
            new_blocks = (new_len + block_size - 1) // block_size
            if new_blocks > old_blocks:
                blk = blocks.take()
                if blk is None:
                    for b in st["table"]:
                        blocks.put(b)
                    del active[rid]
                    stats["aborted_midlife"] += 1
                    stats["reject_no_space"] += 1
                    return
                st["table"].append(blk)
            st["len"] = new_len
            st["reserved"] = new_blocks * block_size

    def release(rid: int) -> None:
        st = active[rid]
        if policy in ("contig-reserve-global", "contig-reserve-own",
                      "contig-exact-malloc"):
            arena.free_span(st["start"], st["reserved"])
        else:
            for b in st["table"]:
                blocks.put(b)
        del active[rid]
        finished.add(rid)
        stats["served"] += 1

    arrivals = sorted(requests, key=lambda r: (r["arrive"], r["id"]))
    ai = 0
    frag_sum = 0.0
    waste_sum = 0.0
    waste_steps = 0
    while step < horizon or active:
        while ai < len(arrivals) and arrivals[ai]["arrive"] == step:
            admit(arrivals[ai])
            ai += 1
        for rid in list(active.keys()):
            req = next(r for r in requests if r["id"] == rid)
            if active[rid]["len"] < req["final"]:
                append_token(rid)
            if rid in active and active[rid]["len"] >= req["final"]:
                release(rid)
        used = sum(st["reserved"] for st in active.values())
        act = len(active)
        if act > 0:
            waste_sum += sum(st["reserved"] - st["len"] for st in active.values())
            waste_steps += 1
        if act > stats["peak_active"]:
            stats["peak_active"] = act
        if used > stats["peak_used_tokens"]:
            stats["peak_used_tokens"] = used
            stats["internal_waste_at_peak"] = sum(
                st["reserved"] - st["len"] for st in active.values())
        frag_sum += arena.ext_fragmentation()
        stats["frag_samples"] += 1
        stats["ext_frag_max"] = max(stats["ext_frag_max"], arena.ext_fragmentation())
        step += 1

    stats["ext_frag_mean"] = round(frag_sum / max(stats["frag_samples"], 1), 6)
    stats["ext_frag_max"] = round(stats["ext_frag_max"], 6)
    stats["internal_waste_mean_active"] = round(
        waste_sum / max(waste_steps, 1), 3)
    total_demand = sum(r["final"] for r in requests)
    stats["workload"] = {
        "num_requests": len(requests),
        "total_tokens_demanded": total_demand,
        "max_len": max_len_decl,
        "mean_len": round(total_demand / len(requests), 1),
        "peak_active": stats["peak_active"],
    }
    stats["timeline"] = None  # 时间线过重不入 JSON（样本数已计入 frag_samples）
    return stats


def capacity_sweep(requests: list[dict[str, int]], block_size: int,
                   fractions: list[float],
                   pool_free_mode: str = "seq") -> list[dict[str, Any]]:
    """容量扫描：不同容量预算下各方案的 served 数（可服务请求数差异曲线）。"""
    total_demand = sum(r["final"] for r in requests)
    rows = []
    for f in fractions:
        cap = int(total_demand * f)
        row: dict[str, Any] = {"capacity_fraction": f, "capacity_tokens": cap}
        for policy in METHOD["policy_names"]:
            r = simulate(policy, requests, cap, block_size, pool_free_mode)
            row[f"served_{policy}"] = r["served"]
            row[f"rejected_{policy}"] = (r["rejected_at_admission"]
                                         + r["aborted_midlife"])
        rows.append(row)
    return rows


def block_size_sweep(requests: list[dict[str, int]], capacity_tokens: int,
                     block_sizes: list[int] = cfg.FRAG_BLOCK_SIZE_SWEEP) -> list[dict]:
    rows = []
    for bs in block_sizes:
        r = simulate("paged", requests, capacity_tokens, bs)
        rows.append(r)
    return rows


if __name__ == "__main__":
    wl = make_workload()
    demand = sum(r["final"] for r in wl)
    print(f"workload: {len(wl)} req, total demand = {demand} tokens, "
          f"max = {max(r['final'] for r in wl)}")
    for frac in (0.5, 0.8, 1.0):
        cap = int(demand * frac)
        print(f"\n== capacity = {cap} tokens ({frac:.0%} of demand) ==")
        for policy in METHOD["policy_names"]:
            r = simulate(policy, wl, cap, cfg.DEFAULT_BLOCK_SIZE)
            print(f"  {policy:<22} served={r['served']:>2} "
                  f"rej={r['rejected_at_admission']:>2} "
                  f"(space={r['reject_no_space']}, frag={r['reject_no_contiguous_span']}) "
                  f"abort={r['aborted_midlife']} copy={r['copy_tokens']:>6} "
                  f"peak_used={r['peak_used_tokens']:>6} "
                  f"waste@peak={r['internal_waste_at_peak']:>6} "
                  f"extfrag(mean/max)={r['ext_frag_mean']:.3f}/{r['ext_frag_max']:.3f}")
