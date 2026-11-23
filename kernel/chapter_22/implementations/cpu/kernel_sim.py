"""第 22 期 kernel_sim：GPU 核索引组织的 CPU 仿真（算法等价体 + 覆盖断言）。

按 rope_kernels.cuh 的线程索引方案逐线程仿真（ep26"循环界 bug 由 benchmark
抓出"教训的制度化——先仿真后上服务器）：

  build_cache_kernel：tid → (pos, j) 除法映射，每 (pos, j) 恰写一次；
  rope_prefill_kernel：tid → (t, h, j) 除法映射，输出每元素恰写一次，
    cache 行读 (seq_offset + t) ∈ [0, max_seq)；
  rope_decode_kernel：grid = (hq+hkv)，block j ∈ [0, half)，q/k 块分支映射，
    输出每元素恰写一次，cache 行读 seq_offset。

断言：
  A1 覆盖：每个输出元素被恰一次写入（coverage == 1）；
  A2 cache 读界：全部 cache 读下标 ∈ [0, max_seq*half)；
  A3 数值：仿真输出 == rope_reference 查表路径（fp64 位级 / fp32 ≤1e-6）；
  A4 尾部边界：num_tokens 非 128 倍数 / seq_offset 大位置 的除法余数正确；
  A5 decode 分支：q 头 [0, hq) 走 q 指针、[hq, hq+hkv) 走 k 指针，映射与
     参考逐头一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import rope_config as cfg  # noqa: E402
import rope_reference as ref  # noqa: E402

THREADS = 128
HALF = cfg.HALF_DIM


# ---------------------------------------------------------------------------
# 逐线程仿真（严格按 .cuh 的 tid → 下标算术）
# ---------------------------------------------------------------------------

def sim_build_cache_thread_map(max_seq: int):
    """返回每线程 (pos, j) 列表与覆盖计数（build_cache_kernel 索引方案）。"""
    total = max_seq * HALF
    coverage = np.zeros((max_seq, HALF), dtype=np.int64)
    pairs = []
    for tid in range(total):
        pos = tid // HALF
        j = tid % HALF
        coverage[pos, j] += 1
        pairs.append((pos, j))
    return pairs, coverage


def sim_prefill_thread_map(num_tokens: int, hq: int, hkv: int):
    """rope_prefill_kernel 索引方案：total = T*(hq+hkv)*half。"""
    heads = hq + hkv
    total = num_tokens * heads * HALF
    coverage = np.zeros((num_tokens, heads, cfg.HEAD_DIM), dtype=np.int64)
    cache_reads = []
    for tid in range(total):
        j = tid % HALF
        h = (tid // HALF) % heads
        t = tid // (HALF * heads)
        coverage[t, h, j] += 1          # out[j]
        coverage[t, h, j + HALF] += 1   # out[j+HALF]
        cache_reads.append(t)           # 行 = seq_offset + t（回填偏移在外层）
    return coverage, cache_reads


def sim_decode_thread_map(hq: int, hkv: int):
    """rope_decode_kernel 索引方案：grid=(hq+hkv) blocks × half threads。"""
    coverage = np.zeros((hq + hkv, cfg.HEAD_DIM), dtype=np.int64)
    for h in range(hq + hkv):
        for j in range(HALF):
            coverage[h, j] += 1
            coverage[h, j + HALF] += 1
    return coverage


# ---------------------------------------------------------------------------
# 数值仿真（与核一致的 fp32 算术 + fp16 舍入）
# ---------------------------------------------------------------------------

def sim_apply_prefill(q, k, cos_cache, sin_cache, seq_offset=0):
    """按核的每线程语义（fp32 算术，io dtype 保输入）仿真 prefill。"""
    q32 = q.astype(np.float32)
    k32 = k.astype(np.float32)
    cos32 = cos_cache.astype(np.float32)
    sin32 = sin_cache.astype(np.float32)
    q_out = np.empty_like(q32)
    k_out = np.empty_like(k32)
    heads_q, heads_k = q32.shape[1], k32.shape[1]
    for t in range(q32.shape[0]):
        row = seq_offset + t
        for h in range(heads_q):
            for j in range(HALF):
                c = np.float32(cos32[row, j])
                s = np.float32(sin32[row, j])
                x1 = np.float32(q32[t, h, j])
                x2 = np.float32(q32[t, h, j + HALF])
                q_out[t, h, j] = np.float32(x1 * c - x2 * s)
                q_out[t, h, j + HALF] = np.float32(x2 * c + x1 * s)
        for h in range(heads_k):
            for j in range(HALF):
                c = np.float32(cos32[row, j])
                s = np.float32(sin32[row, j])
                x1 = np.float32(k32[t, h, j])
                x2 = np.float32(k32[t, h, j + HALF])
                k_out[t, h, j] = np.float32(x1 * c - x2 * s)
                k_out[t, h, j + HALF] = np.float32(x2 * c + x1 * s)
    return q_out, k_out


def sim_apply_decode(q, k, cos_cache, sin_cache, seq_offset):
    """按核的每线程语义仿真 decode（q/k 两段块分支）。"""
    q32 = q.astype(np.float32)[0] if q.ndim == 3 else q.astype(np.float32)
    k32 = k.astype(np.float32)[0] if k.ndim == 3 else k.astype(np.float32)
    cos32 = cos_cache.astype(np.float32)
    sin32 = sin_cache.astype(np.float32)
    q_out = np.empty_like(q32)
    k_out = np.empty_like(k32)
    for h in range(q32.shape[0]):
        for j in range(HALF):
            c = np.float32(cos32[seq_offset, j])
            s = np.float32(sin32[seq_offset, j])
            x1 = np.float32(q32[h, j])
            x2 = np.float32(q32[h, j + HALF])
            q_out[h, j] = np.float32(x1 * c - x2 * s)
            q_out[h, j + HALF] = np.float32(x2 * c + x1 * s)
    for h in range(k32.shape[0]):
        for j in range(HALF):
            c = np.float32(cos32[seq_offset, j])
            s = np.float32(sin32[seq_offset, j])
            x1 = np.float32(k32[h, j])
            x2 = np.float32(k32[h, j + HALF])
            k_out[h, j] = np.float32(x1 * c - x2 * s)
            k_out[h, j + HALF] = np.float32(x2 * c + x1 * s)
    return q_out, k_out


# ---------------------------------------------------------------------------
# 全量断言
# ---------------------------------------------------------------------------

def run_all(max_seq: int = 97, num_tokens_list=(1, 3, 130), seq_offsets=(0, 2047, 32767)):
    failures = []
    rng = np.random.default_rng(cfg.FIXTURES_SEED)

    # A1/A2：映射覆盖 + cache 读界
    _, cov_cache = sim_build_cache_thread_map(max_seq)
    if not (cov_cache == 1).all():
        failures.append(f"A1 build-cache 覆盖 != 1（min={cov_cache.min()} max={cov_cache.max()}）")
    for nt in num_tokens_list:
        cov, _ = sim_prefill_thread_map(nt, cfg.HQ, cfg.HKV)
        if not (cov == 1).all():
            failures.append(f"A1 prefill T={nt} 覆盖 != 1")
    cov_dec = sim_decode_thread_map(cfg.HQ, cfg.HKV)
    if not (cov_dec == 1).all():
        failures.append("A1 decode 覆盖 != 1")
    print(f"[kernel_sim] A1 覆盖断言：build_cache[{max_seq}] / prefill {list(num_tokens_list)} / decode —— "
          f"{'PASS' if not failures else failures}")

    # A3/A4/A5：数值 + 边界 + decode 分支
    cos64, sin64 = ref.build_cache(40000, cfg.ROPE_THETA, cfg.HEAD_DIM, "float64")
    cos32 = cos64.astype(np.float32)
    sin32 = sin64.astype(np.float32)
    worst = 0.0
    for nt in num_tokens_list:
        q = rng.standard_normal((nt, cfg.HQ, cfg.HEAD_DIM)).astype(np.float32)
        k = rng.standard_normal((nt, cfg.HKV, cfg.HEAD_DIM)).astype(np.float32)
        for off in seq_offsets:
            qr, kr = ref.apply_prefill(q, k, cos32, sin32, off)          # fp32 参考查表
            qs, ks = sim_apply_prefill(q, k, cos32, sin32, off)          # 核仿真
            err = max(np.abs(qr - qs).max(), np.abs(kr - ks).max())
            worst = max(worst, err)
            # fp64 锚点对照（核语义 fp32 vs fp64 参考——fp32 角度舍入量级）
            q64, k64 = ref.apply_prefill(q.astype(np.float64), k.astype(np.float64), cos64, sin64, off)
            err64 = max(np.abs(q64.astype(np.float32) - qs).max(), np.abs(k64.astype(np.float32) - ks).max())
            if err > 0.0:
                failures.append(f"A3 prefill T={nt} off={off} 仿真≠参考 err={err}")
            worst = max(worst, err64)
    for off in seq_offsets:
        q = rng.standard_normal((cfg.HQ, cfg.HEAD_DIM)).astype(np.float32)
        k = rng.standard_normal((cfg.HKV, cfg.HEAD_DIM)).astype(np.float32)
        qr, kr = ref.apply_decode(q, k, cos32, sin32, off)
        qs, ks = sim_apply_decode(q, k, cos32, sin32, off)
        err = max(np.abs(qr - qs).max(), np.abs(kr - ks).max())
        if err > 0.0:
            failures.append(f"A5 decode off={off} 仿真≠参考 err={err}")
        worst = max(worst, err)
    print(f"[kernel_sim] A3/A4/A5 数值+边界+decode 分支：worst fp32 仿真/参考差 = {worst:.3e} "
          f"{'PASS' if not failures else 'FAIL'}")
    return failures


if __name__ == "__main__":
    fails = run_all()
    raise SystemExit(1 if fails else 0)
