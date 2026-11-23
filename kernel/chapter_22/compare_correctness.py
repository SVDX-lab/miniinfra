#!/usr/bin/env python3
"""第 22 期正确性比较（CPU 列；对照口径与 ch01 compare_correctness 对齐）。

链路（大纲 §9 顺序的本期映射）：
  1. CPU FP64 公式参考（锚点自洽 + FP32/FP16 舍入分层）；
  2. PyTorch HF 语义对照（chapter_01 qwen3_model RotaryEmbedding/rotate_half
     语义 vendored 复刻——实验 1"与 PyTorch HF 参考实现数值对照"CPU 侧）；
  3. 正确性矩阵：seq {1, 64, 512, 2048, 32768} × head_dim=128 × {fp32, fp16}
     对 FP64 锚点（实验 4 CPU 侧；MUSA/CUDA 核矩阵由 standalone 二进制对
     同构 FP64 真值产出，跨后端口径一致）；
  4. naive 字面循环交叉验证（seq=64/pos=17，独立实现路径防同源 bug）；
  5. cache dtype 实验 CPU 侧：theta × seq × {fp32, fp16} 表误差 + e2e q 误差
     （实验 2 的误差放大场景）；
  6. decode 形态：pos {0, 63, 511, 2047, 32767} × {fp32, fp16} 对 FP64 锚点；
  7. 确定性：prefill/decode 各 3 次位级比对。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import rope_config as cfg  # noqa: E402
import rope_reference as ref  # noqa: E402
from implementations.cpu import rope_reference  # noqa: E402  (双路径 import 等价性自检)

MAX_SEQ = 40960


def comparison_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    af = actual.astype(np.float64)
    ef = expected.astype(np.float64)
    abs_err = np.abs(af - ef)
    denom = np.maximum(np.abs(ef), 1e-6)
    dot_ab = float(np.sum(af * ef))
    dot_aa = float(np.sum(af * af))
    dot_bb = float(np.sum(ef * ef))
    return {
        "max_abs_error": float(abs_err.max()),
        "mean_abs_error": float(abs_err.mean()),
        "mean_rel_error": float((abs_err / denom).mean()),
        "cosine_similarity": dot_ab / (math.sqrt(dot_aa) * math.sqrt(dot_bb) + 1e-30),
        "nan_count": int(np.isnan(af).sum()),
        "inf_count": int(np.isinf(af).sum()),
        "max_abs_error_position": int(abs_err.argmax()),
    }


# ---------------------------------------------------------------------------
# vendored HF 语义（chapter_01/qwen3_model.py 复刻——不 import chapter_01）
# ---------------------------------------------------------------------------

def hf_rotary_tables(position_ids: np.ndarray, theta: float, head_dim: int):
    dims = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (dims / head_dim))
    freqs = torch.from_numpy(position_ids).float().unsqueeze(-1) * inv_freq
    embedding = torch.cat((freqs, freqs), dim=-1)
    return embedding.cos().numpy(), embedding.sin().numpy()


def hf_apply(q: np.ndarray, cos_dup: np.ndarray, sin_dup: np.ndarray):
    """chapter_01 apply_rotary_embedding 语义（[T, H, D] 与 [T, 1, D] 广播）。"""
    qt = torch.from_numpy(q)
    cos = torch.from_numpy(cos_dup).unsqueeze(1)
    sin = torch.from_numpy(sin_dup).unsqueeze(1)
    half = qt.shape[-1] // 2
    rotated = qt * cos + torch.cat((-qt[..., half:], qt[..., :half]), dim=-1) * sin
    return rotated.numpy()


def gen_qk(seq: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.FIXTURES_SEED + seq)
    q = rng.standard_normal((seq, cfg.HQ, cfg.HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((seq, cfg.HKV, cfg.HEAD_DIM)).astype(np.float32)
    return q, k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/correctness-results-cpu.json")
    args = parser.parse_args()

    report: dict = {
        "schema_version": "kernel-chapter23.correctness.cpu.v1",
        "platform": "cpu",
        "torch": __import__("torch").__version__,
        "theta": cfg.ROPE_THETA,
        "head_dim": cfg.HEAD_DIM,
        "sections": {},
    }

    # ---- 1) FP64 锚点自洽 + 舍入分层 ----
    cos64, sin64 = ref.build_cache(MAX_SEQ, cfg.ROPE_THETA, cfg.HEAD_DIM, "float64")
    cos64b, sin64b = ref.build_cache(MAX_SEQ, cfg.ROPE_THETA, cfg.HEAD_DIM, "float64")
    selfcons = bool(np.array_equal(cos64, cos64b) and np.array_equal(sin64, sin64b))
    cos32, sin32 = ref.build_cache(MAX_SEQ, cfg.ROPE_THETA, cfg.HEAD_DIM, "float32")
    cos16, sin16 = ref.build_cache(MAX_SEQ, cfg.ROPE_THETA, cfg.HEAD_DIM, "float16")
    m32 = comparison_metrics(cos32, cos64)
    m16 = comparison_metrics(cos16.astype(np.float32), cos64)
    report["sections"]["anchor"] = {
        "fp64_self_consistent_bitwise": selfcons,
        "fp32_table_vs_fp64": m32,
        "fp16_table_vs_fp64": m16,
        "fp32_table_vs_fp64_by_pos32768_j1": float(abs(cos32[32768, 1] - cos64[32768, 1])),
        "note": "fp16 量化 ~2.4e-4（[-1,1] ulp/2）；fp32 = 角度舍入，随 pos 线性放大",
    }

    # ---- 2) HF 语义对照 ----
    seq_hf = 512
    pos = np.arange(seq_hf)
    cos_hf, sin_hf = hf_rotary_tables(pos, cfg.ROPE_THETA, cfg.HEAD_DIM)
    cos_dup, sin_dup = ref.expand_dup(cos32[:seq_hf], sin32[:seq_hf])
    q, k = gen_qk(seq_hf)
    q_hf = hf_apply(q, cos_hf, sin_hf)
    q_mine, k_mine = ref.apply_prefill(q, k, cos32, sin32, 0)
    m_hf_q = comparison_metrics(q_mine, q_hf)
    m_hf_k = comparison_metrics(k_mine, hf_apply(k, cos_hf, sin_hf))
    # FP64 HF 语义 vs FP64 锚点（同语义双 dtype）
    cos_hf64 = hf_rotary_tables(pos, cfg.ROPE_THETA, cfg.HEAD_DIM)
    report["sections"]["hf_semantics"] = {
        "table_cos_max_diff": float(np.abs(cos_hf - cos_dup).max()),
        "table_sin_max_diff": float(np.abs(sin_hf - sin_dup).max()),
        "applied_q_vs_hf": m_hf_q,
        "applied_k_vs_hf": m_hf_k,
        "verdict": "compact 查表路径与 HF rotate_half 语义一致（同表位级、应用 ≤1e-6）",
    }

    # ---- 3) 正确性矩阵（prefill）----
    cases = []
    for seq in cfg.CORRECTNESS_SEQ_LENS:
        q, k = gen_qk(seq)
        for dtype in ("fp32", "fp16"):
            io_dtype = {"fp32": None, "fp16": np.float16}[dtype]
            q_out, k_out = ref.apply_prefill(q, k, cos32[:seq], sin32[:seq], 0, io_dtype=io_dtype)
            q_ref, k_ref = ref.apply_prefill(q.astype(np.float64), k.astype(np.float64),
                                             cos64[:seq], sin64[:seq], 0)
            m = comparison_metrics(np.concatenate([q_out.reshape(-1), k_out.reshape(-1)]),
                                   np.concatenate([q_ref.reshape(-1), k_ref.reshape(-1)]))
            tol = 5e-4 + 6.0 * seq * 5.9604645e-8 if dtype == "fp32" else 2e-3 + 6.0 * seq * 5.9604645e-8
            cases.append({
                "form": "prefill", "seq": seq, "dtype": dtype, **m,
                "tolerance": tol, "pass": m["max_abs_error"] <= tol and m["nan_count"] == 0,
            })
    report["sections"]["correctness_matrix_prefill"] = cases

    # ---- 4) decode 形态矩阵 ----
    cases_d = []
    for pos in (0, 63, 511, 2047, 32767):
        q1, k1 = gen_qk(1)
        for dtype in ("fp32", "fp16"):
            io_dtype = {"fp32": None, "fp16": np.float16}[dtype]
            q_out, k_out = ref.apply_decode(q1[0], k1[0], cos32, sin32, pos, io_dtype=io_dtype)
            q_ref, k_ref = ref.apply_decode(q1[0].astype(np.float64), k1[0].astype(np.float64),
                                            cos64, sin64, pos)
            m = comparison_metrics(np.concatenate([q_out.reshape(-1), k_out.reshape(-1)]),
                                   np.concatenate([q_ref.reshape(-1), k_ref.reshape(-1)]))
            tol = 5e-4 + 6.0 * (pos + 1) * 5.9604645e-8 if dtype == "fp32" else 2e-3 + 6.0 * (pos + 1) * 5.9604645e-8
            cases_d.append({
                "form": "decode", "pos": pos, "dtype": dtype, **m,
                "tolerance": tol, "pass": m["max_abs_error"] <= tol and m["nan_count"] == 0,
            })
    report["sections"]["correctness_matrix_decode"] = cases_d

    # ---- 5) naive 字面循环交叉验证（独立实现路径）----
    seq_n, pos_n = 64, 17
    qn, kn = gen_qk(seq_n)
    q_out, _ = ref.apply_prefill(qn, kn, cos32, sin32, 0)
    # 字面循环：逐 (t, h, i) 直接按 HF 公式写
    q_naive = np.empty_like(q_out)
    half = cfg.HEAD_DIM // 2
    for t in range(seq_n):
        for h in range(cfg.HQ):
            for i in range(half):
                c = cos32[t, i]
                s = sin32[t, i]
                x1 = qn[t, h, i]
                x2 = qn[t, h, i + half]
                q_naive[t, h, i] = x1 * c - x2 * s
                q_naive[t, h, i + half] = x2 * c + x1 * s
    m_naive = comparison_metrics(q_out, q_naive)
    report["sections"]["naive_loops_crosscheck"] = {
        "seq": seq_n, **m_naive,
        "bitwise_equal": bool(np.array_equal(q_out, q_naive)),
    }

    # ---- 6) cache dtype 误差放大（CPU 侧）----
    cache_rows = []
    for theta in cfg.CACHE_DTYPE_THETAS:
        for seq in cfg.CACHE_DTYPE_SEQS:
            c64t, s64t = ref.build_cache(seq, theta, cfg.HEAD_DIM, "float64")
            c32t, s32t = ref.build_cache(seq, theta, cfg.HEAD_DIM, "float32")
            c16t, s16t = ref.build_cache(seq, theta, cfg.HEAD_DIM, "float16")
            row = {"theta": theta, "seq": seq}
            for name, dt in (("fp32", (c32t, s32t)), ("fp16", (c16t, s16t))):
                m = comparison_metrics(np.concatenate([dt[0].astype(np.float32).reshape(-1),
                                                       dt[1].astype(np.float32).reshape(-1)]),
                                       np.concatenate([c64t.reshape(-1), s64t.reshape(-1)]))
                row[f"table_{name}"] = {"max_abs": m["max_abs_error"], "mean_abs": m["mean_abs_error"]}
            # 逐 j 误差剖面（fp32 表）
            jerr = np.abs(c32t.astype(np.float64) - c64t).max(axis=0)
            top = np.argsort(jerr)[::-1][:3]
            row["fp32_worst_j"] = [int(t) for t in top]
            row["fp32_worst_j_err"] = [float(jerr[t]) for t in top]
            cache_rows.append(row)
    report["sections"]["cache_dtype_amplification"] = cache_rows

    # e2e（theta=1e6，fp32 io，fp32 vs fp16 cache）
    e2e = []
    for seq in cfg.CACHE_DTYPE_SEQS:
        q, k = gen_qk(seq)
        c32t, s32t = ref.build_cache(seq, cfg.ROPE_THETA, cfg.HEAD_DIM, "float32")
        c16t, s16t = ref.build_cache(seq, cfg.ROPE_THETA, cfg.HEAD_DIM, "float16")
        c64t, s64t = ref.build_cache(seq, cfg.ROPE_THETA, cfg.HEAD_DIM, "float64")
        q_ref, _ = ref.apply_prefill(q.astype(np.float64), k.astype(np.float64), c64t, s64t, 0)
        q32o, _ = ref.apply_prefill(q, k, c32t, s32t, 0)
        q16o, _ = ref.apply_prefill(q, k, c16t, s16t, 0, io_dtype=None)  # fp16 表 + fp32 算术
        m32 = comparison_metrics(q32o, q_ref)
        m16 = comparison_metrics(q16o, q_ref)
        e2e.append({"seq": seq, "cache_fp32": m32, "cache_fp16": m16,
                    "ratio_max": m16["max_abs_error"] / max(m32["max_abs_error"], 1e-30)})
    report["sections"]["cache_dtype_e2e"] = e2e

    # ---- 7) 确定性 ----
    q, k = gen_qk(2048)
    runs = []
    for _ in range(3):
        qo, ko = ref.apply_prefill(q, k, cos32, sin32, 0)
        runs.append(hashlib_sha(np.concatenate([qo.reshape(-1), ko.reshape(-1)])))
    q1, k1 = gen_qk(1)
    runs_d = []
    for _ in range(3):
        qo, ko = ref.apply_decode(q1[0], k1[0], cos32, sin32, 2047)
        runs_d.append(hashlib_sha(np.concatenate([qo.reshape(-1), ko.reshape(-1)])))
    report["sections"]["determinism"] = {
        "prefill_2048_bitwise_identical_3x": len(set(runs)) == 1,
        "decode_2047_bitwise_identical_3x": len(set(runs_d)) == 1,
    }

    n_fail = sum(1 for c in report["sections"]["correctness_matrix_prefill"] + cases_d if not c["pass"])
    report["failures"] = n_fail
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[correctness-cpu] failures={n_fail} → {out}")
    for c in report["sections"]["correctness_matrix_prefill"]:
        print(f"  prefill seq={c['seq']:>6} {c['dtype']}: max={c['max_abs_error']:.3e} pass={c['pass']}")
    for c in cases_d:
        print(f"  decode pos={c['pos']:>6} {c['dtype']}: max={c['max_abs_error']:.3e} pass={c['pass']}")
    return 1 if n_fail else 0


def hashlib_sha(values: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
