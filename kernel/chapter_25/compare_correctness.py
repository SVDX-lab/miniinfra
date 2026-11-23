#!/usr/bin/env python3
"""第 25 期正确性对照（复用第 01 期比较口径）。

对照链（按大纲 §9 DSA 条："给定索引集合时，稀疏 gather Attention 与'对同一
子集的 dense Attention'对照验证；索引器与参考实现对照；空集与非整块行为单独
记录"；参考实现以 CPU/PyTorch eager 为准）：

CPU 后端：
  A. §9 主对：fixtures 逐 N × 6 组索引集（strided/identity/single/random×3）：
     sparse gather（einsum 教学形式）vs 子集 dense（bmm 形式）vs FP64 锚点
  B. 索引器：FP32 主链 vs FP64 锚点（任务要求：得分容差单独报告）；
     FP32-on-fp16io 口径（CUDA 核的参考口径）附注
  C. 空集 k=0 / k>N 截断行为记录（§9）
  D. 确定性：3 次位级比对
  E. 端到端：DSA 步（索引器+topk+稀疏）vs dense 步（N=2048，k=512，
     信息误差=检索质量）+ FP64 真值锚点

CUDA 后端（4090）：
  F. 索引器核 vs CPU 参考链（fp16 io/fp32 math 容差）+ vs FP64 锚点（两条
     容差分别报告）
  G. topk 核 vs torch.topk（集合相等，k ∈ {1,33,127,512,2048}）
  H. gather 核：给定索引集 vs torch 子集 dense（fp16 io 容差）；identity 模式
     vs dense；k=N 恒等；FP64 链锚点（fp32 io 参考链）
  I. 边界：k=0 全零 / splits=64 强制 / 非整块 k
  J. 确定性：3 次位级比对

用法：
    python compare_correctness.py --backend cpu
    python compare_correctness.py --backend cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    comparison_metrics,
    device_environment_record,
    resolve_device,
    write_results_json,
)
import dsa_config as cfg  # noqa: E402
from dsa_reference import (  # noqa: E402
    SEED_BASE,
    absorbed_query,
    apply_rope,
    dense_attention_absorbed_bmm,
    dense_attention_absorbed_einsum,
    dense_decode_step,
    dsa_decode_step,
    index_quality,
    index_query_stage,
    index_scores,
    index_weights_stage,
    make_weights,
    q_projection_stage,
    rope_freqs,
    select_topk,
    sparse_gather_attention_bmm,
    sparse_gather_attention_einsum,
)

FIXTURES_DEFAULT = HERE / "fixtures" / "correctness.npz"


def load_fixtures(path: Path) -> tuple[dict, dict]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    with path.with_name("manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return arrays, manifest


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.detach().cpu().contiguous().numpy().tobytes() == \
        b.detach().cpu().contiguous().numpy().tobytes()


def load_cuda_module():
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    return build_ext.get_module()


def fixture_env(arrays: dict, device: torch.device, dtype: torch.dtype):
    """返回 (weights, x, freqs, per-N 缓存字典)——dtype 决定主链精度。"""
    weights = make_weights()
    weights = {k: v.to(device=device, dtype=dtype) for k, v in weights.items()}
    x = torch.randn(1, cfg.D_MODEL, generator=torch.Generator().manual_seed(SEED_BASE + 1)).to(device=device, dtype=dtype)
    freqs = rope_freqs().to(device=device)
    caches = {}
    for n in [int(k[1:].split("_")[0]) for k in arrays if k.startswith("n") and k[1:].split("_")[0].isdigit()]:
        key = f"n{n}_"
        caches[n] = {
            "c_kv": torch.from_numpy(arrays[key + "c_kv"]).to(device=device, dtype=dtype),
            "k_rope": torch.from_numpy(arrays[key + "k_rope_rot"]).to(device=device, dtype=dtype),
            "k_i": torch.from_numpy(arrays[key + "k_i"]).to(device=device, dtype=dtype),
        }
    return weights, x, freqs, caches


# ---------------------------------------------------------------------------
# A/E：CPU 链
# ---------------------------------------------------------------------------

def cpu_chain(arrays: dict) -> dict:
    weights32, x32, freqs, _ = fixture_env(arrays, torch.device("cpu"), torch.float32)
    w64 = {k: v.double() for k, v in weights32.items()}
    x64 = x32.double()
    groups = []
    set_names = ["strided", "identity", "single", "random0", "random1", "random2"]
    for n in sorted({int(k[1:].split("_")[0]) for k in arrays if k.startswith("n")}):
        prefix = f"n{n}_"
        c_kv = torch.from_numpy(arrays[prefix + "c_kv"])
        k_rope = torch.from_numpy(arrays[prefix + "k_rope_rot"])
        k_i = torch.from_numpy(arrays[prefix + "k_i"])
        pos_q = n
        q_nope, q_rope_raw = q_projection_stage(x32, weights32)
        q_rope = apply_rope(q_rope_raw, torch.tensor([pos_q], dtype=torch.float32), freqs)
        q_abs = absorbed_query(q_nope, weights32)
        q_nope64, q_rope_raw64 = q_projection_stage(x64, w64)
        q_abs64 = absorbed_query(q_nope64, w64)
        q_rope64 = apply_rope(q_projection_stage(x64, w64)[1].double(),
                              torch.tensor([pos_q], dtype=torch.float64), freqs.double())

        # A：给定索引集（§9 主对）
        set_results = []
        for name in set_names:
            tag = f"{prefix}set_{name}_"
            if tag + "indices" not in arrays:
                continue
            idx = torch.from_numpy(arrays[tag + "indices"]).long()
            sp32 = sparse_gather_attention_einsum(q_abs, q_rope, c_kv, k_rope, idx, weights32)
            sub32 = dense_attention_absorbed_bmm(q_abs, q_rope,
                                                 c_kv.index_select(0, idx),
                                                 k_rope.index_select(0, idx), weights32)
            sp64 = sparse_gather_attention_einsum(q_abs64, q_rope64,
                                                  c_kv.double(), k_rope.double(), idx, w64)
            entry = {
                "set": name,
                "k_selected": int(idx.numel()),
                "sparse_vs_subset_dense": comparison_metrics(sp32["s"], sub32["s"]),
                "sparse_vs_fp64": comparison_metrics(sp32["s"].double(), sp64["s"]),
                "out_sparse_vs_subset_dense": comparison_metrics(sp32["out_h"], sub32["out_h"]),
            }
            if idx.numel() == 0:
                entry["note"] = "空集：两侧均全零（§9 空集行为）"
            set_results.append(entry)
            print(f"[set/cpu] N={n} {name}: sparse-vs-subdense s max="
                  f"{entry['sparse_vs_subset_dense']['max_abs_error']:.3e} "
                  f"vs-FP64 max={entry['sparse_vs_fp64']['max_abs_error']:.3e}")

        # B：索引器 FP32 vs FP64（单独容差）
        q_i = index_query_stage(x32 @ weights32["W_QA"], weights32)
        w_used = index_weights_stage(x32, weights32)
        s32 = index_scores(q_i, w_used, k_i)
        s64 = torch.from_numpy(arrays[prefix + "index_scores_fp64"])
        fp32_err = comparison_metrics(s32.double(), s64)
        s32_fp16io = index_scores(q_i.half().float(), w_used, k_i.half().float())
        fp16io_err = comparison_metrics(s32_fp16io.double(), s64)

        # E：端到端
        k_e2e = min(512, n)
        dsa = dsa_decode_step(x32, weights32, c_kv, k_rope, k_i, k_e2e, pos_q, freqs)
        dense_step = dense_decode_step(x32, weights32, c_kv, k_rope, pos_q, freqs)
        truth64 = torch.from_numpy(arrays[prefix + "dense_out_fp64"])
        qual = index_quality(dense_step["scores"], dsa["indices"],
                             torch.from_numpy(arrays[prefix + "relevance_ids"]))
        e2e = {
            "k": k_e2e,
            "dsa_vs_dense_out": comparison_metrics(dsa["out"], dense_step["out"]),
            "dense_vs_truth_fp64": comparison_metrics(dense_step["out"], truth64),
            "quality": qual,
        }
        print(f"[e2e/cpu] N={n} k={k_e2e}: sparse-vs-dense out max="
              f"{e2e['dsa_vs_dense_out']['max_abs_error']:.4f} "
              f"captured={qual['captured_mass_mean']:.4f} recall={qual['relevance_recall']:.2f}")
        groups.append({"kv_len": n, "given_set": set_results,
                       "indexer_fp32_vs_fp64": fp32_err,
                       "indexer_fp32onfp16io_vs_fp64": fp16io_err,
                       "end_to_end": e2e})

    # C：空集 / 截断行为
    n = 128
    prefix = f"n{n}_"
    c_kv = torch.from_numpy(arrays[prefix + "c_kv"])
    k_rope = torch.from_numpy(arrays[prefix + "k_rope_rot"])
    k_i = torch.from_numpy(arrays[prefix + "k_i"])
    scores_i = torch.from_numpy(arrays[prefix + "index_scores_fp32"])
    q_nope, q_rope_raw = q_projection_stage(x32, weights32)
    q_rope = apply_rope(q_rope_raw, torch.tensor([n], dtype=torch.float32), freqs)
    q_abs = absorbed_query(q_nope, weights32)
    empty = sparse_gather_attention_einsum(q_abs, q_rope, c_kv, k_rope,
                                           torch.zeros(0, dtype=torch.long), weights32)
    idx_trunc = select_topk(scores_i, 256)
    behavior = {
        "empty_set": {
            "s_all_zero": bool((empty["s"] == 0).all()),
            "out_all_zero": bool((empty["out_h"] == 0).all()),
            "note": "torch softmax 空维 → s=0 → out=0；CUDA 核同语义返回全零（合同）",
        },
        "k_over_n_truncation": {
            "k_requested": 256, "n": n,
            "k_selected": int(idx_trunc.numel()),
            "matches_min_rule": idx_trunc.numel() == n,
            "note": "官方 topk = min(index_topk, seq_len) 同式",
        },
    }

    # D：确定性
    runs = [dsa_decode_step(x32, weights32, c_kv, k_rope, k_i, 128, n, freqs)["out"]
            for _ in range(3)]
    determinism = {
        "dtype": "float32", "kv_len": n,
        "bitwise_identical": bitwise_equal(runs[0], runs[1]) and bitwise_equal(runs[0], runs[2]),
    }
    return {"groups": groups, "edge_behavior": behavior, "determinism": determinism}


# ---------------------------------------------------------------------------
# F–J：CUDA 链
# ---------------------------------------------------------------------------

def cuda_chain(device: torch.device, arrays: dict) -> dict:
    module = load_cuda_module()
    weights16, x16, freqs, caches = fixture_env(arrays, device, torch.float16)
    results: dict = {"indexer": [], "topk": [], "attention": [], "edge": {}}

    for n in sorted(caches):
        prefix = f"n{n}_"
        cache = caches[n]
        q_nope, q_rope_raw = q_projection_stage(x16, weights16)
        q_rope = apply_rope(q_rope_raw, torch.tensor([n], dtype=torch.float32, device=device), freqs)
        q_abs = absorbed_query(q_nope, weights16)
        q_i = index_query_stage(x16 @ weights16["W_QA"], weights16)
        w_used = index_weights_stage(x16.float(), {k: v.float() for k, v in weights16.items()}).float()

        # F：索引器核（两档容差分别报告）
        s_kernel = module.dsa_indexer_scores(q_i, w_used, cache["k_i"], cfg.INDEX_SCALE)
        ref_fp16io = index_scores(q_i.float().cpu(), w_used.cpu(), cache["k_i"].float().cpu())
        s64 = torch.from_numpy(arrays[prefix + "index_scores_fp64"])
        entry = {
            "kv_len": n,
            "kernel_vs_cpu_ref_fp16io": comparison_metrics(s_kernel.cpu(), ref_fp16io),
            "kernel_vs_fp64_anchor": comparison_metrics(s_kernel.cpu().double(), s64),
            "cpu_ref_fp32_vs_fp64": comparison_metrics(ref_fp16io.double(), s64),
        }
        results["indexer"].append(entry)
        print(f"[indexer/cuda] N={n}: vs-ref fp16io max="
              f"{entry['kernel_vs_cpu_ref_fp16io']['max_abs_error']:.3e} "
              f"rel={entry['kernel_vs_cpu_ref_fp16io']['mean_rel_error']:.3e} | "
              f"vs-FP64 max={entry['kernel_vs_fp64_anchor']['max_abs_error']:.3e}")

        # G：topk 集合相等 + 确定性分段序（>T 段升序、==T 段升序两段拼接——
        # 非全局升序是设计属性：紧凑两段写，集合与 torch.topk 相等为准）
        for k in (1, 33, 127, 512, 2048):
            ours = module.dsa_topk(s_kernel, min(k, n)).cpu()
            ref = torch.topk(s_kernel.cpu(), min(k, n)).indices
            gt_len = int((ours[: len(ours)]).numel())  # 全长
            seg_asc = bool((ours[1:] >= ours[:-1]).all())  # 仅记录（预期多为 False）
            results["topk"].append({
                "kv_len": n, "k": min(k, n),
                "set_equal": set(ours.tolist()) == set(ref.tolist()),
                "segmented_note": "gt 段与 eq 段各自升序、eq 段接在 gt 段后——非全局升序",
                "globally_ascending_observed": seg_asc,
            })

        # H：gather 核给定索引集
        for name in ("strided", "identity", "single", "random0"):
            idx_key = f"{prefix}set_{name}_indices"
            if idx_key not in arrays:
                continue
            idx = torch.from_numpy(arrays[idx_key]).long().to(device=device)
            s_kernel_out = module.dsa_sparse_attention(q_abs, q_rope, cache["c_kv"],
                                                       cache["k_rope"], idx.int(), False,
                                                       cfg.SOFTMAX_SCALE, 0)
            sub_torch = sparse_gather_attention_bmm(q_abs, q_rope, cache["c_kv"],
                                                    cache["k_rope"], idx, weights16)
            s64 = torch.from_numpy(arrays[f"{prefix}set_{name}_sparse_fp64_s"])
            entry = {
                "set": name, "kv_len": n, "k_selected": int(idx.numel()),
                "kernel_vs_torch_subset_fp16": comparison_metrics(s_kernel_out.float(), sub_torch["s"].float()),
                "kernel_vs_fp64_s": comparison_metrics(s_kernel_out.float().double(), s64),
            }
            results["attention"].append(entry)
            print(f"[attention/cuda] N={n} {name}: vs-torch-subset max="
                  f"{entry['kernel_vs_torch_subset_fp16']['max_abs_error']:.3e} "
                  f"vs-FP64 s max={entry['kernel_vs_fp64_s']['max_abs_error']:.3e}")

        # identity 模式 vs dense torch（k=N）
        s_ident = module.dsa_sparse_attention(q_abs, q_rope, cache["c_kv"], cache["k_rope"],
                                              torch.zeros(0, device=device, dtype=torch.int32),
                                              True, cfg.SOFTMAX_SCALE, 0)
        dense16 = dense_attention_absorbed_bmm(q_abs, q_rope, cache["c_kv"], cache["k_rope"], weights16)
        results["attention"].append({
            "set": "identity_dense_mode", "kv_len": n, "k_selected": n,
            "kernel_vs_torch_dense_fp16": comparison_metrics(s_ident.float(), dense16["s"].float()),
        })

    # I：边界（N=128）
    n = 128
    cache = caches[n]
    q_nope, q_rope_raw = q_projection_stage(x16, weights16)
    q_rope = apply_rope(q_rope_raw, torch.tensor([n], dtype=torch.float32, device=device), freqs)
    q_abs = absorbed_query(q_nope, weights16)
    s0 = module.dsa_sparse_attention(q_abs, q_rope, cache["c_kv"], cache["k_rope"],
                                     torch.zeros(0, device=device, dtype=torch.int32),
                                     False, cfg.SOFTMAX_SCALE, 0)
    s64f = module.dsa_sparse_attention(q_abs, q_rope, cache["c_kv"], cache["k_rope"],
                                       torch.arange(min(3, n), device=device, dtype=torch.int32),
                                       False, cfg.SOFTMAX_SCALE, 64)
    s1f = module.dsa_sparse_attention(q_abs, q_rope, cache["c_kv"], cache["k_rope"],
                                      torch.arange(min(3, n), device=device, dtype=torch.int32),
                                      False, cfg.SOFTMAX_SCALE, 1)
    results["edge"] = {
        "empty_set_all_zero": bool((s0 == 0).all()),
        "splits64_vs_splits1_k3_max_abs": float((s64f.float() - s1f.float()).abs().max()),
    }
    print(f"[edge/cuda] k=0 全零={results['edge']['empty_set_all_zero']} "
          f"splits64-vs-1 k=3 max={results['edge']['splits64_vs_splits1_k3_max_abs']:.3e}")

    # J：确定性
    runs_i = [module.dsa_indexer_scores(
                  index_query_stage(x16 @ weights16["W_QA"], weights16),
                  index_weights_stage(x16.float(),
                                      {k: v.float() for k, v in weights16.items()}).float(),
                  caches[128]["k_i"], cfg.INDEX_SCALE) for _ in range(3)]
    runs_a = [module.dsa_sparse_attention(q_abs, q_rope, caches[128]["c_kv"], caches[128]["k_rope"],
                                          runs_i[0].topk(33).indices.int(), False,
                                          cfg.SOFTMAX_SCALE, 0) for _ in range(3)]
    results["determinism"] = {
        "indexer_bitwise": bitwise_equal(runs_i[0], runs_i[1]) and bitwise_equal(runs_i[0], runs_i[2]),
        "attention_bitwise": bitwise_equal(runs_a[0], runs_a[1]) and bitwise_equal(runs_a[0], runs_a[2]),
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--fixtures", default=str(FIXTURES_DEFAULT))
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    arrays, manifest = load_fixtures(Path(arguments.fixtures))
    results: dict = {
        "schema_version": "kernel-chapter26.correctness.v1",
        "backend": arguments.backend,
        "config": manifest["config"],
        "config_source": manifest["config_source"],
        "fixture_kv_lens": manifest["kv_lens"],
        "relevance_setting": manifest["relevance_setting"],
    }
    if arguments.backend == "cpu":
        results["cpu_chain"] = cpu_chain(arrays)
    else:
        results["cuda_chain"] = cuda_chain(device, arrays)
    results["environment"] = device_environment_record(device)
    output = HERE / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
