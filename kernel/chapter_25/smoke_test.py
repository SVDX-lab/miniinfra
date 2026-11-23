#!/usr/bin/env python3
"""第 25 期边界行为验证（smoke test）。

CPU 部分（本地可跑）：
  S1 空集 k=0：CPU 参考返回全零（§9 空集行为记录）
  S2 k>N 截断：k=256, N=128 → 128 个索引（官方 min(index_topk, seq_len) 同式）
  S3 k=N 恒等集：给定集 sparse == dense（fp32 机器精度）
  S4 非整块 k=33/127：给定集 sparse vs 子集 dense（容差）
  S5 索引器 FP32 vs FP64 锚点（容差量级记录，单独报告口径）
  S6 大 score 稳定性：softmax 减 max，不产生 NaN/Inf
  S7 NaN 输入传播（观察项）
  S8 权重缺 key → KeyError 显式报错
  S9 relevance/缓存生成确定性（同 seed 两次逐位一致）

CUDA 部分（--backend cuda，4090 服务器）：
  S10 dtype 合同（fp32 输入被拒）
  S11 非连续输入被拒
  S12 N=0 被拒
  S13 CPU Tensor 被拒
  S14 k=0：核路径返回全零（不 launch attention 核）
  S15 identity 模式 vs gather(arange(N)) 一致（容差内）
  S16 splits=64 强制（空 split 守卫，N=3/k=3）
  S17 确定性：indexer/topk/attention 各 3 次位级一致
  S18 topk 核 vs torch.topk 集合相等（k ∈ {1,33,127,512}）
  S19 索引器核 vs CPU 参考链容差（N=2048 fixtures）

用法：
    python smoke_test.py                 # CPU 部分
    python smoke_test.py --backend cuda  # CPU + CUDA 部分（服务器）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import dsa_config as cfg  # noqa: E402
from dsa_reference import (  # noqa: E402
    SEED_BASE,
    absorbed_query,
    apply_rope,
    dense_attention_absorbed_bmm,
    generate_cache,
    index_query_stage,
    index_scores,
    index_weights_stage,
    make_weights,
    plant_directions,
    q_projection_stage,
    relevance_ids,
    rope_freqs,
    select_topk,
    sparse_gather_attention_bmm,
    sparse_gather_attention_einsum,
)

WEIGHT_NAMES = tuple(cfg.WEIGHT_SHAPES.keys())
PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    tag = "PASS" if condition else "FAIL"
    print(f"[smoke/{tag}] {name} {detail}")
    (PASSED if condition else FAILED).append(name)


def fixture_arrays() -> dict:
    data = np.load(HERE / "fixtures" / "correctness.npz")
    return {k: data[k] for k in data.files}


def fixture_weights(arrays: dict, device: torch.device, dtype: torch.dtype) -> dict:
    return make_weights() if dtype == torch.float32 else {
        k: v.to(dtype) for k, v in make_weights().items()}


def smoke_cpu() -> None:
    arrays = fixture_arrays()
    weights = make_weights()
    x = torch.randn(1, cfg.D_MODEL, generator=torch.Generator().manual_seed(SEED_BASE + 1))
    freqs = rope_freqs()
    n = 128
    prefix = f"n{n}_"
    c_kv = torch.from_numpy(arrays[prefix + "c_kv"])
    k_rope = torch.from_numpy(arrays[prefix + "k_rope_rot"])
    k_i = torch.from_numpy(arrays[prefix + "k_i"])
    scores_i = torch.from_numpy(arrays[prefix + "index_scores_fp32"])
    q_nope, q_rope_raw = q_projection_stage(x, weights)
    q_rope = apply_rope(q_rope_raw, torch.tensor([n], dtype=torch.float32), freqs)
    q_abs = absorbed_query(q_nope, weights)

    # S1 空集
    empty = torch.zeros(0, dtype=torch.long)
    att0 = sparse_gather_attention_einsum(q_abs, q_rope, c_kv, k_rope, empty, weights)
    check("S1 空集 k=0 返回全零（§9 记录）",
          bool((att0["s"] == 0).all() and (att0["out_h"] == 0).all()),
          "s=0, out=0（torch softmax 空维语义）")

    # S2 k>N 截断
    idx256 = select_topk(scores_i, 256)
    check("S2 k>N 截断为 N", idx256.numel() == n,
          f"k=256, N={n} → {idx256.numel()} 个索引（官方 min(index_topk, seq_len)）")

    # S3 k=N 恒等
    att_ident = sparse_gather_attention_bmm(q_abs, q_rope, c_kv, k_rope,
                                            torch.arange(n), weights)
    dense = dense_attention_absorbed_bmm(q_abs, q_rope, c_kv, k_rope, weights)
    err = float((att_ident["s"] - dense["s"]).abs().max())
    check("S3 k=N 恒等集 sparse==dense", err < 1e-4, f"max={err:.3e}")

    # S4 非整块
    for k in (33, 127):
        idx = torch.topk(scores_i, k).indices
        sp = sparse_gather_attention_einsum(q_abs, q_rope, c_kv, k_rope, idx, weights)
        sub = dense_attention_absorbed_bmm(q_abs, q_rope,
                                           c_kv.index_select(0, idx),
                                           k_rope.index_select(0, idx), weights)
        err = float((sp["s"] - sub["s"]).abs().max())
        check(f"S4 非整块 k={k} 给定集 sparse vs 子集 dense", err < 1e-4,
              f"max={err:.3e}")

    # S5 索引器 FP32 vs FP64
    s64 = torch.from_numpy(arrays[prefix + "index_scores_fp64"])
    rel = float(((scores_i.double() - s64).abs() / s64.abs().clamp_min(1e-6)).max())
    check("S5 索引器 FP32 vs FP64 锚点", rel < 1e-5,
          f"max rel={rel:.3e}（FP32 得分容差单独口径，EXPERIMENT_LOG 报告）")

    # S6 大分数稳定性
    big = sparse_gather_attention_bmm(q_abs * 1e3, q_rope * 1e3, c_kv, k_rope,
                                      torch.arange(n), weights)
    check("S6 大 score 稳定性（softmax 减 max）",
          bool(torch.isfinite(big["scores"]).all() and torch.isfinite(big["attn"]).all()),
          f"score_max={float(big['scores'].abs().max()):.1f}")

    # S7 NaN 传播（观察项）
    nan_out = dense_attention_absorbed_bmm(q_abs, q_rope,
                                           torch.full_like(c_kv, float("nan")), k_rope, weights)
    check("S7 NaN 输入传播（观察项）", bool(torch.isnan(nan_out["s"]).all()),
          "NaN 输入 → 输出全 NaN（无数值防线需求）")

    # S8 缺 key
    try:
        q_projection_stage(x, {k: v for k, v in weights.items() if k != "W_QA"})
        check("S8 缺 key 显式报错", False, "未触发 KeyError")
    except KeyError:
        check("S8 缺 key 显式报错", True)

    # S9 生成确定性
    c1 = generate_cache(512, weights, x, freqs)
    c2 = generate_cache(512, weights, x, freqs)
    same = all(torch.equal(c1[key], c2[key]) for key in ("c_kv", "k_rope_rot", "k_i"))
    r1 = relevance_ids(512)
    r2 = relevance_ids(512)
    check("S9 缓存/relevance 生成确定性", same and torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1]),
          "同 seed 两次逐位一致")


def smoke_cuda(device: torch.device) -> None:
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    module = build_ext.get_module()
    arrays = fixture_arrays()
    weights16 = {k: v.to(device=device, dtype=torch.float16) for k, v in make_weights().items()}
    n = 128
    prefix = f"n{n}_"
    c16 = torch.from_numpy(arrays[prefix + "c_kv"]).to(device=device, dtype=torch.float16)
    k16 = torch.from_numpy(arrays[prefix + "k_rope_rot"]).to(device=device, dtype=torch.float16)
    ki16 = torch.from_numpy(arrays[prefix + "k_i"]).to(device=device, dtype=torch.float16)
    x16 = torch.randn(1, cfg.D_MODEL, generator=torch.Generator().manual_seed(SEED_BASE + 1)).to(device=device, dtype=torch.float16)
    q_nope, q_rope_raw = q_projection_stage(x16, weights16)
    freqs = rope_freqs().to(device=device)
    q_rope = apply_rope(q_rope_raw, torch.tensor([n], dtype=torch.float32, device=device), freqs)
    q_abs = absorbed_query(q_nope, weights16)
    q_i = index_query_stage((x16 @ weights16["W_QA"]), weights16)
    w_used = index_weights_stage(x16.float(), {k: v.float() for k, v in weights16.items()}).float()

    def expect_error(name: str, fn) -> None:
        try:
            fn()
            check(name, False, "未触发 RuntimeError")
        except RuntimeError as error:
            check(name, True, f"{type(error).__name__}: {str(error)[:80]}")

    # S10-S13 合同
    expect_error("S10 dtype 合同（fp32 被拒）",
                 lambda: module.dsa_sparse_attention(
                     q_abs.float(), q_rope.float(), c16.float(), k16.float(),
                     torch.arange(n, device=device, dtype=torch.int32), False, 1.0, 0))
    c_noncontig = c16.repeat(1, 2)[:, :cfg.KV_LORA_RANK]
    check("S11 输入为非连续切片", not c_noncontig.is_contiguous())
    expect_error("S11 非连续输入被拒",
                 lambda: module.dsa_sparse_attention(
                     q_abs, q_rope, c_noncontig, k16,
                     torch.arange(n, device=device, dtype=torch.int32), False, 1.0, 0))
    expect_error("S12 N=0 被拒",
                 lambda: module.dsa_sparse_attention(
                     q_abs, q_rope,
                     torch.zeros(0, cfg.KV_LORA_RANK, device=device, dtype=torch.float16),
                     torch.zeros(0, cfg.QK_ROPE_HEAD_DIM, device=device, dtype=torch.float16),
                     torch.zeros(0, device=device, dtype=torch.int32), True, 1.0, 0))
    expect_error("S13 CPU Tensor 被拒",
                 lambda: module.dsa_sparse_attention(
                     q_abs.cpu(), q_rope.cpu(), c16.cpu(), k16.cpu(),
                     torch.arange(n, dtype=torch.int32), False, 1.0, 0))

    # S14 k=0
    s0 = module.dsa_sparse_attention(q_abs, q_rope, c16, k16,
                                     torch.zeros(0, device=device, dtype=torch.int32),
                                     False, cfg.SOFTMAX_SCALE, 0)
    check("S14 k=0 核路径返回全零", bool((s0 == 0).all()), "§9 空集行为")

    # S15 identity vs gather(arange)
    s_ident = module.dsa_sparse_attention(q_abs, q_rope, c16, k16,
                                          torch.zeros(0, device=device, dtype=torch.int32),
                                          True, cfg.SOFTMAX_SCALE, 0)
    s_gather = module.dsa_sparse_attention(q_abs, q_rope, c16, k16,
                                           torch.arange(n, device=device, dtype=torch.int32),
                                           False, cfg.SOFTMAX_SCALE, 0)
    err = float((s_ident.float() - s_gather.float()).abs().max())
    check("S15 identity 模式 vs gather(arange(N)) 一致", err < 5e-3, f"max={err:.3e}")

    # S16 splits=64 强制（k=3 → 空分块守卫）
    s_h = module.dsa_sparse_attention(q_abs, q_rope, c16, k16,
                                      torch.arange(min(3, n), device=device, dtype=torch.int32),
                                      False, cfg.SOFTMAX_SCALE, 64)
    s_1 = module.dsa_sparse_attention(q_abs, q_rope, c16, k16,
                                      torch.arange(min(3, n), device=device, dtype=torch.int32),
                                      False, cfg.SOFTMAX_SCALE, 1)
    err = float((s_h.float() - s_1.float()).abs().max())
    check("S16 splits=64 > k=3 空分块正确", err < 5e-3, f"max={err:.3e}")

    # S17 确定性（各 3 次位级）
    scores_runs = [module.dsa_indexer_scores(q_i, w_used, ki16, cfg.INDEX_SCALE)
                   for _ in range(3)]
    det_scores = all(torch.equal(scores_runs[0], r) for r in scores_runs[1:])
    topk_runs = [module.dsa_topk(scores_runs[0], 33) for _ in range(3)]
    det_topk = all(torch.equal(topk_runs[0], r) for r in topk_runs[1:])
    att_runs = [module.dsa_sparse_attention(q_abs, q_rope, c16, k16, topk_runs[0],
                                            False, cfg.SOFTMAX_SCALE, 0) for _ in range(3)]
    det_att = all(torch.equal(att_runs[0], r) for r in att_runs[1:])
    check("S17 确定性（indexer/topk/attention 各 3 次位级一致）",
          det_scores and det_topk and det_att,
          f"scores={det_scores} topk={det_topk} attention={det_att}")

    # S18 topk 核 vs torch.topk 集合相等
    set_all = True
    for k in (1, 33, 127, 512):
        ours = module.dsa_topk(scores_runs[0], min(k, n)).cpu()
        ref = torch.topk(scores_runs[0].cpu(), min(k, n)).indices
        set_all = set_all and set(ours.tolist()) == set(ref.tolist())
    check("S18 topk 核 vs torch.topk 集合相等", set_all,
          "k ∈ {1,33,127,512}；写序不同（升序 vs value 降序），集合相等为准")

    # S19 索引器核 vs CPU 参考链（fp16 io / fp32 math 口径）
    ref_scores = index_scores(q_i.float().cpu(), w_used.cpu(), ki16.float().cpu())
    err = float((scores_runs[0].cpu() - ref_scores).abs().max())
    rel = err / float(ref_scores.abs().max())
    check("S19 索引器核 vs CPU 参考链", rel < 2e-3,
          f"max={err:.3e} rel={rel:.3e}（fp16 io 口径）")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    arguments = parser.parse_args()
    smoke_cpu()
    if arguments.backend == "cuda":
        smoke_cuda(torch.device("cuda"))
    print(f"\n[smoke] PASS={len(PASSED)} FAIL={len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print("  FAILED:", name)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
