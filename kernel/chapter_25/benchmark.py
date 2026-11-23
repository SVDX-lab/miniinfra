#!/usr/bin/env python3
"""第 25 期统一计时口径基准（decode 形态：单 query，N×k 扫描矩阵）。

实验映射：
  实验 2（索引器开销占总时长比例）：per (N,k) 记录 indexer/topk/sparse 三段
     P50，ratio = (indexer+topk)/(indexer+topk+sparse)；
  实验 3（三元关系曲线）：同一次扫描输出 误差（sparse vs dense full
     attention，同一合成权重序列）+ 延迟 + 检索质量上下文
     （captured/oracle/recall）；
  附加：dense 基线（identity 模式同核 + torch eager 链）、Design A 代理
     （k_I 每步从 latent 重算的 GEMM）、理论口径（roofline_dsa.sweep_matrix：
     MAC/字节比、break-even、工作集 L2/DRAM 分区、延迟下限）。

口径：
  - 计时器复用 ch01 口径（P50/P95，warmup/repeats 入结果）；CUDA 事件对逐次
    记录、末尾一次同步；
  - CUDA io fp16（fp32 片上累加）；CPU 列为参考实现 torch eager（fp32），
    非 kernel 级——声明为"参考实现计时"；
  - 输入运行时生成（seed 规则同 fixtures：SEED_BASE+1000+N 等，planted
    relevance 同一规则——三 N 的支撑集 S=256/410/1638）；
  - 误差定义（实验 3）：sparse（k 个选中 token）vs dense full attention
    （全部 N 个 token，FP32 参考），同一合成权重/同一 cache；另记
    sparse vs dense-fp16kernel 以分离数值噪声与信息误差；
  - 索引器开销口径（实验 2 主口径）：indexer_score + topk（每 query 必须
    支付的索引段）；Design A（重算 k_I 的 GEMM）单列，不进主口径
    （官方对齐 Design B：索引 K 独立缓存）。

用法：
    python benchmark.py --backend cpu
    python benchmark.py --backend cuda
    python benchmark.py --backend cuda --profile-mode core --kv-len 65536
                        --repeats 10   （nsys 被测最小进程模式）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    OperatorTimer,
    device_environment_record,
    resolve_device,
    write_results_json,
)
import dsa_config as cfg  # noqa: E402
from dsa_reference import (  # noqa: E402
    SEED_BASE,
    absorbed_query,
    apply_rope,
    calibrate_plants,
    dense_attention_absorbed_bmm,
    generate_cache,
    index_quality,
    index_query_stage,
    index_scores,
    index_weights_stage,
    make_weights,
    plant_directions,
    q_projection_stage,
    rope_freqs,
    sparse_gather_attention_bmm,
)
from roofline_dsa import PLATFORM_PEAKS, sweep_matrix  # noqa: E402

N_SWEEP = list(cfg.N_SWEEP)
K_SWEEP = list(cfg.K_SWEEP)
TOPK_SWEEP = list(cfg.TOPK_SWEEP)


def load_cuda_module():
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    return build_ext.get_module()


def prepare_case(backend: str, device: torch.device, dtype: torch.dtype, n: int):
    """运行时生成 N 的完整输入（与 fixtures 同 seed 规则），返回全部张量。"""
    weights32 = make_weights()
    x32 = torch.randn(1, cfg.D_MODEL, generator=torch.Generator().manual_seed(SEED_BASE + 1))
    freqs32 = rope_freqs()
    dirs = plant_directions(x32, weights32)
    calib = calibrate_plants(n, weights32, x32, dirs)
    cache = generate_cache(n, weights32, x32, freqs32, calib)
    relevance = cache["relevance_ids"]

    weights_on_dev32 = {k: v.to(device=device) for k, v in weights32.items()}
    weights = {k: v.to(device=device, dtype=dtype) for k, v in weights32.items()}
    x = x32.to(device=device, dtype=dtype)
    freqs = freqs32.to(device=device)
    c_kv = cache["c_kv"].to(device=device, dtype=dtype)
    k_rope = cache["k_rope_rot"].to(device=device, dtype=dtype)
    k_i = cache["k_i"].to(device=device, dtype=dtype)
    w_kl = weights32["W_KL"].to(device=device, dtype=dtype)
    relevance_d = relevance.to(device=device)

    q_nope, q_rope_raw = q_projection_stage(x, weights)
    q_rope = apply_rope(q_rope_raw, torch.tensor([n], dtype=torch.float32, device=device), freqs)
    q_abs = absorbed_query(q_nope, weights)
    q_i = index_query_stage(x @ weights["W_QA"], weights)
    w_used = index_weights_stage(x.float(), weights_on_dev32).float()

    # FP32 参考（误差定义用：dense full attention，同 cache）
    c_kv32 = cache["c_kv"].to(device=device)
    k_rope32 = cache["k_rope_rot"].to(device=device)
    weights_on_dev = weights_on_dev32
    x_dev = x32.to(device=device)
    q_nope32, q_rope_raw32 = q_projection_stage(x_dev, weights_on_dev)
    q_abs32 = absorbed_query(q_nope32, weights_on_dev)
    q_rope32 = apply_rope(q_rope_raw32,
                          torch.tensor([n], dtype=torch.float32, device=device),
                          freqs32.to(device))
    return {
        "weights": weights, "weights32": weights_on_dev, "x": x, "freqs": freqs,
        "c_kv": c_kv, "k_rope": k_rope, "k_i": k_i, "w_kl": w_kl,
        "q_abs": q_abs, "q_rope": q_rope, "q_i": q_i, "w_used": w_used,
        "relevance": relevance_d, "relevance_cpu": relevance,
        "c_kv32": c_kv32, "k_rope32": k_rope32, "q_abs32": q_abs32, "q_rope32": q_rope32,
    }


def dense_scores_fp32(case) -> torch.Tensor:
    """dense full attention 分数（FP32，误差/质量参考）。"""
    scores = (torch.matmul(case["q_abs32"], case["c_kv32"].t())
              + case["q_rope32"] @ case["k_rope32"].t()) * cfg.SOFTMAX_SCALE
    return scores


def bench_backend(backend: str, profile_mode: str | None, profile_kv_len: int | None,
                  warmup: int, repeats: int) -> dict:
    device = resolve_device(backend)
    dtype = torch.float16 if backend == "cuda" else torch.float32
    module = load_cuda_module() if backend == "cuda" else None
    timer = OperatorTimer(device, warmup, repeats)
    results: dict = {
        "schema_version": "kernel-chapter26.benchmark.v1",
        "backend": backend,
        "dtype": str(dtype).replace("torch.", ""),
        "n_sweep": N_SWEEP, "k_sweep": K_SWEEP, "topk_sweep": TOPK_SWEEP,
        "timing": {"warmup": warmup, "repeats": repeats},
        "indexer_overhead_definition":
            "ratio = (indexer_score + topk) / (indexer_score + topk + sparse_attention)"
            "（Design B：索引 K 独立缓存，官方对齐）；k_I 重算 GEMM 单列 design_a_proxy",
        "platform_peaks": PLATFORM_PEAKS,
        "per_n": [],
    }
    lens = [profile_kv_len] if profile_kv_len else N_SWEEP
    theory_rows = {(row["n"], row["k"]): row for row in sweep_matrix()} if profile_mode is None else {}

    for n in lens:
        case = prepare_case(backend, device, dtype, n)
        entry: dict = {"n": n, "support": cfg.relevance_support(n),
                       "tier_sizes": cfg.tier_sizes(n)}

        # dense 参考（fp32 分数 + fp16/fp32 dense 核延迟）
        scores32 = dense_scores_fp32(case)
        attn32 = torch.softmax(scores32, dim=-1)
        dense_ref_s = torch.matmul(attn32, case["c_kv32"])                      # [H,512] fp32
        dense_ref_out = torch.matmul(dense_ref_s, case["weights32"]["W_UV"])

        if profile_mode in (None, "core"):
            if backend == "cuda":
                zero_idx = torch.zeros(0, device=device, dtype=torch.int32)
                entry["dense_identity_kernel"] = timer.measure(
                    lambda: module.dsa_sparse_attention(
                        case["q_abs"], case["q_rope"], case["c_kv"], case["k_rope"],
                        zero_idx, True, cfg.SOFTMAX_SCALE, 0))
            entry["dense_torch_bmm"] = timer.measure(
                lambda: dense_attention_absorbed_bmm(case["q_abs"], case["q_rope"],
                                                     case["c_kv"], case["k_rope"],
                                                     case["weights"]))
            # Design A 代理：k_I 每步从 latent 重算（写侧 GEMM 的成本下界代理）
            entry["design_a_kI_gemm_proxy"] = timer.measure(
                lambda: case["c_kv"] @ case["w_kl"])

        per_k = []
        ks = [k for k in K_SWEEP if k <= n]
        for k in ks:
            k_entry: dict = {"k": k}
            # ---- 三段计时（每 query 管线）----
            if backend == "cuda":
                k_entry["indexer_score"] = timer.measure(
                    lambda: module.dsa_indexer_scores(case["q_i"], case["w_used"],
                                                      case["k_i"], cfg.INDEX_SCALE))
                scores_gpu = module.dsa_indexer_scores(case["q_i"], case["w_used"],
                                                       case["k_i"], cfg.INDEX_SCALE)
                k_entry["topk_naive"] = timer.measure(lambda: module.dsa_topk(scores_gpu, k))
                k_entry["topk_torch"] = timer.measure(lambda: torch.topk(scores_gpu, k).indices)
                indices = module.dsa_topk(scores_gpu, k)
                k_entry["sparse_gather"] = timer.measure(
                    lambda: module.dsa_sparse_attention(
                        case["q_abs"], case["q_rope"], case["c_kv"], case["k_rope"],
                        indices, False, cfg.SOFTMAX_SCALE, 0))
                s_sparse = module.dsa_sparse_attention(
                    case["q_abs"], case["q_rope"], case["c_kv"], case["k_rope"],
                    indices, False, cfg.SOFTMAX_SCALE, 0).float()
            else:
                k_entry["indexer_score"] = timer.measure(
                    lambda: index_scores(case["q_i"], case["w_used"], case["k_i"]))
                scores_cpu = index_scores(case["q_i"], case["w_used"], case["k_i"])
                k_entry["topk_torch"] = timer.measure(lambda: torch.topk(scores_cpu, k).indices)
                indices = torch.topk(scores_cpu, k).indices
                k_entry["sparse_gather"] = timer.measure(
                    lambda: sparse_gather_attention_bmm(case["q_abs"], case["q_rope"],
                                                        case["c_kv"], case["k_rope"],
                                                        indices, case["weights"]))
                s_sparse = sparse_gather_attention_bmm(
                    case["q_abs"], case["q_rope"], case["c_kv"], case["k_rope"],
                    indices, case["weights"])["s"]
                scores_gpu = scores_cpu

            # ---- 误差（三元关系第 2 维）----
            p50 = lambda name: k_entry[name]["p50_ms"]
            total_ms = p50("indexer_score") + p50("topk_naive" if backend == "cuda" else "topk_torch") + p50("sparse_gather")
            index_ms = p50("indexer_score") + p50("topk_naive" if backend == "cuda" else "topk_torch")
            k_entry["dsa_total_ms"] = round(total_ms, 4)
            k_entry["indexer_overhead_ratio"] = round(index_ms / total_ms, 4)
            k_entry["indexer_overhead_ratio_score_only"] = round(p50("indexer_score") / total_ms, 4)
            dense16_s = dense_attention_absorbed_bmm(case["q_abs"], case["q_rope"],
                                                     case["c_kv"], case["k_rope"],
                                                     case["weights"])["s"]
            k_entry["error_sparse_vs_dense_fp32"] = {
                "s_max_abs": float((s_sparse - dense_ref_s).abs().max()),
                "s_mean_abs": float((s_sparse - dense_ref_s).abs().mean()),
                "out_max_abs": float((torch.matmul(s_sparse, case["weights32"]["W_UV"])
                                      - dense_ref_out).abs().max()),
                "cosine_s": float(torch.nn.functional.cosine_similarity(
                    s_sparse.flatten(), dense_ref_s.flatten(), dim=0)),
            }
            k_entry["error_sparse_vs_dense_fp16kernel"] = {
                "s_max_abs": float((s_sparse - dense16_s.float()).abs().max()),
            }
            k_entry["quality"] = index_quality(scores32, indices.cpu(), case["relevance_cpu"])
            # 理论口径
            if (n, k) in theory_rows:
                row = theory_rows[(n, k)]
                k_entry["theory"] = {
                    "dsa_b_macs": row["dsa_b_macs"], "dense_macs": row["dense_macs"],
                    "dsa_b_bytes": row["dsa_b_bytes"], "dense_bytes": row["dense_bytes"],
                    "index_share_of_macs": row["index_share_of_macs"],
                    "mac_ratio_vs_dense": row["mac_ratio_vs_dense"],
                    "byte_ratio_vs_dense": row["byte_ratio_vs_dense"],
                    "working_set": row["working_set"],
                    "lower_bounds_ms": row["lower_bounds_ms"],
                }
            per_k.append(k_entry)
            print(f"[bench/{backend}] N={n} k={k}: indexer={p50('indexer_score'):.3f} "
                  f"topk={p50('topk_naive' if backend == 'cuda' else 'topk_torch'):.3f} "
                  f"sparse={p50('sparse_gather'):.3f} ms | ratio={k_entry['indexer_overhead_ratio']:.3f} "
                  f"err_s={k_entry['error_sparse_vs_dense_fp32']['s_max_abs']:.4f} "
                  f"captured={k_entry['quality']['captured_mass_mean']:.3f}")
        entry["per_k"] = per_k
        results["per_n"].append(entry)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--profile-mode", choices=["full", "core"], default=None)
    parser.add_argument("--kv-len", type=int, default=None)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    if arguments.backend == "cpu":
        warmup = arguments.warmup if arguments.warmup is not None else 2
        repeats = arguments.repeats if arguments.repeats is not None else 5
    else:
        warmup = arguments.warmup if arguments.warmup is not None else 3
        repeats = arguments.repeats if arguments.repeats is not None else 10

    results = bench_backend(arguments.backend, arguments.profile_mode,
                            arguments.kv_len, warmup, repeats)
    results["environment"] = device_environment_record(resolve_device(arguments.backend))
    suffix = f"-{arguments.profile_mode}-n{arguments.kv_len}" if arguments.profile_mode else ""
    output = Path(arguments.out) if arguments.out else \
        HERE / "results" / f"benchmark-results-{arguments.backend}{suffix}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
