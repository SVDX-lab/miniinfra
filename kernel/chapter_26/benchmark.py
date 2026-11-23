#!/usr/bin/env python3
"""第 26 期实验 2-5（统一计时口径，见 contract 与 EXPERIMENT_LOG）。

exp2  接受率 vs draft 长度 k：k ∈ {1,2,3,4,6,8} × 6 个 prompt（fixtures），
      逐位置接受率曲线（验证"首 token 高后递减"形态）。
exp3  接受率 vs draft 质量：reuse / reuse_hidden / random / copy 四种 draft
      在 k ∈ {4, 8} 的对照（reuse 的 k∈{4,8} 数据从 exp2 复用）。
exp4  加速比实测 vs 理论：P01 全生成 wall-clock（fast_greedy 基线 vs 投机
      k 扫描，repeats=3）；前向成本标定曲线（T_target(n) / T_draft(n)）；
      理论 (αk+1)/(1+ck) 与全标定模型对照实测。
exp5  verify 批量化收益归因：一次 (k+1) 位置批量前向 vs 逐 token (k+1) 次
      前向（贪心路径）的时间比；CPU 列实测，CUDA/MUSA 待补。

计时口径：CPU 用 time.perf_counter（OperatorTimer 同款：warmup 后逐样本，
P50/P95 入结果）；端到端生成 wall-clock 单次/多次指定。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ep27_common  # noqa: E402

ep27_common.bootstrap()

import torch  # noqa: E402

from experiment_utils import device_environment_record, write_results_json  # noqa: E402
from mtp_reference import (  # noqa: E402
    SpeculativeDecoder,
    backbone_hidden,
    build_draft,
    fast_greedy_generate,
    idealized_speedup,
    logits_at_rows,
    summarize_spec_stats,
)

SCHEMA_VERSION = "kernel-chapter27.benchmark.v1"
K_SWEEP = [1, 2, 3, 4, 6, 8]


def p50(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2]


def timed_forward_hidden(model, ids: list[int], warmup: int, repeats: int) -> dict:
    """测量目标模型 backbone 前向（不含 LM Head）。CPU 报 min（噪声鲁棒）+ P50。"""
    tensor = torch.tensor([ids], dtype=torch.long)
    for _ in range(warmup):
        backbone_hidden(model, tensor)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        backbone_hidden(model, tensor)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "length": len(ids),
        "p50_ms": round(p50(samples), 3),
        "min_ms": round(min(samples), 3),
        "repeats": repeats,
    }


def timed_draft_step(draft_model, ids: list[int], warmup: int, repeats: int) -> dict:
    """测量 draft 单步（单层前向 + LM 头投影）。min + P50。"""
    for _ in range(warmup):
        draft_model.forward_hidden(ids, None)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        draft_model.forward_hidden(ids, None)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "length": len(ids),
        "p50_ms": round(p50(samples), 4),
        "min_ms": round(min(samples), 4),
        "repeats": repeats,
    }


def run_exp2(model, fixtures) -> dict:
    """接受率 vs k（6 prompt × k 扫描 × 2 个 draft）。

    draft=reuse：未训练 MTP 形态（argmax 接受率接近 0 的真实下界）；
    draft=lookup：prompt-lookup 启发式（重复文本下接受率首位置高、后递减
    的形态来源）。两条曲线同图对照即"接受率—k—draft 质量"三元关系。
    """
    eos = fixtures["prompts"][0]["eos_token_id"]
    per_draft = {}
    for mode in ("reuse", "lookup"):
        draft_model = build_draft(model, mode)
        decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos)
        per_prompt = []
        for entry in fixtures["prompts"]:
            prompt_ids = entry["prompt_token_ids"]
            per_k = []
            for k in K_SWEEP:
                generated, finished, stats = decoder.generate(prompt_ids, 32, k)
                summary = summarize_spec_stats(stats)
                summary["k"] = k
                summary["finished_by_eos"] = finished
                summary["generated_token_count"] = len(generated) - len(prompt_ids)
                per_k.append(summary)
                print(
                    f"[exp2] {mode} {entry['id']} k={k}: alpha={summary['alpha_over_drafted']} "
                    f"pos1={summary['per_position'][0]['accept_rate_over_evaluated'] if summary['per_position'] else None}"
                )
            per_prompt.append({"prompt_id": entry["id"], "note": entry["note"], "per_k": per_k})
        per_draft[mode] = per_prompt
    return {"per_draft": per_draft}


def run_exp3(model, fixtures) -> dict:
    """draft 质量对照（reuse_hidden / random / copy / lookup2，k ∈ {4,8}）。

    reuse 与 lookup 的 k∈{4,8} 数据从 exp2 复用（同模型同 prompt 同口径）。
    """
    eos = fixtures["prompts"][0]["eos_token_id"]
    per_draft = {}
    for mode in ["reuse_hidden", "random", "copy", "lookup2"]:
        draft_model = build_draft(model, mode, seed=fixtures["draft_seeds"]["random_main"])
        decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos)
        rows = []
        for k in (4, 8):
            aggregate_accepted = 0
            aggregate_drafted = 0
            per_prompt = []
            for entry in fixtures["prompts"]:
                generated, _, stats = decoder.generate(entry["prompt_token_ids"], 32, k)
                summary = summarize_spec_stats(stats)
                per_prompt.append(
                    {
                        "prompt_id": entry["id"],
                        "alpha": summary["alpha_over_drafted"],
                        "mean_accepted_per_iteration": summary["mean_accepted_per_iteration"],
                        "iterations": summary["iterations"],
                        "histogram": summary["iteration_accept_histogram"],
                    }
                )
                aggregate_accepted += summary["accepted_tokens"]
                aggregate_drafted += summary["drafted_tokens"]
            alpha = aggregate_accepted / aggregate_drafted if aggregate_drafted else 0.0
            rows.append({"k": k, "alpha_total": round(alpha, 6), "per_prompt": per_prompt})
            print(f"[exp3] {mode} k={k}: alpha_total={alpha}")
        per_draft[mode] = rows
    return {"per_draft": per_draft}


def run_exp4(model, fixtures) -> dict:
    """加速比实测 vs 理论（P01 + P07 两个 regime；draft = reuse + lookup）。"""

    def speedup_rows(prompt_id: str, drafts: tuple[str, ...]) -> tuple[dict, list]:
        entry = next(item for item in fixtures["prompts"] if item["id"] == prompt_id)
        prompt_ids = entry["prompt_token_ids"]
        eos = entry["eos_token_id"]
        prompt_length = len(prompt_ids)

        # 端到端 wall-clock：fast_greedy 基线 + 投机 k 扫描（各 repeats=3，取 P50）。
        repeats = 3
        greedy_samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fast_greedy_generate(model, prompt_ids, 32, eos)
            greedy_samples.append((time.perf_counter() - start) * 1000.0)
        greedy_ms = p50(greedy_samples)
        # 生成 token 数取一次运行的实测（EOS 收尾）。
        generated_ref, _ = fast_greedy_generate(model, prompt_ids, 32, eos)
        greedy_tokens = len(generated_ref) - prompt_length
        print(f"[exp4] {prompt_id} greedy p50={greedy_ms:.0f}ms for {greedy_tokens} tokens")

        rows = []
        for mode in drafts:
            draft_model = build_draft(model, mode)
            decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos)
            for k in (2, 4, 8):
                spec_samples = []
                stats_last = None
                for _ in range(repeats):
                    start = time.perf_counter()
                    generated, _, stats = decoder.generate(prompt_ids, 32, k)
                    spec_samples.append((time.perf_counter() - start) * 1000.0)
                    stats_last = stats
                spec_ms = p50(spec_samples)
                emitted = stats_last.emitted_tokens
                alpha = stats_last.accepted_tokens / stats_last.drafted_tokens
                measured_speedup = greedy_ms / spec_ms
                rows.append(
                    {
                        "draft": mode,
                        "k": k,
                        "spec_p50_ms": round(spec_ms, 1),
                        "spec_samples_ms": [round(sample, 1) for sample in spec_samples],
                        "emitted_tokens": emitted,
                        "alpha": round(alpha, 6),
                        "measured_speedup": round(measured_speedup, 4),
                        "tokens_per_second_ratio_vs_greedy": round(
                            (emitted / (spec_ms / 1000.0)) / (greedy_tokens / (greedy_ms / 1000.0)), 4
                        ),
                        "idealized_speedup_alpha1": round(idealized_speedup(1.0, cost_ratio_c, k), 4),
                        "idealized_speedup_measured_alpha": round(
                            idealized_speedup(alpha, cost_ratio_c, k), 4
                        ),
                        "draft_ms": round(stats_last.draft_time_ms, 1),
                        "verify_ms": round(stats_last.verify_time_ms, 1),
                        "iterations": stats_last.iterations,
                    }
                )
                print(
                    f"[exp4] {prompt_id} {mode} k={k}: spec={spec_ms:.0f}ms "
                    f"measured_speedup={measured_speedup:.3f} alpha={alpha:.3f} "
                    f"iters={stats_last.iterations}"
                )
        baseline = {
            "greedy_p50_ms": round(greedy_ms, 1),
            "greedy_samples_ms": [round(sample, 1) for sample in greedy_samples],
            "greedy_tokens": greedy_tokens,
        }
        return baseline, rows

    # 前向成本标定曲线（P01 上下文长度区间；min 口径，repeats=8 抗噪）。
    p01 = next(item for item in fixtures["prompts"] if item["id"] == "P01")
    prompt_ids = p01["prompt_token_ids"]
    prompt_length = len(prompt_ids)
    calibration_lengths = list(range(prompt_length, prompt_length + 37, 4))
    target_curve = [
        timed_forward_hidden(model, prompt_ids + [13] * (n - prompt_length), warmup=2, repeats=8)
        for n in calibration_lengths
    ]
    print("[exp4] target curve min:", [row["min_ms"] for row in target_curve])

    draft_model = build_draft(model, "reuse")
    draft_curve = [
        timed_draft_step(draft_model, prompt_ids + [13] * (n - prompt_length), warmup=3, repeats=10)
        for n in calibration_lengths
    ]
    print("[exp4] draft curve min:", [row["min_ms"] for row in draft_curve])

    # 成本比 c = draft 单步 / 目标单步（同长度 min 中点）。
    mid = target_curve[len(target_curve) // 2]
    mid_draft = draft_curve[len(draft_curve) // 2]
    cost_ratio_c = mid_draft["min_ms"] / mid["min_ms"]

    # Regime A：P01（未训练 draft α≈0 → 负收益）；Regime B：P07 重复域（lookup α 高）。
    baseline_p01, rows_p01 = speedup_rows("P01", ("reuse", "lookup"))
    baseline_p07, rows_p07 = speedup_rows("P07", ("lookup",))

    return {
        "cost_ratio_c_draft_over_target": round(cost_ratio_c, 6),
        "target_forward_curve": target_curve,
        "draft_forward_curve": draft_curve,
        "P01": {**baseline_p01, "rows": rows_p01},
        "P07": {**baseline_p07, "rows": rows_p07},
    }


def run_exp5(model, fixtures) -> dict:
    """verify 批量化收益：一次 (k+1) 位置前向 vs 贪心路径逐 token (k+1) 次前向。

    序列内容取 P01 prompt + 真值生成序列（真实上下文），n=20（prompt 长度）。
    逐 token 路径 = 对前缀 n+0, n+1, ..., n+k 各做一次完整前向（ch01 无
    KV Cache 贪心的真实成本结构）；批量路径 = 对 n+k 一次前向并切片末 k+1
    行 logits（投机 verify 的真实成本结构）。两侧 LM Head 同口径（末 k+1 行
    vs 各末行——LM Head 差异另行注明：逐 token 侧每步只算自己末行）。
    """
    entry = next(item for item in fixtures["prompts"] if item["id"] == "P01")
    truth = ep27_common.load_truth_manifest()
    base_ids = entry["prompt_token_ids"] + truth["generated_token_ids"]
    n = entry["prompt_token_count"]
    warmup, repeats = 1, 5
    rows = []
    for k in (1, 2, 4, 8, 16):
        context = base_ids[:n]
        candidates = base_ids[n : n + k]
        # 逐 token（贪心路径）：前缀逐步增长的前向。
        for _ in range(warmup):
            for i in range(k + 1):
                backbone_hidden(model, torch.tensor([context + candidates[:i]], dtype=torch.long))
        seq_samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for i in range(k + 1):
                backbone_hidden(model, torch.tensor([context + candidates[:i]], dtype=torch.long))
            seq_samples.append((time.perf_counter() - start) * 1000.0)
        # 批量（verify）：一次前向覆盖 n+k 个位置。
        batched_ids = torch.tensor([context + candidates], dtype=torch.long)
        for _ in range(warmup):
            backbone_hidden(model, batched_ids)
        batch_samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            backbone_hidden(model, batched_ids)
            batch_samples.append((time.perf_counter() - start) * 1000.0)
        seq_p50 = p50(seq_samples)
        batch_p50 = p50(batch_samples)
        rows.append(
            {
                "k": k,
                "n": n,
                "verify_positions": k + 1,
                "sequential_p50_ms": round(seq_p50, 2),
                "batched_p50_ms": round(batch_p50, 2),
                "batching_speedup": round(seq_p50 / batch_p50, 4),
                "sequential_samples_ms": [round(sample, 2) for sample in seq_samples],
                "batched_samples_ms": [round(sample, 2) for sample in batch_samples],
            }
        )
        print(
            f"[exp5] k={k}: seq={seq_p50:.1f}ms batch={batch_p50:.1f}ms "
            f"speedup={seq_p50 / batch_p50:.2f}x"
        )
    return {"rows": rows, "note": "CPU 列；CUDA/MUSA 列待补（verify_batch_bench.cu）"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=ep27_common.default_model_dir())
    parser.add_argument(
        "--exps", default="2,3,4,5", help="要运行的实验编号，逗号分隔（默认全部）"
    )
    parser.add_argument("--out-prefix", default="results")
    arguments = parser.parse_args()
    selected = {int(item) for item in arguments.exps.split(",")}

    fixtures = ep27_common.load_fixtures()
    model = ep27_common.load_verified_model(arguments.model_dir)

    if 2 in selected or 3 in selected:
        acceptance_payload = {
            "schema_version": SCHEMA_VERSION,
            "episode": 27,
            "backend": "cpu",
            "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "max_new_tokens": 32,
            "draft_seed": fixtures["draft_seeds"]["random_main"],
        }
        if 2 in selected:
            print("=== exp2: 接受率 vs k（draft=reuse） ===")
            acceptance_payload["exp2_k_sweep"] = run_exp2(model, fixtures)
        if 3 in selected:
            print("=== exp3: draft 质量对照 ===")
            acceptance_payload["exp3_draft_quality"] = run_exp3(model, fixtures)
        write_results_json(
            Path(arguments.out_prefix) / "acceptance-results-cpu.json", acceptance_payload
        )

    if 4 in selected:
        print("=== exp4: 加速比实测 vs 理论 ===")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episode": 27,
            "backend": "cpu",
            "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "repeats_end_to_end": 3,
            "exp4_speedup": run_exp4(model, fixtures),
            "environment": device_environment_record(torch.device("cpu")),
        }
        write_results_json(
            Path(arguments.out_prefix) / "benchmark-results-cpu.json", payload
        )

    if 5 in selected:
        print("=== exp5: verify 批量化收益 ===")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episode": 27,
            "backend": "cpu",
            "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "exp5_verify_batching": run_exp5(model, fixtures),
        }
        write_results_json(
            Path(arguments.out_prefix) / "verify-batch-results-cpu.json", payload
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
