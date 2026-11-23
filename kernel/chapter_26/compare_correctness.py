#!/usr/bin/env python3
"""第 26 期实验 1：投机解码输出 vs 第 01 期 Greedy 32 Token 真值——逐 Token 一致。

greedy 投机解码不改分布的实验证明：
- 对三种 draft（单层复用权重 / 单层随机权重 / 恒等复制）× k ∈ {1,2,3,4,6,8}
  全组合：投机输出必须与 ch01 真值逐 token 相等（含 EOS 收尾行为）；
  draft 质量可以任意差，输出分布不允许变——这就是"draft 只提议位置安排、
  token 内容全部来自目标模型 argmax"的实证。
- 同时记录每轮接受率统计（接受数 / draft 数）。

运行（本地 venv）：/home/aiuser/venvs/ml/bin/python compare_correctness.py
可选：--model-dir models/qwen3-0.6b --out results/correctness-results-cpu.json
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
    build_draft,
    fast_greedy_generate,
    summarize_spec_stats,
)

SCHEMA_VERSION = "kernel-chapter27.correctness.v1"
DRAFT_MODES = ["reuse", "reuse_hidden", "random", "copy", "lookup", "lookup2"]
K_SWEEP = [1, 2, 3, 4, 6, 8]


def token_comparison(spec_tokens: list[int], truth_tokens: list[int]) -> dict:
    count_match = len(spec_tokens) == len(truth_tokens)
    first_divergence = None
    for index, (a, b) in enumerate(zip(spec_tokens, truth_tokens)):
        if a != b:
            first_divergence = index
            break
    all_match = count_match and first_divergence is None
    return {
        "token_count_match": count_match,
        "all_match": all_match,
        "first_divergence_index": first_divergence,
        "spec_token_count": len(spec_tokens),
        "truth_token_count": len(truth_tokens),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=ep27_common.default_model_dir())
    parser.add_argument("--out", default="results/correctness-results-cpu.json")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    arguments = parser.parse_args()

    fixtures = ep27_common.load_fixtures()
    truth = ep27_common.load_truth_manifest()
    prompt_entry = next(entry for entry in fixtures["prompts"] if entry["id"] == "P01")
    prompt_ids = prompt_entry["prompt_token_ids"]
    if prompt_ids != truth["prompt_token_ids"]:
        raise RuntimeError("P01 prompt 与 ch01 真值 prompt 不一致，锚点破坏")
    eos_token_id = truth["eos_token_id"]
    truth_tokens = truth["generated_token_ids"]

    started = time.perf_counter()
    model = ep27_common.load_verified_model(arguments.model_dir)
    print(f"[model] FP32 CPU 加载 + 真值 logits SHA256 对拍通过（{time.perf_counter() - started:.1f}s）")

    runs = []

    # 基线 0：ch01 原版 greedy_generate（全序列 logits 口径）。
    started = time.perf_counter()
    ch01_ids, ch01_finished = model.greedy_generate(
        torch.tensor([prompt_ids], dtype=torch.long),
        max_new_tokens=arguments.max_new_tokens,
        eos_token_id=eos_token_id,
    )
    ch01_seconds = time.perf_counter() - started
    ch01_tokens = ch01_ids[0].tolist()[len(prompt_ids):]
    runs.append(
        {
            "run": "baseline_ch01_greedy_full_lmhead",
            "comparison": token_comparison(ch01_tokens, truth_tokens),
            "finished_by_eos": ch01_finished,
            "wall_seconds": round(ch01_seconds, 3),
            "tokens": ch01_tokens,
        }
    )
    print(f"[run] ch01 greedy: all_match={runs[-1]['comparison']['all_match']} ({ch01_seconds:.1f}s)")

    # 基线 1：fast_greedy（仅末位 logits 口径——投机实验的时基线）。
    started = time.perf_counter()
    fast_ids, fast_finished = fast_greedy_generate(
        model, prompt_ids, arguments.max_new_tokens, eos_token_id
    )
    fast_seconds = time.perf_counter() - started
    fast_tokens = fast_ids[len(prompt_ids):]
    runs.append(
        {
            "run": "baseline_fast_greedy_last_logits",
            "comparison": token_comparison(fast_tokens, truth_tokens),
            "finished_by_eos": fast_finished,
            "wall_seconds": round(fast_seconds, 3),
            "tokens": fast_tokens,
        }
    )
    print(f"[run] fast greedy: all_match={runs[-1]['comparison']['all_match']} ({fast_seconds:.1f}s)")

    # 实验矩阵：4 种 draft × 6 个 k。
    for mode in DRAFT_MODES:
        draft_model = build_draft(model, mode, seed=fixtures["draft_seeds"]["random_main"])
        decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos_token_id)
        for k in K_SWEEP:
            started = time.perf_counter()
            generated, finished, stats = decoder.generate(
                prompt_ids, arguments.max_new_tokens, k
            )
            wall = time.perf_counter() - started
            spec_tokens = generated[len(prompt_ids):]
            comparison = token_comparison(spec_tokens, truth_tokens)
            record = {
                "run": f"spec_{mode}_k{k}",
                "draft_mode": mode,
                "k": k,
                "comparison": comparison,
                "finished_by_eos": finished,
                "wall_seconds": round(wall, 3),
                "acceptance": summarize_spec_stats(stats),
            }
            runs.append(record)
            print(
                f"[run] {mode} k={k}: all_match={comparison['all_match']} "
                f"alpha={record['acceptance']['alpha_over_drafted']} "
                f"iters={record['acceptance']['iterations']} ({wall:.1f}s)"
            )

    aggregate = {
        "total_runs": len(runs),
        "all_match_runs": sum(1 for run in runs if run["comparison"]["all_match"]),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "episode": 27,
        "backend": "cpu",
        "prompt_id": "P01",
        "prompt_token_count": len(prompt_ids),
        "max_new_tokens": arguments.max_new_tokens,
        "truth_reference": str(ep27_common.TRUTH_MANIFEST),
        "truth_tokens": truth_tokens,
        "eos_token_id": eos_token_id,
        "dtype": "float32",
        "torch_threads": torch.get_num_threads(),
        "runs": runs,
        "aggregate": aggregate,
        "conclusion": "greedy 投机解码不改分布：全组合逐 token 一等（all_match_runs == total_runs）"
        if aggregate["all_match_runs"] == aggregate["total_runs"]
        else "存在不一致运行——投机解码正确性破坏，禁止继续",
        "environment": device_environment_record(torch.device("cpu")),
    }
    write_results_json(Path(arguments.out), payload)
    print(f"\n[exp1] {aggregate['all_match_runs']}/{aggregate['total_runs']} runs all_match")
    return 0 if aggregate["all_match_runs"] == aggregate["total_runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
