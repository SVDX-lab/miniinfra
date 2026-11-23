#!/usr/bin/env python3
"""第 26 期补充诊断：draft 质量归因 + prompt-lookup 对照（快查版）。

回答两个问题（结果进 EXPERIMENT_LOG 实验 3 归因）：
1. 未训练单层 draft 的 logits 与目标模型差在哪一级——draft top-1 == 目标
   top-1（argmax 级，greedy 接受的门槛）vs draft top-1 ∈ 目标 top-5
   （排序级接近）的比例。若排序级明显高而 argmax 级 ≈0，说明差异主要在
   "深度不够 → 分布锋利度/排序尾部"而非"完全无关"。
2. prompt-lookup draft（1/2 gram）的接受率形态——预期首位置高、后递减。

运行：/home/aiuser/venvs/ml/bin/python draft_quality_probe.py
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

from experiment_utils import write_results_json  # noqa: E402
from mtp_reference import (  # noqa: E402
    SpeculativeDecoder,
    backbone_hidden,
    build_draft,
    logits_at_rows,
    summarize_spec_stats,
)

SCHEMA_VERSION = "kernel-chapter27.draft-probe.v1"


@torch.no_grad()
def probe_rank_agreement(model, decoder, prompt_ids, k, eos_token_id, max_new_tokens=32):
    """重放投机循环，但在每个 verify 位置记录目标 top-5 与 draft token 的关系。

    返回：{proposed, argmax_hit, top5_hit, top20_hit}（位置 0 单独统计）。
    """
    counts = {
        "proposed": 0,
        "argmax_hit": 0,
        "top5_hit": 0,
        "top20_hit": 0,
        "proposed_pos0": 0,
        "argmax_hit_pos0": 0,
        "top5_hit_pos0": 0,
    }
    ids = list(prompt_ids)
    seed_hidden = None
    if decoder.draft.needs_seed_hidden:
        seed_hidden = backbone_hidden(model, torch.tensor([ids], dtype=torch.long))[0, -1, :].view(1, 1, -1)
    emitted_total = 0
    while emitted_total < max_new_tokens:
        remaining = max_new_tokens - emitted_total
        step_k = min(k, remaining)
        draft_ids = decoder.draft.draft(ids, step_k, seed_hidden)
        full_ids = torch.tensor([ids + draft_ids], dtype=torch.long)
        hidden = backbone_hidden(model, full_ids)
        n = len(ids)
        logits = logits_at_rows(model, hidden, slice(n - 1, n + k))[0]  # [k+1, V]
        top20 = torch.topk(logits, 20, dim=-1).indices.tolist()
        emitted, accepted, info = decoder.verify_and_accept(ids, draft_ids)
        emitted = emitted[:remaining]
        for position in range(step_k):
            draft_token = draft_ids[position]
            counts["proposed"] += 1
            if position == 0:
                counts["proposed_pos0"] += 1
            if draft_token in top20[position]:
                counts["top20_hit"] += 1
                if draft_token in top20[position][:5]:
                    counts["top5_hit"] += 1
                    if position == 0:
                        counts["top5_hit_pos0"] += 1
                    if draft_token == top20[position][0]:
                        counts["argmax_hit"] += 1
                        if position == 0:
                            counts["argmax_hit_pos0"] += 1
        if decoder.draft.needs_seed_hidden:
            verify_hidden = info["verify_hidden"]
            seed_index = min(len(ids) + len(emitted) - 2, info["verify_length"] - 1)
            seed_hidden = verify_hidden[seed_index].unsqueeze(0).unsqueeze(0)
        ids.extend(emitted)
        emitted_total += len(emitted)
        if eos_token_id in emitted:
            break
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=ep27_common.default_model_dir())
    arguments = parser.parse_args()

    fixtures = ep27_common.load_fixtures()
    truth = ep27_common.load_truth_manifest()
    entry = next(item for item in fixtures["prompts"] if item["id"] == "P01")
    prompt_ids = entry["prompt_token_ids"]
    eos = truth["eos_token_id"]

    model = ep27_common.load_verified_model(arguments.model_dir)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "prompt_id": "P01",
        "dtype": "float32",
        "rank_agreement": {},
        "lookup_acceptance": {},
    }

    # 1) 排序级对齐（draft=reuse / reuse_hidden，k=4）
    for mode in ("reuse", "reuse_hidden"):
        draft_model = build_draft(model, mode)
        decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos)
        counts = probe_rank_agreement(model, decoder, prompt_ids, 4, eos)
        proposed = max(counts["proposed"], 1)
        pos0 = max(counts["proposed_pos0"], 1)
        payload["rank_agreement"][mode] = {
            **counts,
            "argmax_hit_rate": round(counts["argmax_hit"] / proposed, 6),
            "top5_hit_rate": round(counts["top5_hit"] / proposed, 6),
            "top20_hit_rate": round(counts["top20_hit"] / proposed, 6),
            "argmax_hit_rate_pos0": round(counts["argmax_hit_pos0"] / pos0, 6),
            "top5_hit_rate_pos0": round(counts["top5_hit_pos0"] / pos0, 6),
        }
        print(f"[probe] {mode}: {payload['rank_agreement'][mode]}")

    # 2) prompt-lookup 接受率形态（k=4）
    for mode in ("lookup", "lookup2"):
        draft_model = build_draft(model, mode)
        decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos)
        started = time.perf_counter()
        generated, finished, stats = decoder.generate(prompt_ids, 32, 4)
        wall = time.perf_counter() - started
        summary = summarize_spec_stats(stats)
        payload["lookup_acceptance"][mode] = summary
        tokens_ok = generated[len(prompt_ids):] == truth["generated_token_ids"]
        print(
            f"[probe] {mode}: alpha={summary['alpha_over_drafted']} "
            f"pos_rates={[p['accept_rate_over_evaluated'] for p in summary['per_position']]} "
            f"truth_match={tokens_ok} wall={wall:.1f}s"
        )

    write_results_json(Path("results") / "draft-quality-probe-cpu.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
