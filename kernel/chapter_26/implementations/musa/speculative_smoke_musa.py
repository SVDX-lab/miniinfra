#!/usr/bin/env python3
"""第 26 期 MUSA 小验证：draft+verify 全链在 S4000 上小规模跑通（可选列）。

前置（前序情报）：ep05/ch01 已验证 musa 上完整手写 Qwen3 推理可行（BF16，
greedy 与 CPU FP32 真值逐 token 一致——见 chapter_01/results/correctness-
results-musa.json all_match=True）。本脚本把投机解码全链搬到 musa：
- 目标模型 BF16 + musa；draft = 单层复用（MTPDraftModel reuse）+ lookup；
- 正确性口径：投机输出 == musa 自己的 fast_greedy 输出（同 dtype 同设备
  自洽）+ == ch01 CPU FP32 真值（跨 dtype 一致性，ch01 已证可行）；
- 小规模：max_new_tokens=16（控制锁内时长），k=4。

本地预验证：--device cpu --dtype float32 可在本地把同一条代码路径跑通
（上传前自检），服务器上用默认 --device musa --dtype bfloat16。

运行（S4000，锁内）：
  python3 speculative_smoke_musa.py --model-dir ../chapter_01/models/qwen3-0.6b
输出：results/speculative-smoke-musa.json（stdout 摘要）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / ".."))

import ep27_common  # noqa: E402

ep27_common.bootstrap()

import torch  # noqa: E402

from experiment_utils import backend_sync, resolve_device, write_results_json  # noqa: E402
from mtp_reference import (  # noqa: E402
    SpeculativeDecoder,
    build_draft,
    fast_greedy_generate,
    summarize_spec_stats,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="musa", choices=["musa", "cpu", "cuda"])
    parser.add_argument("--dtype", default=None, choices=[None, "bfloat16", "float32"])
    parser.add_argument("--model-dir", default="../chapter_01/models/qwen3-0.6b")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    device = resolve_device(arguments.device)
    dtype = {
        None: (torch.bfloat16 if device.type == "musa" else torch.float32),
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[arguments.dtype]

    # ep27_common.load_verified_model 固定 CPU FP32（权重校验路径），musa 列
    # 在此基础上搬运：先 CPU 校验（同机无 ch01 真值时跳过校验）再 .to(device)。
    model = ep27_common.load_verified_model(arguments.model_dir, dtype=torch.float32)
    truth = ep27_common.load_truth_manifest()
    fixtures = ep27_common.load_fixtures()
    entry = next(item for item in fixtures["prompts"] if item["id"] == "P01")
    prompt_ids = entry["prompt_token_ids"]
    eos = truth["eos_token_id"]

    model = model.to(device=device, dtype=dtype)
    backend_sync(device)
    print(f"[musa-smoke] model on {device} dtype={dtype}")

    # musa 自己的贪心（同 dtype 同设备）。
    greedy_ids, greedy_finished = fast_greedy_generate(
        model, prompt_ids, arguments.max_new_tokens, eos
    )
    greedy_tokens = greedy_ids[len(prompt_ids):]

    rows = []
    for mode in ("reuse", "lookup"):
        draft_model = build_draft(model, mode)
        if isinstance(draft_model, torch.nn.Module):
            draft_model = draft_model.to(device=device, dtype=dtype)
        decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos)
        started = time.perf_counter()
        generated, finished, stats = decoder.generate(
            prompt_ids, arguments.max_new_tokens, arguments.k
        )
        backend_sync(device)
        wall = time.perf_counter() - started
        spec_tokens = generated[len(prompt_ids):]
        record = {
            "draft_mode": mode,
            "k": arguments.k,
            "tokens_equal_to_musa_greedy": spec_tokens == greedy_tokens,
            "tokens_equal_to_cpu_truth_prefix": spec_tokens == truth["generated_token_ids"][: len(spec_tokens)],
            "finished_by_eos": finished,
            "wall_seconds": round(wall, 3),
            "acceptance": summarize_spec_stats(stats),
        }
        rows.append(record)
        print(
            f"[musa-smoke] {mode} k={arguments.k}: ==musa_greedy={record['tokens_equal_to_musa_greedy']} "
            f"==cpu_truth_prefix={record['tokens_equal_to_cpu_truth_prefix']} wall={wall:.1f}s"
        )

    payload = {
        "schema_version": "kernel-chapter27.speculative-smoke.v1",
        "episode": 27,
        "backend": str(device),
        "dtype": str(dtype),
        "model_sha_source": "chapter_01/models/qwen3-0.6b（MANIFEST 校验）",
        "max_new_tokens": arguments.max_new_tokens,
        "musa_greedy_tokens": greedy_tokens,
        "cpu_truth_tokens_32": truth["generated_token_ids"],
        "rows": rows,
    }
    out = arguments.out or str(
        ep27_common.ROOT / "results" / f"speculative-smoke-{device.type}.json"
    )
    write_results_json(Path(out), payload)
    ok = all(row["tokens_equal_to_musa_greedy"] for row in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
