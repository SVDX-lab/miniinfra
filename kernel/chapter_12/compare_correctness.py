#!/usr/bin/env python3
"""第 12 期实验 1：三后端同一输入运行与结果复核（vs 第 01 期 truth 共享锚点）。

口径：
  - 输入：第 01 期 truth prompt（token ids 直接对账——tokenizer 平台无关性证据）；
  - 真值：chapter_01/truth/truth-manifest.json（+ probes.npz 首	Token Logits，
    在位时附加向量级 metrics）；
  - 运行：native 与 fused 两种模式各跑一次 Greedy（上限与 truth 相同，
    eos 同口径），首 Token Logits 三层 metrics；
  - 复核判据：generated token ids 与 truth 逐位一致（含 eos 收尾位级一致）；
    FP32 CPU 列附 logits SHA256 位级口径（KI-27-07 回退口径兜底）。

输出：results/correctness-results-<backend>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import (  # noqa: E402
    backend_sync,
    comparison_metrics,
    device_environment_record,
    sha256_of_tensor,
    write_results_json,
)
from implementations.runtime import Qwen3Runtime  # noqa: E402
from implementations.runtime.qwen3_runtime import (  # noqa: E402
    TRUTH_PROBES,
    bootstrap_compat,
    load_truth,
)


def token_comparison(run_tokens: list[int], truth_tokens: list[int]) -> dict:
    length_match = len(run_tokens) == len(truth_tokens)
    mismatches = [
        {"position": index, "run": token, "truth": expected}
        for index, (token, expected) in enumerate(zip(run_tokens, truth_tokens))
        if token != expected
    ]
    return {
        "match": bool(length_match and not mismatches),
        "length_match": length_match,
        "run_count": len(run_tokens),
        "truth_count": len(truth_tokens),
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
    }


def logits_verification(logits_fp32: torch.Tensor, truth: dict,
                        truth_logits: np.ndarray | None) -> dict:
    """三层口径：SHA256 位级 → 可移植口径（KI-27-07）→ 向量 metrics。"""
    result: dict = {}
    sha = sha256_of_tensor(logits_fp32)
    result["sha256"] = sha
    result["sha_bit_exact_vs_truth"] = sha == truth["first_token_logits_sha256"]
    if not result["sha_bit_exact_vs_truth"]:
        argmax_id = int(logits_fp32.argmax().item())
        top5 = torch.topk(logits_fp32, 5)
        top5_ids = sorted(int(v) for v in top5.indices.tolist())
        truth_argmax = truth["first_token_id"]
        truth_top5_ids = sorted(item["id"] for item in truth["first_token_top5"])
        result["portable_check"] = {
            "argmax_match": argmax_id == truth_argmax,
            "argmax_run": argmax_id,
            "argmax_truth": truth_argmax,
            "top5_ids_match": top5_ids == truth_top5_ids,
            "top5_ids_run": top5_ids,
            "top5_ids_truth": truth_top5_ids,
        }
    if truth_logits is not None:
        expected = torch.from_numpy(truth_logits).float()
        result["vs_truth_logits"] = comparison_metrics(logits_fp32, expected)
        run_top5 = torch.topk(logits_fp32, 5)
        result["top5_ids"] = [int(v) for v in run_top5.indices.tolist()]
        result["top5_match_truth_manifest"] = result["top5_ids"] == [
            item["id"] for item in truth["first_token_top5"]
        ]
    return result


def run_mode(runtime: Qwen3Runtime, truth: dict, truth_logits: np.ndarray | None,
             eos_token_id: int) -> tuple[dict, torch.Tensor]:
    """单模式（native/fused）完整复核：encode → 首 token logits → Greedy。

    返回 (结果 dict, 首 Token Logits 张量)——后者供 native/fused 交叉 metrics。
    """
    try:
        prompt_ids = runtime.encode_chat_prompt(truth["prompt"])
        encode_match = prompt_ids == truth["prompt_token_ids"]
        encode_checked = True
    except ModuleNotFoundError as error:  # regex 缺失环境兜底：直接用 truth ids
        prompt_ids = list(truth["prompt_token_ids"])
        encode_match = True
        encode_checked = False
        print(f"[encode] tokenizer 不可用（{error}），改用 truth prompt ids（跳过重编码对账）")
    input_ids = runtime.input_ids_tensor(prompt_ids)

    # 首 Token Logits（真值锚点向量级对账）。
    logits = runtime.first_token_logits(input_ids)
    logits_result = logits_verification(logits, truth, truth_logits)

    # Greedy 解码（Backend.greedy_decode 统一循环；上限与 truth 同）。
    generate_result = runtime.generate(
        input_ids, max_new_tokens=truth["max_new_tokens"], eos_token_id=eos_token_id
    )
    run_tokens = generate_result.generated_ids[len(prompt_ids):]
    tokens = token_comparison(run_tokens, truth["generated_token_ids"])

    summary = generate_result.to_dict()
    result = {
        "encode_prompt_ids_match_truth": encode_match,
        "encode_reencoding_checked": encode_checked,
        "first_token_logits": logits_result,
        "greedy": {
            "tokens_vs_truth": tokens,
            "finished_by_eos": generate_result.finished_by_eos,
            "finished_by_eos_truth": truth["finished_by_eos"],
            "total_wall_ms": summary["total_wall_ms"],
            "mean_step_ms": summary["mean_step_ms"],
            "step_times_ms": summary["step_times_ms"],
            "op_call_counts": summary["op_call_counts"],
        },
        "generated_ids": run_tokens,
        "decoded_text": runtime.tokenizer.decode(run_tokens, skip_special_tokens=True),
    }
    return result, logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--dtype", default=None, choices=[None, "float32", "float16"])
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    bootstrap_compat()
    from implementations.runtime.qwen3_runtime import DEFAULT_MODEL_DIR

    if arguments.backend == "musa":
        import torch_musa  # noqa: F401  (env record 前注册设备)

    model_dir = Path(arguments.model_dir) if arguments.model_dir else DEFAULT_MODEL_DIR
    truth = load_truth()
    truth_logits = None
    if TRUTH_PROBES.exists():
        with np.load(TRUTH_PROBES) as archive:  # type: ignore[assignment]
            truth_logits = archive["first_token_logits"].copy()
        print("[truth] probes.npz 在位（首 Token Logits 向量级对账开启）")
    else:
        print("[truth] probes.npz 不在位（manifest + SHA256/可移植口径）")

    dtype = None
    if arguments.dtype:
        dtype = getattr(torch, arguments.dtype)

    payload: dict = {
        "schema_version": "kernel-chapter13.correctness.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": arguments.backend,
        "truth_reference": {
            "manifest": "chapter_01/truth/truth-manifest.json",
            "probes_present": truth_logits is not None,
            "prompt": truth["prompt"],
            "truth_generated_count": truth["generated_token_count"],
            "finished_by_eos_truth": truth["finished_by_eos"],
        },
    }

    eos_token_id = truth["eos_token_id"]

    # native 模式（无融合 op，第 01 期逐字路径）。
    native_runtime = Qwen3Runtime.create(
        arguments.backend, model_dir=model_dir, dtype=dtype, fused=False)
    native_evidence = native_runtime.load()
    payload["environment"] = device_environment_record(
        native_runtime.backend.resolve_device())
    payload["native"], native_logits = run_mode(
        native_runtime, truth, truth_logits, eos_token_id)
    payload["load_evidence_native"] = native_evidence
    del native_runtime

    # fused 模式（第 04 期注册融合 op；CPU=composite / CUDA|MUSA=torch.ops）。
    fused_runtime = Qwen3Runtime.create(
        arguments.backend, model_dir=model_dir, dtype=dtype, fused=True)
    fused_evidence = fused_runtime.load()
    payload["fused"], fused_logits = run_mode(
        fused_runtime, truth, truth_logits, eos_token_id)
    payload["load_evidence_fused"] = fused_evidence

    # fused 路由留痕：op 调用计数应 >0 且只在声明设备上。
    router = fused_runtime.backend.fused_ops_protocol()
    payload["router_summary"] = router.summary()

    # native vs fused 交叉复核（首 Token Logits 向量级 + tokens 集合级）。
    payload["fused_vs_native"] = {
        "tokens_identical": payload["native"]["generated_ids"]
        == payload["fused"]["generated_ids"],
        "first_token_logits": comparison_metrics(fused_logits, native_logits),
    }
    payload["verification"] = {
        "encode_match": payload["fused"]["encode_prompt_ids_match_truth"]
        and payload["native"]["encode_prompt_ids_match_truth"],
        "native_tokens_match_truth": payload["native"]["greedy"]["tokens_vs_truth"]["match"],
        "fused_tokens_match_truth": payload["fused"]["greedy"]["tokens_vs_truth"]["match"],
        "fused_equals_native": payload["fused_vs_native"]["tokens_identical"],
    }
    payload["pass"] = all([
        payload["verification"]["encode_match"],
        payload["verification"]["native_tokens_match_truth"],
        payload["verification"]["fused_tokens_match_truth"],
        payload["verification"]["fused_equals_native"],
    ])

    out_path = Path(arguments.out) if arguments.out else \
        HERE / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(out_path, payload)

    print(f"[exp1/{arguments.backend}] encode={payload['verification']['encode_match']} "
          f"native_vs_truth={payload['verification']['native_tokens_match_truth']} "
          f"fused_vs_truth={payload['verification']['fused_tokens_match_truth']} "
          f"fused==native={payload['verification']['fused_equals_native']} "
          f"router_calls={payload['router_summary']['call_counts']}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
