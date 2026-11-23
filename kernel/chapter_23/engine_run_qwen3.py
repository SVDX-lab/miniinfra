# -*- coding: utf-8 -*-
"""实验 2：vllm-musa 引擎跑通 Qwen3-0.6B 并与 ch01 Greedy 真值对拍。

- prompt：ch01 truth 的 20 token（chat template 后 id 序列，直接传 prompt_token_ids，
  规避 tokenizer/template 差异）
- 采样：greedy（temperature=0），max_tokens=32（与 ch01 同上限）
- 对拍口径：引擎生成 id 序列 vs ch01 generated_token_ids
  （ch01 为 27 token 含 EOS 收尾；greedy-vs-greedy，允许引擎侧实现差异，记录即可）
- 附带：TTFT 近似（max_tokens=1 单独一跑）与 TPOT=(t32-t1)/31；启动耗时、峰值显存
产出 results/engine-truth-run-musa.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine24_common import (AccelMem, Timer, build_prompts, dump_json,  # noqa: E402
                             load_json, utcnow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--truth", default="fixtures/truth-ch01.json")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--cudagraph-mode", default=None,
                    choices=["NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE"],
                    help="覆盖引擎默认 FULL_AND_PIECEWISE（MUSA 上 FULL 捕获有 page-size bug）")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", default="results/engine-truth-run-musa.json")
    args = ap.parse_args()

    truth = load_json(args.truth)
    R = {
        "schema_version": "kernel-chapter24.truthrun.v1",
        "created_at_utc": utcnow(),
        "engine": "vllm-musa",
        "model_path": args.model,
        "enforce_eager": bool(args.enforce_eager),
        "cudagraph_mode": args.cudagraph_mode or "engine-default(FULL_AND_PIECEWISE)",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem_util,
        "enable_prefix_caching": False,
    }
    mem = AccelMem()

    from vllm import LLM, SamplingParams

    kw = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="auto",
        enforce_eager=bool(args.enforce_eager),
        enable_prefix_caching=False,
        seed=1234,
    )
    if args.cudagraph_mode:
        kw["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    with Timer() as t_init:
        llm = LLM(**kw)
    R["init_kwargs"] = {k: str(v) for k, v in kw.items()}
    R["init_wall_s"] = round(t_init.elapsed, 3)
    R["init_peak_mem"] = mem.snapshot()

    sp32 = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    sp1 = SamplingParams(temperature=0, max_tokens=1)

    prompts = build_prompts(truth["prompt_token_ids"], len(truth["prompt_token_ids"]), 1)
    assert len(prompts[0]["prompt_token_ids"]) == truth["prompt_token_count"]

    # 热身（引擎编译/捕获路径与测量分离）
    llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)

    # ---- 正式 32-token 生成 ----
    mem.reset_peak()
    with Timer() as t32:
        outs = llm.generate(prompts, sp32, use_tqdm=False)
    gen = outs[0].outputs[0]
    engine_ids = list(gen.token_ids)
    R["gen32"] = {
        "wall_s": round(t32.elapsed, 3),
        "token_ids": engine_ids,
        "token_count": len(engine_ids),
        "finish_reason": gen.finish_reason,
        "text": gen.text,
        "peak_mem": mem.snapshot(),
    }

    # ---- max_tokens=1（TTFT 近似口径）----
    with Timer() as t1:
        outs1 = llm.generate(prompts, sp1, use_tqdm=False)
    R["gen1"] = {"wall_s": round(t1.elapsed, 3)}

    # ---- 与 ch01 truth 对拍 ----
    ref = truth["generated_token_ids"]
    n_ref = len(ref)
    match = 0
    for a, b in zip(engine_ids, ref):
        if a == b:
            match += 1
        else:
            break
    R["truth_compare"] = {
        "reference_token_count": n_ref,
        "engine_token_count": len(engine_ids),
        "exact_match_full": engine_ids == ref,
        "prefix_match_tokens": match,
        "prefix_match_ratio": round(match / max(n_ref, 1), 4),
        "first_token_match": bool(engine_ids and engine_ids[0] == truth["first_token_id"]),
        "engine_stopped_at_eos": bool(engine_ids and engine_ids[-1] == truth["eos_token_id"]),
        "eos_token_id": truth["eos_token_id"],
        "reference_text": truth["decoded_text"],
        "mismatch_positions": [
            {"idx": i, "ref": ref[i], "engine": engine_ids[i]}
            for i in range(min(len(ref), len(engine_ids)))
            if ref[i] != engine_ids[i]
        ][:10],
    }

    # 每步时间（引擎口径近似：TPOT=(t32-t1)/(32-1)，含首 token）
    n_out = R["gen32"]["token_count"]
    if n_out >= 2:
        R["derived"] = {
            "ttft_approx_s": round(t1.elapsed, 4),
            "tpot_approx_s": round((t32.elapsed - t1.elapsed) / (n_out - 1), 6),
            "tpot_approx_ms": round((t32.elapsed - t1.elapsed) / (n_out - 1) * 1000, 3),
            "tokens_per_s_step1": round(n_out / t32.elapsed, 2),
            "method": "two-run difference: (t(max_tokens=N) - t(max_tokens=1)) / (N-1)",
        }

    dump_json(R, args.out)
    print(f"truth run written -> {args.out}")
    print(
        "SUMMARY exact=%s prefix=%d/%d first_token=%s text=%s"
        % (
            R["truth_compare"]["exact_match_full"],
            R["truth_compare"]["prefix_match_tokens"],
            n_ref,
            R["truth_compare"]["first_token_match"],
            gen.text[:60],
        )
    )


if __name__ == "__main__":
    main()
