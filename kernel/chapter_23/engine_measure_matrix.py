# -*- coding: utf-8 -*-
"""实验 3：测量矩阵（MUSA 列）——prompt {20,200,2000} × batch {1,8} × output {32,128}。

口径（大纲 §10.2）：
- TTFT  = t(max_tokens=1, 同 batch) 的墙钟（引擎口径近似，含调度/预热抵消后稳态）
- TPOT  = (t(max_tokens=N) - t(max_tokens=1)) / (N-1)   ← 与 vLLM benchmark_overhead 同法
- 吞吐  = batch*N / t_full（token/s，请求级 = batch/t_full）
- 峰值显存 = torch 加速器 allocator 高水位（每配置 reset 后读）
- 采样 greedy（temperature=0）+ ignore_eos=True（固定输出长度，吞吐/TPOT 可除）
- 每配置 repeats 次取样本数组（P50 可重算），另跑 warmup
产出 results/engine-matrix-musa.json
"""
import argparse
import itertools
import os
import statistics
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
                    choices=["NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE"])
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--plens", default="20,200,2000")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--outlens", default="32,128")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/engine-matrix-musa.json")
    args = ap.parse_args()

    plens = [int(x) for x in args.plens.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    outlens = [int(x) for x in args.outlens.split(",")]

    truth = load_json(args.truth)
    truth_ids = truth["prompt_token_ids"]
    R = {
        "schema_version": "kernel-chapter24.matrix.v1",
        "created_at_utc": utcnow(),
        "engine": "vllm-musa",
        "model_path": args.model,
        "enforce_eager": bool(args.enforce_eager),
        "cudagraph_mode": args.cudagraph_mode or "engine-default(FULL_AND_PIECEWISE)",
        "enable_prefix_caching": False,
        "max_model_len": args.max_model_len,
        "grid": {"prompt_len": plens, "batch": batches, "output_len": outlens},
        "repeats": args.repeats,
        "method": "TTFT=t(max_tokens=1); TPOT=(t(N)-t(1))/(N-1); throughput=batch*N/t(N)",
        "configs": [],
    }
    mem = AccelMem()

    from vllm import LLM, SamplingParams

    llm_kw = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="auto",
        enforce_eager=bool(args.enforce_eager),
        enable_prefix_caching=False,
        seed=1234,
    )
    if args.cudagraph_mode:
        llm_kw["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    llm = LLM(**llm_kw)

    # 全局 warmup（触发编译/JIT/graph 捕获；与测量分离）
    llm.generate(
        build_prompts(truth_ids, 20, 1),
        SamplingParams(temperature=0, max_tokens=8),
        use_tqdm=False,
    )

    # 预构造 prompt 组（同一 id 序列复用，记录 SHA 口径见 README）
    prompt_cache = {}
    for L in set(plens) | {20}:
        prompt_cache[L] = build_prompts(truth_ids, L, 1)[0]

    for plen, batch, olen in itertools.product(plens, batches, outlens):
        prompts = [dict(prompt_cache[plen]) for _ in range(batch)]
        spN = SamplingParams(temperature=0, max_tokens=olen, ignore_eos=True)
        sp1 = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)

        # 配置级 warmup
        llm.generate(prompts, SamplingParams(temperature=0, max_tokens=min(olen, 8), ignore_eos=True), use_tqdm=False)

        t_ttft, t_full = [], []
        for rep in range(args.repeats):
            with Timer() as a:
                llm.generate(prompts, sp1, use_tqdm=False)
            t_ttft.append(round(a.elapsed, 4))
            with Timer() as b:
                outs = llm.generate(prompts, spN, use_tqdm=False)
            t_full.append(round(b.elapsed, 4))
            n_tok = len(outs[0].outputs[0].token_ids)

        # 峰值显存改由外置 mthreads-gmi 采样器承担（V1 引擎跑在子进程，
        # 父进程 allocator 恒 0——首跑教训）；此处仅留父进程口径占位注记
        peak = mem.snapshot()
        peak["note"] = "parent-process allocator (always 0 under V1 engine); real peak from memsample log"

        t1_med = statistics.median(t_ttft)
        tN_med = statistics.median(t_full)
        tpot = (tN_med - t1_med) / max(olen - 1, 1)
        cfg = {
            "prompt_len": plen,
            "batch": batch,
            "output_len": olen,
            "actual_out_tokens_first_req": n_tok,
            "ttft_s": {"samples": t_ttft, "median": round(t1_med, 4)},
            "full_s": {"samples": t_full, "median": round(tN_med, 4)},
            "tpot_s": round(tpot, 6),
            "tpot_ms": round(tpot * 1000, 3),
            "throughput_tok_per_s": round(batch * olen / tN_med, 2),
            "throughput_req_per_s": round(batch / tN_med, 3),
            "peak_mem": peak,
        }
        R["configs"].append(cfg)
        print(
            "CFG plen=%d b=%d out=%d  ttft=%.4fs tpot=%.4fms thr=%.1f tok/s peak=%.0fMiB"
            % (plen, batch, olen, t1_med, tpot * 1000, batch * olen / tN_med,
               peak["peak_allocated_MiB"])
        )
        # 增量写盘（协议 0.5）
        dump_json(R, args.out)

    R["finished_at_utc"] = utcnow()
    dump_json(R, args.out)
    print(f"matrix written -> {args.out}")


if __name__ == "__main__":
    main()
