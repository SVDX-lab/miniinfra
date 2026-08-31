"""第 11 期 Draft Length 扫描主实验。"""

import argparse

import torch

from experiment_utils import (
    environment_snapshot,
    load_target_and_draft,
    memory_snapshot,
    model_parameter_bytes,
    set_seed,
    summarize,
    timed_call,
    write_json,
)
from qwen3_model import DEFAULT_DRAFT_MODEL_ID, DEFAULT_TARGET_MODEL_ID
from qwen3_tokenizer import Qwen3Tokenizer
from speculative_decode import speculative_greedy_generate, target_greedy_generate


DEFAULT_PROMPTS = [
    "请解释推测解码为什么需要验证候选 token，并给出一个简短例子。",
    "请继续下面的代码，只输出函数体：\ndef is_prime(number):\n",
    "下面是一段重复格式的数据：项目1=完成，项目2=完成，项目3=完成。请继续写到项目10。",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL_ID)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL_ID)
    parser.add_argument("--target-revision")
    parser.add_argument("--draft-revision")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--draft-lengths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", default="chapter_11/benchmark-speculative-results.local.json"
    )
    return parser.parse_args()


def reset_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def measure(function, first_token_function, device, repeats):
    e2e = []
    ttft = []
    peaks = []
    results = []
    for _ in range(repeats):
        reset_peak_memory(device)
        first_result, first_seconds = timed_call(first_token_function, device)
        full_result, full_seconds = timed_call(function, device)
        ttft.append(first_seconds)
        e2e.append(full_seconds)
        peaks.append(memory_snapshot().get("max_allocated_bytes", 0))
        results.append(full_result)
        if first_result.token_ids[0] != full_result.token_ids[0]:
            raise RuntimeError("TTFT 与完整生成的首 Token 不一致")
    generated = len(results[-1].token_ids)
    decode_tpot = [
        max(0.0, total - first) / max(1, generated - 1)
        for total, first in zip(e2e, ttft)
    ]
    return {
        "generated_tokens": generated,
        "ttft_seconds": summarize(ttft),
        "decode_tpot_seconds": summarize(decode_tpot),
        "end_to_end_seconds": summarize(e2e),
        "output_tokens_per_second": summarize(
            [generated / value for value in e2e]
        ),
        "peak_allocated_bytes": summarize(peaks),
        "last_stats": results[-1].stats.to_dict(include_rounds=False),
        "token_ids": results[-1].token_ids,
    }


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA GPU")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats 必须大于 0，warmup 不能小于 0")
    set_seed(args.seed)
    target_directory, target, draft_directory, draft = load_target_and_draft(
        args.target_model,
        args.draft_model,
        args.device,
        args.dtype,
        target_revision=args.target_revision,
        draft_revision=args.draft_revision,
    )
    tokenizer = Qwen3Tokenizer(target_directory)
    prompts = DEFAULT_PROMPTS if args.prompts is None else args.prompts
    cases = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode_chat_prompt(prompt)
        baseline_call = lambda: target_greedy_generate(
            target, prompt_ids, args.max_new_tokens, eos_token_id=None
        )
        baseline_first = lambda: target_greedy_generate(
            target, prompt_ids, 1, eos_token_id=None
        )
        for _ in range(args.warmup):
            baseline_call()
        baseline = measure(
            baseline_call, baseline_first, args.device, args.repeats
        )
        speculative = []
        for draft_length in args.draft_lengths:
            spec_call = lambda draft_length=draft_length: speculative_greedy_generate(
                target,
                draft,
                prompt_ids,
                args.max_new_tokens,
                draft_length,
                eos_token_id=None,
                include_round_details=False,
            )
            spec_first = lambda draft_length=draft_length: speculative_greedy_generate(
                target,
                draft,
                prompt_ids,
                1,
                draft_length,
                eos_token_id=None,
                include_round_details=False,
            )
            for _ in range(args.warmup):
                spec_call()
            result = measure(spec_call, spec_first, args.device, args.repeats)
            if result["token_ids"] != baseline["token_ids"]:
                raise RuntimeError("性能实验中 Speculative 输出与 Baseline 不一致")
            result["draft_length"] = draft_length
            result["decode_tpot_speedup"] = (
                baseline["decode_tpot_seconds"]["mean"]
                / result["decode_tpot_seconds"]["mean"]
            )
            speculative.append(result)
        cases.append(
            {
                "prompt": prompt,
                "prompt_tokens": len(prompt_ids),
                "baseline": baseline,
                "speculative": speculative,
            }
        )

    payload = {
        "environment": environment_snapshot(
            target_directory,
            draft_directory,
            args.dtype,
            target_model=args.target_model,
            draft_model=args.draft_model,
        ),
        "config": {
            "draft_lengths": args.draft_lengths,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "eos_enabled": False,
            "thinking_enabled": False,
            "timing": "wall clock with CUDA synchronization; tokenizer excluded",
            "ttft": "双模型初始化路径生成首 Token；包含两套 KV 分配和两次 Prefill",
            "decode_tpot": "(end_to_end - separately measured TTFT) / (tokens - 1)",
        },
        "model_memory": {
            "target_parameter_bytes": model_parameter_bytes(target),
            "draft_parameter_bytes": model_parameter_bytes(draft),
            "cuda_after_load": memory_snapshot(),
        },
        "cases": cases,
    }
    write_json(args.output, payload)
    print("结果已写入", args.output)
    for case in cases:
        print("prompt_tokens=%d" % case["prompt_tokens"])
        print(
            "  target_only TPOT %.3f ms"
            % (case["baseline"]["decode_tpot_seconds"]["mean"] * 1000)
        )
        for result in case["speculative"]:
            print(
                "  γ=%d TPOT %.3f ms speedup %.3fx acceptance %.3f"
                % (
                    result["draft_length"],
                    result["decode_tpot_seconds"]["mean"] * 1000,
                    result["decode_tpot_speedup"],
                    result["last_stats"]["acceptance_rate"],
                )
            )


if __name__ == "__main__":
    main()
