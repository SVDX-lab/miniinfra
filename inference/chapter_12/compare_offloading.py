"""第 12 期真实权重正确性实验。

同一组固定请求分别运行：

- reference：GPU 池足以容纳全部请求（无抢占）；
- swap：人为缩小 GPU 池，swap 抢占 + 换回续算；
- recompute：同一小池，丢弃 KV + 重新 Prefill。

比较逐请求输出 Token 序列和每个生成位置的 Logits。swap 搬移字节不改变已保存
的 KV，但后续输出仍可能受 batch 形状影响；recompute 数学等价，数值既可能相同
也可能出现有限精度差异。float32 与 bfloat16 分列记录。
"""

import argparse

import torch

from engine import run_engine
from experiment_utils import collect_environment, save_results
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer
from scheduler import make_request_specs

PROMPTS = {
    "短": "用一句话解释 KV Cache。",
    "中": "解释 GPU 显存不足时推理引擎可以采取哪些措施，并比较它们的代价。" * 2,
    "长": "详细说明把 KV Cache 从 GPU 换出到 CPU 内存时，需要考虑的带宽、"
          "停顿、正确性与生命周期问题，并解释为什么不能只换出一个请求的"
          "部分 KV。请分层论述。" * 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description="抢占式换出正确性对照")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"),
                        action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--extra-pool-blocks", type=int, default=6,
                        help="缩小池 = 全部 Prompt 块数 + 该余量，"
                             "使抢占必然发生且总上下文超出池")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def first_divergence(reference, candidate):
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def logits_error(reference, candidate):
    """只在双方都实际生成的位置比较，不跨越缺失位置补零。"""
    count = min(len(reference), len(candidate))
    if count == 0:
        return {"compared_positions": 0, "max_abs_error": 0.0}
    maximum = max(
        float(torch.max(torch.abs(reference[index] - candidate[index])).item())
        for index in range(count)
    )
    return {"compared_positions": count, "max_abs_error": maximum}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("正确性实验需要可用的 NVIDIA GPU")
    dtypes = args.dtype or ["float32", "bfloat16"]
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)

    sequences = [
        tokenizer.encode_chat_prompt(PROMPTS[name]) for name in ("短", "中", "长")
    ]
    # 复制为 6 个请求（两组同长），保证 Decode 后期总上下文超过缩小池。
    sequences = sequences + sequences
    prompt_lengths = [len(sequence) for sequence in sequences]
    print("Prompt 长度:", prompt_lengths)

    report = {"environment": collect_environment(device), "runs": []}
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        model = load_handwritten_model(model_directory, device, dtype)
        specs = make_request_specs(sequences, args.max_new_tokens)
        block_size = args.block_size
        prompt_blocks = sum(
            (length + block_size - 1) // block_size for length in prompt_lengths
        )
        pool_blocks = prompt_blocks + args.extra_pool_blocks
        print("\n=== %s：池 %d 块（Prompt 合计 %d 块） ===" % (
            dtype_name, pool_blocks, prompt_blocks,
        ))
        common = dict(
            max_running_requests=6,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
            token_budget=args.token_budget,
            block_size=block_size,
            stop_on_eos=False,
            capture_logits=True,
        )
        reference = run_engine(model, specs, preempt_mode="swap", **common)
        swap = run_engine(
            model, specs, preempt_mode="swap", pool_blocks=pool_blocks, **common
        )
        recompute = run_engine(
            model, specs, preempt_mode="recompute",
            pool_blocks=pool_blocks, **common,
        )
        entry = {
            "dtype": dtype_name,
            "pool_blocks": pool_blocks,
            "reference_preemptions": reference["metrics"]["preemption_count"],
            "swap_preemptions": swap["metrics"]["preemption_count"],
            "recompute_preemptions": recompute["metrics"]["preemption_count"],
            "swap_resume_wall_ms_total": swap["metrics"]["swap_out_wall_ms_total"]
            + swap["metrics"]["swap_in_wall_ms_total"],
            "requests": [],
        }
        for index in range(len(sequences)):
            reference_tokens = reference["new_token_ids"][index]
            swap_match = swap["new_token_ids"][index] == reference_tokens
            recompute_match = (
                recompute["new_token_ids"][index] == reference_tokens
            )
            entry["requests"].append({
                "request_id": str(index),
                "prompt_tokens": prompt_lengths[index],
                "swap_match": swap_match,
                "swap_divergence": first_divergence(
                    reference_tokens, swap["new_token_ids"][index]
                ),
                "recompute_match": recompute_match,
                "recompute_divergence": first_divergence(
                    reference_tokens, recompute["new_token_ids"][index]
                ),
                "swap_logits": logits_error(
                    reference["token_logits"][str(index)],
                    swap["token_logits"][str(index)],
                ),
                "recompute_logits": logits_error(
                    reference["token_logits"][str(index)],
                    recompute["token_logits"][str(index)],
                ),
            })
        entry["swap_all_match"] = all(
            item["swap_match"] for item in entry["requests"]
        )
        entry["recompute_all_match"] = all(
            item["recompute_match"] for item in entry["requests"]
        )
        report["runs"].append(entry)
        print("抢占次数: reference=%d swap=%d recompute=%d" % (
            entry["reference_preemptions"], entry["swap_preemptions"],
            entry["recompute_preemptions"],
        ))
        for item in entry["requests"]:
            print(
                "  请求 %s（%d token）: swap=%s max|Δlogit|=%.3g；"
                "recompute=%s max|Δlogit|=%.3g" % (
                item["request_id"], item["prompt_tokens"],
                "一致" if item["swap_match"] else "分叉@%s" % item["swap_divergence"],
                item["swap_logits"]["max_abs_error"],
                "一致" if item["recompute_match"] else
                "分叉@%s" % item["recompute_divergence"],
                item["recompute_logits"]["max_abs_error"],
            ))
        del reference, swap, recompute
        del model
        torch.cuda.empty_cache()

    if args.output:
        save_results(args.output, report)


if __name__ == "__main__":
    main()
