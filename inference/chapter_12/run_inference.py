"""第 12 期自然语言推理入口。

用缩小的 GPU Block Pool 运行多个请求，展示抢占、换出/换入（或丢弃重算）
事件与最终输出。默认负载在生成阶段必然触发一次以上抢占。
"""

import argparse

import torch

from engine import run_engine
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer
from scheduler import make_request_specs

PREEMPT_MODES = ("swap", "recompute")
ADMISSION_MODES = ("incremental", "conservative")


def parse_args():
    parser = argparse.ArgumentParser(description="运行抢占式换出推理")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt", action="append", help="可重复传入")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--pool-blocks", type=int, default=14,
                        help="GPU Block Pool 大小；默认值使本演示必然触发抢占")
    parser.add_argument("--cpu-pool-blocks", type=int, default=None)
    parser.add_argument("--preempt-mode", choices=PREEMPT_MODES, default="swap")
    parser.add_argument("--admission", choices=ADMISSION_MODES,
                        default="incremental")
    parser.add_argument("--stop-on-eos", action="store_true",
                        help="默认关闭 EOS：固定输出预算保证抢占压力可预测")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("真实权重推理需要可用的 NVIDIA GPU")
    prompts = args.prompt or [
        "用一句话解释 KV Cache 换出。",
        "请列举三种 GPU 显存不足时的处理方式。",
        "解释 PCIe 带宽对 KV 搬运时间的影响。" * 2,
        "简要说明抢占式调度在推理引擎中的作用。" * 3,
    ]
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    model = load_handwritten_model(model_directory, device)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts]
    print("Prompt Token 数:", [len(sequence) for sequence in sequences])
    specs = make_request_specs(sequences, args.max_new_tokens)
    result = run_engine(
        model, specs, args.max_running_requests,
        tokenizer.eos_token_id, device,
        token_budget=args.token_budget,
        preempt_mode=args.preempt_mode,
        admission_mode=args.admission,
        pool_blocks=args.pool_blocks,
        cpu_pool_blocks=args.cpu_pool_blocks,
        block_size=args.block_size, stop_on_eos=args.stop_on_eos,
    )

    print("\n=== 资源事件时间线 ===")
    for event in result["resource_events"]:
        if event["type"] == "admit":
            continue
        description = {
            "swap_out": "换出 %d 块 (%.1f MiB) %.2f ms %.2f GB/s",
            "swap_in": "换入 %d 块 (%.1f MiB) %.2f ms %.2f GB/s",
            "preempt_drop": "丢弃 %d 块 (%.1f MiB) %.2f ms",
            "resume_recompute": "重算恢复 %d token",
        }[event["type"]]
        if event["type"] == "resume_recompute":
            print("[%.1f ms] 请求 %s: %s" % (
                event["clock_ms"], event["request_id"],
                description % (event["redo_tokens"],),
            ))
        elif event["type"] == "preempt_drop":
            print("[%.1f ms] 请求 %s: %s" % (
                event["clock_ms"], event["request_id"],
                description % (
                    event["blocks"], event["bytes"] / 2**20,
                    event["wall_ms"],
                ),
            ))
        else:
            print("[%.1f ms] 请求 %s: %s" % (
                event["clock_ms"], event["request_id"],
                description % (
                    event["blocks"], event["bytes"] / 2**20,
                    event["wall_ms"],
                    event.get("gb_per_second", 0.0),
                ),
            ))

    print("\n=== 输出 ===")
    for prompt, token_ids in zip(prompts, result["new_token_ids"]):
        print("Prompt:", prompt[:40] + ("..." if len(prompt) > 40 else ""))
        print("Output:", tokenizer.decode(token_ids))
    metrics = result["metrics"]
    print("\n抢占次数: %d, 恢复次数: %d, 逻辑并发峰值: %d" % (
        metrics["preemption_count"], metrics["resume_count"],
        metrics["logical_concurrency_peak"],
    ))
    print("swap 换出: %.1f MiB / %.2f ms（%.2f GB/s）" % (
        metrics["swap_out_bytes_total"] / 2**20,
        metrics["swap_out_wall_ms_total"],
        metrics["swap_out_gb_per_second"],
    ))
    print("swap 换入: %.1f MiB / %.2f ms（%.2f GB/s）" % (
        metrics["swap_in_bytes_total"] / 2**20,
        metrics["swap_in_wall_ms_total"],
        metrics["swap_in_gb_per_second"],
    ))
    if metrics["swap_out_bytes_total"]:
        print("逻辑有效 / 尾块碎片（单向换出）: %.1f / %.1f MiB" % (
            metrics["swap_out_logical_bytes_total"] / 2**20,
            metrics["swap_out_tail_fragment_bytes_total"] / 2**20,
        ))
    if metrics["recompute_redo_tokens_total"]:
        print("recompute: %d token / %.2f ms Prefill 暴露时间" % (
            metrics["recompute_redo_tokens_total"],
            metrics["recompute_prefill_wall_ms_total"],
        ))


if __name__ == "__main__":
    main()
