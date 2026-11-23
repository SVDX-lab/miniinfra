"""第 13 期自然语言推理入口，展示同步或异步 KV 传输时间线。"""

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
from transfer import TRANSFER_MODES


def parse_args():
    parser = argparse.ArgumentParser(description="运行异步 KV Offloading 推理")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt", action="append", help="可重复传入")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--pool-blocks", type=int, default=14)
    parser.add_argument("--cpu-pool-blocks", type=int, default=None)
    parser.add_argument(
        "--transfer-mode", choices=TRANSFER_MODES, default="async"
    )
    parser.add_argument("--stop-on-eos", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("真实权重推理需要 NVIDIA GPU")
    prompts = args.prompt or [
        "用一句话解释异步传输。",
        "说明 CUDA Stream 与 Event 的分工。",
        "解释为什么 D2H 完成前不能释放 GPU Block。" * 2,
        "说明 KV Cache 异步换入期间的请求状态和数据依赖。" * 3,
    ]
    device = torch.device("cuda")
    directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(directory)
    model = load_handwritten_model(directory, device)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts]
    print("Prompt Token 数:", [len(sequence) for sequence in sequences])
    result = run_engine(
        model,
        make_request_specs(sequences, args.max_new_tokens),
        args.max_running_requests,
        tokenizer.eos_token_id,
        device,
        token_budget=args.token_budget,
        transfer_mode=args.transfer_mode,
        pool_blocks=args.pool_blocks,
        cpu_pool_blocks=args.cpu_pool_blocks,
        block_size=args.block_size,
        stop_on_eos=args.stop_on_eos,
    )

    print("\n=== 传输事件 ===")
    for event in result["resource_events"]:
        if event["type"] not in ("swap_out", "swap_in"):
            continue
        print(
            "[%7.1f ms] %-8s request=%s blocks=%d device=%.2f ms "
            "enqueue=%.3f ms exposed=%.2f ms"
            % (
                event["clock_ms"], event["type"], event["request_id"],
                event["blocks"], event["device_ms"],
                event["submit_wall_ms"], event["exposed_wait_ms"],
            )
        )

    print("\n=== 输出 ===")
    for prompt, token_ids in zip(prompts, result["new_token_ids"]):
        print("Prompt:", prompt[:48] + ("..." if len(prompt) > 48 else ""))
        print("Output:", tokenizer.decode(token_ids))
    metrics = result["metrics"]
    print(
        "\nmode=%s, preempt=%d, resume=%d, transfer device/exposed="
        "%.2f/%.2f ms, makespan=%.1f ms"
        % (
            metrics["transfer_mode"], metrics["preemption_count"],
            metrics["resume_count"], metrics["transfer_device_ms_total"],
            metrics["transfer_exposed_wait_ms_total"], metrics["makespan_ms"],
        )
    )


if __name__ == "__main__":
    main()
