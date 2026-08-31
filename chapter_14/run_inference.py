"""自然语言演示：recompute、首次 Store 和 warm Load。"""

import argparse

import torch

from engine import run_request
from experiment_utils import (
    add_model_arguments,
    cache_service,
    load_model,
    parse_dtype,
)
from qwen3_tokenizer import Qwen3Tokenizer


def main():
    parser = argparse.ArgumentParser(description="第 14 期外部 KV Cache 推理演示")
    add_model_arguments(parser)
    parser.add_argument("--prompt", default="请用三句话解释什么是 KV Cache。")
    parser.add_argument("--mode", choices=("recompute", "external", "both"), default="both")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--external-chunk-size", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=128)
    parser.add_argument("--capacity-mib", type=float, default=512.0)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    directory, _, model = load_model(args.model, args.revision, device, dtype)
    tokenizer = Qwen3Tokenizer(directory)
    token_ids = tokenizer.encode_chat_prompt(args.prompt)
    common = dict(
        model=model,
        token_ids=token_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        model_id=args.model,
        revision=args.revision,
        block_size=args.block_size,
        external_chunk_size=args.external_chunk_size,
        token_budget=args.token_budget,
        stop_on_eos=True,
    )
    if args.mode in ("recompute", "both"):
        baseline = run_request(mode="recompute", **common)
        print("recompute:", tokenizer.decode(baseline["new_token_ids"], True))
        print("recompute metrics:", baseline["metrics"])
    if args.mode in ("external", "both"):
        with cache_service(args.capacity_mib) as (client, _):
            cold = run_request(mode="external", external_client=client, **common)
            warm = run_request(mode="external", external_client=client, **common)
            print("external cold:", tokenizer.decode(cold["new_token_ids"], True))
            print("external warm:", tokenizer.decode(warm["new_token_ids"], True))
            print("cold metrics:", cold["metrics"])
            print("warm metrics:", warm["metrics"])


if __name__ == "__main__":
    main()
