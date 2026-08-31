"""由 cross_process_validate.py 启动的独立模型进程。"""

import argparse

import torch

from cache_protocol import ExternalCacheClient
from compare_external_cache import synthetic_tokens
from engine import run_request
from experiment_utils import (
    add_model_arguments,
    environment_record,
    load_model,
    parse_dtype,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--role", choices=("producer", "consumer"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prompt-length", type=int, default=513)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--external-chunk-size", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    _, config, model = load_model(args.model, args.revision, device, dtype)
    result = run_request(
        model=model,
        token_ids=synthetic_tokens(args.prompt_length, config.vocab_size),
        max_new_tokens=args.max_new_tokens,
        eos_token_id=-1,
        device=device,
        mode="external",
        external_client=ExternalCacheClient(port=args.port),
        model_id=args.model,
        revision=args.revision,
        block_size=args.block_size,
        external_chunk_size=args.external_chunk_size,
        token_budget=args.token_budget,
        stop_on_eos=False,
    )
    write_json(args.output, {
        "role": args.role,
        "environment": environment_record(args.model, args.revision, device, dtype),
        "new_token_ids": result["new_token_ids"],
        "metrics": result["metrics"],
    })
    print(
        "%s pid=%d hit_tokens=%d stored_chunks=%d"
        % (
            args.role,
            __import__("os").getpid(),
            result["metrics"]["hit_tokens"],
            result["metrics"]["stored_chunks"],
        )
    )


if __name__ == "__main__":
    main()
