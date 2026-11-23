#!/usr/bin/env python3
"""Chapter 43 export: gather the DDP state into single-card inference weights.

Runs a short DDP training (or just loads a checkpoint via ``--resume-from``)
and writes the FULL FP32 state dict as one safetensors file with the plain
Qwen3 layout — loadable by the chapter-16 inference stack without any
distributed runtime.  Rank 0 writes; all ranks synchronize on barriers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from checkpointing import (
    detect_mode,
    export_inference_weights,
    load_for_resume,
    model_checksum,
)
from ddp_core import cleanup_distributed, setup_distributed
from experiment_utils import environment_record, write_json
from train_worker import build_optimizer_and_scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true", help="export the tiny synthetic model")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="load this checkpoint instead of fresh weights")
    parser.add_argument("--resume-mode", default="auto", choices=("auto", "centralized", "sharded", "dcp"))
    parser.add_argument("--steps", type=int, default=1, help="training steps before the export")
    parser.add_argument("--global-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=Path("exported/qwen3-exported.safetensors"))
    parser.add_argument("--report", type=Path, default=Path("export-report.local.json"))
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    use_cuda = torch.cuda.is_available() and not args.tiny
    backend = "nccl" if use_cuda else "gloo"
    rank, world_size, device = setup_distributed(rank, world_size, backend=backend)
    try:
        from sft_core import seed_everything

        seed_everything(args.seed)
        if args.tiny:
            from train_worker import TINY_CONFIG
            from qwen3_model import Qwen3ForCausalLM

            torch.manual_seed(args.seed)
            model = Qwen3ForCausalLM(TINY_CONFIG).to(device)
            config = {"model_id": "tiny-synthetic"}
        else:
            from ddp_core import load_tokenizer_and_model

            _tokenizer, model = load_tokenizer_and_model(device)
            config = {"model_id": "Qwen/Qwen3-0.6B-Base"}

        if args.resume_from:
            optimizer, scheduler = build_optimizer_and_scheduler(model, args.learning_rate, 1, args.steps)
            mode = args.resume_mode if args.resume_mode != "auto" else detect_mode(Path(args.resume_from))
            load_for_resume(Path(args.resume_from), model, optimizer, scheduler, device,
                            rank, world_size, mode)

        if rank == 0:
            metrics = export_inference_weights(model, args.output, config)
            payload = environment_record(world_size=world_size, dp_degree=world_size)
            payload.update({
                "output": str(args.output),
                "resumed_from": str(args.resume_from) if args.resume_from else None,
                "model_checksum_before_export": model_checksum(model),
                "export_bytes": metrics["bytes"],
                "export_seconds": metrics["seconds"],
                "tensors": metrics["tensors"],
                "layout": "plain Qwen3ForCausalLM state dict (model.* / lm_head.weight)",
            })
            write_json(args.report, payload)
            print(f"exported {metrics['tensors']:.0f} tensors ({metrics['bytes']} bytes) to {args.output}")
        torch.distributed.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
