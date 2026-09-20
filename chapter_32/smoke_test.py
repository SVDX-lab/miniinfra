#!/usr/bin/env python3
"""CPU-only Chapter 32 checks: shard math and the hand-written sharded AdamW.

1. ``shard_boundaries`` partitions every parameter exactly, distributes the
   remainder, and never overlaps.
2. On a tiny linear stack (the outline's "small linear layer first"), the
   ShardedStateAdamW + AllGather step is BITWISE equal to torch's AdamW on
   the same full gradients — across three steps, so bias correction at
   t=1,2,3 is all exercised.
3. A tiny Qwen3 model behaves the same way.
4. Two Gloo ranks end with identical parameters and only half the state.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ddp_core import all_reduce_gradients, free_port
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from zero1_core import ShardedStateAdamW, all_gather_parameters, memory_ledger, shard_boundaries


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)


def check_shard_boundaries() -> None:
    for numel, world in ((10, 2), (11, 2), (9, 4), (16, 4), (1, 2), (0, 2)):
        bounds = [shard_boundaries(numel, world, rank) for rank in range(world)]
        assert bounds[0][0] == 0 and bounds[-1][1] == numel, (numel, world, bounds)
        for (_, end_a), (start_b, _) in zip(bounds, bounds[1:]):
            assert end_a == start_b, (numel, world, bounds)
        total = sum(end - start for start, end in bounds)
        assert total == numel, (numel, world, bounds)
    assert shard_boundaries(11, 2, 0) == (0, 6) and shard_boundaries(11, 2, 1) == (6, 11)


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


def linear_stack(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16))


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(5)

    # (2) linear stack: sharded update + all-gather equals torch AdamW bitwise.
    torch.manual_seed(11)
    model_sharded = linear_stack(11)
    torch.manual_seed(11)
    model_full = linear_stack(11)

    optimizer_sharded = ShardedStateAdamW(
        list(model_sharded.parameters()), world_size, rank,
        lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
    )
    optimizer_full = build_optimizer(model_full)

    for step in range(3):
        inputs = torch.randn(8, 16, generator=generator)
        targets = torch.randn(8, 16, generator=generator)

        for model in (model_sharded, model_full):
            (model_sharded if model is model_sharded else model_full).zero_grad(set_to_none=True)
            loss = ((model(inputs) - targets) ** 2).mean()
            loss.backward()
            all_reduce_gradients(model, world_size)

        optimizer_sharded.step()
        allgather_ms, allgather_bytes = all_gather_parameters(list(model_sharded.parameters()), world_size, rank)
        optimizer_full.step()

        for parameter_sharded, parameter_full in zip(model_sharded.parameters(), model_full.parameters()):
            assert torch.equal(parameter_sharded.detach(), parameter_full.detach()), f"step {step}: sharded update must match torch AdamW bitwise"

    # (3) tiny Qwen3 model, one step.
    torch.manual_seed(11)
    model_sharded = tiny_model()
    torch.manual_seed(11)
    model_full = tiny_model()
    optimizer_sharded = ShardedStateAdamW(list(model_sharded.parameters()), world_size, rank, lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer_full = build_optimizer(model_full)

    tokens = torch.randint(4, 64, (2, 12), generator=generator)
    labels = tokens.clone()
    labels[:, :2] = -100
    for model in (model_sharded, model_full):
        optimizer = optimizer_sharded if model is model_sharded else optimizer_full
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            model(tokens)[:, :-1, :].reshape(-1, 64), labels[:, 1:].reshape(-1), ignore_index=-100,
        )
        loss.backward()
        all_reduce_gradients(model, world_size)
    optimizer_sharded.step()
    all_gather_parameters(list(model_sharded.parameters()), world_size, rank)
    optimizer_full.step()
    for parameter_sharded, parameter_full in zip(model_sharded.parameters(), model_full.parameters()):
        assert torch.equal(parameter_sharded.detach(), parameter_full.detach()), "tiny Qwen3 one-step parity failed"

    # (4) cross-rank identity and the state ledger.
    checksum = float(sum(parameter.detach().double().sum() for parameter in model_sharded.parameters()))
    checksums = [None] * world_size
    dist.all_gather_object(checksums, checksum)
    assert len(set(checksums)) == 1, f"ranks diverged: {checksums}"

    ledger = memory_ledger(list(model_sharded.parameters()), optimizer_sharded, world_size)
    assert abs(ledger["state_shard_fraction"] - 1.0 / world_size) < 0.01, ledger

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"linear stack 3 steps + tiny Qwen3 one step: bitwise parity; allgather {allgather_bytes} B/rank, {allgather_ms:.1f} ms")
        print(f"state shard fraction {ledger['state_shard_fraction']:.3f}")
        print("chapter_32 smoke tests passed")


def main() -> None:
    check_shard_boundaries()
    port = free_port()
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
