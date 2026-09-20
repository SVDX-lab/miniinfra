#!/usr/bin/env python3
"""CPU-only Chapter 33 checks: Reduce-Scatter shards and the ZeRO-2 optimizer.

1. ``reduce_scatter`` of two ranks' tensors delivers the owner exactly the
   bitwise-equal chunk of the AllReduce result.
2. On a tiny linear stack (including an ODD-sized bias, exercising the
   padding path), the ZeRO-2 flow — backward, Reduce-Scatter into shard
   buffers, ShardedGradAdamW step, parameter AllGather — matches torch AdamW
   on the same global gradients bitwise across three steps.
3. A tiny Qwen3 model behaves the same way.
4. The state ledger shows gradients AND moments at 1/world_size.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ddp_core import all_reduce_gradients, free_port
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from zero2_core import ShardedGradAdamW, all_gather_parameters, memory_ledger, reduce_scatter_gradients, ring_reduce_scatter


def build_full_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)


def linear_stack(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 33))


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(5)

    # (1) ring reduce-scatter chunk equals the allreduce slice, bitwise.
    tensor = torch.randn(101, generator=generator) * (rank + 1)  # odd size: padding path
    padded = torch.zeros(102)
    padded[:101] = tensor
    chunk = 51
    output = torch.empty(chunk)
    ring_reduce_scatter(padded, output, world_size, rank)
    reference = padded.clone()
    dist.all_reduce(reference[:101])
    if rank == 0:
        expected = reference[:chunk]
    else:
        expected = reference[chunk:101]
        expected = torch.cat([expected, torch.zeros(chunk - expected.numel())])
    assert torch.equal(output, expected), "reduce_scatter chunk must equal the allreduce slice bitwise"

    # (2) linear stack through the full ZeRO-2 flow.
    torch.manual_seed(11)
    model_zero2 = linear_stack(11)
    torch.manual_seed(11)
    model_full = linear_stack(11)
    optimizer_zero2 = ShardedGradAdamW(list(model_zero2.parameters()), world_size, rank, lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer_full = build_full_optimizer(model_full)

    for step in range(3):
        inputs = torch.randn(8, 16, generator=generator)
        targets = torch.randn(8, 33, generator=generator)
        for model, optimizer in ((model_zero2, optimizer_zero2), (model_full, optimizer_full)):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(inputs) - targets) ** 2).mean()
            loss.backward()
            if model is model_full:
                # Only the torch-AdamW reference needs the flat AllReduce; the
                # ZeRO-2 path must Reduce-Scatter the LOCAL gradients exactly
                # as the real step does.
                all_reduce_gradients(model, world_size)
        reduce_scatter_gradients(list(model_zero2.parameters()), optimizer_zero2, world_size)
        optimizer_zero2.step()
        all_gather_parameters(list(model_zero2.parameters()), world_size, rank)
        optimizer_full.step()
        for parameter_zero2, parameter_full in zip(model_zero2.parameters(), model_full.parameters()):
            assert torch.equal(parameter_zero2.detach(), parameter_full.detach()), f"step {step}: ZeRO-2 update must match torch AdamW bitwise"

    # (3) tiny Qwen3 model, one step.
    torch.manual_seed(11)
    model_zero2 = tiny_model()
    torch.manual_seed(11)
    model_full = tiny_model()
    optimizer_zero2 = ShardedGradAdamW(list(model_zero2.parameters()), world_size, rank, lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer_full = build_full_optimizer(model_full)
    tokens = torch.randint(4, 64, (2, 12), generator=generator)
    labels = tokens.clone()
    labels[:, :2] = -100
    for model, optimizer in ((model_zero2, optimizer_zero2), (model_full, optimizer_full)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            model(tokens)[:, :-1, :].reshape(-1, 64), labels[:, 1:].reshape(-1), ignore_index=-100,
        )
        loss.backward()
        if model is model_full:
            all_reduce_gradients(model, world_size)
    reduce_scatter_gradients(list(model_zero2.parameters()), optimizer_zero2, world_size)
    optimizer_zero2.step()
    all_gather_parameters(list(model_zero2.parameters()), world_size, rank)
    optimizer_full.step()
    for parameter_zero2, parameter_full in zip(model_zero2.parameters(), model_full.parameters()):
        assert torch.equal(parameter_zero2.detach(), parameter_full.detach()), "tiny Qwen3 one-step parity failed"

    # (4) ledger and cross-rank identity.
    ledger = memory_ledger(list(model_zero2.parameters()), optimizer_zero2)
    assert abs(ledger["grad_shard_fraction"] - 1.0 / world_size) < 0.01, ledger
    assert abs(ledger["state_shard_fraction"] - 1.0 / world_size) < 0.01, ledger
    checksum = float(sum(parameter.detach().double().sum() for parameter in model_zero2.parameters()))
    checksums = [None] * world_size
    dist.all_gather_object(checksums, checksum)
    assert len(set(checksums)) == 1, f"ranks diverged: {checksums}"

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"reduce_scatter parity, 3-step ZeRO-2 flow, tiny Qwen3: bitwise OK; ledger fractions {ledger['grad_shard_fraction']:.3f}/{ledger['state_shard_fraction']:.3f}")
        print("chapter_33 smoke tests passed")


def main() -> None:
    port = free_port()
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
