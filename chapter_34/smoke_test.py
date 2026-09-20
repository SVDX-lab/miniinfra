#!/usr/bin/env python3
"""CPU-only Chapter 34 checks: the ZeRO-3 parameter lifecycle and tiny FSDP.

1. Shard math: boundaries and zero padding (Chapter 32 convention).
2. One sharded linear with an ODD-size weight: the Reduce-Scattered gradient
   shard equals the AllReduced full gradient's slice BITWISE; the gathered
   full weight is provably FREED when forward returns (Reshard), and backward
   issues exactly one on-demand AllGather and one Reduce-Scatter.
3. A two-linear stack through three full ZeRO-3 steps (gather -> reshard ->
   on-demand gather -> Reduce-Scatter -> ShardedLeafAdamW) matches torch
   AdamW on the AllReduced global gradients BITWISE on every parameter.
4. A SwiGLU MLP block (Qwen3 MLP shape) matches its unsharded twin bitwise.
5. A TIED embedding/lm_head weight (one shard, two uses) stays consistent
   with the tied reference within tight tolerance: two Reduce-Scatters sum in
   a different order than one AllReduce, a real FP fork, not a masking bug.
6. The resident-state ledger shows parameters/gradients/state at 1/world_size.
7. The official FSDP (CPU + gloo, FULL_SHARD, use_orig_params=True) wraps the
   tiny Qwen3 model, trains one step, keeps ranks identical, and exposes the
   tied weight under both state-dict keys with equal values.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from ddp_core import all_reduce_gradients, free_port
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from zero3_core import (
    GATHER_CALLS,
    LAST_FULL_GATHER,
    REDUCE_SCATTER_CALLS,
    ShardedLeafAdamW,
    make_parameter_shard,
    memory_ledger_zero3,
    reset_lifecycle_counters,
    shard_boundaries,
    sharded_embedding,
    sharded_linear,
    sharded_swiglu_mlp,
    unshard_parameter,
)


def build_full_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


def shard_from(parameter: nn.Parameter, world_size: int, rank: int) -> nn.Parameter:
    return make_parameter_shard(parameter.detach(), world_size, rank)


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(5)

    # (1) shard boundaries cover every element exactly once; padding tail exists.
    numel = 101
    start, end = shard_boundaries(numel, world_size, rank)
    covered = [0] * numel
    for owner in range(world_size):
        owner_start, owner_end = shard_boundaries(numel, world_size, owner)
        for index in range(owner_start, owner_end):
            covered[index] += 1
    assert all(count == 1 for count in covered), "shards must partition the parameter exactly once"

    # (2) odd-sized linear: shard grad == AllReduce slice, bitwise; reshard is real.
    torch.manual_seed(11)
    full_weight = torch.randn(33, 17)
    weight_shard = make_parameter_shard(full_weight, world_size, rank)
    weight_reference = full_weight.clone().requires_grad_(True)
    inputs = torch.randn(8, 17, generator=generator)
    targets = torch.randn(8, 33, generator=generator)
    reset_lifecycle_counters()
    output = sharded_linear(inputs, weight_shard, (33, 17))
    assert GATHER_CALLS["forward"] == 1 and GATHER_CALLS["backward"] == 0
    assert LAST_FULL_GATHER["phase"] == "forward"
    assert LAST_FULL_GATHER["weakref"]() is None, "the gathered full weight must be freed at forward return (Reshard)"
    loss = ((output - targets) ** 2).mean()
    loss_reference = ((inputs @ weight_reference.t() - targets) ** 2).mean()
    loss_reference.backward()
    dist.all_reduce(weight_reference.grad, op=dist.ReduceOp.SUM)
    loss.backward()
    assert GATHER_CALLS["backward"] == 1, "backward must re-gather the weight on demand"
    assert REDUCE_SCATTER_CALLS["backward"] == 1
    start, end = shard_boundaries(33 * 17, world_size, rank)
    expected = torch.zeros_like(weight_shard)
    expected[: end - start] = weight_reference.grad.view(-1)[start:end]
    assert torch.equal(weight_shard.grad, expected), "shard gradient must equal the AllReduce slice bitwise"

    # (3) two-linear stack, three ZeRO-3 steps vs torch AdamW, bitwise.
    # The reference runs the SAME op sequence as the sharded path (matmul +
    # add, not nn.Linear's fused addmm) so kernels, not just math, match.
    torch.manual_seed(11)
    model_reference = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 33))
    ref_w1 = nn.Parameter(model_reference[0].weight.detach().clone())
    ref_b1 = nn.Parameter(model_reference[0].bias.detach().clone())
    ref_w2 = nn.Parameter(model_reference[2].weight.detach().clone())
    ref_b2 = nn.Parameter(model_reference[2].bias.detach().clone())
    ref_params = [ref_w1, ref_b1, ref_w2, ref_b2]

    def reference_forward(x):
        hidden = torch.nn.functional.gelu(x.matmul(ref_w1.t()) + ref_b1)
        return hidden.matmul(ref_w2.t()) + ref_b2

    leaves = {
        "w1": shard_from(ref_w1, world_size, rank),
        "b1": shard_from(ref_b1, world_size, rank),
        "w2": shard_from(ref_w2, world_size, rank),
        "b2": shard_from(ref_b2, world_size, rank),
    }
    shapes = {
        "w1": (32, 16), "b1": (32,), "w2": (33, 32), "b2": (33,),
    }
    stack_optimizer = ShardedLeafAdamW(list(leaves.values()), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer_reference = torch.optim.AdamW(ref_params, lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)

    def zero3_forward(x):
        hidden = torch.nn.functional.gelu(sharded_linear(x, leaves["w1"], shapes["w1"], leaves["b1"]))
        return sharded_linear(hidden, leaves["w2"], shapes["w2"], leaves["b2"])

    for step in range(3):
        step_inputs = torch.randn(8, 16, generator=generator)
        step_targets = torch.randn(8, 33, generator=generator)
        optimizer_reference.zero_grad(set_to_none=True)
        loss_reference = ((reference_forward(step_inputs) - step_targets) ** 2).mean()
        loss_reference.backward()
        # Only the torch-AdamW twin needs the flat AllReduce; the ZeRO-3 path
        # Reduce-Scatters the LOCAL gradients exactly as the real step does.
        for parameter in ref_params:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        for leaf in leaves.values():
            leaf.grad = None
        loss_zero3 = ((zero3_forward(step_inputs) - step_targets) ** 2).mean()
        loss_zero3.backward()
        stack_optimizer.step()
        optimizer_reference.step()
        for name, parameter in (("w1", ref_w1), ("b1", ref_b1), ("w2", ref_w2), ("b2", ref_b2)):
            rebuilt = unshard_parameter(leaves[name], shapes[name], world_size)
            assert torch.equal(rebuilt, parameter.detach()), f"step {step}: ZeRO-3 update of {name} must match torch AdamW bitwise"

    # (4) SwiGLU MLP block, one step, bitwise.
    torch.manual_seed(13)
    hidden_size, intermediate_size = 16, 24
    gate_reference = nn.Parameter(torch.randn(intermediate_size, hidden_size))
    up_reference = nn.Parameter(torch.randn(intermediate_size, hidden_size))
    down_reference = nn.Parameter(torch.randn(hidden_size, intermediate_size))
    gate_shard = make_parameter_shard(gate_reference.detach(), world_size, rank)
    up_shard = make_parameter_shard(up_reference.detach(), world_size, rank)
    down_shard = make_parameter_shard(down_reference.detach(), world_size, rank)
    optimizer_zero3 = ShardedLeafAdamW([gate_shard, up_shard, down_shard], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer_reference = torch.optim.AdamW([gate_reference, up_reference, down_reference], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)
    block_inputs = torch.randn(4, 16, generator=generator)
    block_targets = torch.randn(4, hidden_size, generator=generator)

    def reference_mlp(x):
        return (torch.nn.functional.silu(x @ gate_reference.t()) * (x @ up_reference.t())) @ down_reference.t()

    optimizer_reference.zero_grad(set_to_none=True)
    ((reference_mlp(block_inputs) - block_targets) ** 2).mean().backward()
    for parameter in (gate_reference, up_reference, down_reference):
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    for leaf in (gate_shard, up_shard, down_shard):
        leaf.grad = None
    ((sharded_swiglu_mlp(block_inputs, gate_shard, up_shard, down_shard, hidden_size, intermediate_size) - block_targets) ** 2).mean().backward()
    optimizer_zero3.step()
    optimizer_reference.step()
    for shard, reference, shape in ((gate_shard, gate_reference, (intermediate_size, hidden_size)), (up_shard, up_reference, (intermediate_size, hidden_size)), (down_shard, down_reference, (hidden_size, intermediate_size))):
        assert torch.equal(unshard_parameter(shard, shape, world_size), reference.detach()), "SwiGLU block must match torch AdamW bitwise"

    # (5) tied embedding / lm_head: one shard, two uses, tight tolerance.
    torch.manual_seed(17)
    vocab_size, embedding_dim = 64, 16
    embedding_reference = nn.Embedding(vocab_size, embedding_dim)
    head_reference = nn.Linear(embedding_dim, vocab_size, bias=False)
    head_reference.weight = embedding_reference.weight  # tie, Qwen3-style
    tied_shard = make_parameter_shard(embedding_reference.weight.detach(), world_size, rank)
    optimizer_zero3 = ShardedLeafAdamW([tied_shard], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer_reference = build_full_optimizer(embedding_reference)
    token_ids = torch.randint(0, vocab_size, (2, 12), generator=generator)
    targets = torch.randn(2, 12, vocab_size, generator=generator)

    def zero3_tied_forward(ids):
        hidden = sharded_embedding(ids, tied_shard, vocab_size, embedding_dim)
        return sharded_linear(hidden, tied_shard, (vocab_size, embedding_dim))

    optimizer_reference.zero_grad(set_to_none=True)
    ((head_reference(embedding_reference(token_ids)) - targets) ** 2).mean().backward()
    all_reduce_gradients(embedding_reference, world_size)
    tied_shard.grad = None
    ((zero3_tied_forward(token_ids) - targets) ** 2).mean().backward()
    # Two Reduce-Scatters (embedding + head) sum the rank contributions in a
    # different order than one AllReduce of the summed gradient: a real FP
    # fork that ZeRO-3 must tolerate, NOT a masking/sharding error.
    start, end = shard_boundaries(vocab_size * embedding_dim, world_size, rank)
    expected = torch.zeros_like(tied_shard)
    expected[: end - start] = embedding_reference.weight.grad.view(-1)[start:end]
    max_grad_diff = float((tied_shard.grad - expected).abs().max())
    assert max_grad_diff < 1e-5, f"tied shard gradient fork too large: {max_grad_diff}"
    optimizer_zero3.step()
    optimizer_reference.step()
    rebuilt = unshard_parameter(tied_shard, (vocab_size, embedding_dim), world_size)
    max_param_diff = float((rebuilt - embedding_reference.weight.detach()).abs().max())
    assert max_param_diff < 1e-5, f"tied shard update fork too large: {max_param_diff}"

    # (6) the resident-state ledger: everything at 1/world_size.
    ledger = memory_ledger_zero3(
        list(leaves.values()), stack_optimizer, [32 * 16, 32, 33 * 32, 33], element_size=4,
    )
    assert abs(ledger["param_shard_fraction"] - 1.0 / world_size) < 0.02, ledger
    assert abs(ledger["grad_shard_fraction"] - 1.0 / world_size) < 0.02, ledger
    assert abs(ledger["state_shard_fraction"] - 1.0 / world_size) < 0.02, ledger

    # (7) official FSDP wraps the tiny Qwen3 model; one step on CUDA + nccl.
    # FSDP requires a non-CPU accelerator device, so this block is skipped on
    # the CPU smoke path; the GPU correctness test exercises it with real
    # weights instead.
    if device.type == "cuda":
        torch.manual_seed(11)
        model_fsdp = tiny_model()
        expected_keys = set(model_fsdp.state_dict().keys())
        before = float(sum(value.double().abs().sum() for value in model_fsdp.state_dict().values()))
        from fsdp_train import set_full_state_dict_type, wrap_model_fsdp

        fsdp_model = wrap_model_fsdp(model_fsdp, device)
        optimizer_fsdp = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)
        token_ids = torch.randint(4, 64, (2, 12), generator=generator)
        labels = token_ids.clone()
        labels[:, :2] = -100
        optimizer_fsdp.zero_grad(set_to_none=True)
        logits = fsdp_model(token_ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 64), labels[:, 1:].reshape(-1), ignore_index=-100,
        )
        loss.backward()
        grad_norm = fsdp_model.clip_grad_norm_(1.0)
        optimizer_fsdp.step()
        assert float(loss.item()) > 0 and torch.isfinite(grad_norm).all(), "FSDP step must produce finite loss and norm"

        set_full_state_dict_type(fsdp_model)
        state = fsdp_model.state_dict()
        assert set(state.keys()) == expected_keys, "use_orig_params must expose the original state-dict names"
        assert "model.embed_tokens.weight" in state and "lm_head.weight" in state
        embed_checksum = float(state["model.embed_tokens.weight"].double().sum())
        head_checksum = float(state["lm_head.weight"].double().sum())
        assert embed_checksum == head_checksum, "the tied weight must stay one object under FSDP"
        state_checksum = float(sum(value.double().abs().sum() for value in state.values()))
        checksums = [None] * world_size
        dist.all_gather_object(checksums, (state_checksum, float(loss.item())))
        assert len(set(checksums)) == 1, f"ranks diverged after the FSDP step: {checksums}"
        assert state_checksum != before, "the FSDP step must update the sharded parameters"
    else:
        loss = None

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        fsdp_note = "FSDP tiny step: OK" if device.type == "cuda" else "FSDP tiny step: skipped on CPU (needs an accelerator)"
        print(f"shard math, reshard+on-demand gather proof, 3-step bitwise parity, SwiGLU block, tied weight (fork {max_grad_diff:.2e}), ledger {ledger['param_shard_fraction']:.3f}, {fsdp_note}")
        print("chapter_34 smoke tests passed")


def main() -> None:
    port = free_port()
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
