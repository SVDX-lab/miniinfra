#!/usr/bin/env python3
"""CPU-only Chapter 35 checks: MLP shard math, gradients, and one update.

Runs a 2-rank gloo group with a tiny Qwen3 (hidden 32, intermediate 64,
vocab 64, 2 layers, tp=2):

1. ParallelMLP(tp=1) equals the replicated SwiGLU definition (no-op comm path).
2. ParallelMLP(tp=2) forward equals the full MLP up to FP32 reduction order
   (the AllReduce sums rank partials in a different association).
3. The input gradient through the parallel block equals the reference input
   gradient (the "f" op AllReduce makes the partial sums exact again).
4. Every shard weight gradient equals the matching slice of the full weight
   gradient — shard gradients are rank-local by construction.
5. One AdamW step on the tiny model: reassembled TP parameters match the
   replicated reference model trained with plain AdamW + clip_grad_norm_.
6. Ownership proof: replicated-parameter gradients are IDENTICAL across ranks
   without any communication, while shard gradients are NOT (they must not be
   synchronized either).
7. Divisibility violations raise errors that carry shape and tp_size.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from assistant_loss import IGNORE_INDEX
from mlp_tp import (
    ParallelMLP,
    assemble_tensor_parallel_weights,
    full_mlp_reference,
    gather_tensor_parallel_parameters,
    split_local_parameters,
    split_tensor_parallel_weights,
    tp_train_step,
)
from mini_batch import PaddedBatch, batch_loss
from qwen3_model import Qwen3Config, Qwen3ForCausalLM


HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 64
LAYERS = 2


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )


def mlp_weights(generator: torch.Generator) -> dict:
    return {
        "mlp.gate_proj.weight": torch.randn(INTERMEDIATE, HIDDEN, generator=generator) * 0.3,
        "mlp.up_proj.weight": torch.randn(INTERMEDIATE, HIDDEN, generator=generator) * 0.3,
        "mlp.down_proj.weight": torch.randn(HIDDEN, INTERMEDIATE, generator=generator) * 0.3,
    }


def tiny_batch(generator: torch.Generator) -> PaddedBatch:
    tokens = torch.randint(4, VOCAB, (2, 12), generator=generator)
    labels = tokens.clone()
    labels[:, :2] = IGNORE_INDEX
    supervised = [(row[1:] != IGNORE_INDEX).sum().item() for row in labels]
    return PaddedBatch(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        labels=labels,
        source_lines=(1, 2),
        lengths=(12, 12),
        supervised_tokens_per_sample=tuple(supervised),
    )


def expect_error(action, *needles: str) -> None:
    try:
        action()
    except ValueError as error:
        message = str(error)
        missing = [needle for needle in needles if needle not in message]
        assert not missing, f"error message lacks context {missing}: {message}"
        return
    raise AssertionError(f"expected a ValueError containing {needles}")


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_rank, tp_size = rank, world_size
    generator = torch.Generator().manual_seed(5)

    # (1) tp=1 is the replicated baseline: same math, no collective calls.
    weights = mlp_weights(generator)
    parallel_tp1 = ParallelMLP(HIDDEN, INTERMEDIATE, 0, 1)
    parallel_tp1.load_state_dict(_strip_prefix(weights))
    x0 = torch.randn(3, 5, HIDDEN, generator=generator)
    torch.testing.assert_close(
        parallel_tp1(x0),
        full_mlp_reference(x0, weights["mlp.gate_proj.weight"], weights["mlp.up_proj.weight"], weights["mlp.down_proj.weight"]),
        rtol=1e-6, atol=1e-7, msg="tp=1 ParallelMLP must equal the replicated SwiGLU",
    )

    # (2)-(4) tp=2 against the replicated reference, same inputs everywhere.
    shards = split_tensor_parallel_weights(weights, tp_size)
    parallel = ParallelMLP(HIDDEN, INTERMEDIATE, tp_rank, tp_size)
    parallel.load_state_dict(_strip_prefix(shards[tp_rank]))
    slice_ = parallel.intermediate_slice()

    x = torch.randn(4, 6, HIDDEN, generator=generator)
    upstream = torch.randn(4, 6, HIDDEN, generator=generator)

    x_parallel = x.clone().requires_grad_(True)
    out_parallel = parallel(x_parallel)
    (out_parallel * upstream).sum().backward()

    gate_ref = weights["mlp.gate_proj.weight"].clone().requires_grad_(True)
    up_ref = weights["mlp.up_proj.weight"].clone().requires_grad_(True)
    down_ref = weights["mlp.down_proj.weight"].clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)
    out_ref = full_mlp_reference(x_ref, gate_ref, up_ref, down_ref)
    (out_ref * upstream).sum().backward()

    # Forward: AllReduce sums rank partials; FP32 association differs, hence
    # the tolerance instead of bitwise equality.
    torch.testing.assert_close(out_parallel, out_ref, rtol=1e-5, atol=1e-6,
                               msg="parallel MLP forward must match the full MLP")
    # Input gradient: rank partials are AllReduced by the "f" op.
    torch.testing.assert_close(x_parallel.grad, x_ref.grad, rtol=1e-5, atol=1e-6,
                               msg="parallel MLP input gradient must match the full MLP")
    # Weight gradients: rank-local shards of the full gradient.
    torch.testing.assert_close(parallel.gate_proj.weight.grad, gate_ref.grad[slice_], rtol=1e-5, atol=1e-7,
                               msg="gate shard gradient must match the full gradient slice")
    torch.testing.assert_close(parallel.up_proj.weight.grad, up_ref.grad[slice_], rtol=1e-5, atol=1e-7,
                               msg="up shard gradient must match the full gradient slice")
    torch.testing.assert_close(parallel.down_proj.weight.grad, down_ref.grad[:, slice_], rtol=1e-5, atol=1e-7,
                               msg="down shard gradient must match the full gradient slice")

    gathered_outputs = [None] * world_size
    dist.all_gather_object(gathered_outputs, out_parallel.detach())
    for other_rank, other_output in enumerate(gathered_outputs):
        assert torch.equal(out_parallel.detach(), other_output), \
            f"forward output differs between rank {tp_rank} and rank {other_rank}"

    # (6) ownership on the tiny model BEFORE any optimizer step.
    torch.manual_seed(11)
    full_model = Qwen3ForCausalLM(tiny_config())
    torch.manual_seed(11)
    tp_model = Qwen3ForCausalLM(tiny_config(), mlp_factory=lambda cfg: ParallelMLP(
        cfg.hidden_size, cfg.intermediate_size, tp_rank, tp_size))
    full_sd = full_model.state_dict()
    tp_model.load_state_dict(split_tensor_parallel_weights(full_sd, tp_size)[tp_rank])

    batch = tiny_batch(generator)
    tp_logits = tp_model(batch.input_ids, attention_mask=batch.attention_mask)
    batch_loss(tp_logits, batch.labels, "token_mean").backward()
    replicated, shard = split_local_parameters(tp_model)
    assert len(shard) == LAYERS * 3 and len(replicated) > 0, (len(replicated), len(shard))

    replicated_sums = [None] * world_size
    dist.all_gather_object(replicated_sums, [float(p.grad.double().sum()) for p in replicated])
    for other_rank, other_sums in enumerate(replicated_sums):
        for own, other in zip((float(p.grad.double().sum()) for p in replicated), other_sums):
            assert abs(own - other) <= 1e-9 * max(1.0, abs(own)), \
                f"replicated gradients differ between rank {tp_rank} and rank {other_rank}: {own} vs {other}"
    shard_sums = [None] * world_size
    dist.all_gather_object(shard_sums, [float(p.grad.double().sum()) for p in shard])
    assert len({repr(sums) for sums in shard_sums}) == world_size, \
        "shard gradients are rank-local; identical sums across ranks would mean the MLP was never split"

    # (5) one AdamW step: TP group vs replicated reference (plain clip).
    full_model.zero_grad(set_to_none=True)
    tp_model.zero_grad(set_to_none=True)
    optimizer_ref = torch.optim.AdamW(full_model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)
    optimizer_tp = torch.optim.AdamW(tp_model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)

    ref_logits = full_model(batch.input_ids, attention_mask=batch.attention_mask)
    ref_loss = batch_loss(ref_logits, batch.labels, "token_mean")
    ref_loss.backward()
    torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
    optimizer_ref.step()

    metrics = tp_train_step(tp_model, optimizer_tp, batch, 1.0, torch.device("cpu"), 1.0, tp_size, tp_rank)
    torch.testing.assert_close(
        torch.tensor(metrics["global_loss"]), torch.tensor(float(ref_loss.item())),
        rtol=1e-5, atol=1e-7, msg="TP step loss must match the replicated reference loss",
    )
    assert metrics["tp_allreduce_forward_count"] == LAYERS, metrics
    assert metrics["tp_allreduce_backward_count"] == LAYERS, metrics

    # ``gather_tensor_parallel_parameters`` already returns the FULL state on
    # every rank: replicated parameters locally, shard parameters AllGathered
    # and concatenated in rank order.
    reassembled = gather_tensor_parallel_parameters(tp_model, tp_size)
    for name, parameter in full_model.named_parameters():
        torch.testing.assert_close(
            reassembled[name], parameter.detach(),
            rtol=1e-4, atol=1e-7,
            msg=f"reassembled {name} must match the replicated reference after one AdamW step",
        )

    # (7) divisibility violations must fail loudly with shape/tp_size context.
    expect_error(lambda: split_tensor_parallel_weights(weights, 3), "tp_size=3", "full_weight_shape")
    expect_error(lambda: ParallelMLP(HIDDEN, INTERMEDIATE, tp_rank, 3), "tp_size=3", "tp_rank")

    # (8) split -> assemble round trip is bitwise exact, replicated keys included.
    reassembled_weights = assemble_tensor_parallel_weights(shards)
    for key, weight in weights.items():
        assert torch.equal(reassembled_weights[key], weight), f"round trip changed {key}"

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("MLP TP forward/input-grad/weight-grad shards, one-step parity, ownership, divisibility errors: OK")
        print("chapter_35 smoke tests passed")


def _strip_prefix(state_dict: dict) -> dict:
    """Standalone ParallelMLP modules drop the model-level ``mlp.`` prefix."""

    return {key.removeprefix("mlp."): value for key, value in state_dict.items()}


def main() -> None:
    port = _free_port()
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


if __name__ == "__main__":
    main()
