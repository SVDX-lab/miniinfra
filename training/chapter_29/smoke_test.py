#!/usr/bin/env python3
"""CPU-only Chapter 29 checks: ring AllReduce, sharding, scaling math, one step.

Everything runs on CPU with the Gloo backend and a tiny random model:

1. Ring AllReduce volume formula against hand-computed values.
2. Ring AllReduce parity with ``dist.all_reduce``: bitwise on int64, FP32
   tolerance on float32.
3. Batch-sharding invariants: rank shards partition every global batch, and
   every sample appears exactly once per epoch.
4. The supervised-token loss scaling: sharded gradients after AllReduce SUM
   equal the big-batch ``token_mean`` gradient; the naive "mean of means"
   weighting is a proven counterexample that must deviate.
5. One full optimizer step: DDP across two ranks stays within tolerance of
   the single-process big batch, and both ranks end bitwise identical.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from ddp_core import all_reduce_gradients, all_reduce_scalars, free_port, shard_members
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from ring_allreduce import allreduce_volume, ring_all_reduce


TOLERANCE = 1e-5
LEARNING_RATE = 1e-2  # larger than the tolerance so the step check can discriminate


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )


def build_tiny_model(generator_seed: int) -> torch.nn.Module:
    torch.manual_seed(generator_seed)
    return Qwen3ForCausalLM(tiny_config())


def build_optimizer_cpu(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)


def synthetic_group(seed: int, size: int, length: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(4, 64, (size, length), generator=generator)
    labels = tokens.clone()
    labels[:, :2] = -100
    return tokens, labels


def synthetic_sample(seed: int, length: int):
    from assistant_loss import EncodedConversation

    tokens, labels = synthetic_group(seed, 1, length)
    return EncodedConversation(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        labels=labels,
        assistant_loss_mask=labels != -100,
        source_line=seed,
        rendered_text="",
        token_spans=(),
        loss_mode="assistant_only",
    )


def check_volume_formula() -> None:
    numbers = allreduce_volume(1_000, 1, world_size=2)
    assert numbers["steps"] == 2, numbers
    assert numbers["bytes_per_link"] == 1_000.0, numbers  # 2*(W-1)/W * D = D for W=2
    numbers4 = allreduce_volume(1_000, 1, world_size=4)
    assert numbers4["steps"] == 6, numbers4
    assert numbers4["bytes_per_link"] == 1_000 * 1.5, numbers4  # 2*(W-1)/W = 3/2
    numbers8 = allreduce_volume(8, 1, world_size=8)
    assert numbers8["bytes_per_link"] == 8 * 1.75, numbers8


def check_sharding_invariants() -> None:
    plan = [tuple(sorted(range(i, i + 4))) for i in range(0, 12, 4)]
    for members in plan:
        shards = [shard_members(members, rank, 2) for rank in range(2)]
        union = sorted(sum((list(shard) for shard in shards), []))
        assert union == list(members), (shards, members)
        assert shards[0] == tuple(members[0::2]) and shards[1] == tuple(members[1::2])
    covered = [index for members in plan for index in members]
    assert sorted(covered) == list(range(12)), "every sample must appear exactly once per epoch"


def check_counterexample() -> None:
    """Mean-of-means weighting must deviate from the global token_mean objective."""

    from mini_batch import batch_loss, pad_samples

    model_a = build_tiny_model(generator_seed=17)
    model_b = build_tiny_model(generator_seed=17)
    sample_small = synthetic_sample(seed=19, length=8)    # few supervised tokens
    sample_large = synthetic_sample(seed=23, length=32)   # many supervised tokens
    batch = pad_samples([sample_small, sample_large], pad_token_id=0)

    def global_token_mean(model):
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        return batch_loss(logits, batch.labels, "token_mean")

    def mean_of_means(model):
        losses = []
        for sample in (sample_small, sample_large):
            single = pad_samples([sample], pad_token_id=0)
            logits = model(single.input_ids, attention_mask=single.attention_mask)
            losses.append(batch_loss(logits, single.labels, "token_mean"))
        return torch.stack(losses).mean()

    global_token_mean(model_a).backward()
    mean_of_means(model_b).backward()
    worst = max(
        float((gradient_a - gradient_b).abs().max())
        for gradient_a, gradient_b in zip(
            (parameter.grad for parameter in model_a.parameters()),
            (parameter.grad for parameter in model_b.parameters()),
        )
    )
    assert worst > 1e-4, f"counterexample must deviate, got {worst}"
    print(f"counterexample separation: OK (max grad deviation {worst:.3e})")


def _big_batch_step(model, samples, optimizer):
    from mini_batch import batch_loss, pad_samples

    batch = pad_samples(list(samples), pad_token_id=0)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.input_ids, attention_mask=batch.attention_mask)
    loss = batch_loss(logits, batch.labels, "token_mean")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item()), float(grad_norm), batch.supervised_tokens


def _sharded_step(model, samples, rank, world_size, optimizer):
    from mini_batch import batch_loss, pad_samples

    supervised_total = sum(sample.supervised_tokens for sample in samples)
    optimizer.zero_grad(set_to_none=True)
    scaled_loss_sum = 0.0
    for sample in samples[rank::world_size]:
        single = pad_samples([sample], pad_token_id=0)
        logits = model(single.input_ids, attention_mask=single.attention_mask)
        loss = batch_loss(logits, single.labels, "token_mean")
        (loss * (single.supervised_tokens / supervised_total)).backward()
        scaled_loss_sum += float(loss.item()) * (single.supervised_tokens / supervised_total)
    all_reduce_gradients(model, world_size)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return scaled_loss_sum, float(grad_norm)


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    generator = torch.Generator().manual_seed(7)

    # (2) ring parity: integer bitwise, float tolerance.
    integers = torch.arange(1000, dtype=torch.int64) * (rank + 1)
    ringed = ring_all_reduce(integers, rank, world_size)
    reference = integers.clone()
    dist.all_reduce(reference)
    assert torch.equal(ringed, reference), "int64 ring AllReduce must be bitwise exact"
    floats = torch.randn(1000, generator=generator) * (rank + 1)
    ringed_float = ring_all_reduce(floats, rank, world_size)
    reference_float = floats.clone()
    dist.all_reduce(reference_float)
    assert torch.allclose(ringed_float, reference_float, rtol=1e-6, atol=1e-6), "float ring AllReduce drifted"

    # (4)+(5) scaling math and one optimizer step on a tiny model.
    model_sharded = build_tiny_model(generator_seed=11)
    model_reference = build_tiny_model(generator_seed=11)
    samples = [synthetic_sample(seed=29, length=8), synthetic_sample(seed=31, length=32)]
    optimizer_sharded = build_optimizer_cpu(model_sharded)
    optimizer_reference = build_optimizer_cpu(model_reference)

    big_loss, big_norm, _supervised_total = _big_batch_step(model_reference, samples, optimizer_reference)
    sharded_loss, sharded_norm = _sharded_step(model_sharded, samples, rank, world_size, optimizer_sharded)
    global_loss = all_reduce_scalars(sharded_loss, world_size, torch.device("cpu"))

    if rank == 0:
        loss_rel = abs(big_loss - global_loss) / abs(big_loss)
        assert loss_rel < TOLERANCE, f"global loss mismatch: {big_loss} vs {global_loss}"
        max_error = max(
            float((parameter_sharded.detach() - parameter_reference.detach()).abs().max())
            for parameter_sharded, parameter_reference in zip(model_sharded.parameters(), model_reference.parameters())
        )
        assert max_error < TOLERANCE, f"one-step parameter error {max_error}"
        norm_rel = abs(big_norm - sharded_norm) / big_norm
        assert norm_rel < TOLERANCE, f"gradient norm error {norm_rel}"
        print(f"scaling math and one-step consistency: OK (big loss {big_loss:.6f}, global loss {global_loss:.6f}, max param error {max_error:.2e})")

    checksum = float(sum(parameter.detach().double().sum() for parameter in model_sharded.parameters()))
    checksums = [None] * world_size
    dist.all_gather_object(checksums, checksum)
    assert len(set(checksums)) == 1, f"ranks diverged after one step: {checksums}"
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("ring parity and cross-rank identity: OK")


def main() -> None:
    check_volume_formula()
    check_sharding_invariants()
    check_counterexample()
    port = free_port()
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)
    print("chapter_29 smoke tests passed")


if __name__ == "__main__":
    main()
