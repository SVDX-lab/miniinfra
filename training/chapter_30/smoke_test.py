#!/usr/bin/env python3
"""CPU-only Chapter 30 checks: bucket layout, hook coverage, value parity.

1. Bucket layout on a tiny model: reverse parameter order, param granularity,
   full coverage, target-size respected, an oversized parameter gets its own
   bucket.
2. Two Gloo ranks run the hook-driven bucketed Backward on identical models:
   after the tail wait, every parameter has a gradient, and the reduced
   gradients are BITWISE equal to the Chapter 29 flat post-backward AllReduce
   (elementwise SUM is order-independent for two ranks).
3. Both a many-bucket layout and a single-bucket layout produce the same
   values; one full optimizer step keeps ranks identical.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bucketed_ddp import BucketCoordinator, build_buckets
from ddp_core import all_reduce_gradients, free_port
from mini_batch import batch_loss, pad_samples
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from assistant_loss import EncodedConversation


def build_optimizer_cpu(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)


def synthetic_sample(seed: int, length: int) -> EncodedConversation:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(4, 64, (1, length), generator=generator)
    labels = tokens.clone()
    labels[:, :2] = -100
    return EncodedConversation(
        input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=labels,
        assistant_loss_mask=labels != -100, source_line=seed, rendered_text="",
        token_spans=(), loss_mode="assistant_only",
    )


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


def check_bucket_layout() -> None:
    model = tiny_model()
    parameters = list(model.parameters())
    small = build_buckets(model, bucket_bytes=16 * 1024)
    assert len(small) > 3, f"small target should split into many buckets, got {len(small)}"
    covered = [id(parameter) for bucket in small for parameter, _, _ in bucket.members]
    assert sorted(covered) == sorted(id(parameter) for parameter in parameters), "coverage broken"

    flat = build_buckets(model, bucket_bytes=2 ** 40)
    assert len(flat) == 1 and len(flat[0].members) == len(parameters), "huge target must yield one bucket"

    # Reverse order: the first bucket starts with the LAST parameter of the model.
    first_member = small[0].members[0][0]
    last_parameter = parameters[-1]
    assert first_member is last_parameter, "buckets must be filled in reverse parameter order"

    # Every bucket respects the target unless a single parameter exceeds it.
    for bucket in small:
        size = bucket.size_bytes
        assert size <= 16 * 1024 or len(bucket.members) == 1, (bucket.index, size)


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.manual_seed(11)

    sample = synthetic_sample(seed=29, length=24)
    batch = pad_samples([sample], pad_token_id=0)

    for bucket_bytes, expect_buckets in ((2 * 1024, "many"), (2 ** 40, "one")):
        torch.manual_seed(11)
        model_bucketed = tiny_model()
        torch.manual_seed(11)
        model_flat = tiny_model()
        coordinator = BucketCoordinator(model_bucketed, bucket_bytes=bucket_bytes, world_size=world_size)
        buckets = coordinator.buckets
        assert (len(buckets) > 1) == (expect_buckets == "many"), (len(buckets), expect_buckets)

        optimizer_bucketed = build_optimizer_cpu(model_bucketed)
        optimizer_flat = build_optimizer_cpu(model_flat)
        optimizer_bucketed.zero_grad(set_to_none=True)
        optimizer_flat.zero_grad(set_to_none=True)

        logits_flat = model_flat(batch.input_ids, attention_mask=batch.attention_mask)
        loss_flat = batch_loss(logits_flat, batch.labels, "token_mean")
        loss_flat.backward()
        all_reduce_gradients(model_flat, world_size)

        logits = model_bucketed(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean")
        loss.backward()
        tail = coordinator.wait_and_attach()
        assert tail["buckets_fired"] == len(buckets), tail
        for parameter in model_bucketed.parameters():
            assert parameter.grad is not None, "tail wait must re-attach every gradient"

        worst = max(
            float((parameter_bucketed.grad - parameter_flat.grad).abs().max())
            for parameter_bucketed, parameter_flat in zip(model_bucketed.parameters(), model_flat.parameters())
        )
        assert worst == 0.0, f"bucketed grads must match flat AllReduce bitwise, got {worst}"

        torch.nn.utils.clip_grad_norm_(model_bucketed.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model_flat.parameters(), 1.0)
        optimizer_bucketed.step()
        optimizer_flat.step()

        checksum = float(sum(parameter.detach().double().sum() for parameter in model_bucketed.parameters()))
        checksums = [None] * world_size
        dist.all_gather_object(checksums, checksum)
        assert len(set(checksums)) == 1, f"ranks diverged: {checksums}"

        coordinator.remove_hooks()
        if rank == 0:
            print(f"bucket layout {expect_buckets} ({len(buckets)} buckets): parity OK, cross-rank identical")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("chapter_30 smoke tests passed")


def main() -> None:
    check_bucket_layout()
    port = free_port()
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
