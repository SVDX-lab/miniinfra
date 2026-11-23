#!/usr/bin/env python3
"""CPU-only Chapter 31 checks: both sync modes reproduce the big-batch step.

Two Gloo ranks, a tiny model, a global batch of four synthetic samples
sharded 2+2 and accumulated over K=2 micro-steps:

1. ``every`` and ``last`` must produce equal gradients within FP32 tolerance
   (the two modes group the same SUM differently, and FP32 addition is not
   associative) and within tolerance of the single-process big-batch
   ``token_mean`` gradient.
2. ``allreduce_count`` must be K for ``every`` and 1 for ``last`` — the whole
   point of the chapter.
3. One optimizer step keeps ranks identical in both modes, and both modes
   land within tolerance of the big-batch step.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ddp_core import free_port
from mini_batch import batch_loss, pad_samples
from no_sync_core import accumulate_step, build_micro_batches
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from assistant_loss import EncodedConversation


TOLERANCE = 1e-5


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


def build_optimizer(model):
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


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")

    samples = [synthetic_sample(seed=41 + index, length=8 + 4 * index) for index in range(4)]
    members = (0, 1, 2, 3)
    supervised_global = sum(sample.supervised_tokens for sample in samples)

    # Reference: single process, big padded batch, global token_mean.
    torch.manual_seed(11)
    model_reference = tiny_model()
    batch = pad_samples(samples, pad_token_id=0)
    optimizer_reference = build_optimizer(model_reference)
    optimizer_reference.zero_grad(set_to_none=True)
    loss = batch_loss(model_reference(batch.input_ids, attention_mask=batch.attention_mask), batch.labels, "token_mean")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_reference.parameters(), 1.0)
    # accumulate_step also compares post-clip gradients; mirror that here.
    reference_grads = {name: parameter.grad.detach().clone() for name, parameter in model_reference.named_parameters()}
    optimizer_reference.step()

    reports = {}
    for mode in ("every", "last"):
        torch.manual_seed(11)
        model = tiny_model()
        optimizer = build_optimizer(model)
        micros, accounting = build_micro_batches(samples, members, rank, world_size, num_micro=2, pad_token_id=0)
        assert accounting["supervised_global"] == supervised_global
        metrics = accumulate_step(model, optimizer, micros, supervised_global, device, 1.0, world_size, rank, sync_mode=mode, collect_telemetry=False)
        reports[mode] = (metrics, model)

        expected_count = 2 if mode == "every" else 1
        assert metrics["allreduce_count"] == expected_count, (mode, metrics["allreduce_count"])
        grad_error = max(
            float((parameter.grad - reference_grads[name]).abs().max())
            for name, parameter in model.named_parameters()
        )
        assert grad_error < TOLERANCE, f"{mode}: gradient error {grad_error}"
        step_error = max(
            float((parameter_new - parameter_reference).abs().max())
            for parameter_new, parameter_reference in zip(model.parameters(), model_reference.parameters())
        )
        assert step_error < TOLERANCE, f"{mode}: one-step parameter error {step_error}"

    metrics_every, model_every = reports["every"]
    metrics_last, model_last = reports["last"]
    grad_gap = max(
        float((parameter_a.grad - parameter_b.grad).abs().max())
        for parameter_a, parameter_b in zip(model_every.parameters(), model_last.parameters())
    )
    assert grad_gap < TOLERANCE, f"modes must agree within tolerance, got {grad_gap}"

    checksum = float(sum(parameter.detach().double().sum() for parameter in model_last.parameters()))
    checksums = [None] * world_size
    dist.all_gather_object(checksums, checksum)
    assert len(set(checksums)) == 1, f"ranks diverged: {checksums}"

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"modes agree; allreduce counts {metrics_every['allreduce_count']:.0f} vs {metrics_last['allreduce_count']:.0f}; "
              f"bytes {metrics_every['allreduce_bytes_total']:.0f} vs {metrics_last['allreduce_bytes_total']:.0f}")
        print("chapter_31 smoke tests passed")


def main() -> None:
    port = free_port()
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
