#!/usr/bin/env python3
"""CPU-only Chapter 36 checks: GQA head shards, gradients, and one update.

Runs a tiny Qwen3 (hidden 32, intermediate 64, vocab 64, 2 layers, 8 Q heads,
4 KV heads, head_dim 8) on CPU gloo groups — a 2-rank round and a 4-rank round:

1. ParallelAttention(tp=1) equals the replicated GQA definition (no-op comm).
2. ParallelAttention(tp=2) forward equals the full attention up to FP32
   reduction order; the "f" op makes the input gradient exact; every q/k/v/o
   shard gradient equals the matching head slice of the full gradient.
3. GQA head mapping: each rank records its Q/KV head index lists, the local
   group count equals the global one (4), and every owned Q head groups into
   an owned KV head; the union over ranks is the full head set.
4. Full tiny model (attention + MLP sharded): forward loss, cross-rank loss
   identity, one AdamW step parity against the replicated reference, and the
   ownership proof — replicated gradients (norms/embedding/LM head) are
   identical across ranks WITHOUT communication, shard gradients are NOT.
5. tp=4 round: same parity checks with 2 Q + 1 KV head per rank.
6. Divisibility violations raise errors that carry shape and tp_size.
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from assistant_loss import IGNORE_INDEX
from attention_tp import (
    K_KEY_SUFFIX,
    O_KEY_SUFFIX,
    Q_KEY_SUFFIX,
    V_KEY_SUFFIX,
    ParallelAttention,
    full_attention_reference,
    gqa_head_layout,
    split_attention_parallel_weights,
    split_transformer_parallel_weights,
    assemble_transformer_parallel_weights,
    validate_gqa_layout,
)
from mlp_tp import (
    ParallelMLP,
    gather_tensor_parallel_parameters,
    split_local_parameters,
    tp_train_step,
)
from mini_batch import PaddedBatch, batch_loss
from qwen3_model import Qwen3Config, Qwen3ForCausalLM, RotaryEmbedding


HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 64
LAYERS = 2
NUM_Q_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 8


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=LAYERS,
        num_attention_heads=NUM_Q_HEADS, num_key_value_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )


def attention_weights(generator: torch.Generator) -> dict:
    return {
        Q_KEY_SUFFIX: torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN, generator=generator) * 0.3,
        K_KEY_SUFFIX: torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN, generator=generator) * 0.3,
        V_KEY_SUFFIX: torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN, generator=generator) * 0.3,
        O_KEY_SUFFIX: torch.randn(HIDDEN, NUM_Q_HEADS * HEAD_DIM, generator=generator) * 0.3,
        "self_attn.q_norm.weight": torch.randn(HEAD_DIM, generator=generator).abs() * 0.5 + 0.5,
        "self_attn.k_norm.weight": torch.randn(HEAD_DIM, generator=generator).abs() * 0.5 + 0.5,
    }


def rotary_and_mask(sequence_length: int, batch_size: int = 1):
    rotary = RotaryEmbedding(HEAD_DIM, 10000.0)
    position_ids = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)
    cosine, sine = rotary(position_ids, torch.float32)
    mask = torch.full((sequence_length, sequence_length), torch.finfo(torch.float32).min)
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    return cosine, sine, mask


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


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}


def check_head_mapping(layout: dict) -> None:
    """This rank's head lists: correct sizes, local grouping, global union."""

    assert validate_gqa_layout(layout), f"rank {layout['tp_rank']} Q/KV mapping broken"
    assert layout["local_q_heads"] == NUM_Q_HEADS // layout["tp_size"]
    assert layout["local_kv_heads"] == NUM_KV_HEADS // layout["tp_size"]
    assert layout["gqa_groups"] == NUM_Q_HEADS // NUM_KV_HEADS


def _round_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_rank, tp_size = rank, world_size
    generator = torch.Generator().manual_seed(5)

    # (3) per-rank GQA head mapping, checked before anything else runs.
    layout = gqa_head_layout(tp_rank, tp_size, NUM_Q_HEADS, NUM_KV_HEADS)
    check_head_mapping(layout)
    all_layouts = [None] * world_size
    dist.all_gather_object(all_layouts, layout)
    if tp_rank == 0:
        union_q = sorted(head for entry in all_layouts for head in entry["q_heads"])
        union_kv = sorted(head for entry in all_layouts for head in entry["kv_heads"])
        assert union_q == list(range(NUM_Q_HEADS)), union_q
        assert union_kv == list(range(NUM_KV_HEADS)), union_kv

    # (1) tp=1 is the replicated baseline: same math, no collective calls.
    weights = attention_weights(generator)
    parallel_tp1 = ParallelAttention(HIDDEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, 1e-6, 0, 1)
    parallel_tp1.load_state_dict(_strip_prefix(weights, "self_attn."))
    x0 = torch.randn(1, 5, HIDDEN, generator=generator)
    cosine0, sine0, mask0 = rotary_and_mask(5)
    reference0 = full_attention_reference(
        x0, weights[Q_KEY_SUFFIX], weights[K_KEY_SUFFIX], weights[V_KEY_SUFFIX], weights[O_KEY_SUFFIX],
        weights["self_attn.q_norm.weight"], weights["self_attn.k_norm.weight"],
        cosine0, sine0, mask0, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, 1e-6,
    )
    torch.testing.assert_close(
        parallel_tp1(x0, cosine0, sine0, mask0), reference0,
        rtol=1e-6, atol=1e-7, msg="tp=1 ParallelAttention must equal the replicated GQA attention",
    )

    # (2) tp=N against the replicated reference, same inputs everywhere.
    shards = split_attention_parallel_weights(weights, tp_size, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
    parallel = ParallelAttention(HIDDEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, 1e-6, tp_rank, tp_size)
    parallel.load_state_dict(_strip_prefix(shards[tp_rank], "self_attn."))

    x = torch.randn(2, 6, HIDDEN, generator=generator)
    upstream = torch.randn(2, 6, HIDDEN, generator=generator)
    cosine, sine, mask = rotary_and_mask(6, batch_size=2)

    x_parallel = x.clone().requires_grad_(True)
    out_parallel = parallel(x_parallel, cosine, sine, mask)
    (out_parallel * upstream).sum().backward()

    refs = {key: value.clone().requires_grad_(True) for key, value in weights.items()}
    x_ref = x.clone().requires_grad_(True)
    out_ref = full_attention_reference(
        x_ref, refs[Q_KEY_SUFFIX], refs[K_KEY_SUFFIX], refs[V_KEY_SUFFIX], refs[O_KEY_SUFFIX],
        refs["self_attn.q_norm.weight"], refs["self_attn.k_norm.weight"],
        cosine, sine, mask, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, 1e-6,
    )
    (out_ref * upstream).sum().backward()

    # Forward: the o_proj AllReduce sums rank partials; FP32 association
    # differs (tp=2 happens to stay bitwise equal, tp=4 does not), hence
    # tolerance instead of bitwise equality.
    torch.testing.assert_close(out_parallel, out_ref, rtol=1e-5, atol=1e-6,
                               msg="parallel attention forward must match the full attention")
    # Input gradient: rank partials are AllReduced by the "f" op.
    torch.testing.assert_close(x_parallel.grad, x_ref.grad, rtol=1e-5, atol=1e-6,
                               msg="attention input gradient must match the full attention")
    # Weight gradients: rank-local head slices of the full gradient.
    local_q_rows = layout["local_q_heads"] * HEAD_DIM
    local_kv_rows = layout["local_kv_heads"] * HEAD_DIM
    q_begin, kv_begin = tp_rank * local_q_rows, tp_rank * local_kv_rows
    # Slice comparisons ARE the GQA-mapping proof: a rank holding the wrong
    # heads would compare against the wrong rows/columns and fail at O(1),
    # while correct shards only differ by allreduce reassociation (~1e-6 at
    # tp=4, which is why atol covers near-zero gradient entries).
    torch.testing.assert_close(
        parallel.q_proj.weight.grad, refs[Q_KEY_SUFFIX].grad[q_begin : q_begin + local_q_rows],
        rtol=1e-4, atol=1e-5, msg="q shard gradient must match the full gradient's Q-head rows")
    torch.testing.assert_close(
        parallel.k_proj.weight.grad, refs[K_KEY_SUFFIX].grad[kv_begin : kv_begin + local_kv_rows],
        rtol=1e-4, atol=1e-5, msg="k shard gradient must match the full gradient's KV-head rows")
    torch.testing.assert_close(
        parallel.v_proj.weight.grad, refs[V_KEY_SUFFIX].grad[kv_begin : kv_begin + local_kv_rows],
        rtol=1e-4, atol=1e-5, msg="v shard gradient must match the full gradient's KV-head rows")
    torch.testing.assert_close(
        parallel.o_proj.weight.grad, refs[O_KEY_SUFFIX].grad[:, q_begin : q_begin + local_q_rows],
        rtol=1e-4, atol=1e-5, msg="o shard gradient must match the full gradient's Q-head columns")

    # Cross-rank identity of the attention OUTPUT (post-AllReduce): every rank
    # must hold the same full attention output.
    gathered_outputs = [None] * world_size
    dist.all_gather_object(gathered_outputs, out_parallel.detach())
    for other_rank, other_output in enumerate(gathered_outputs):
        assert torch.equal(out_parallel.detach(), other_output), \
            f"attention output differs between rank {tp_rank} and rank {other_rank}"

    # (4) ownership + one AdamW step on the tiny model, attention + MLP sharded.
    torch.manual_seed(11)
    full_model = Qwen3ForCausalLM(tiny_config())
    torch.manual_seed(11)
    tp_model = Qwen3ForCausalLM(
        tiny_config(),
        mlp_factory=lambda cfg: ParallelMLP(cfg.hidden_size, cfg.intermediate_size, tp_rank, tp_size),
        attn_factory=lambda cfg: ParallelAttention(
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            cfg.rms_norm_eps, tp_rank, tp_size,
        ),
    )
    tp_model.load_state_dict(split_transformer_parallel_weights(
        full_model.state_dict(), tp_size, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)[tp_rank])

    batch = tiny_batch(generator)

    # (4a) PRE-STEP gradient parity, the actual sharding-semantics check and
    # therefore held to a tight bar: every layer-0 shard gradient (attention
    # AND MLP) must equal its slice of the full-model gradient.  This is done
    # BEFORE any optimizer step, where FP32 noise is still at the raw
    # allreduce-reassociation level (~1e-7 here).
    tp_logits = tp_model(batch.input_ids, attention_mask=batch.attention_mask)
    tp_loss = batch_loss(tp_logits, batch.labels, "token_mean")
    ref_logits = full_model(batch.input_ids, attention_mask=batch.attention_mask)
    ref_loss = batch_loss(ref_logits, batch.labels, "token_mean")
    torch.testing.assert_close(tp_loss, ref_loss, rtol=1e-5, atol=1e-6,
                               msg="TP forward loss must match the replicated reference")
    tp_loss.backward()
    ref_loss.backward()
    named_tp = dict(tp_model.named_parameters())
    named_ref = dict(full_model.named_parameters())
    local_q_rows = layout["local_q_heads"] * HEAD_DIM
    local_kv_rows = layout["local_kv_heads"] * HEAD_DIM
    q_begin, kv_begin = tp_rank * local_q_rows, tp_rank * local_kv_rows
    for proj, ref_grad, shard_grad in (
        ("q_proj", named_ref["model.layers.0.self_attn.q_proj.weight"].grad, named_tp["model.layers.0.self_attn.q_proj.weight"].grad),
        ("k_proj", named_ref["model.layers.0.self_attn.k_proj.weight"].grad, named_tp["model.layers.0.self_attn.k_proj.weight"].grad),
        ("v_proj", named_ref["model.layers.0.self_attn.v_proj.weight"].grad, named_tp["model.layers.0.self_attn.v_proj.weight"].grad),
    ):
        rows = local_q_rows if proj == "q_proj" else local_kv_rows
        begin = q_begin if proj == "q_proj" else kv_begin
        torch.testing.assert_close(shard_grad, ref_grad[begin : begin + rows],
                                   rtol=1e-4, atol=1e-6, msg=f"layer-0 {proj} shard gradient must match the reference slice")
    torch.testing.assert_close(
        named_tp["model.layers.0.self_attn.o_proj.weight"].grad,
        named_ref["model.layers.0.self_attn.o_proj.weight"].grad[:, q_begin : q_begin + local_q_rows],
        rtol=1e-4, atol=1e-6, msg="layer-0 o_proj shard gradient must match the reference columns")
    for proj, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
        tp_grad = named_tp[f"model.layers.0.mlp.{proj}.weight"].grad
        ref_grad = named_ref[f"model.layers.0.mlp.{proj}.weight"].grad
        rows = tp_grad.shape[dim]
        begin = tp_rank * rows
        expected = ref_grad[begin : begin + rows, :] if dim == 0 else ref_grad[:, begin : begin + rows]
        torch.testing.assert_close(tp_grad, expected, rtol=1e-4, atol=1e-6,
                                   msg=f"layer-0 {proj} shard gradient must match the reference slice")

    # (4b) ownership proof: replicated-parameter gradients (norms, embedding,
    # LM head — including the SYNCED q/k norms) are identical across ranks
    # without any extra communication, while shard gradients are NOT.
    replicated, shard = split_local_parameters(tp_model)
    assert len(shard) == LAYERS * 7 and len(replicated) > 0, (len(replicated), len(shard))

    replicated_sums = [None] * world_size
    dist.all_gather_object(replicated_sums, [float(p.grad.double().sum()) for p in replicated])
    for other_rank, other_sums in enumerate(replicated_sums):
        for own, other in zip((float(p.grad.double().sum()) for p in replicated), other_sums):
            assert abs(own - other) <= 1e-9 * max(1.0, abs(own)), \
                f"replicated gradients differ between rank {tp_rank} and rank {other_rank}: {own} vs {other}"
    shard_sums = [None] * world_size
    dist.all_gather_object(shard_sums, [float(p.grad.double().sum()) for p in shard])
    assert len({repr(sums) for sums in shard_sums}) == world_size, \
        "shard gradients are rank-local; identical sums across ranks would mean attention was never split"

    # (4c) one AdamW step: TP group vs replicated reference (plain clip).
    full_model.zero_grad(set_to_none=True)
    tp_model.zero_grad(set_to_none=True)
    optimizer_ref = torch.optim.AdamW(full_model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.1, foreach=False, fused=False)
    optimizer_tp = torch.optim.AdamW(tp_model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                     weight_decay=0.1, foreach=False, fused=False)

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
    # TWO parallel blocks per layer now: attention (o_proj g / input f) and
    # MLP (down_proj g / input f) on the forward side; on the backward side
    # the two input-gradient f-ops PLUS the q_norm/k_norm gradient syncs
    # (replicated value, per-rank partial gradient — the chapter's trap).
    assert metrics["tp_allreduce_forward_count"] == 2 * LAYERS, metrics
    assert metrics["tp_allreduce_backward_count"] == 4 * LAYERS, metrics

    reassembled = gather_tensor_parallel_parameters(tp_model, tp_size)
    for name, parameter in full_model.named_parameters():
        # Post-AdamW comparison uses the same scale-aware metric as
        # compare_correctness (max abs diff over max abs reference): elements
        # with near-zero gradients have m/sqrt(v) dominated by FP32 noise, so
        # elementwise bars are seed-fragile, while ANY wrong shard produces a
        # deviation comparable to the weight scale itself.
        difference = (reassembled[name].double() - parameter.detach().double()).abs().max().item()
        scale = parameter.detach().double().abs().max().item()
        assert difference / max(scale, 1e-9) < 5e-3, \
            (f"reassembled {name} deviates from the replicated reference after one AdamW step: "
             f"abs {difference:.3e} vs scale {scale:.3e}")

    # (6) divisibility violations must fail loudly with shape/tp_size context.
    # tp_rank=0 keeps the range check out of the way so the divisibility
    # branch is what fires.
    expect_error(lambda: gqa_head_layout(0, 3, NUM_Q_HEADS, NUM_KV_HEADS), "tp_size=3", "8")
    expect_error(lambda: gqa_head_layout(0, 3, NUM_Q_HEADS, 3), "tp_size=3")
    expect_error(
        lambda: ParallelAttention(HIDDEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, 1e-6, 0, 3),
        "tp_size=3",
    )
    expect_error(
        lambda: split_transformer_parallel_weights(full_model.state_dict(), 3, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM),
        "tp_size=3",
    )

    # split -> assemble round trip is bitwise exact, replicated keys included.
    reassembled_weights = assemble_transformer_parallel_weights(
        split_transformer_parallel_weights(full_model.state_dict(), tp_size, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM),
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
    )
    for key, weight in full_model.state_dict().items():
        assert torch.equal(reassembled_weights[key], weight), f"round trip changed {key}"

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"tp={tp_size}: attention forward/input-grad/head-slice grads, GQA mapping, "
              f"one-step parity, ownership, divisibility errors: OK")


def main() -> None:
    for world_size in (2, 4):
        port = _free_port()
        mp.spawn(_round_worker, args=(world_size, port), nprocs=world_size, join=True)
    print("chapter_36 smoke tests passed")


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


if __name__ == "__main__":
    main()
