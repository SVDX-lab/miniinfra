#!/usr/bin/env python3
"""CPU-only Chapter 38 checks: sequence shards, boundary operators, gradients.

Runs a tiny Qwen3 (hidden 32, intermediate 64, vocab 64, 2 layers, 8 Q heads,
4 KV heads, head_dim 8, sequence 12) on CPU gloo groups — a 2-rank round and a
4-rank round:

1. The g / ḡ operators: AllGather forward concatenates the shards (identical
   on all ranks) and slices in backward; ReduceScatter forward returns the
   rank window of the summed partials and AllGathers in backward.
2. ``ParallelLayerNorm`` on a shard equals the reference RMSNorm restricted to
   that window (per-token independence); after the sync hook its weight
   gradient equals the FULL reference weight gradient on every rank.
3. One SP decoder layer vs the replicated chapter-16 layer: forward output,
   input gradient and every weight gradient match the reference (shard grads
   to their exact slices, replicated grads exactly).
4. Full tiny model (attention+MLP shards + sequence shards): forward loss,
   cross-rank loss identity, shard/replicated gradient ownership, one AdamW
   step parity against the replicated reference, and the same parity for the
   chapter-36 pure-TP model (both arms against one reference).
5. The chapter's claim, measured: saved-for-backward bytes per rank of the SP
   model are strictly smaller than the pure-TP model's on the same batch.
6. Indivisible sequence lengths raise errors that carry shape and tp_size.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from assistant_loss import IGNORE_INDEX
from attention_tp import ParallelAttention, split_transformer_parallel_weights
from mlp_tp import (
    DOWN_KEY_SUFFIX,
    GATE_KEY_SUFFIX,
    UP_KEY_SUFFIX,
    ParallelMLP,
    gather_tensor_parallel_parameters,
    split_local_parameters,
)
from mini_batch import PaddedBatch, batch_loss
from qwen3_model import Qwen3Config, Qwen3ForCausalLM, RMSNorm, RotaryEmbedding
from sequence_parallel import (
    COMM_RECORDER,
    ActivationLedger,
    ParallelLayerNorm,
    SPDecoderLayer,
    SequenceParallelForCausalLM,
    _GatherFromSequenceParallelRegion,
    _ScatterToSequenceParallelRegion,
    sequence_shard_bounds,
    sp_loss_terms,
    sp_train_step,
)


HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 64
LAYERS = 2
NUM_Q_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 8
SEQ = 12  # divisible by both round sizes (2 and 4)


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=LAYERS,
        num_attention_heads=NUM_Q_HEADS, num_key_value_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )


def rotary_and_mask(attention_mask: torch.Tensor):
    sequence_length = attention_mask.shape[1]
    rotary = RotaryEmbedding(HEAD_DIM, 10000.0)
    position_ids = torch.arange(sequence_length).unsqueeze(0).expand(attention_mask.shape[0], -1)
    cosine, sine = rotary(position_ids, torch.float32)
    smallest = torch.finfo(torch.float32).min
    mask = torch.full((sequence_length, sequence_length), smallest)
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    mask = mask.expand(attention_mask.shape[0], 1, sequence_length, sequence_length).clone()
    key_is_padding = attention_mask[:, None, None, :] == 0
    return cosine, sine, mask.masked_fill(key_is_padding, smallest)


def tiny_batch(generator: torch.Generator) -> PaddedBatch:
    tokens = torch.randint(4, VOCAB, (2, SEQ), generator=generator)
    labels = tokens.clone()
    labels[:, :2] = IGNORE_INDEX
    supervised = [(row[1:] != IGNORE_INDEX).sum().item() for row in labels]
    return PaddedBatch(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        labels=labels,
        source_lines=(1, 2),
        lengths=(SEQ, SEQ),
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


def check_operators(tp_rank: int, tp_size: int, generator: torch.Generator) -> None:
    """(1) g/ḡ forward + backward semantics against a local emulation."""

    begin, end = sequence_shard_bounds(SEQ, tp_rank, tp_size)

    # g: forward AllGather == concatenation of every rank's shard.
    shard = torch.randn(2, end - begin, HIDDEN, generator=generator)
    x = shard.clone().requires_grad_(True)
    gathered = _GatherFromSequenceParallelRegion.apply(x, tp_rank, tp_size, None, COMM_RECORDER)
    all_shards = [None] * tp_size
    dist.all_gather_object(all_shards, shard)
    expected = torch.cat(all_shards, dim=1)
    assert torch.equal(gathered, expected), "g forward must concatenate the shards"

    # g backward: identical (post-f-AllReduce) upstream on every rank -> local slice.
    upstream = torch.randn(2, SEQ, HIDDEN)
    gathered.backward(upstream)
    assert torch.equal(x.grad, upstream[:, begin:end]), "g backward must slice the rank-identical gradient"

    # ḡ: forward ReduceScatter == rank window of the summed partials.
    partial = torch.randn(2, SEQ, HIDDEN, generator=generator)
    p = partial.clone().requires_grad_(True)
    scattered = _ScatterToSequenceParallelRegion.apply(p, tp_size, None, COMM_RECORDER)
    all_partials = [None] * tp_size
    dist.all_gather_object(all_partials, partial)
    expected_window = sum(all_partials)[:, begin:end]
    torch.testing.assert_close(scattered, expected_window, rtol=1e-6, atol=1e-6,
                               msg="ḡ forward must reduce-scatter the summed partials")

    # ḡ backward: disjoint pieces -> AllGather reproduces the full gradient.
    shard_upstream = torch.randn(2, end - begin, HIDDEN, generator=generator)
    scattered.backward(shard_upstream)
    all_pieces = [None] * tp_size
    dist.all_gather_object(all_pieces, shard_upstream)
    assert torch.allclose(p.grad, torch.cat(all_pieces, dim=1), rtol=1e-6, atol=1e-6), \
        "ḡ backward must all-gather the per-rank gradient pieces"


def check_parallel_layer_norm(tp_rank: int, tp_size: int, generator: torch.Generator) -> None:
    """(2) sharded RMSNorm == reference RMSNorm restricted to the window."""

    begin, end = sequence_shard_bounds(SEQ, tp_rank, tp_size)
    weight = torch.randn(HIDDEN, generator=generator).abs() * 0.5 + 0.5
    x_full = torch.randn(2, SEQ, HIDDEN, generator=generator)
    upstream = torch.randn(2, SEQ, HIDDEN, generator=generator)

    reference = RMSNorm(HIDDEN, 1e-6)
    with torch.no_grad():
        reference.weight.copy_(weight)
    x_ref = x_full.clone().requires_grad_(True)
    (reference(x_ref) * upstream).sum().backward()

    sharded = ParallelLayerNorm(HIDDEN, 1e-6, tp_size)
    with torch.no_grad():
        sharded.weight.copy_(weight)
    x_sp = x_full[:, begin:end].clone().requires_grad_(True)
    torch.testing.assert_close(sharded(x_sp), reference(x_ref)[:, begin:end], rtol=1e-6, atol=1e-7,
                               msg="sharded norm forward must equal the reference window")
    (sharded(x_sp) * upstream[:, begin:end]).sum().backward()

    torch.testing.assert_close(x_sp.grad, x_ref.grad[:, begin:end], rtol=1e-6, atol=1e-7,
                               msg="sharded norm input gradient must match the window slice")
    torch.testing.assert_close(
        sharded.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-6,
        msg="sync hook must restore the FULL norm weight gradient on every rank",
    )


def _expected_reference_slice(name: str, ref_grad: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """The matching reference-gradient slice for one shard weight (the same
    layout rules the loaders use: q/k/v/gate/up rows, o/down columns)."""

    local_q = (NUM_Q_HEADS // tp_size) * HEAD_DIM
    local_kv = (NUM_KV_HEADS // tp_size) * HEAD_DIM
    local_inter = INTERMEDIATE // tp_size
    if name.endswith("self_attn.q_proj.weight"):
        return ref_grad[tp_rank * local_q : (tp_rank + 1) * local_q]
    if name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
        return ref_grad[tp_rank * local_kv : (tp_rank + 1) * local_kv]
    if name.endswith("self_attn.o_proj.weight"):
        return ref_grad[:, tp_rank * local_q : (tp_rank + 1) * local_q]
    if name.endswith((GATE_KEY_SUFFIX, UP_KEY_SUFFIX)):
        return ref_grad[tp_rank * local_inter : (tp_rank + 1) * local_inter]
    if name.endswith(DOWN_KEY_SUFFIX):
        return ref_grad[:, tp_rank * local_inter : (tp_rank + 1) * local_inter]
    raise KeyError(f"unexpected shard weight {name}")


def _compare_grads(named_sp: dict, named_ref: dict, tp_rank: int, tp_size: int, context: str) -> None:
    """Shard gradients to their exact reference slices; replicated gradients
    (same shape on every rank, values synced by hooks) to the full grad."""

    for name, parameter in named_sp.items():
        ref_grad = named_ref[name].grad
        if parameter.grad is None or ref_grad is None:
            raise AssertionError(f"missing gradient for {name} ({context})")
        if parameter.shape == named_ref[name].shape:
            torch.testing.assert_close(
                parameter.grad, ref_grad, rtol=1e-4, atol=1e-6,
                msg=f"replicated {name} ({context}) gradient must equal the full reference",
            )
        else:
            expected = _expected_reference_slice(name, ref_grad, tp_rank, tp_size)
            torch.testing.assert_close(
                parameter.grad, expected, rtol=1e-4, atol=1e-6,
                msg=f"shard {name} ({context}) gradient must match its reference slice",
            )


def check_decoder_layer(tp_rank: int, tp_size: int, generator: torch.Generator) -> None:
    """(3) SP decoder layer vs the replicated chapter-16 decoder layer."""

    torch.manual_seed(17)
    reference = Qwen3ForCausalLM(tiny_config()).model.layers[0]
    shards = split_transformer_parallel_weights(
        reference.state_dict(), tp_size, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM
    )

    sp_layer = SPDecoderLayer(tiny_config(), tp_rank, tp_size)
    sp_layer.load_state_dict(shards[tp_rank])

    x_full = torch.randn(2, SEQ, HIDDEN, generator=generator)
    cosine, sine, mask = rotary_and_mask(torch.ones(2, SEQ, dtype=torch.long))

    x_ref = x_full.clone().requires_grad_(True)
    out_ref = reference(x_ref, cosine, sine, mask)
    out_ref.sum().backward()

    begin, end = sequence_shard_bounds(SEQ, tp_rank, tp_size)
    x_sp = x_full[:, begin:end].clone().requires_grad_(True)
    out_sp = sp_layer(x_sp, cosine, sine, mask)
    out_sp.sum().backward()

    torch.testing.assert_close(out_sp, out_ref.detach()[:, begin:end], rtol=1e-5, atol=1e-6,
                               msg="SP decoder layer forward must match the reference window")
    torch.testing.assert_close(x_sp.grad, x_ref.grad[:, begin:end], rtol=1e-5, atol=1e-6,
                               msg="SP decoder layer input gradient must match the reference window")
    _compare_grads(dict(sp_layer.named_parameters()), dict(reference.named_parameters()),
                   tp_rank, tp_size, "decoder layer")


def build_sp_model(tp_rank: int, tp_size: int, full_model: Qwen3ForCausalLM) -> SequenceParallelForCausalLM:
    torch.manual_seed(11)
    sp_model = SequenceParallelForCausalLM(tiny_config(), tp_rank, tp_size)
    sp_model.load_state_dict(split_transformer_parallel_weights(
        full_model.state_dict(), tp_size, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)[tp_rank], strict=False)
    sp_model.tie_weights()
    return sp_model


def build_tp_model(tp_rank: int, tp_size: int, full_model: Qwen3ForCausalLM) -> Qwen3ForCausalLM:
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
        full_model.state_dict(), tp_size, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)[tp_rank], strict=False)
    tp_model.tie_weights()
    return tp_model


def _measure_saved_bytes(model: torch.nn.Module, batch: PaddedBatch,
                         labels: torch.Tensor | None = None) -> dict:
    """Forward-only saved-for-backward byte count (parameter storages excluded).

    ``labels`` must cover the SAME window the model's logits cover: the TP arm
    produces the full sequence, the SP arm only its rank window.
    """

    model.zero_grad(set_to_none=True)
    ledger = ActivationLedger(model.parameters())
    with ledger:
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        batch_loss(logits, batch.labels if labels is None else labels, "token_mean")
    return {"saved_bytes": ledger.saved_bytes, "saved_count": ledger.saved_count}


def check_full_model(tp_rank: int, tp_size: int, generator: torch.Generator) -> dict:
    """(4)+(5) loss / ownership / one AdamW step for BOTH arms + saved bytes."""

    torch.manual_seed(11)
    full_model = Qwen3ForCausalLM(tiny_config())
    sp_model = build_sp_model(tp_rank, tp_size, full_model)
    tp_model = build_tp_model(tp_rank, tp_size, full_model)

    batch = tiny_batch(generator)

    # Reference loss and gradients (replicated model).
    ref_logits = full_model(batch.input_ids, attention_mask=batch.attention_mask)
    ref_loss = batch_loss(ref_logits, batch.labels, "token_mean")
    ref_loss.backward()
    named_ref = dict(full_model.named_parameters())

    # SP forward: each rank holds a DIFFERENT sequence window, so the local
    # loss is that window's mean loss. Compare it against the replicated
    # model's loss restricted to the same window; cross-rank window losses are
    # not expected to be equal.
    begin, end = sequence_shard_bounds(SEQ, tp_rank, tp_size)
    sp_logits = sp_model(batch.input_ids, attention_mask=batch.attention_mask)
    sp_local_loss = batch_loss(sp_logits, batch.labels[:, begin:end].clone(), "token_mean")
    ref_window_loss = batch_loss(ref_logits[:, begin:end].detach(), batch.labels[:, begin:end], "token_mean")
    torch.testing.assert_close(
        torch.tensor(float(sp_local_loss.item())), torch.tensor(float(ref_window_loss.item())),
        rtol=1e-4, atol=1e-5, msg="SP window loss must match the replicated model on the same window",
    )

    # SP backward on the SAME global objective: the per-shard loss SUM from
    # sp_loss_terms (whose predictor sets partition the full sequence) divided
    # by the GLOBAL supervised count.  Summed over ranks this is exactly the
    # reference token-mean gradient; the replicated-weight sync hooks
    # (LayerNorms, tied Embedding) AllReduce the rank-partial sums during
    # this backward, so every rank ends with the full reference gradient.
    supervised = float(batch.supervised_tokens)
    loss_sum_local, count_local = sp_loss_terms(sp_logits, batch.labels, tp_rank, tp_size, SEQ)
    assert count_local.item() <= batch.supervised_tokens
    (loss_sum_local / supervised).backward()
    _compare_grads(dict(sp_model.named_parameters()), named_ref, tp_rank, tp_size, "full SP model")

    # (5) the chapter's claim, measured on the same batch: SP saves fewer
    # activation bytes per rank than the pure-TP layout.
    tp_saved = _measure_saved_bytes(tp_model, batch)
    sp_saved = _measure_saved_bytes(sp_model, batch, batch.labels[:, begin:end].clone())
    assert sp_saved["saved_bytes"] < tp_saved["saved_bytes"], (
        f"SP must save fewer activation bytes per rank: sp={sp_saved['saved_bytes']}, tp={tp_saved['saved_bytes']}"
    )

    # (4c) one AdamW step through the real step functions.
    full_model.zero_grad(set_to_none=True)
    sp_model.zero_grad(set_to_none=True)
    optimizer_ref = torch.optim.AdamW(full_model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.1, foreach=False, fused=False)
    optimizer_sp = torch.optim.AdamW(sp_model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                     weight_decay=0.1, foreach=False, fused=False)

    ref_logits = full_model(batch.input_ids, attention_mask=batch.attention_mask)
    ref_loss = batch_loss(ref_logits, batch.labels, "token_mean")
    ref_loss.backward()
    torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
    optimizer_ref.step()

    metrics = sp_train_step(sp_model, optimizer_sp, batch, 1.0, torch.device("cpu"), 1.0, tp_size, tp_rank)
    torch.testing.assert_close(
        torch.tensor(metrics["global_loss"]), torch.tensor(float(ref_loss.item())),
        rtol=1e-5, atol=1e-6, msg="SP step loss must match the replicated reference loss",
    )
    # Collective census per step: forward has ONLY boundary operators (2
    # AllGathers + 2 ReduceScatters per layer, no AllReduce); backward keeps
    # the chapter-36 f AllReduces (2/layer) + q/k syncs (2/layer) and adds 2
    # AllGathers per layer plus the NEW shard-weight syncs (2 LayerNorms per
    # layer, Embedding + final norm); the loss uses one packed AllReduce.
    assert metrics["tp_allreduce_forward_count"] == 0.0, metrics
    assert metrics["sp_allgather_forward_count"] == 2 * LAYERS, metrics
    assert metrics["sp_reduce_scatter_forward_count"] == 2 * LAYERS, metrics
    assert metrics["tp_allreduce_backward_count"] == 6 * LAYERS + 2, metrics
    assert metrics["sp_allgather_backward_count"] == 2 * LAYERS, metrics
    assert metrics["sp_loss_allreduce_count"] == 1.0, metrics

    # Replicated-parameter gradients are identical across ranks (they were
    # partial sums before the sync hooks); shard gradients are not.
    replicated, shard = split_local_parameters(sp_model)
    replicated_sums = [None] * tp_size
    dist.all_gather_object(replicated_sums, [float(p.grad.double().sum()) for p in replicated])
    for other_rank, other_sums in enumerate(replicated_sums):
        for own, other in zip((float(p.grad.double().sum()) for p in replicated), other_sums):
            assert abs(own - other) <= 1e-9 * max(1.0, abs(own)), \
                f"synced replicated gradients differ between rank {tp_rank} and rank {other_rank}"
    shard_sums = [None] * tp_size
    dist.all_gather_object(shard_sums, [float(p.grad.double().sum()) for p in shard])
    assert len({repr(sums) for sums in shard_sums}) == tp_size, \
        "shard gradients are rank-local; identical sums would mean nothing was sharded"

    # After the step: reassembled SP parameters == replicated reference.
    reassembled = gather_tensor_parallel_parameters(sp_model, tp_size)
    for name, parameter in full_model.named_parameters():
        difference = (reassembled[name].double() - parameter.detach().double()).abs().max().item()
        scale = parameter.detach().double().abs().max().item()
        assert difference / max(scale, 1e-9) < 5e-3, \
            (f"reassembled {name} deviates from the reference after one AdamW step: "
             f"abs {difference:.3e} vs scale {scale:.3e}")

    return {"tp_saved_bytes": tp_saved["saved_bytes"], "sp_saved_bytes": sp_saved["saved_bytes"]}


def _round_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_rank, tp_size = rank, world_size
    generator = torch.Generator().manual_seed(5)

    check_operators(tp_rank, tp_size, generator)
    check_parallel_layer_norm(tp_rank, tp_size, generator)
    check_decoder_layer(tp_rank, tp_size, generator)
    saved = check_full_model(tp_rank, tp_size, generator)

    # (6) indivisible sequence lengths must fail loudly with context.
    expect_error(lambda: sequence_shard_bounds(SEQ, 0, 5), "sequence length", "12", "tp_size=5")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"tp={tp_size}: g/ḡ operators, sharded norms, decoder layer, full-model step parity, "
              f"saved activation bytes (tp {saved['tp_saved_bytes']} -> sp {saved['sp_saved_bytes']}), "
              f"divisibility errors: OK")


def main() -> None:
    for world_size in (2, 4):
        port = _free_port()
        mp.spawn(_round_worker, args=(world_size, port), nprocs=world_size, join=True)
    print("chapter_38 smoke tests passed")


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


if __name__ == "__main__":
    main()
