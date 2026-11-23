#!/usr/bin/env python3
"""CPU-only Chapter 37 checks: vocab shards, hand-computed CE, one update.

Runs a tiny Qwen3 (hidden 32, intermediate 64, 2 layers, 8 Q heads, 4 KV
heads, head_dim 8, vocab 64) on a 2-rank CPU gloo group:

1. HAND-COMPUTED cross entropy on a 4-word vocab (2 words per rank): the
   distributed loss and the sharded logit gradients must equal the
   closed-form values derived in ``_hand_ce_case`` and computed with ``math``
   only — for a target owned by rank 1, a target owned by rank 0, and an
   IGNORE_INDEX position (zero loss, zero gradient).
2. VocabParallelEmbedding(tp=2) forward equals the full embedding lookup
   through the AllReduce; the weight gradient is exactly the rank's row
   slice, so tokens owned by the OTHER rank contribute nothing locally.
3. Full tiny model (vocab + attention + MLP sharded): the model returns
   SHARDED logits [batch, seq, 32]; embedding/LM head are ONE tied Parameter
   per rank; forward loss matches the replicated reference; pre-step shard
   gradients (sharded logits, embedding rows, attention heads, MLP) match
   the reference slices; one AdamW step reproduces the reference trajectory.
4. Ownership: only the LayerNorms (+ the synced q/k norms) are replicated —
   their gradients are identical across ranks without communication; every
   shard class (vocab/attention/MLP) has rank-local gradients that DIFFER.
5. Communication: the vocab arm's forward adds the embedding "g" op and 3
   vocab-CE collectives (one MAX, two SUM) per STEP to the per-layer block
   pairs; the backward adds the LM head's input-gradient AllReduce (its
   dlogits_r @ W_r is a partial sum) and NOTHING for the CE itself.
6. Logits memory accounting: local_logits_bytes = full/tp_size.
7. Divisibility violations raise errors that carry shape and tp_size.
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from assistant_loss import IGNORE_INDEX
from attention_tp import ParallelAttention
from mlp_tp import ParallelMLP
from mini_batch import PaddedBatch, batch_loss
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from vocab_tp import (
    VocabParallelEmbedding,
    VocabParallelLMHead,
    _VocabParallelCrossEntropy,
    batch_loss_vocab_parallel,
    gather_all_tensor_parallel_parameters,
    split_all_parameters,
    tp_train_step,
)


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


def _hand_ce_case(local_row: torch.Tensor, rank: int) -> None:
    """Hand-computed two-rank cross entropy for ONE row of logits.

    Full logits (hand-picked): ``l = [2.0, 1.0, 0.5, -1.0]``; rank 0 owns
    words {0, 1} (its shard ``[2.0, 1.0]``), rank 1 owns {2, 3} (shard
    ``[0.5, -1.0]``).  With global max ``m = 2.0``:

        Z    = exp(0) + exp(-1) + exp(-1.5) + exp(-3) = 1.6407966696877361
        loss = log(Z) + m - l_t          (t: the target word's logit)

        softmax = [1, e^-1, e^-1.5, e^-3] / Z
        dloss/dl_j = softmax_j - 1{j == t}

    so for target t = 2 (owned by rank 1): rank 0's gradient is
    ``[softmax_0, softmax_1]`` and rank 1's is ``[softmax_2 - 1, softmax_3]``;
    for t = 0 the roles swap.  IGNORE_INDEX rows contribute zero loss and
    zero gradient.  Every expected value below comes from ``math`` on these
    closed forms — torch only runs the distributed computation under test.
    """

    full_logits = [2.0, 1.0, 0.5, -1.0]
    max_value = 2.0
    z = sum(math.exp(value - max_value) for value in full_logits)
    softmax = [math.exp(value - max_value) / z for value in full_logits]

    begin = rank * 2

    def expected_gradient_row(label: int) -> list[float]:
        if label == IGNORE_INDEX:
            return [0.0, 0.0]  # ignored positions: zero loss AND zero gradient
        return [softmax[begin] - (1.0 if label == begin else 0.0),
                softmax[begin + 1] - (1.0 if label == begin + 1 else 0.0)]

    cases = [
        # (labels, expected per-row losses, expected per-row gradient rows)
        ([2, 2], [math.log(z) + max_value - 0.5] * 2, [expected_gradient_row(2)] * 2),
        ([0, IGNORE_INDEX], [math.log(z) + max_value - 2.0, 0.0],
         [expected_gradient_row(0), expected_gradient_row(IGNORE_INDEX)]),
    ]
    for labels, expected_losses, expected_grads in cases:
        label_tensor = torch.tensor(labels, dtype=torch.long)
        logits = local_row.repeat(len(labels), 1).clone().requires_grad_(True)
        losses = _VocabParallelCrossEntropy.apply(logits, label_tensor, begin, begin + 2, 2, None, _recorder())
        expected = torch.tensor(expected_losses, dtype=torch.float32)
        torch.testing.assert_close(losses, expected, rtol=1e-6, atol=1e-7,
                                   msg=f"rank {rank}: distributed CE must equal the hand computation")
        losses.sum().backward()
        torch.testing.assert_close(
            logits.grad, torch.tensor(expected_grads, dtype=torch.float32), rtol=1e-6, atol=1e-7,
            msg=f"rank {rank}: CE gradients must equal softmax - onehot (hand computed)")


def _recorder():
    from mlp_tp import COMM_RECORDER

    return COMM_RECORDER


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_rank, tp_size = rank, world_size
    generator = torch.Generator().manual_seed(5)

    # (1) hand-computed CE on a 4-word vocab (2 per rank).
    local_row = torch.tensor([[2.0, 1.0], [0.5, -1.0]])[tp_rank].unsqueeze(0)
    _hand_ce_case(local_row, tp_rank)

    # (2) vocab-parallel embedding lookup and its rank-local weight gradient.
    torch.manual_seed(7)
    full_weight = torch.randn(VOCAB, HIDDEN) * 0.3
    ids = torch.randint(0, VOCAB, (3, 6), generator=generator)
    embedding = VocabParallelEmbedding(VOCAB, HIDDEN, tp_rank, tp_size)
    with torch.no_grad():
        embedding.weight.copy_(full_weight[embedding.vocab_begin : embedding.vocab_end])
    lookup = embedding(ids)
    lookup.sum().backward()
    reference_weight = full_weight.clone().requires_grad_(True)
    F.embedding(ids, reference_weight).sum().backward()
    begin, end = embedding.vocab_begin, embedding.vocab_end
    torch.testing.assert_close(
        lookup, F.embedding(ids, full_weight), rtol=1e-6, atol=1e-7,
        msg="vocab-parallel embedding lookup must equal the full lookup through the AllReduce")
    torch.testing.assert_close(
        embedding.weight.grad, reference_weight.grad[begin:end],
        rtol=1e-5, atol=1e-7,
        msg="embedding gradient must be exactly this rank's row slice")
    embedding.zero_grad(set_to_none=True)
    reference_weight.grad = None

    # (3) full tiny model: vocab + attention + MLP sharded, tied head.
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
        embed_factory=lambda cfg: VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size, tp_rank, tp_size),
        lm_head_factory=lambda cfg: VocabParallelLMHead(cfg.vocab_size, cfg.hidden_size, tp_rank, tp_size),
    )
    assert tp_model.lm_head.weight is tp_model.model.embed_tokens.weight, \
        "the vocab shard must be ONE tied Parameter per rank"
    from attention_tp import split_transformer_parallel_weights

    full_sd = full_model.state_dict()
    tp_sd = split_transformer_parallel_weights(full_sd, tp_size, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)[tp_rank]
    local_vocab = tp_model.lm_head.local_vocab
    vocab_begin = tp_rank * local_vocab
    # The generic splitter replicates the embedding; replace it with this
    # rank's rows BEFORE loading so the tied Parameter gets the right shard.
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        tp_sd[key] = full_sd[key][vocab_begin : vocab_begin + local_vocab].clone()
    tp_model.load_state_dict(tp_sd)

    batch = tiny_batch(generator)

    # Sharded logits, matching losses, and pre-step gradient parity (tight).
    captured_logits_grad = {}
    def head_backward_hook(module, grad_input, grad_output):
        if "grad" not in captured_logits_grad:
            captured_logits_grad["grad"] = grad_output[0].detach().clone()

    handle = tp_model.lm_head.register_full_backward_hook(head_backward_hook)
    tp_logits = tp_model(batch.input_ids, attention_mask=batch.attention_mask)
    assert tp_logits.shape == (2, 12, local_vocab), tp_logits.shape
    tp_loss = batch_loss_vocab_parallel(tp_logits, batch.labels, vocab_begin, tp_size)
    ref_logits = full_model(batch.input_ids, attention_mask=batch.attention_mask)
    ref_logits.retain_grad()  # non-leaf: keep the output gradient for the slice check
    ref_loss = batch_loss(ref_logits, batch.labels, "token_mean")
    torch.testing.assert_close(tp_loss, ref_loss, rtol=1e-5, atol=1e-6,
                               msg="vocab-parallel CE loss must match the replicated full-logits loss")
    tp_loss.backward()
    ref_loss.backward()
    handle.remove()

    # Sharded-logits gradient vs the reference slice: the numeric proof that
    # the MAX/SUM/target collectives and the backward formula are exact.
    torch.testing.assert_close(
        captured_logits_grad["grad"], ref_logits.grad[:, :, vocab_begin : vocab_begin + local_vocab],
        rtol=1e-4, atol=1e-6,
        msg="sharded logits gradient must equal the full logits gradient slice")
    named_tp = dict(tp_model.named_parameters())
    named_ref = dict(full_model.named_parameters())
    torch.testing.assert_close(
        named_tp["model.embed_tokens.weight"].grad,
        named_ref["model.embed_tokens.weight"].grad[vocab_begin : vocab_begin + local_vocab],
        rtol=1e-4, atol=1e-6,
        msg="tied embedding gradient must accumulate lookup AND head paths into the correct slice")
    local_q_rows = (NUM_Q_HEADS // tp_size) * HEAD_DIM
    local_kv_rows = (NUM_KV_HEADS // tp_size) * HEAD_DIM
    q_begin = tp_rank * local_q_rows
    kv_begin = tp_rank * local_kv_rows
    for suffix, ref_grad, shard_grad in (
        ("q_proj", named_ref["model.layers.0.self_attn.q_proj.weight"].grad, named_tp["model.layers.0.self_attn.q_proj.weight"].grad),
        ("k_proj", named_ref["model.layers.0.self_attn.k_proj.weight"].grad, named_tp["model.layers.0.self_attn.k_proj.weight"].grad),
    ):
        rows = local_q_rows if suffix == "q_proj" else local_kv_rows
        offset = q_begin if suffix == "q_proj" else kv_begin
        torch.testing.assert_close(shard_grad, ref_grad[offset : offset + rows, :],
                                   rtol=1e-4, atol=1e-6, msg=f"layer-0 {suffix} shard gradient mismatch")
    for proj, dim in (("gate_proj", 0), ("down_proj", 1)):
        tp_grad = named_tp[f"model.layers.0.mlp.{proj}.weight"].grad
        ref_grad = named_ref[f"model.layers.0.mlp.{proj}.weight"].grad
        rows = tp_grad.shape[dim]
        offset = tp_rank * rows
        expected = ref_grad[offset : offset + rows, :] if dim == 0 else ref_grad[:, offset : offset + rows]
        torch.testing.assert_close(tp_grad, expected, rtol=1e-4, atol=1e-6,
                                   msg=f"layer-0 {proj} shard gradient mismatch")

    # (4) ownership: replicated = LayerNorms only (+ synced q/k norms).
    replicated, shard = split_all_parameters(tp_model)
    assert len(shard) == LAYERS * 7 + 1, (len(replicated), len(shard))  # attn 4 + mlp 3 per layer, + tied vocab
    replicated_sums = [None] * world_size
    dist.all_gather_object(replicated_sums, [float(p.grad.double().sum()) for p in replicated])
    for other_rank, other_sums in enumerate(replicated_sums):
        for own, other in zip((float(p.grad.double().sum()) for p in replicated), other_sums):
            assert abs(own - other) <= 1e-9 * max(1.0, abs(own)), \
                f"replicated gradients differ between rank {tp_rank} and rank {other_rank}: {own} vs {other}"
    shard_sums = [None] * world_size
    dist.all_gather_object(shard_sums, [float(p.grad.double().sum()) for p in shard])
    assert len({repr(sums) for sums in shard_sums}) == world_size, \
        "shard gradients are rank-local; identical sums would mean nothing was split"

    # (3c) one AdamW step through the real step function.
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

    metrics = tp_train_step(tp_model, optimizer_tp, batch, 1.0, torch.device("cpu"), 1.0, tp_size, tp_rank,
                            collect_telemetry=False, vocab_parallel=True)
    torch.testing.assert_close(
        torch.tensor(metrics["global_loss"]), torch.tensor(float(ref_loss.item())),
        rtol=1e-5, atol=1e-7, msg="TP step loss must match the replicated reference loss")
    # forward: attention g + mlp g per layer, embedding g once, and the three
    # vocab-CE collectives (MAX + SUM + target SUM) once per step.
    # backward: attention f + mlp f + q/k norm syncs per layer, the LM head's
    # input-gradient f op once per step — and NOTHING for the CE itself.
    assert metrics["tp_allreduce_forward_count"] == 2 * LAYERS + 1 + 3, metrics
    assert metrics["tp_allreduce_backward_count"] == 4 * LAYERS + 1, metrics
    # (6) logits memory accounting: the whole point of the vocab split.
    assert metrics["local_logits_bytes"] * tp_size == metrics["full_logits_bytes"], metrics

    reassembled = gather_all_tensor_parallel_parameters(tp_model, tp_size)
    for name, parameter in full_model.named_parameters():
        difference = (reassembled[name].double() - parameter.detach().double()).abs().max().item()
        scale = parameter.detach().double().abs().max().item()
        assert difference / max(scale, 1e-9) < 5e-3, \
            (f"reassembled {name} deviates from the replicated reference after one AdamW step: "
             f"abs {difference:.3e} vs scale {scale:.3e}")

    # (7) divisibility violations must fail loudly with shape/tp_size context.
    expect_error(lambda: VocabParallelEmbedding(VOCAB, HIDDEN, 0, 3), "tp_size=3", "full_weight_shape")
    expect_error(lambda: VocabParallelLMHead(VOCAB, HIDDEN, 0, 3), "tp_size=3", "full_weight_shape")
    expect_error(lambda: VocabParallelEmbedding(VOCAB, HIDDEN, 5, 4), "tp_rank must be in [0, 4)", "got 5")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("vocab shards, hand-computed CE, tied head, one-step parity, ownership, "
              "logits memory accounting, divisibility errors: OK")


def main() -> None:
    port = _free_port()
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)
    print("chapter_37 smoke tests passed")


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


if __name__ == "__main__":
    main()
