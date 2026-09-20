"""Chapter 38 copy: Attention Tensor Parallel with GQA head sharding (ch. 21).

This file is the stable chapter-36 implementation, reused unchanged except
for one constructor flag: ``ParallelAttention`` forwards ``reduce_output`` to
its Row-parallel ``o_proj``.  Sequence Parallel needs the o_proj output as
the UNREDUCED rank partial (``reduce_output=False``) so the SP decoder layer
in ``sequence_parallel.py`` can Reduce-Scatter it along the sequence dim; the
default ``True`` keeps the chapter-36 AllReduce behavior.  This chapter's own
variable — whether the activations BETWEEN the parallel regions are also
sequence-sharded — lives in ``sequence_parallel.py``.

Chapter-21 mechanism summary (unchanged here).  For tp_size = N, per
decoder layer:

- ``q_proj`` is COLUMN parallel along the Q-head dim: the full weight
  ``[num_q_heads * head_dim, hidden]`` is split into N contiguous row blocks
  of ``[local_q_heads * head_dim, hidden]``.  Because each head occupies
  exactly ``head_dim`` consecutive rows, a contiguous row block IS a whole
  number of heads — no padding and no gather inside the head.
- ``k_proj`` / ``v_proj`` are COLUMN parallel along the KV-head dim in the
  same way: ``[local_kv_heads * head_dim, hidden]`` row blocks.
- ``o_proj`` is ROW parallel along the Q-head input dim: ``[hidden,
  local_q_heads * head_dim]`` column blocks; the partial outputs are summed
  across ranks with one AllReduce (the "g" op of the attention block).
- ``q_norm`` / ``k_norm`` are per-head RMSNorm over ``head_dim``; ``head_dim``
  is never split, so they stay rank-local with the head they belong to.
- RoPE acts INSIDE each head (a fixed 2D rotation per position), so applying
  it to the local head shard reproduces exactly the reference rotation for
  those heads — "local RoPE" is not an approximation.

GQA CONSISTENCY (the error-prone core of this chapter)
------------------------------------------------------
Qwen3 uses ``repeat_interleave`` grouping: with G = num_q_heads //
num_key_value_heads, Q head h attends through KV head h // G.  Rank r owns
the contiguous Q heads ``[r*Lq, (r+1)*Lq)`` and KV heads ``[r*Lkv,
(r+1)*Lkv)`` with ``Lq = num_q_heads/N`` and ``Lkv = num_kv_heads/N``.  The
Q heads of rank r need exactly the KV heads ``{h // G} = [r*Lq/G,
(r+1)*Lq/G)``; this equals the rank's contiguous KV shard iff ``Lq / G ==
Lkv``, which holds automatically once BOTH head counts divide by N — the
local group count ``Lq // Lkv`` equals the global G and the local
``repeat_kv`` reproduces the global grouping.  Both divisibility constraints
(num_q_heads % N == 0 and num_kv_heads % N == 0) are checked explicitly and
raise a ``shard_error`` carrying shape/tp context; there is no silent
fallback.  ``gqa_head_layout`` returns the per-rank head index lists and
``validate_gqa_layout`` re-derives the mapping; both are recorded into the
correctness JSON so the sharding is auditable, not assumed.

COMMUNICATION per layer (all exposed on the critical path, all reuse the
chapter-35 operators — the same f/g pair per parallel block):

- attention block: one forward AllReduce (o_proj output sum, "g") and one
  backward AllReduce (attention-input gradient sum, "f", issued ONCE for the
  three q/k/v branches at the block entry — three branches consuming the same
  input must not pay for the same sum three times);
- MLP block: the chapter-35 pair (down_proj output sum, MLP-input sum).

GRADIENT OWNERSHIP after this change:

1. Shard weights (q/k/v/o and gate/up/down of every layer): gradients are
   rank-local, NO communication.  Each rank's shard gradient equals the
   matching head/intermediate slice of the full gradient — verified against
   the single-GPU reference slice by ``compare_correctness.py``.
2. Block-input residual-stream gradients: one "f" AllReduce per parallel
   block (two per layer now).  Skipping either one would silently propagate
   a partial gradient along the residual stream.
3. Replicated weights — the LayerNorms OUTSIDE the parallel region, the
   Embedding and the LM Head — keep chapter-35 class-3 semantics: their
   gradients are rank-identical and an AllReduce SUM would be WRONG.
   THE EXCEPTION, and the trap of this chapter: ``q_norm``/``k_norm`` live
   INSIDE the column-parallel region.  Their weights are replicated in
   value, but each rank only accumulates the gradient of its OWN heads, so
   the raw gradients are PARTIAL sums that differ across ranks.  Without a
   backward AllReduce SUM per norm weight, every rank would apply a
   different update to an identical parameter and the copies would silently
   diverge after the first optimizer step.  ``ParallelAttention`` attaches a
   post-accumulate-grad hook that performs exactly this AllReduce (counted
   as backward communication), which restores the reference gradient and
   the class-3 contract for everything downstream of it.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from mlp_tp import (
    DOWN_KEY_SUFFIX,
    GATE_KEY_SUFFIX,
    UP_KEY_SUFFIX,
    ColumnParallelLinear,
    RowParallelLinear,
    _CopyToTensorParallelRegion,
    shard_error,
    shard_weight as shard_mlp_weight,
    validate_shard_key as validate_mlp_key,
)
from qwen3_model import RMSNorm, apply_rotary_embedding, repeat_key_value


Q_KEY_SUFFIX = "self_attn.q_proj.weight"
K_KEY_SUFFIX = "self_attn.k_proj.weight"
V_KEY_SUFFIX = "self_attn.v_proj.weight"
O_KEY_SUFFIX = "self_attn.o_proj.weight"
ATTENTION_KEY_SUFFIXES = (Q_KEY_SUFFIX, K_KEY_SUFFIX, V_KEY_SUFFIX, O_KEY_SUFFIX)


def gqa_head_layout(tp_rank: int, tp_size: int, num_q_heads: int, num_kv_heads: int) -> dict:
    """The head indices this rank owns, with every divisibility made explicit.

    Qwen3-0.6B (32 Q heads, 8 KV heads): tp=2 -> 16 Q + 4 KV per rank,
    tp=4 -> 8 Q + 2 KV.  ``q_heads``/``kv_heads`` are contiguous index lists;
    the corresponding weight rows are ``q_row_begin:q_row_end`` (and the KV
    analog) of the full q/k/v projections.
    """

    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    if num_q_heads % tp_size != 0:
        raise shard_error(
            f"Q head count {num_q_heads} is not divisible by tp_size {tp_size}",
            (num_q_heads,), torch.float32, torch.device("cpu"), tp_rank, tp_size,
        )
    if num_kv_heads % tp_size != 0:
        raise shard_error(
            f"KV head count {num_kv_heads} is not divisible by tp_size {tp_size}",
            (num_kv_heads,), torch.float32, torch.device("cpu"), tp_rank, tp_size,
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"Q head count {num_q_heads} must be divisible by KV head count {num_kv_heads}")
    local_q = num_q_heads // tp_size
    local_kv = num_kv_heads // tp_size
    groups = num_q_heads // num_kv_heads
    if local_q != local_kv * groups:
        # Unreachable while both divisibility checks pass, but the invariant
        # that makes the local repeat_kv correct deserves its own guard.
        raise ValueError(
            f"local GQA group count must stay {groups}: local_q={local_q}, local_kv={local_kv}"
        )
    return {
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_groups": groups,
        "local_q_heads": local_q,
        "local_kv_heads": local_kv,
        "q_heads": list(range(tp_rank * local_q, (tp_rank + 1) * local_q)),
        "kv_heads": list(range(tp_rank * local_kv, (tp_rank + 1) * local_kv)),
    }


def validate_gqa_layout(layout: dict) -> bool:
    """Re-derive the GQA mapping: every owned Q head must group into an owned
    KV head under ``repeat_interleave`` (Q head h -> KV head h // groups)."""

    groups = layout["gqa_groups"]
    owned_kv = set(layout["kv_heads"])
    return all(head // groups in owned_kv for head in layout["q_heads"])


def _attention_split(key: str, num_q_heads: int, num_kv_heads: int, head_dim: int) -> tuple[int, int]:
    """(split_dim, full_dim) of one attention projection weight."""

    if key.endswith(Q_KEY_SUFFIX):
        return 0, num_q_heads * head_dim
    if key.endswith(K_KEY_SUFFIX) or key.endswith(V_KEY_SUFFIX):
        return 0, num_kv_heads * head_dim
    if key.endswith(O_KEY_SUFFIX):
        return 1, num_q_heads * head_dim
    raise KeyError(f"{key!r} is not an attention projection weight")


def validate_attention_shard_key(key: str, weight: torch.Tensor, tp_size: int,
                                 num_q_heads: int, num_kv_heads: int, head_dim: int) -> None:
    """Divisibility check for one attention weight of a full state dict."""

    if not any(key.endswith(suffix) for suffix in ATTENTION_KEY_SUFFIXES):
        return
    split_dim, full_dim = _attention_split(key, num_q_heads, num_kv_heads, head_dim)
    if weight.shape[split_dim] != full_dim:
        raise shard_error(
            f"state dict key {key!r} has shape {tuple(weight.shape)}, expected dim {split_dim} = {full_dim}",
            tuple(weight.shape), weight.dtype, weight.device, -1, tp_size,
        )
    if full_dim % tp_size != 0:
        raise shard_error(
            f"state dict key {key!r} splits along dim {split_dim} of size {full_dim}, "
            f"which is not divisible by tp_size {tp_size}",
            tuple(weight.shape), weight.dtype, weight.device, -1, tp_size,
        )


def shard_attention_weight(key: str, weight: torch.Tensor, tp_rank: int, tp_size: int,
                           num_q_heads: int, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """This rank's shard of one attention weight.

    q/k/v split along dim 0 by contiguous head rows; o splits along dim 1 by
    the matching Q-head columns.  Contiguity is what makes a plain row/column
    slice equal to the per-head blocks.  Any other key (q_norm/k_norm and
    friends) is replicated unchanged, mirroring ``mlp_tp.shard_weight``.
    """

    if not any(key.endswith(suffix) for suffix in ATTENTION_KEY_SUFFIXES):
        return weight
    split_dim, full_dim = _attention_split(key, num_q_heads, num_kv_heads, head_dim)
    local = full_dim // tp_size
    begin = tp_rank * local
    if split_dim == 0:
        return weight[begin : begin + local, :]
    return weight[:, begin : begin + local]


def split_attention_parallel_weights(state_dict: dict, tp_size: int, num_q_heads: int,
                                     num_kv_heads: int, head_dim: int) -> list[dict]:
    """One state dict per TP rank for a standalone (attention-only) module."""

    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    for key, weight in state_dict.items():
        validate_attention_shard_key(key, weight, tp_size, num_q_heads, num_kv_heads, head_dim)
    return [
        {
            key: shard_attention_weight(key, weight, tp_rank, tp_size, num_q_heads, num_kv_heads, head_dim).clone()
            for key, weight in state_dict.items()
        }
        for tp_rank in range(tp_size)
    ]


def split_transformer_parallel_weights(state_dict: dict, tp_size: int, num_q_heads: int,
                                       num_kv_heads: int, head_dim: int) -> list[dict]:
    """One state dict per TP rank of a WHOLE model: attention AND MLP shards.

    Attention keys are validated/sharded here; MLP keys delegate to the
    chapter-35 helpers; every other tensor (norms, embedding, LM head) is
    replicated unchanged into all ranks.
    """

    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    for key, weight in state_dict.items():
        validate_attention_shard_key(key, weight, tp_size, num_q_heads, num_kv_heads, head_dim)
    return [
        {
            key: (
                shard_attention_weight(key, weight, tp_rank, tp_size, num_q_heads, num_kv_heads, head_dim).clone()
                if any(key.endswith(suffix) for suffix in ATTENTION_KEY_SUFFIXES)
                else _shard_mlp_or_replicate(key, weight, tp_rank, tp_size)
            )
            for key, weight in state_dict.items()
        }
        for tp_rank in range(tp_size)
    ]


def _shard_mlp_or_replicate(key: str, weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    validate_mlp_key(key, weight, tp_size)
    return shard_mlp_weight(key, weight, tp_rank, tp_size).clone()


def _concat_shards(key: str, parts: list[torch.Tensor]) -> torch.Tensor:
    if any(key.endswith(suffix) for suffix in (Q_KEY_SUFFIX, K_KEY_SUFFIX, V_KEY_SUFFIX)):
        return torch.cat(parts, dim=0).contiguous()
    if key.endswith(O_KEY_SUFFIX):
        return torch.cat(parts, dim=1).contiguous()
    if key.endswith(GATE_KEY_SUFFIX) or key.endswith(UP_KEY_SUFFIX):
        return torch.cat(parts, dim=0).contiguous()
    if key.endswith(DOWN_KEY_SUFFIX):
        return torch.cat(parts, dim=1).contiguous()
    for rank, part in enumerate(parts[1:], start=1):
        if not torch.equal(part, parts[0]):
            raise ValueError(
                f"replicated key {key!r} differs between rank 0 and rank {rank}; "
                "the TP group does not share identical non-sharded weights"
            )
    return parts[0].clone()


def assemble_transformer_parallel_weights(shards: Sequence[dict], num_q_heads: int,
                                          num_kv_heads: int, head_dim: int) -> dict:
    """Inverse of ``split_transformer_parallel_weights`` (rank order preserved)."""

    if not shards:
        raise ValueError("need at least one shard state dict")
    return {key: _concat_shards(key, [shard[key] for shard in shards]) for key in shards[0]}


class ParallelAttention(nn.Module):
    """GQA attention with Q/KV heads sharded over the TP group.

    Forward per rank: local Q/K/V projections (this rank's heads only), local
    RoPE, local GQA (repeat_kv INSIDE the local head set — the local group
    count equals the global one, see the module docstring), local attention
    over the full sequence, then a Row-parallel o_proj whose partial outputs
    the "g" AllReduce sums into the full attention output.  Mathematically
    identical to the replicated GQA attention up to floating-point reduction
    order.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        tp_rank: int,
        tp_size: int,
        group=None,
        recorder=None,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        self.layout = gqa_head_layout(tp_rank, tp_size, num_attention_heads, num_key_value_heads)
        if num_attention_heads * head_dim % tp_size != 0:  # defensive; implied by head divisibility
            raise shard_error(
                f"Q projection output {num_attention_heads * head_dim} is not divisible by tp_size {tp_size}",
                (num_attention_heads * head_dim, hidden_size), torch.float32, torch.device("cpu"), tp_rank, tp_size,
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.local_q_heads = self.layout["local_q_heads"]
        self.local_kv_heads = self.layout["local_kv_heads"]
        self.local_num_key_value_groups = self.local_q_heads // self.local_kv_heads
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = group
        from mlp_tp import COMM_RECORDER

        self.recorder = recorder if recorder is not None else COMM_RECORDER
        self.q_proj = ColumnParallelLinear(
            hidden_size, num_attention_heads * head_dim, tp_rank, tp_size,
            input_gradient_allreduce=False, group=group, recorder=self.recorder,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_key_value_heads * head_dim, tp_rank, tp_size,
            input_gradient_allreduce=False, group=group, recorder=self.recorder,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_key_value_heads * head_dim, tp_rank, tp_size,
            input_gradient_allreduce=False, group=group, recorder=self.recorder,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size, tp_rank, tp_size,
            # reduce_output=False leaves the rank partial UNREDUCED (chapter 23
            # Sequence Parallel Reduce-Scatters it along the sequence dim);
            # the default True keeps the chapter-36 AllReduce behavior.
            reduce_output=reduce_output, group=group, recorder=self.recorder,
        )
        # Per-head RMSNorm over head_dim: head_dim is never split, so the
        # local norm weights are exactly the reference weights (replicated
        # VALUES, but only this rank's heads consume them — see the
        # gradient-sync hook below).
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)
        # q/k norm weights are replicated in value but each rank accumulates
        # only its own heads' gradient terms: the raw gradients are partial
        # sums that differ across ranks.  One SUM AllReduce per weight inside
        # backward restores the reference gradient; skipping it lets the
        # replicated copies diverge from the FIRST optimizer step on.
        for norm_weight in (self.q_norm.weight, self.k_norm.weight):
            norm_weight.register_post_accumulate_grad_hook(self._sync_norm_gradient)

    def _sync_norm_gradient(self, parameter: torch.Tensor) -> None:
        if self.tp_size > 1 and parameter.grad is not None:
            self.recorder.all_reduce(parameter.grad, self.tp_size, self.group, "backward")

    def forward(self, hidden_states: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor,
                causal_mask: torch.Tensor) -> torch.Tensor:
        # The "f" op, issued ONCE for the three q/k/v branches: its backward
        # AllReduces the attention-block input gradient (ownership class 2).
        hidden_states = _CopyToTensorParallelRegion.apply(
            hidden_states, self.tp_size, self.group, self.recorder
        )
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        query = query.view(batch_size, sequence_length, self.local_q_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)

        key = self.k_proj(hidden_states)
        key = key.view(batch_size, sequence_length, self.local_kv_heads, self.head_dim)
        key = self.k_norm(key).transpose(1, 2)

        value = self.v_proj(hidden_states)
        value = value.view(batch_size, sequence_length, self.local_kv_heads, self.head_dim)
        value = value.transpose(1, 2)

        # RoPE rotates inside each head: the local heads get exactly the
        # reference rotation.  The [batch, 1, seq, head_dim] broadcast works
        # unchanged over any head count.
        query, key = apply_rotary_embedding(query, key, cosine, sine)
        key = repeat_key_value(key, self.local_num_key_value_groups)
        value = repeat_key_value(value, self.local_num_key_value_groups)

        attention_scores = torch.matmul(query, key.transpose(2, 3))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_scores = attention_scores + causal_mask
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_probs = attention_probs.to(query.dtype)
        attention_output = torch.matmul(attention_probs, value)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, sequence_length, -1)
        # Row-parallel o_proj: the "g" AllReduce inside sums the rank partials
        # into the full attention output (ownership class 1 weights, class 3
        # semantics restored for everything downstream).
        return self.o_proj(attention_output)


def _rms_norm_reference(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The RMSNorm definition used by the reference path (float32 statistics)."""

    input_dtype = hidden_states.dtype
    float_states = hidden_states.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def full_attention_reference(
    hidden_states: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    o_w: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    causal_mask: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    """The replicated GQA definition this chapter must reproduce, written next
    to the parallel implementation so the equivalence is one glance away."""

    batch_size, sequence_length, _ = hidden_states.shape
    query = (hidden_states @ q_w.transpose(0, 1)).view(batch_size, sequence_length, num_q_heads, head_dim)
    query = _rms_norm_reference(query, q_norm_w, eps).transpose(1, 2)
    key = (hidden_states @ k_w.transpose(0, 1)).view(batch_size, sequence_length, num_kv_heads, head_dim)
    key = _rms_norm_reference(key, k_norm_w, eps).transpose(1, 2)
    value = (hidden_states @ v_w.transpose(0, 1)).view(batch_size, sequence_length, num_kv_heads, head_dim)
    value = value.transpose(1, 2)
    query, key = apply_rotary_embedding(query, key, cosine, sine)
    groups = num_q_heads // num_kv_heads
    key = repeat_key_value(key, groups)
    value = repeat_key_value(value, groups)
    scores = torch.matmul(query, key.transpose(2, 3)) / (head_dim ** 0.5) + causal_mask
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    context = torch.matmul(probs, value).transpose(1, 2).contiguous()
    context = context.view(batch_size, sequence_length, -1)
    return context @ o_w.transpose(0, 1)
