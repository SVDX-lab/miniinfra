"""Chapter 38 core: Sequence Parallel over the non-TP regions (Megatron style).

The only variable of this chapter: on top of the chapter-36 attention+MLP TP
layout, are the activations BETWEEN the parallel regions (LayerNorm, residual
adds — Qwen3 has no dropout in the decoder) also sharded along the SEQUENCE
dim?  With tp_size = N every rank then holds [batch, seq/N, hidden] residual
stream states and normalizes only its own sequence shard, while the TP
regions keep their head/intermediate sharding and still run over the FULL
sequence.

Region layout of one decoder layer (b = batch, s = sequence, h = hidden):

    x_r [b, s/N, h]                      sequence shard, rank-resident
    normed_r = RMSNorm(x_r)              per-token: exact on the shard
    normed   = AllGather_seq(normed_r)   "g" op, enters the TP region
    attn_r   = o_proj_partial(...)       col-parallel q/k/v, row-parallel o,
                                         NO AllReduce inside (reduce_output=False)
    out_r    = ReduceScatter_seq(attn_r) "ḡ" op, leaves the TP region
    x_r      = x_r + out_r               residual add, still shard-local

The MLP block repeats the same g/ḡ pair.  Weight layouts are UNCHANGED from
chapter 21: this chapter only re-arranges activations and swaps two
collective operators per parallel block.

WHY THE NORMS ARE SAFE TO SHARD: RMSNorm is strictly per-token (the mean is
over the hidden dim only), and the residual add is elementwise per token, so
computing them on a sequence shard equals computing them on the full sequence
and slicing — no cross-token dependency anywhere outside the TP regions.
Attention is the only sequence-coupled op, and it runs on the gathered FULL
sequence inside the TP region.

COMMUNICATION per parallel block (all exposed on the critical path):

- forward:  one AllGather along seq (g) + one ReduceScatter along seq (ḡ)
            replacing the chapter-36 output AllReduce;
- backward: the chapter-36 "f" op still AllReduces the block-input gradient
            (the gathered activations feed all local heads, so each rank's
            gradient is partial), then the g backward SLICES the now
            rank-identical full gradient locally — no communication.

DEVIATION FROM MEGATRON, kept on purpose: Megatron fuses the "f" AllReduce
and the g backward into a single ReduceScatter (RS(AR(x)) == RS(x) when the
per-rank gradients are partial sums), saving roughly half of the backward
bytes.  This chapter keeps the chapter-36 f operator unchanged and slices
locally instead: the module copies stay byte-compatible and each operator is
readable on its own.  The fusion is recorded as an out-of-scope optimization.

GRADIENT OWNERSHIP (the chapter's trap) — Sequence Parallel BREAKS the
chapter-35/21 class-3 premise for the non-TP weights:

1. Shard weights (q/k/v/o, gate/up/down): rank-local gradients, NO
   communication — unchanged from chapter 21.
2. Block-input residual-stream gradients: still one "f" AllReduce per block.
3. The q/k per-head norm weights inside ParallelAttention: still partial per
   rank (own heads only) and still synced by the chapter-36 hook.
4. NEW: every weight OUTSIDE the TP regions now sees only its own sequence
   shard — input/post_attention LayerNorms, the final norm, and the tied
   Embedding/LM Head all accumulate rank-PARTIAL gradients.  Left alone, the
   replicated copies would diverge from the first optimizer step.
   ``ParallelLayerNorm`` and ``register_replicated_gradient_sync`` attach one
   post-accumulate-grad SUM AllReduce per weight, which restores the exact
   reference gradient (partial_r summed over ranks) and the class-3 contract
   for everything downstream.

LOSS: each rank scores exactly the predictors it owns — every position of
its own sequence window.  The window's last predictor reads its target from
the replicated labels one step BEYOND the window, so the per-shard loss SUMs
and supervised-token COUNTs partition the global ones (no boundary term
dropped, none counted twice).  The pair is AllReduced once (two scalars) and
the backward loss is ``sum_local / count_global``, so the gradient is exactly
the global token-mean gradient.  Full logits are never materialized beyond
the local shard.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from attention_tp import (
    K_KEY_SUFFIX,
    O_KEY_SUFFIX,
    Q_KEY_SUFFIX,
    V_KEY_SUFFIX,
    ParallelAttention,
    gqa_head_layout,
)
from mlp_tp import (
    DOWN_KEY_SUFFIX,
    GATE_KEY_SUFFIX,
    UP_KEY_SUFFIX,
    ParallelMLP,
    TPCommRecorder,
    clip_grad_norm_tp,
    shard_error,
)
from mini_batch import PaddedBatch
from qwen3_model import (
    Qwen3Config,
    RotaryEmbedding,
)
from assistant_loss import IGNORE_INDEX


def sequence_shard_bounds(sequence_length: int, tp_rank: int, tp_size: int) -> tuple[int, int]:
    """This rank's [begin, end) window of the sequence dim; fails loudly on
    indivisible sequence lengths (a silent ragged shard would corrupt every
    slice-based comparison)."""

    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    if sequence_length % tp_size != 0:
        raise shard_error(
            f"sequence length {sequence_length} is not divisible by tp_size {tp_size}",
            (sequence_length,),
            torch.float32,
            torch.device("cpu"),
            tp_rank,
            tp_size,
        )
    local = sequence_length // tp_size
    return tp_rank * local, (tp_rank + 1) * local


def ring_bytes(numel: int, element_size: int, tp_size: int, op: str) -> dict[str, float]:
    """Ring-model per-rank wire bytes for one collective (accounting only).

    These are the analytic per-rank send/receive volumes of the NCCL ring
    algorithms — a ledger for comparing arms and operators, NOT a measurement.
    AllReduce moves 2(N-1)/N of the tensor out and (N-1)/N in; AllGather and
    ReduceScatter each move (N-1)/N of the FULL tensor.
    """

    full = numel * element_size
    if tp_size <= 1:
        moved = 0.0
    elif op == "allreduce":
        moved = 2.0 * (tp_size - 1) / tp_size * full
    elif op in ("allgather", "reduce_scatter"):
        moved = (tp_size - 1) / tp_size * full
    else:
        raise ValueError(f"unknown collective op {op!r}")
    return {"bytes_sent": moved, "bytes_received": moved}


class SPCommRecorder(TPCommRecorder):
    """Chapter-23 recorder: the chapter-36 AllReduce timing PLUS AllGather /
    ReduceScatter timing and ring-model byte ledgers for every op type.

    Drop-in compatible with the chapter-36 modules: ``all_reduce`` keeps the
    parent's timing keys so the TP baseline arm can use this recorder too and
    both arms report identically named fields.  AllGather/ReduceScatter get
    their own CUDA event pairs, resolved together in ``resolve``; on CPU
    (gloo smoke tests) ``time.perf_counter`` is used instead.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_pending: list[tuple[str, str, object, object]] = []
        self._op_ms: dict[tuple[str, str], float] = {}
        self._op_count: dict[tuple[str, str], int] = {}
        self._op_bytes: dict[tuple[str, str], float] = {}
        self.loss_ms = 0.0
        self.loss_count = 0

    def reset(self) -> None:
        super().reset()
        self._op_pending.clear()
        self._op_ms.clear()
        self._op_count.clear()
        self._op_bytes.clear()
        self.loss_ms = 0.0
        self.loss_count = 0

    def _accumulate(self, phase: str, elapsed_ms: float) -> None:
        if phase == "loss":
            # The single scalar AllReduce pair assembling the global token-mean
            # loss; kept out of the TP AllReduce totals but still timed.
            self.loss_ms += elapsed_ms
            self.loss_count += 1
        else:
            super()._accumulate(phase, elapsed_ms)

    def _record_op_direct(self, op: str, phase: str, elapsed_ms: float, accounting_numel: int,
                          element_size: int, tp_size: int) -> None:
        key = (op, phase)
        self._op_ms[key] = self._op_ms.get(key, 0.0) + elapsed_ms
        self._op_count[key] = self._op_count.get(key, 0) + 1
        self._op_bytes[key] = self._op_bytes.get(key, 0.0) + sum(
            ring_bytes(accounting_numel, element_size, tp_size, op).values()
        )

    def _run_op(self, op: str, phase: str, tensor: torch.Tensor, accounting_numel: int,
                tp_size: int, group, action) -> None:
        if tp_size <= 1:
            return
        if tensor.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            action()
            end.record()
            self._op_pending.append((op, phase, start, end))
            self._op_count[(op, phase)] = self._op_count.get((op, phase), 0) + 1
            self._op_bytes[(op, phase)] = self._op_bytes.get((op, phase), 0.0) + sum(
                ring_bytes(accounting_numel, tensor.element_size(), tp_size, op).values()
            )
        else:
            begin = time.perf_counter()
            action()
            self._record_op_direct(op, phase, (time.perf_counter() - begin) * 1000,
                                   accounting_numel, tensor.element_size(), tp_size)

    def all_gather(self, tensor_list: list[torch.Tensor], tensor: torch.Tensor,
                   tp_size: int, group, phase: str) -> None:
        # Accounting uses the FULL gathered size (shard x tp_size): the ring
        # moves (N-1)/N of the full tensor per rank.
        self._run_op("allgather", phase, tensor, tensor.numel() * tp_size,
                     tp_size, group,
                     lambda: dist.all_gather(tensor_list, tensor.contiguous(), group=group))

    def reduce_scatter(self, output: torch.Tensor, input_list: list[torch.Tensor],
                       tp_size: int, group, phase: str) -> None:
        # Accounting uses the FULL input size (chunk x tp_size); input_list
        # holds the per-rank chunks of that full tensor.
        self._run_op("reduce_scatter", phase, output, output.numel() * tp_size,
                     tp_size, group,
                     lambda: dist.reduce_scatter(output, [chunk.contiguous() for chunk in input_list],
                                                 op=dist.ReduceOp.SUM, group=group))

    def resolve(self) -> dict[str, float]:
        base = dict(super().resolve())
        base["sp_loss_allreduce_ms"] = self.loss_ms
        base["sp_loss_allreduce_count"] = float(self.loss_count)
        if self._op_pending:
            torch.cuda.synchronize()
            for op, phase, start, end in self._op_pending:
                key = (op, phase)
                self._op_ms[key] = self._op_ms.get(key, 0.0) + float(start.elapsed_time(end))
            self._op_pending.clear()
        for (op, phase), value in self._op_ms.items():
            base[f"sp_{op}_{phase}_ms"] = value
        for (op, phase), value in self._op_count.items():
            base[f"sp_{op}_{phase}_count"] = float(value)
        for (op, phase), value in self._op_bytes.items():
            base[f"sp_{op}_{phase}_ring_bytes"] = value
        return base


# The single recorder instance used by this chapter's scripts (module-global
# so tp_train_step's lookup and the SP modules observe the same counters).
COMM_RECORDER = SPCommRecorder()


def _all_gather_seq(tensor: torch.Tensor, tp_size: int, group, recorder, phase: str) -> torch.Tensor:
    """Concatenate every rank's sequence shard along dim 1 (identical output
    on all ranks).  List-based collectives keep the batch-first layout; the
    `_tensor` variants would need a seq-first transpose (out of scope)."""

    if tp_size <= 1:
        return tensor
    shards = [torch.empty_like(tensor) for _ in range(tp_size)]
    recorder.all_gather(shards, tensor, tp_size, group, phase)
    return torch.cat(shards, dim=1)


def _reduce_scatter_seq(tensor: torch.Tensor, local_seq: int, tp_size: int, group,
                        recorder, phase: str) -> torch.Tensor:
    """Sum rank partials and keep only this rank's sequence window:
    rank r receives sum_r' tensor_r'[:, r*local:(r+1)*local]."""

    if tp_size <= 1:
        return tensor
    output = torch.empty(
        (tensor.shape[0], local_seq, tensor.shape[2]),
        dtype=tensor.dtype, device=tensor.device,
    )
    chunks = list(tensor.chunk(tp_size, dim=1))
    recorder.reduce_scatter(output, chunks, tp_size, group, phase)
    return output


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """The SP "g" op: forward AllGather(seq), backward local slice.

    The backward receives the block-input gradient AFTER the "f" op's
    AllReduce, i.e. rank-identical and complete, so the correct per-rank
    shard gradient is simply this rank's slice — no second sum (a
    ReduceScatter over identical copies would multiply them by tp_size).
    Megatron fuses f's AllReduce with a ReduceScatter here; the fusion and
    its byte saving are out of scope for this chapter.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_rank: int, tp_size: int, group, recorder):
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        ctx.group = group
        ctx.recorder = recorder
        ctx.local_seq = input_.shape[1]
        return _all_gather_seq(input_, tp_size, group, recorder, "forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        begin = ctx.tp_rank * ctx.local_seq
        return grad_output[:, begin : begin + ctx.local_seq].contiguous(), None, None, None, None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """The SP "ḡ" op: forward ReduceScatter(seq), backward AllGather(seq).

    Forward inputs are rank PARTIAL sums over the full sequence (the
    row-parallel output with reduce_output=False); ReduceScatter sums them and
    keeps this rank's window, exactly the shard of the true output.  Backward
    inputs are DISJOINT pieces of the true gradient (one per rank), so the
    correct backward is a plain AllGather, not a sum.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_size: int, group, recorder):
        ctx.tp_size = tp_size
        ctx.group = group
        ctx.recorder = recorder
        ctx.local_seq = input_.shape[1] // tp_size
        return _reduce_scatter_seq(input_, ctx.local_seq, tp_size, group, recorder, "forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = _all_gather_seq(grad_output, ctx.tp_size, ctx.group, ctx.recorder, "backward")
        return grad, None, None, None


class ParallelLayerNorm(nn.Module):
    """Qwen3 RMSNorm on a sequence shard, with replicated weights.

    RMSNorm normalizes over the hidden dim of EACH token independently, so
    running it on [b, s/N, h] equals running it on [b, s, h] and slicing —
    the mathematical justification for sharding the non-TP region (Qwen3 has
    no dropout; the residual adds are equally per-token).  The weight is
    replicated in VALUE but after sequence sharding each rank accumulates only
    its own tokens' gradient terms: the raw gradients are partial sums that
    differ across ranks.  A post-accumulate-grad SUM AllReduce restores the
    exact reference gradient; skipping it lets the copies diverge from the
    FIRST optimizer step on (same trap as chapter 21's q/k norms).
    """

    def __init__(self, hidden_size: int, eps: float, tp_size: int, group=None,
                 recorder: SPCommRecorder | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.tp_size = tp_size
        self.group = group
        self.recorder = recorder if recorder is not None else COMM_RECORDER
        self.weight.register_post_accumulate_grad_hook(self._sync_gradient)

    def _sync_gradient(self, parameter: torch.Tensor) -> None:
        if self.tp_size > 1 and parameter.grad is not None:
            self.recorder.all_reduce(parameter.grad, self.tp_size, self.group, "backward")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        float_states = hidden_states.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


def register_replicated_gradient_sync(parameters: Sequence[nn.Parameter], tp_size: int,
                                      group=None, recorder: SPCommRecorder | None = None) -> int:
    """Attach the backward SUM AllReduce to every non-TP weight that sees only
    a sequence shard but has NO sync of its own — in this model that is the
    tied Embedding/LM Head (the shared tensor gets ONE hook; ParallelLayerNorm
    already syncs its own weight).  Returns the number of hooks."""

    recorder = recorder if recorder is not None else COMM_RECORDER
    seen: set[int] = set()

    def _sync(parameter: torch.Tensor) -> None:
        if tp_size > 1 and parameter.grad is not None:
            recorder.all_reduce(parameter.grad, tp_size, group, "backward")

    attached = 0
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parameter.register_post_accumulate_grad_hook(_sync)
        attached += 1
    return attached


class SPDecoderLayer(nn.Module):
    """One Qwen3 decoder layer with the non-TP regions sequence-sharded.

    Same weight layout and same state-dict names as the chapter-36 layer
    (input_layernorm / self_attn / post_attention_layernorm / mlp); only the
    activation layout and the two boundary operators differ.  The parallel
    blocks are built with reduce_output=False so the row-parallel outputs stay
    UNREDUCED rank partials for the ḡ ReduceScatter.
    """

    def __init__(self, config: Qwen3Config, tp_rank: int, tp_size: int,
                 group=None, recorder: SPCommRecorder | None = None) -> None:
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = group
        self.recorder = recorder if recorder is not None else COMM_RECORDER
        self.input_layernorm = ParallelLayerNorm(
            config.hidden_size, config.rms_norm_eps, tp_size, group, self.recorder
        )
        self.post_attention_layernorm = ParallelLayerNorm(
            config.hidden_size, config.rms_norm_eps, tp_size, group, self.recorder
        )
        self.self_attn = ParallelAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
            tp_rank,
            tp_size,
            group=group,
            recorder=self.recorder,
            reduce_output=False,
        )
        self.mlp = ParallelMLP(
            config.hidden_size,
            config.intermediate_size,
            tp_rank,
            tp_size,
            group=group,
            recorder=self.recorder,
            reduce_output=False,
        )

    def _gather(self, shard: torch.Tensor) -> torch.Tensor:
        return _GatherFromSequenceParallelRegion.apply(
            shard, self.tp_rank, self.tp_size, self.group, self.recorder
        )

    def _scatter(self, partial: torch.Tensor) -> torch.Tensor:
        return _ScatterToSequenceParallelRegion.apply(
            partial, self.tp_size, self.group, self.recorder
        )

    def forward(self, hidden_states: torch.Tensor, cosine: torch.Tensor,
                sine: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer on the shard; the TP region sees the full sequence.
        residual = hidden_states                                      # [b, s/N, h]
        normed_full = self._gather(self.input_layernorm(hidden_states))  # [b, s, h]
        attn_partial = self.self_attn(normed_full, cosine, sine, causal_mask)  # [b, s, h] rank partial
        hidden_states = residual + self._scatter(attn_partial)        # [b, s/N, h]

        # MLP sub-layer, same pair of boundary operators.
        residual = hidden_states
        normed_full = self._gather(self.post_attention_layernorm(hidden_states))
        mlp_partial = self.mlp(normed_full)
        hidden_states = residual + self._scatter(mlp_partial)
        return hidden_states


class SequenceParallelModel(nn.Module):
    """Transformer trunk with the residual stream sequence-sharded.

    Attribute and state-dict names mirror ``qwen3_model.Qwen3Model`` so both
    the tiny-model smoke checks and the real-weight loader can reuse the
    chapter-36 shard helpers verbatim.  The Embedding lookup and the final
    norm run on the local sequence window; RoPE and the causal mask are
    computed for the FULL sequence (identical on every rank) because attention
    inside the TP regions runs over all positions.
    """

    def __init__(self, config: Qwen3Config, tp_rank: int, tp_size: int,
                 group=None, recorder: SPCommRecorder | None = None) -> None:
        super().__init__()
        self.config = config
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = group
        self.recorder = recorder if recorder is not None else COMM_RECORDER
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                SPDecoderLayer(config, tp_rank, tp_size, group, self.recorder)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = ParallelLayerNorm(
            config.hidden_size, config.rms_norm_eps, tp_size, group, self.recorder
        )
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
        # The tied Embedding/LM Head sees only the local shard: its gradient
        # is a rank-partial sum that needs the sync AllReduce.  The final norm
        # is a ParallelLayerNorm and already syncs its own weight —
        # registering it here TOO would AllReduce twice and multiply the
        # gradient by tp_size.
        register_replicated_gradient_sync(
            [self.embed_tokens.weight], tp_size, group, self.recorder
        )

    def make_attention_mask(self, attention_mask, sequence_length, dtype, device):
        """The chapter-16 causal + key-padding mask over the FULL sequence."""

        smallest_value = torch.finfo(dtype).min
        mask = torch.full(
            (sequence_length, sequence_length),
            smallest_value,
            dtype=dtype,
            device=device,
        )
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is None:
            return mask
        if attention_mask.shape[1] != sequence_length:
            raise ValueError("attention_mask 与 input_ids 的序列长度不一致")
        mask = mask.expand(attention_mask.shape[0], 1, sequence_length, sequence_length).clone()
        key_is_padding = attention_mask[:, None, None, :] == 0
        return mask.masked_fill(key_is_padding, smallest_value)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 [batch, sequence] 的二维 Tensor")
        batch_size, sequence_length = input_ids.shape
        begin, end = sequence_shard_bounds(sequence_length, self.tp_rank, self.tp_size)

        hidden_states = self.embed_tokens(input_ids[:, begin:end])      # [b, s/N, h]

        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape != input_ids.shape:
            raise ValueError("position_ids 的 Shape 必须与 input_ids 相同")
        # Full-sequence RoPE: the TP regions attend over ALL positions.
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        causal_mask = self.make_attention_mask(
            attention_mask, sequence_length, hidden_states.dtype, hidden_states.device
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, cosine, sine, causal_mask)
        return self.norm(hidden_states)                                  # [b, s/N, h]


class SequenceParallelForCausalLM(nn.Module):
    """Sequence-sharded trunk + replicated LM Head evaluated on the local shard.

    Each rank produces logits ONLY for its own sequence window; the global
    token-mean loss is assembled from the per-shard sums/counts by
    ``sp_loss_terms``.  Full-vocab logits are therefore never materialized
    beyond [b, s/N, vocab] — and chapter 22's vocab sharding stays out.
    """

    def __init__(self, config: Qwen3Config, tp_rank: int, tp_size: int,
                 group=None, recorder: SPCommRecorder | None = None) -> None:
        super().__init__()
        self.config = config
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.model = SequenceParallelModel(config, tp_rank, tp_size, group, recorder)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        # Qwen3-0.6B ties input Embedding and LM Head; the shared tensor gets
        # ONE gradient-sync hook (registered by SequenceParallelModel).
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids
        )
        return self.lm_head(hidden_states)                               # [b, s/N, vocab]


def sp_loss_terms(logits_shard: torch.Tensor, labels_full: torch.Tensor,
                  tp_rank: int, tp_size: int, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unreduced shifted cross entropy for the predictors THIS rank owns.

    Returns ``(loss_sum, token_count)``.  The rank owns every predictor in its
    window [begin, end): positions begin..end-2 predict labels inside the
    window, and the LAST position end-1 predicts ``labels_full[:, end]`` —
    one step beyond the window, read from the replicated labels without any
    communication.  Only the global final position (which has no next token)
    is dropped, by the last rank.  The per-rank predictor sets therefore
    PARTITION the full-sequence predictors [0, s-1): summed over ranks,
    ``loss_sum`` and ``token_count`` are exactly the global token-mean's
    numerator and denominator — no shard-boundary term is dropped and none is
    counted twice.  (A window-internal shift ``logits[:, :-1]`` vs
    ``labels[:, 1:]`` would silently drop one supervised term per boundary.)
    """

    begin, end = sequence_shard_bounds(sequence_length, tp_rank, tp_size)
    if logits_shard.shape[1] != end - begin:
        raise ValueError(
            f"logits shard 的序列长度 {logits_shard.shape[1]} 与 rank 窗口 [{begin}, {end}) 不一致"
        )
    label_end = min(end + 1, sequence_length)
    shifted_logits = logits_shard[:, : label_end - begin - 1].contiguous()
    shifted_labels = labels_full[:, begin + 1 : label_end].contiguous()
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    count = (shifted_labels != IGNORE_INDEX).sum()
    return losses, count


def sp_global_loss_scale(loss_sum_local: torch.Tensor, count_local: torch.Tensor,
                         tp_size: int, group=None, recorder: SPCommRecorder | None = None):
    """Assemble the global token-mean loss from per-shard sums/counts.

    The pair ``(loss_sum, count)`` is packed into ONE float64 scalar tensor so
    a single AllReduce delivers the global sum and count together (timed by
    the recorder under the ``loss`` phase, no extra device synchronization).
    The backward loss is ``sum_local / count_global``: its gradient summed
    over ranks is exactly the global token-mean gradient, because every rank
    divides by the same constant.
    """

    recorder = recorder if recorder is not None else COMM_RECORDER
    packed = torch.stack([loss_sum_local.detach().double(), count_local.double()])
    recorder.all_reduce(packed, tp_size, group, "loss")
    sum_global, count_global = packed[0], packed[1]
    denominator = count_global.clamp(min=1.0)
    global_loss = float((sum_global / denominator).item())
    backward_loss = loss_sum_local / denominator.to(loss_sum_local.dtype)
    metrics = {
        "supervised_global": float(count_global.item()),
    }
    return backward_loss, global_loss, metrics


class ActivationLedger:
    """Exact saved-for-backward byte accounting via saved_tensors_hooks.

    ``pack`` runs for every tensor an op stashes for backward; parameter
    storages are excluded by data pointer so the number is an ACTIVATION
    proxy comparable across arms (both arms save their own weight shards, but
    the weight layout is identical, so the delta between arms is purely the
    activation layout this chapter changes).  Use it around forward (+loss);
    the count is final once forward returns.
    """

    def __init__(self, exclude_parameters: Sequence[torch.nn.Parameter] = ()) -> None:
        self.saved_bytes = 0
        self.saved_count = 0
        self._param_ptrs = {parameter.data_ptr() for parameter in exclude_parameters}
        self._handle = None

    def _pack(self, tensor):
        if torch.is_tensor(tensor) and tensor.data_ptr() not in self._param_ptrs:
            self.saved_bytes += tensor.numel() * tensor.element_size()
            self.saved_count += 1
        return tensor

    def _unpack(self, tensor):
        return tensor

    def __enter__(self) -> "ActivationLedger":
        self._handle = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._handle.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._handle.__exit__(*exc)
        self._handle = None


def sp_memory_ledger(model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                     tp_size: int = 1) -> dict:
    """State ledger (parameters/optimizer) of this rank, SP layout.

    Weight ownership is IDENTICAL to chapter 21 (only activations moved), so
    the chapter-36 ledger applies verbatim; re-exported here so the chapter's
    scripts depend on one module for their accounting.
    """

    from mlp_tp import tp_memory_ledger

    return tp_memory_ledger(model, optimizer, tp_size)


def sp_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    host_batch: PaddedBatch,
    loss_scale: float,
    device: torch.device,
    max_grad_norm: float,
    tp_size: int,
    rank: int,
    collect_telemetry: bool = True,
    recorder: SPCommRecorder | None = None,
) -> dict[str, float]:
    """One Sequence Parallel optimizer step with every segment timed.

    Data is replicated (TP convention): every rank receives the SAME batch and
    slices its own sequence window; ``loss_scale`` stays an argument for
    interface parity and is exactly 1.0 in this chapter.

    Communication inside the step:
    forward  - one seq-AllGather + one seq-ReduceScatter per parallel block;
    backward - the chapter-36 "f" AllReduce per block, the q/k norm syncs, the
               NEW sequence-shard syncs (LayerNorm weights, Embedding/final
               norm), and the ḡ backward AllGathers;
    plus     - ONE scalar AllReduce pair assembling the global token-mean loss.
    After backward there is deliberately NO parameter-gradient AllReduce:
    every weight's gradient is already the exact reference value (shard
    weights are rank-local slices; replicated weights were synced by hooks).
    """

    from mlp_tp import gpu_telemetry

    recorder = recorder if recorder is not None else COMM_RECORDER
    model.train()
    optimizer.zero_grad(set_to_none=True)
    recorder.reset()

    sequence_length = host_batch.input_ids.shape[1]
    begin, end = sequence_shard_bounds(sequence_length, rank, tp_size)
    labels = host_batch.labels.to(device)
    # Data is replicated (TP convention): the model receives the FULL batch
    # and slices its own sequence window internally (SequenceParallelModel.
    # forward applies sequence_shard_bounds to the input it receives).
    input_ids = host_batch.input_ids.to(device)
    attention_mask = host_batch.attention_mask.to(device)

    if device.type != "cuda":
        start = time.perf_counter()
        h2d_begin = time.perf_counter()
        h2d_ms = (time.perf_counter() - h2d_begin) * 1000
        forward_begin = time.perf_counter()
        logits = model(input_ids, attention_mask=attention_mask)
        loss_sum, count_local = sp_loss_terms(logits, labels, rank, tp_size, sequence_length)
        loss, global_loss, loss_comm = sp_global_loss_scale(loss_sum, count_local, tp_size, None, recorder)
        forward_incl_ms = (time.perf_counter() - forward_begin) * 1000
        backward_begin = time.perf_counter()
        loss.backward()
        backward_incl_ms = (time.perf_counter() - backward_begin) * 1000
        comm = recorder.resolve()
        loss_ms = comm["sp_loss_allreduce_ms"]
        forward_ms = forward_incl_ms - comm["tp_allreduce_forward_ms"] - loss_ms
        backward_ms = backward_incl_ms - comm["tp_allreduce_backward_ms"]
        optimizer_begin = time.perf_counter()
        grad_norm = clip_grad_norm_tp(model, max_grad_norm, tp_size)
        optimizer.step()
        optimizer_ms = (time.perf_counter() - optimizer_begin) * 1000
        step_ms = (time.perf_counter() - start) * 1000
    else:
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        events = {
            name: torch.cuda.Event(enable_timing=True)
            for name in ("h2d_start", "h2d_end", "forward_end", "backward_end", "optimizer_start", "optimizer_end")
        }
        events["h2d_start"].record()
        events["h2d_end"].record()
        logits = model(input_ids, attention_mask=attention_mask)
        loss_sum, count_local = sp_loss_terms(logits, labels, rank, tp_size, sequence_length)
        loss, global_loss, loss_comm = sp_global_loss_scale(loss_sum, count_local, tp_size, None, recorder)
        events["forward_end"].record()
        loss.backward()
        events["backward_end"].record()
        # resolve() synchronizes once and folds the AllReduce, boundary-op and
        # loss-phase event pairs into the comm totals.
        comm = recorder.resolve()
        loss_ms = comm["sp_loss_allreduce_ms"]
        forward_ms = float(events["h2d_end"].elapsed_time(events["forward_end"])) \
            - comm["tp_allreduce_forward_ms"] - loss_ms
        backward_ms = float(events["forward_end"].elapsed_time(events["backward_end"])) \
            - comm["tp_allreduce_backward_ms"]
        h2d_ms = float(events["h2d_start"].elapsed_time(events["h2d_end"]))
        events["optimizer_start"].record()
        grad_norm = clip_grad_norm_tp(model, max_grad_norm, tp_size)
        optimizer.step()
        events["optimizer_end"].record()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - wall_start) * 1000
        optimizer_ms = float(events["optimizer_start"].elapsed_time(events["optimizer_end"]))

    metrics = {
        "h2d_time_ms": h2d_ms,
        "forward_time_ms": forward_ms,
        "backward_time_ms": backward_ms,
        "optimizer_time_ms": optimizer_ms,
        "step_time_ms": step_ms,
        "local_loss": global_loss,     # the assembled global token-mean loss
        "global_loss": global_loss,
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(host_batch.physical_tokens),
        "non_padding_tokens": float(host_batch.non_padding_tokens),
        "supervised_tokens": float(host_batch.supervised_tokens),
        "padding_tokens": float(host_batch.padding_tokens),
        "padding_ratio": host_batch.padding_ratio,
        "batch_max_length": float(max(host_batch.lengths)),
        "samples": float(host_batch.batch_size),
        "sequence_shard_local": float(end - begin),
    }
    metrics.update(comm)
    metrics.update(loss_comm)
    sp_comm_total_ms = sum(
        value for key, value in comm.items() if key.startswith("sp_") and key.endswith("_ms")
    )
    metrics["sp_boundary_comm_total_ms"] = sp_comm_total_ms
    metrics["tp_allreduce_total_ms"] = comm["tp_allreduce_forward_ms"] + comm["tp_allreduce_backward_ms"]
    metrics["sp_total_comm_ms"] = (
        metrics["tp_allreduce_total_ms"]
        + sp_comm_total_ms
        + comm["sp_loss_allreduce_ms"]
    )
    metrics["sp_comm_exposed_ratio"] = (
        metrics["sp_total_comm_ms"] / metrics["step_time_ms"] if metrics["step_time_ms"] > 0 else 0.0
    )
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "physical_tokens_per_second": host_batch.physical_tokens / seconds,
        "non_padding_tokens_per_second": host_batch.non_padding_tokens / seconds,
        "supervised_tokens_per_second": host_batch.supervised_tokens / seconds,
    })
    if device.type == "cuda" and collect_telemetry:
        metrics.update(gpu_telemetry(rank))
    return metrics


def load_handwritten_model_sp(model_directory, device, tp_rank, tp_size, dtype=None,
                              group=None, recorder: SPCommRecorder | None = None):
    """Build the Sequence Parallel model with chapter-36 weight shard reads.

    Attention (q/k/v row head blocks, o columns) and MLP (gate/up rows, down
    columns) weights are read through the safetensors ``PySafeSlice`` API as
    this rank's window only; norms, Embedding and LM Head are read fully and
    stay replicated.  Weight layout == chapter 21; ONLY the activation layout
    differs (sequence shards + g/ḡ operators), which is this chapter's single
    variable against the ``tp_wN`` arm.
    """

    config = Qwen3Config.from_model_directory(model_directory)
    if config.intermediate_size % tp_size != 0:
        raise shard_error(
            f"model intermediate size {config.intermediate_size} is not divisible by tp_size {tp_size}",
            (config.intermediate_size, config.hidden_size),
            torch.float32,
            torch.device("cpu"),
            tp_rank,
            tp_size,
        )
    # Head divisibility (num_q_heads and num_kv_heads vs tp_size) fails loudly
    # before any weight byte is read.
    gqa_head_layout(tp_rank, tp_size, config.num_attention_heads, config.num_key_value_heads)
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")

    model = SequenceParallelForCausalLM(config, tp_rank, tp_size, group=group, recorder=recorder)

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    from safetensors import safe_open

    layout = gqa_head_layout(tp_rank, tp_size, config.num_attention_heads, config.num_key_value_heads)
    local_q_rows = layout["local_q_heads"] * config.head_dim
    local_kv_rows = layout["local_kv_heads"] * config.head_dim
    q_begin = tp_rank * local_q_rows
    kv_begin = tp_rank * local_kv_rows
    local_intermediate = config.intermediate_size // tp_size
    shard_begin = tp_rank * local_intermediate
    shard_end = shard_begin + local_intermediate

    state_dict = {}
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            source = handle.get_slice(key)
            if key.endswith(Q_KEY_SUFFIX) or key.endswith(O_KEY_SUFFIX):
                state_dict[key] = (
                    source[q_begin : q_begin + local_q_rows, :]
                    if key.endswith(Q_KEY_SUFFIX)
                    else source[:, q_begin : q_begin + local_q_rows]
                )
            elif key.endswith(K_KEY_SUFFIX) or key.endswith(V_KEY_SUFFIX):
                state_dict[key] = source[kv_begin : kv_begin + local_kv_rows, :]
            elif key.endswith(GATE_KEY_SUFFIX) or key.endswith(UP_KEY_SUFFIX):
                state_dict[key] = source[shard_begin:shard_end, :]
            elif key.endswith(DOWN_KEY_SUFFIX):
                state_dict[key] = source[:, shard_begin:shard_end]
            else:
                state_dict[key] = source[:]

    load_result = model.load_state_dict(state_dict, strict=False)
    # lm_head.weight is tied to embed_tokens.weight and missing from the file.
    allowed_missing_keys = {"lm_head.weight"}
    unexpected_missing = set(load_result.missing_keys) - allowed_missing_keys
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "权重名称不匹配，missing="
            + str(sorted(unexpected_missing))
            + ", unexpected="
            + str(sorted(load_result.unexpected_keys))
        )

    model.tie_weights()
    model_dtype = dtype or torch.float32

    # model.to(dtype=...) would also convert the float RoPE buffer; keep its
    # float32 precision by restoring it after the move (chapter-16 rule).
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    return model


def arm_byte_ledger(batch_size: int, sequence_length: int, hidden_size: int,
                    tp_size: int, num_layers: int, dtype_bytes: int = 4) -> dict:
    """Analytic per-step communication byte ledger for BOTH arms (ring model).

    Op counts per decoder layer come straight from the implementations:
    tp_wN (chapter-36 layout): forward 2 AllReduce ([b,s,h] o_proj + down_proj
    outputs); backward 2 block-input AllReduces + 2 tiny q/k norm syncs.
    sp_wN adds, per parallel block: forward AllGather + ReduceScatter; backward
    AllGather (ḡ) — and the f AllReduce stays — plus tiny syncs for the
    LayerNorm weights (2/layer) and Embedding/final norm (once per step).
    """

    activation = batch_size * sequence_length * hidden_size * dtype_bytes
    hidden_vec = hidden_size * dtype_bytes
    head_sync = 128 * dtype_bytes  # q_norm/k_norm weight vectors (head_dim)

    def allreduce(numel):
        return 2.0 * sum(ring_bytes(numel, dtype_bytes, tp_size, "allreduce").values())

    def allgather(numel):
        return 2.0 * sum(ring_bytes(numel, dtype_bytes, tp_size, "allgather").values())

    def reduce_scatter(numel):
        return 2.0 * sum(ring_bytes(numel, dtype_bytes, tp_size, "reduce_scatter").values())

    tp_per_layer = 2 * allreduce(activation) + 2 * allreduce(head_sync)
    sp_per_layer = (
        2 * allgather(activation)      # g forward (attention + MLP entries)
        + 2 * reduce_scatter(activation)  # ḡ forward
        + 2 * allgather(activation)    # ḡ backward
        + 2 * allreduce(activation)    # f backward (kept from chapter 21)
        + 2 * allreduce(head_sync)     # q/k norm syncs
        + 2 * allreduce(hidden_vec)    # LayerNorm weight syncs (new in SP)
    )
    per_step_extra_syncs = 2 * allreduce(hidden_vec)  # Embedding + final norm syncs (SP only)
    return {
        "ring_model": "per-rank sent+received bytes; NCCL ring approximation, not a wire measurement",
        "activation_tensor_bytes": float(activation),
        "tp_arm_per_layer_bytes": float(tp_per_layer),
        "sp_arm_per_layer_bytes": float(sp_per_layer),
        "sp_arm_per_step_extra_sync_bytes": float(per_step_extra_syncs),
        "tp_arm_total_bytes": float(num_layers * tp_per_layer),
        "sp_arm_total_bytes": float(num_layers * sp_per_layer + per_step_extra_syncs),
        "sp_over_tp_bytes_ratio": float(
            (num_layers * sp_per_layer + per_step_extra_syncs) / max(num_layers * tp_per_layer, 1.0)
        ),
    }
