"""Chapter 37 core: Vocab Parallel Embedding, sharded LM head, and the
distributed cross entropy that never materializes full logits.

The only variable of this chapter: on top of the chapter-36 layout (attention
and MLP sharded), is the Embedding/LM Head vocab dimension ALSO sharded?  For
tp_size = N, rank r owns the vocab rows ``[r*V/N, (r+1)*V/N)``:

- ``VocabParallelEmbedding``: rows r of the embedding matrix.  Lookup masks
  token ids outside the range to zero contributions, then ONE forward
  AllReduce SUM (the embedding "g" op) reconstructs the full hidden states
  ``E[t] = sum_r E_r[t]``.  Its backward is the identity, so each rank's
  embedding gradient receives only the masked local rows — a rank-local shard
  of the full gradient, NO communication.
- ``VocabParallelLMHead``: the same row blocks consumed from above,
  ``logits_r = h @ W_r^T`` producing SHARDED logits ``[batch, seq, V/N]``.
  There is deliberately NO allreduce over logits: the full ``[batch, seq,
  151936]`` logits tensor is exactly what this chapter refuses to materialize
  (batch 2 x seq 512 x 151936 x FP32 is ~622 MiB per rank, larger than most
  of the model).  The price is ONE backward AllReduce: like every
  column-parallel layer without an output reduction, the head input gradient
  ``d(h) = dlogits_r @ W_r`` is a PARTIAL sum on each rank, so the head wraps
  its input in the chapter-35 "f" op (forward identity, backward AllReduce
  SUM).  With a REPLICATED head (chapter 21 and the ``attn_tp`` arm) each
  rank holds the full weight and the full gradient, so that communication
  simply does not exist — its appearance here is part of the vocab split's
  cost, and it is why the LayerNorm gradients stay rank-identical.
- Tie: Qwen3-0.6B ties embedding and LM head, so both consume the SAME
  sharded Parameter; gradients from the lookup path and the logits path
  accumulate into one rank-local shard.

VOCAB PARALLEL CROSS ENTROPY (the error-prone core of this chapter)
-------------------------------------------------------------------
For one position with logits ``l_v`` and target ``t`` the loss is
``-l_t + log sum_v exp(l_v)``.  With sharded logits every rank can compute
the exact same value without ever gathering V:

    m    = max_v l_v            local max -> AllReduce MAX   -> global max
    Z    = sum_v exp(l_v - m)   local sum  -> AllReduce SUM  -> global Z
    l_t  = target logit         owned by ONE rank; every rank masks its
                local target logits to zero and an AllReduce SUM collects it
    loss = log(Z) + m - l_t

Backward (dloss/dl_j = softmax_j - 1{j==t}) needs ONLY quantities this rank
already has — the local shard, the scalar m and Z, and the target — so the
backward performs NO communication at all.  This is the payoff of the
vocab split: three tiny ``[token]``-sized collectives on the forward path
replace a full-vocab AllReduce or AllGather, and gradients stay local.

The implementation is a single ``torch.autograd.Function`` whose forward
returns per-token losses (zeros at ``IGNORE_INDEX`` positions, matching
``F.cross_entropy(ignore_index=...)`` reduction="none") and whose backward
applies the formula directly.  ``unreduced_vocab_parallel_cross_entropy``
performs the causal shift and ``reduce_losses`` (the shared chapter-18
reduction) keeps the global ``token_mean`` semantics identical to the
replicated path.

EXTENDED GRADIENT OWNERSHIP: attention/MLP shards follow chapters 21/20;
embedding/LM head shards join ownership class 1 (rank-local gradients, no
communication).  The only replicated parameters left are the LayerNorms,
whose gradients stay rank-identical (class 3, no sync, an AllReduce would be
wrong).  ``split_all_parameters``, ``clip_grad_norm_all``,
``gather_all_tensor_parallel_parameters`` and ``tp_memory_ledger_all`` extend
the chapter-35 helpers with the vocab shard class, and ``tp_train_step``
dispatches the loss between full logits (``batch_loss``, ``attn_tp`` arm) and
the vocab-parallel path (``batch_loss_vocab_parallel``, ``vocab_tp`` arm).
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from assistant_loss import IGNORE_INDEX
from mlp_tp import (
    ColumnParallelLinear,
    RowParallelLinear,
    _CopyToTensorParallelRegion,
    _ReduceFromTensorParallelRegion,
    shard_error,
)
from mini_batch import reduce_losses


EMBED_KEY_SUFFIX = "embed_tokens.weight"


class VocabParallelEmbedding(nn.Module):
    """Embedding rows sharded over the TP group; output AllReduced once.

    Ids outside this rank's range are masked to row 0 BEFORE the lookup and
    their outputs are zeroed after it — the lookup itself stays gather-free
    and in-range, and the AllReduce sums exactly one nonzero contribution per
    token.  The output IS the full hidden state: it is rank-identical after
    the "g" op, so everything downstream needs no further communication.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        tp_rank: int,
        tp_size: int,
        group=None,
        recorder=None,
    ) -> None:
        super().__init__()
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
        if vocab_size % tp_size != 0:
            raise shard_error(
                f"vocab size {vocab_size} is not divisible by tp_size {tp_size}",
                (vocab_size, hidden_size),
                torch.float32,
                torch.device("cpu"),
                tp_rank,
                tp_size,
            )
        local_vocab = vocab_size // tp_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.local_vocab = local_vocab
        self.vocab_begin = tp_rank * local_vocab
        self.vocab_end = self.vocab_begin + local_vocab
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = group
        from mlp_tp import COMM_RECORDER

        self.recorder = recorder if recorder is not None else COMM_RECORDER
        self.weight = nn.Parameter(torch.empty(local_vocab, hidden_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        in_range = (input_ids >= self.vocab_begin) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_begin).masked_fill(~in_range, 0)
        output = F.embedding(local_ids, self.weight)
        output = output * in_range.unsqueeze(-1).to(output.dtype)
        # The embedding "g" op: forward sums the one-hot place results across
        # ranks, backward is the identity (the input is integer ids and has
        # no gradient; the weight gradient is rank-local via the mask).
        return _ReduceFromTensorParallelRegion.apply(output, self.tp_size, self.group, self.recorder)


class VocabParallelLMHead(nn.Module):
    """Logits sharded over the TP group; the weight IS the embedding shard.

    Qwen3-0.6B ties embedding and LM head, so ``tie_weights`` assigns the
    ``VocabParallelEmbedding`` weight Parameter here; the lookup and the
    projection then accumulate gradients into one rank-local shard.  The
    output is DELIBERATELY not reduced: the sharded logits feed the
    vocab-parallel cross entropy, and materializing full logits is exactly
    the memory cost this chapter removes.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        tp_rank: int,
        tp_size: int,
        group=None,
        recorder=None,
    ) -> None:
        super().__init__()
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
        if vocab_size % tp_size != 0:
            raise shard_error(
                f"vocab size {vocab_size} is not divisible by tp_size {tp_size}",
                (vocab_size, hidden_size),
                torch.float32,
                torch.device("cpu"),
                tp_rank,
                tp_size,
            )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.local_vocab = vocab_size // tp_size
        self.vocab_begin = tp_rank * self.local_vocab
        self.vocab_end = self.vocab_begin + self.local_vocab
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = group
        from mlp_tp import COMM_RECORDER

        self.recorder = recorder if recorder is not None else COMM_RECORDER
        self.weight = nn.Parameter(torch.empty(self.local_vocab, hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The "f" op of the column-parallel head: forward identity, backward
        # AllReduce of the final-hidden gradient (each rank's dlogits_r @ W_r
        # is only the partial sum of its own vocab columns).  The final norm
        # output feeds ONLY this head, so the in-place reduction of its
        # gradient is safe — same single-consumer argument as the chapter-35
        # block entries.
        hidden_states = _CopyToTensorParallelRegion.apply(
            hidden_states, self.tp_size, self.group, self.recorder
        )
        return F.linear(hidden_states, self.weight)


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Per-token CE over sharded logits: 3 forward AllReduces, 0 backward.

    Forward: global max (MAX AllReduce), global exp-sum (SUM AllReduce) and
    the target logit (SUM AllReduce over the per-rank masked gather).  The
    backward recomputes the local softmax from the saved shard and the saved
    global scalars, then applies ``grad * (softmax - onehot)`` — every rank
    holds everything the formula needs, so backward needs no communication.
    """

    @staticmethod
    def forward(ctx, local_logits, labels, vocab_begin, vocab_end, tp_size, group, recorder):
        # Global max of each row: local max, then MAX AllReduce.
        row_max = local_logits.max(dim=-1).values.contiguous()
        recorder.all_reduce(row_max, tp_size, group, "forward")
        exp_local = (local_logits - row_max.unsqueeze(-1)).exp()
        # Global normalizer: local partial sum, then SUM AllReduce.
        denom = exp_local.sum(dim=-1).contiguous()
        recorder.all_reduce(denom, tp_size, group, "forward")
        # Target logit, collected from whichever rank owns the target token:
        # every rank masks its out-of-range targets to zero contributions.
        in_range = (labels >= vocab_begin) & (labels < vocab_end)
        local_targets = (labels - vocab_begin).masked_fill(~in_range, 0)
        target_logit = local_logits.gather(1, local_targets.unsqueeze(1)).squeeze(1)
        target_logit = (target_logit * in_range.to(target_logit.dtype)).contiguous()
        recorder.all_reduce(target_logit, tp_size, group, "forward")
        losses = denom.log() + row_max - target_logit
        ignored = labels == IGNORE_INDEX
        losses = torch.where(ignored, torch.zeros_like(losses), losses)
        ctx.save_for_backward(local_logits, labels, row_max, denom)
        ctx.vocab_begin = vocab_begin
        ctx.vocab_end = vocab_end
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        local_logits, labels, row_max, denom = ctx.saved_tensors
        with torch.no_grad():
            # exp/denom reproduce the forward softmax bit-for-bit because both
            # use the SAVED global scalars (recomputing the local max here
            # would give a different-but-equivalent softmax).
            softmax = (local_logits - row_max.unsqueeze(-1)).exp()
            softmax = softmax / denom.unsqueeze(-1)
            in_range = (labels >= ctx.vocab_begin) & (labels < ctx.vocab_end)
            local_targets = (labels - ctx.vocab_begin).masked_fill(~in_range, 0)
            onehot = torch.zeros_like(softmax)
            onehot.scatter_(1, local_targets.unsqueeze(1), 1.0)
            onehot = onehot * in_range.unsqueeze(-1).to(onehot.dtype)
            grad = (softmax - onehot) * grad_output.unsqueeze(-1).to(softmax.dtype)
            grad = grad * (labels != IGNORE_INDEX).unsqueeze(-1).to(grad.dtype)
        return grad, None, None, None, None, None, None


def unreduced_vocab_parallel_cross_entropy(
    shard_logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_begin: int,
    tp_size: int,
    group=None,
    recorder=None,
) -> torch.Tensor:
    """Per-token losses ``[batch, seq-1]`` from SHARDED logits ``[batch, seq, V/N]``.

    Performs the same causal shift as
    ``mini_batch.unreduced_shifted_cross_entropy`` and returns zeros at
    IGNORE_INDEX positions, so ``reduce_losses`` keeps the global token_mean
    semantics identical to the replicated path.  ``vocab_begin`` is this
    rank's first vocab index — ``model.lm_head.vocab_begin`` for the tied
    Qwen3 head, an explicit number for the standalone smoke checks.
    """

    if shard_logits.ndim != 3 or labels.ndim != 2 or shard_logits.shape[:2] != labels.shape:
        raise ValueError("expected shard logits [batch, sequence, local_vocab] and matching labels")
    batch_size, sequence_length, local_vocab = shard_logits.shape
    flat_logits = shard_logits[:, :-1, :].reshape(-1, local_vocab)
    flat_labels = labels[:, 1:].reshape(-1)
    if not bool((flat_labels != IGNORE_INDEX).any()):
        raise ValueError("no supervised next-token targets")
    from mlp_tp import COMM_RECORDER

    recorder = recorder if recorder is not None else COMM_RECORDER
    losses = _VocabParallelCrossEntropy.apply(
        flat_logits, flat_labels, vocab_begin, vocab_begin + local_vocab, tp_size, group, recorder
    )
    return losses.view(batch_size, sequence_length - 1)


def batch_loss_vocab_parallel(
    shard_logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_begin: int,
    tp_size: int,
    group=None,
    recorder=None,
    reduction: str = "token_mean",
) -> torch.Tensor:
    """The global training loss over sharded logits — same reduction as
    ``mini_batch.batch_loss``, computed WITHOUT full logits ever existing."""

    return reduce_losses(
        unreduced_vocab_parallel_cross_entropy(shard_logits, labels, vocab_begin, tp_size, group, recorder),
        labels,
        reduction,
    )


def is_vocab_parallel_module(module: nn.Module) -> bool:
    return isinstance(module, (VocabParallelEmbedding, VocabParallelLMHead))


def split_all_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Chapter-20 ``split_local_parameters`` extended with the vocab class.

    Replicated parameters exist identically on every rank (LayerNorms only,
    in this chapter's variant); shard parameters are this rank's piece of an
    attention, MLP, embedding or LM head weight.  The tied embedding/LM head
    Parameter appears once (PyTorch deduplicates shared parameters).
    """

    from mlp_tp import is_tensor_parallel_shard

    shard_ids = set()
    for module in model.modules():
        if is_tensor_parallel_shard(module) or is_vocab_parallel_module(module):
            shard_ids.update(id(parameter) for parameter in module.parameters())
    replicated: list[nn.Parameter] = []
    shard: list[nn.Parameter] = []
    for parameter in model.parameters():
        (shard if id(parameter) in shard_ids else replicated).append(parameter)
    return replicated, shard


def clip_grad_norm_all(
    model: nn.Module,
    max_norm: float,
    tp_size: int,
    group=None,
) -> torch.Tensor:
    """Chapter-20 exact-norm clip over the EXTENDED ownership split.

    Replicated (LayerNorm) gradients are counted once; attention/MLP/vocab
    shard gradients are summed across ranks — one scalar AllReduce total.  On
    a model without vocab shards this is bit-for-bit the chapter-35 routine,
    so the ``attn_tp`` baseline arm can share this step function.
    """

    from mlp_tp import _first_device

    device = _first_device(model)
    replicated, shard = split_all_parameters(model)
    replicated_sq = torch.zeros((), dtype=torch.float64, device=device)
    shard_sq = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in replicated:
        if parameter.grad is not None:
            replicated_sq += parameter.grad.double().pow(2).sum()
    for parameter in shard:
        if parameter.grad is not None:
            shard_sq += parameter.grad.double().pow(2).sum()
    if tp_size > 1:
        dist.all_reduce(shard_sq, op=dist.ReduceOp.SUM, group=group)
    total_norm = torch.sqrt(replicated_sq + shard_sq)
    clip_coef = float(max_norm) / (float(total_norm.item()) + 1e-6)
    if clip_coef < 1.0:
        for parameter in replicated + shard:
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coef)
    return total_norm


def _shard_dim_of_owner(model: nn.Module, parameter: torch.Tensor) -> int:
    for module in model.modules():
        if isinstance(module, (ColumnParallelLinear, VocabParallelEmbedding, VocabParallelLMHead)) \
                and module.weight is parameter:
            return 0
        if isinstance(module, RowParallelLinear) and module.weight is parameter:
            return 1
    raise ValueError("parameter does not belong to any parallel layer")


def gather_all_tensor_parallel_parameters(model: nn.Module, tp_size: int, group=None) -> dict:
    """Reassemble the FULL parameter state on every rank (all shard classes).

    Used by correctness checks — never by the training hot path.
    """

    replicated, shard = split_all_parameters(model)
    del replicated
    gathered: dict[int, torch.Tensor] = {}
    for parameter in shard:
        if tp_size > 1:
            buffer = [torch.empty_like(parameter) for _ in range(tp_size)]
            dist.all_gather(buffer, parameter.detach(), group=group)
        else:
            buffer = [parameter.detach()]
        gathered[id(parameter)] = torch.cat(buffer, dim=_shard_dim_of_owner(model, parameter)).contiguous()
    state: dict = {}
    for name, parameter in model.named_parameters():
        state[name] = gathered[id(parameter)] if id(parameter) in gathered else parameter.detach().clone()
    return state


def tp_memory_ledger_all(model: nn.Module, optimizer: torch.optim.Optimizer | None = None, tp_size: int = 1) -> dict:
    """Chapter-20 ownership ledger extended with the vocab shard class."""

    replicated, shard = split_all_parameters(model)
    replicated_bytes = sum(parameter.numel() * parameter.element_size() for parameter in replicated)
    shard_local_bytes = sum(parameter.numel() * parameter.element_size() for parameter in shard)
    ledger = {
        "replicated_parameter_bytes": float(replicated_bytes),
        "shard_parameter_bytes_local": float(shard_local_bytes),
        "shard_parameter_bytes_full": float(shard_local_bytes * tp_size),
        "local_parameter_bytes": float(replicated_bytes + shard_local_bytes),
        "full_model_parameter_bytes": float(replicated_bytes + shard_local_bytes * tp_size),
        "tp_size": float(tp_size),
    }
    if optimizer is not None:
        shard_ids = {id(parameter) for parameter in shard}
        replicated_state = 0
        shard_state_local = 0
        for parameter, state in optimizer.state.items():
            state_bytes = sum(
                value.numel() * value.element_size() for value in state.values() if torch.is_tensor(value)
            )
            if id(parameter) in shard_ids:
                shard_state_local += state_bytes
            else:
                replicated_state += state_bytes
        ledger["replicated_optimizer_state_bytes"] = float(replicated_state)
        ledger["shard_optimizer_state_bytes_local"] = float(shard_state_local)
        ledger["local_optimizer_state_bytes"] = float(replicated_state + shard_state_local)
    return ledger


def to_device_batch(batch, device: torch.device):
    """Move one host batch to the compute device (chapter-35 helper re-export)."""

    from mlp_tp import to_device_batch as _to_device_batch

    return _to_device_batch(batch, device)


def gpu_telemetry(rank: int) -> dict[str, float]:
    """Local-GPU power/temperature/clock (chapter-35 helper re-export)."""

    from mlp_tp import gpu_telemetry as _gpu_telemetry

    return _gpu_telemetry(rank)


def tp_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    host_batch,
    loss_scale: float,
    device: torch.device,
    max_grad_norm: float,
    tp_size: int,
    rank: int,
    collect_telemetry: bool = True,
    vocab_parallel: bool = True,
) -> dict[str, float]:
    """One optimizer step, dispatching the loss by arm.

    ``vocab_parallel=True`` (the ``vocab_tp_wN`` arm) computes the loss with
    the distributed cross entropy over SHARDED logits; ``False`` (the
    ``attn_tp_wN`` baseline arm) materializes full logits and uses the plain
    ``batch_loss`` — the memory those full logits occupy is precisely the
    variable under test, so both arms record ``local_logits_bytes`` and
    ``full_logits_bytes``.  Communication: embedding "g" + 2 block "g" ops
    per layer + 3 vocab-CE scalars on the forward side (vocab arm only), the
    2 block "f" ops + 2 q/k norm syncs per layer on the backward side, plus
    the single clip-norm scalar AllReduce.  The vocab-CE backward adds NO
    communication.
    """

    from mlp_tp import COMM_RECORDER, batch_loss

    model.train()
    optimizer.zero_grad(set_to_none=True)
    recorder = COMM_RECORDER
    recorder.reset()
    if device.type != "cuda":
        start = time.perf_counter()
        h2d_begin = time.perf_counter()
        batch = to_device_batch(host_batch, device)
        h2d_ms = (time.perf_counter() - h2d_begin) * 1000
        forward_begin = time.perf_counter()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        if vocab_parallel:
            loss = batch_loss_vocab_parallel(
                logits, batch.labels, model.lm_head.vocab_begin, tp_size, recorder=recorder
            ) * loss_scale
        else:
            loss = batch_loss(logits, batch.labels, "token_mean") * loss_scale
        forward_incl_ms = (time.perf_counter() - forward_begin) * 1000
        backward_begin = time.perf_counter()
        loss.backward()
        backward_incl_ms = (time.perf_counter() - backward_begin) * 1000
        comm = recorder.resolve()
        forward_ms = forward_incl_ms - comm["tp_allreduce_forward_ms"]
        backward_ms = backward_incl_ms - comm["tp_allreduce_backward_ms"]
        loss_value = float(loss.item())
        optimizer_begin = time.perf_counter()
        grad_norm = clip_grad_norm_all(model, max_grad_norm, tp_size)
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
        batch = to_device_batch(host_batch, device)
        events["h2d_end"].record()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        if vocab_parallel:
            loss = batch_loss_vocab_parallel(
                logits, batch.labels, model.lm_head.vocab_begin, tp_size, recorder=recorder
            ) * loss_scale
        else:
            loss = batch_loss(logits, batch.labels, "token_mean") * loss_scale
        events["forward_end"].record()
        loss.backward()
        events["backward_end"].record()
        # resolve() synchronizes the device once; the per-call event pairs of
        # the TP collectives are folded into comm totals here.
        comm = recorder.resolve()
        loss_value = float(loss.item())
        events["optimizer_start"].record()
        grad_norm = clip_grad_norm_all(model, max_grad_norm, tp_size)
        optimizer.step()
        events["optimizer_end"].record()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - wall_start) * 1000
        h2d_ms = float(events["h2d_start"].elapsed_time(events["h2d_end"]))
        forward_ms = float(events["h2d_end"].elapsed_time(events["forward_end"])) - comm["tp_allreduce_forward_ms"]
        backward_ms = float(events["forward_end"].elapsed_time(events["backward_end"])) - comm["tp_allreduce_backward_ms"]
        optimizer_ms = float(events["optimizer_start"].elapsed_time(events["optimizer_end"]))
    full_vocab_size = getattr(model.config, "vocab_size", None)
    if full_vocab_size is None:
        raise ValueError("the model must expose config.vocab_size for the logits memory accounting")
    element_size = logits.element_size()
    local_logits_bytes = float(logits.numel() * element_size)
    full_logits_bytes = float(logits.shape[0] * logits.shape[1] * full_vocab_size * element_size)
    metrics = {
        "h2d_time_ms": h2d_ms,
        "forward_time_ms": forward_ms,
        "backward_time_ms": backward_ms,
        "optimizer_time_ms": optimizer_ms,
        "step_time_ms": step_ms,
        "local_loss": loss_value,
        "global_loss": loss_value,  # data is replicated: the local loss IS the global loss
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(batch.physical_tokens),
        "non_padding_tokens": float(batch.non_padding_tokens),
        "supervised_tokens": float(batch.supervised_tokens),
        "padding_tokens": float(batch.padding_tokens),
        "padding_ratio": batch.padding_ratio,
        "batch_max_length": float(max(batch.lengths)),
        "samples": float(batch.batch_size),
        "local_logits_bytes": local_logits_bytes,
        "full_logits_bytes": full_logits_bytes,
        "logits_shard_ratio": local_logits_bytes / full_logits_bytes,
    }
    metrics.update(comm)
    total_comm_ms = comm["tp_allreduce_forward_ms"] + comm["tp_allreduce_backward_ms"]
    metrics["tp_allreduce_total_ms"] = total_comm_ms
    metrics["tp_allreduce_exposed_ratio"] = (
        total_comm_ms / metrics["step_time_ms"] if metrics["step_time_ms"] > 0 else 0.0
    )
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "physical_tokens_per_second": batch.physical_tokens / seconds,
        "non_padding_tokens_per_second": batch.non_padding_tokens / seconds,
        "supervised_tokens_per_second": batch.supervised_tokens / seconds,
    })
    if device.type == "cuda" and collect_telemetry:
        metrics.update(gpu_telemetry(rank))
    return metrics

