"""Chapter 40 core: non-interleaved 1F1B pipeline schedule over layer stages.

The only variable of this chapter: the pipeline SCHEDULE.  Model split (every
stage owns one contiguous chunk of decoder layers), stage count, micro-batch
count, P2P payloads and the tied Embedding/LM-Head handling are IDENTICAL to
chapter 24 — ``gpipe_step`` below is the chapter-39 fill-drain baseline kept
verbatim, and ``onef1b_step`` is the variant.  Weights are never replicated:
stage k owns exactly layers [start, end), stage 0 additionally owns the
Embedding, the last stage the final norm + LM Head.

Per micro-batch the P2P contract is unchanged (``dist.send``/``dist.recv``):

- forward: stage k sends its output activation ``[b, s, h]`` to stage k+1;
- backward: the LAST stage drives ``loss.backward()``; every other stage
  receives the input gradient of its output activation from the NEXT stage
  and seeds ``torch.autograd.backward(out, grad_tensors=g)`` with it;
- non-last stages then send their input activation's ``.grad`` to the
  previous stage.

LOSS SHARDS, PARAMETER UPDATE BOUNDARY, the tied-gradient exchange and the
exact global-norm clip are all inherited unchanged from chapter 24 (see
``exchange_tied_gradients`` / ``clip_grad_norm_pp`` / ``pp_train_step``).

THE SCHEDULE (``onef1b_step``): stage s runs

- warmup  ``w = min(num_stages - 1 - s, num_micro)`` forwards,
- steady  ``num_micro - w`` iterations of ONE forward followed by ONE
  backward (the 1F1B alternation),
- cooldown ``w`` backwards.

Every earlier stage starts backwards as soon as its first in-flight
micro-batch drains, so the number of simultaneously ALIVE activation graphs
per stage drops from ``num_micro`` (GPipe fill-drain) to
``min(num_stages - s, num_micro)`` — the residency hypothesis this chapter
measures.

OPERATION ORDER INSIDE AN ITERATION (deadlock note): a steady iteration runs
``[recv activation, forward, recv grad, backward, send grad, send
activation]`` — every blocking recv is posted before the iteration's sends.
With strictly blocking transports (gloo, used by the CPU smoke) a send that
outruns the peer's matching recv can close a send-send cycle between
neighbours; posting recvs first provably breaks that cycle here: a rank only
issues a downstream send after the grad recv of the same iteration, whose
producer could not have been blocked on this rank.  NCCL enqueues
asynchronously either way (stage 0 still runs ahead into later micro-batches
— the actual pipelining), so one code path serves both backends.  Message
order per rank pair per direction is unchanged by this reordering (activations
and gradients still flow in micro-batch order), so the chapter-39 P2P
protocol is preserved byte for byte.

``StageTimeline`` and ``compute_bubble`` are inherited unchanged from
chapter 24: forward/backward/P2P event spans per micro-batch with second
timestamps relative to the step start.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from mini_batch import batch_loss
from qwen3_model import (
    Qwen3Config,
    Qwen3DecoderLayer,
    RMSNorm,
    RotaryEmbedding,
    resolve_model_directory,
)


def split_layers(num_hidden_layers: int, num_stages: int) -> tuple[list[tuple[int, int]], dict]:
    """Contiguous, near-even layer ranges per stage.

    Divisible cases (28 layers / 2 stages -> [14, 14]; / 4 -> [7, 7, 7, 7])
    split exactly; otherwise the EARLIER stages take one extra layer
    (10 layers / 4 stages -> [3, 3, 2, 2]).  The remainder is recorded so
    results can state the imbalance instead of hiding it.  Identical to
    chapter 24: the schedule change must not touch the layer split.
    """

    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if num_stages > num_hidden_layers:
        raise ValueError(
            f"num_stages {num_stages} exceeds num_hidden_layers {num_hidden_layers}: a stage would hold no layer"
        )
    base, remainder = divmod(num_hidden_layers, num_stages)
    counts = [base + (1 if stage < remainder else 0) for stage in range(num_stages)]
    ranges: list[tuple[int, int]] = []
    start = 0
    for count in counts:
        ranges.append((start, start + count))
        start += count
    if start != num_hidden_layers:
        raise AssertionError(f"layer split covers {start} of {num_hidden_layers} layers")
    return ranges, {"uneven": remainder != 0, "stage_layer_counts": counts, "remainder_layers": remainder}


class StageTimeline:
    """Forward/backward/P2P event spans of one stage, in seconds from step start.

    GPU spans use CUDA Event pairs (all resolved by one final synchronize);
    CPU spans use ``perf_counter``.  ``end_step`` returns plain dicts ready
    for the timeline JSON.  Unchanged from chapter 24.
    """

    def __init__(self, device: torch.device) -> None:
        self.is_cuda = device.type == "cuda"
        self._entries: list[dict] = []
        self._step_start = None
        self.step_wall_ms = 0.0

    def start_step(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize()
            self._step_start = torch.cuda.Event(enable_timing=True)
            self._step_start.record()
        else:
            self._step_start = time.perf_counter()

    def begin(self, kind: str, micro: int | None = None, **info) -> int:
        if self.is_cuda:
            if self._step_start is None:
                # Lazy step origin: direct step-function usage (smoke /
                # correctness scripts) may never call start_step().
                self._step_start = torch.cuda.Event(enable_timing=True)
                self._step_start.record()
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            if self._step_start is None:
                self._step_start = time.perf_counter()
            start = time.perf_counter()
        self._entries.append({"kind": kind, "micro": micro, "start": start, "end": None, "info": info})
        return len(self._entries) - 1

    def end(self, handle: int) -> None:
        entry = self._entries[handle]
        if entry["end"] is not None:
            raise AssertionError(f"timeline event {entry['kind']} ended twice")
        if self.is_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            entry["end"] = end
        else:
            entry["end"] = time.perf_counter()

    def end_step(self) -> list[dict]:
        if self.is_cuda:
            torch.cuda.synchronize()
        records = []
        for entry in self._entries:
            if entry["end"] is None:
                raise AssertionError(f"timeline event {entry['kind']} never ended")
            if self.is_cuda:
                start_s = self._step_start.elapsed_time(entry["start"]) / 1000
                end_s = self._step_start.elapsed_time(entry["end"]) / 1000
            else:
                start_s = entry["start"] - self._step_start
                end_s = entry["end"] - self._step_start
            record = {
                "kind": entry["kind"],
                "micro": entry["micro"],
                "start_s": round(start_s, 6),
                "end_s": round(end_s, 6),
            }
            record.update(entry["info"])
            records.append(record)
        self._entries = []
        return records


def compute_bubble(events: list[dict]) -> dict:
    """Per-stage busy/bubble ratios from one stage's timeline events.

    Busy time sums the compute spans (forward/backward); P2P spans are
    reported separately because a blocking recv is idle-by-construction.
    The window is [first start, last end] of this stage's own step.
    Unchanged from chapter 24 so bubble numbers stay directly comparable.
    """

    if not events:
        return {"busy_s": 0.0, "window_s": 0.0, "bubble_ratio": 0.0, "p2p_s": 0.0}
    compute = [event for event in events if not event["kind"].startswith("p2p")]
    p2p = [event for event in events if event["kind"].startswith("p2p")]
    busy = sum(event["end_s"] - event["start_s"] for event in compute)
    p2p_total = sum(event["end_s"] - event["start_s"] for event in p2p)
    window = max(event["end_s"] for event in events) - min(event["start_s"] for event in events)
    return {
        "busy_s": round(busy, 6),
        "window_s": round(window, 6),
        "bubble_ratio": round(1.0 - busy / window, 6) if window > 0 else 0.0,
        "p2p_s": round(p2p_total, 6),
    }


def _send_activation(tensor: torch.Tensor, dst_stage: int, timeline: StageTimeline | None,
                     micro: int | None) -> None:
    payload = tensor.detach().contiguous()
    handle = None
    if timeline is not None:
        handle = timeline.begin("p2p_send", micro, peer_stage=dst_stage,
                                bytes=float(payload.numel() * payload.element_size()))
    dist.send(payload, dst=dst_stage)
    if handle is not None:
        timeline.end(handle)


def _recv_activation(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device,
                     src_stage: int, timeline: StageTimeline | None, micro: int | None) -> torch.Tensor:
    buffer = torch.empty(shape, dtype=dtype, device=device)
    handle = None
    if timeline is not None:
        handle = timeline.begin("p2p_recv", micro, peer_stage=src_stage,
                                bytes=float(buffer.numel() * buffer.element_size()))
    dist.recv(buffer, src=src_stage)
    if handle is not None:
        timeline.end(handle)
    return buffer


class PipelineStage(nn.Module):
    """One contiguous chunk of decoder layers plus the boundary pieces.

    Identical to chapter 24 (the schedule is the only variable): stage 0 owns
    ``embed_tokens``; the last stage owns ``norm`` and an UNTIED ``lm_head``
    loaded from the same safetensors tensor (the tied-embedding gradient
    exchange keeps the two copies on identical trajectories — see
    ``exchange_tied_gradients``).  State-dict keys are ``layers.{j}...`` with
    LOCAL layer indices; ``layer_range`` maps them to global layer numbers.
    """

    def __init__(self, config: Qwen3Config, stage_index: int, num_stages: int) -> None:
        super().__init__()
        ranges, split_meta = split_layers(config.num_hidden_layers, num_stages)
        self.config = config
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.layer_range = ranges[stage_index]
        self.split_meta = split_meta
        self.is_first = stage_index == 0
        self.is_last = stage_index == num_stages - 1
        layer_count = self.layer_range[1] - self.layer_range[0]
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(layer_count)]
        )
        if self.is_first:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if self.is_last:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    @property
    def prev_stage(self) -> int:
        if self.is_first:
            raise ValueError("stage 0 has no previous stage")
        return self.stage_index - 1

    @property
    def next_stage(self) -> int:
        if self.is_last:
            raise ValueError("the last stage has no next stage")
        return self.stage_index + 1

    def make_attention_mask(self, attention_mask, sequence_length, dtype, device):
        """The chapter-16 causal + key-padding mask (full sequence)."""

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

    def forward_layers(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                       position_ids: torch.Tensor = None) -> torch.Tensor:
        """Run this stage's layers over the FULL sequence (PP never shards seq)."""

        batch_size, sequence_length, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        causal_mask = self.make_attention_mask(
            attention_mask, sequence_length, hidden_states.dtype, hidden_states.device
        )
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, cosine, sine, causal_mask)
        return hidden_states

    def head_loss(self, hidden_states: torch.Tensor, labels: torch.Tensor, scale: float) -> torch.Tensor:
        """Final norm + LM Head + scaled token-mean loss (last stage only)."""

        if not self.is_last:
            raise ValueError("only the last stage can compute the loss")
        logits = self.lm_head(self.norm(hidden_states))
        return batch_loss(logits, labels, "token_mean") * scale

    def tied_parameters(self) -> dict[str, nn.Parameter]:
        """The two halves of Qwen3's tied Embedding/LM Head weight."""

        if self.is_first:
            return {"embed": self.embed_tokens.weight}
        if self.is_last:
            return {"lm_head": self.lm_head.weight}
        return {}


def load_pipeline_stage(model_directory, device: torch.device, stage_index: int,
                        num_stages: int, dtype: torch.dtype | None = None) -> PipelineStage:
    """Build one stage and read ONLY its own weights from safetensors.

    Unchanged from chapter 24: layer weights of other stages are never
    materialized; the Embedding is read by the first stage, the final norm by
    the last, and the tied LM Head weight by the last stage under its own
    name (Qwen3-0.6B stores it once as ``model.embed_tokens.weight``).
    """

    directory = str(resolve_model_directory(model_directory))
    config = Qwen3Config.from_model_directory(directory)
    stage = PipelineStage(config, stage_index, num_stages)

    weight_path = Path(directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    from safetensors import safe_open

    begin, end = stage.layer_range
    state: dict = {}
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("model.layers."):
                parts = key.split(".")
                layer_index = int(parts[2])
                if begin <= layer_index < end:
                    # Re-index to LOCAL layer positions: model.layers.{i}.x -> layers.{i-begin}.x
                    state["layers." + str(layer_index - begin) + key[len(f"model.layers.{layer_index}") :]] = handle.get_slice(key)[:]
                continue
            if key == "model.embed_tokens.weight":
                if stage.is_first:
                    state["embed_tokens.weight"] = handle.get_slice(key)[:]
                if stage.is_last and "lm_head.weight" not in handle.keys():
                    # Tied model: the last stage's LM Head shares the Embedding
                    # tensor; load it under the lm_head name (recorded below).
                    state["lm_head.weight"] = handle.get_slice(key)[:]
                continue
            if key == "model.norm.weight":
                if stage.is_last:
                    state["norm.weight"] = handle.get_slice(key)[:]
                continue
            if key == "lm_head.weight":
                if stage.is_last:
                    state["lm_head.weight"] = handle.get_slice(key)[:]
                continue
            raise ValueError(f"未预期的权重键 {key!r}：本期 Stage 划分只覆盖 layers/embed/norm/lm_head")

    load_result = stage.load_state_dict(state, strict=False)
    missing_expected: set[str] = set()
    if not stage.is_first:
        missing_expected.add("embed_tokens.weight")
    if not stage.is_last:
        missing_expected.update({"norm.weight", "lm_head.weight"})
    unexpected_missing = set(load_result.missing_keys) - missing_expected
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Stage 权重名称不匹配，missing="
            + str(sorted(unexpected_missing))
            + ", unexpected="
            + str(sorted(load_result.unexpected_keys))
        )

    stage_dtype = dtype or torch.float32
    rotary_inv_freq = stage.rotary_emb.inv_freq.clone()
    stage = stage.to(device=device, dtype=stage_dtype)
    stage.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    return stage


def _forward_micro(stage: PipelineStage, micro: dict, device: torch.device,
                   timeline: StageTimeline, micro_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One micro-batch through THIS stage; returns (input_holder, output).

    ``input_holder`` is the Embedding output (first stage) or the received
    activation re-rooted as a graph leaf (other stages) — its ``.grad`` is
    the input gradient this stage hands back to the previous stage.  The
    input recv is posted FIRST (before any compute), which is half of the
    recv-greedy ordering that keeps strictly blocking transports deadlock
    free; the OUTPUT send is issued by the caller so the 1F1B schedule can
    defer it behind the iteration's grad recv.
    """

    if stage.is_first:
        handle = timeline.begin("forward", micro_index, layers=f"{stage.layer_range[0]}:{stage.layer_range[1]}")
        holder = stage.embed_tokens(micro["input_ids"].to(device))
        # The Embedding output is a NON-LEAF tensor (an op result on the
        # weight parameter); retain_grad() is what makes ``holder.grad`` —
        # this stage's boundary input gradient, handed to no previous stage
        # but compared against the reference at layer ``begin`` — populate
        # during backward.  The Embedding weight's own gradient is unaffected.
        holder.retain_grad()
        out = stage.forward_layers(holder, micro["attention_mask"].to(device))
        timeline.end(handle)
        return holder, out
    shape = (micro["batch_size"], micro["sequence_length"], stage.config.hidden_size)
    holder_dtype = next(stage.parameters()).dtype  # FP32 in this chapter; the P2P payload dtype follows the stage
    holder = _recv_activation(shape, holder_dtype, device, stage.prev_stage, timeline, micro_index)
    holder.requires_grad_(True)
    handle = timeline.begin("forward", micro_index, layers=f"{stage.layer_range[0]}:{stage.layer_range[1]}")
    out = stage.forward_layers(holder, micro["attention_mask"].to(device))
    timeline.end(handle)
    return holder, out


def _backward_micro(out: torch.Tensor, grad_output: torch.Tensor | None,
                    timeline: StageTimeline, micro_index: int) -> None:
    handle = timeline.begin("backward", micro_index)
    if grad_output is None:
        out.backward()  # scalar loss on the last stage
    else:
        torch.autograd.backward(out, grad_tensors=grad_output)
    timeline.end(handle)


def schedule_1f1b(stage_index: int, num_stages: int, num_micro: int) -> dict:
    """Warmup/steady/cooldown split of the non-interleaved 1F1B schedule.

    Stage s runs ``min(num_stages - 1 - s, num_micro)`` warmup forwards, then
    alternates one forward with one backward for the remaining micro-batches,
    then drains ``warmup`` backwards.  ``in_flight_micros`` is the peak number
    of activation graphs alive on this stage: ``warmup`` after warmup, plus
    the one graph of the steady forward before its matching backward — bounded
    by ``min(num_stages - s, num_micro)`` and strictly below GPipe's
    ``num_micro`` whenever a steady phase exists.
    """

    if num_micro < 1:
        raise ValueError(f"num_micro must be >= 1, got {num_micro}")
    warmup = min(num_stages - 1 - stage_index, num_micro)
    steady = num_micro - warmup
    return {
        "warmup": warmup,
        "steady": steady,
        "cooldown": warmup,
        "in_flight_micros": min(num_stages - stage_index, num_micro),
    }


def onef1b_step(stage: PipelineStage, micro_batches: list[dict], device: torch.device,
                timeline: StageTimeline) -> dict:
    """Non-interleaved 1F1B: warmup forwards, alternating 1F1B, cooldown.

    Identical P2P contract and loss scaling as ``gpipe_step``; only the
    ORDER of the per-stage operations differs, which is the single variable.
    Steady iterations run ``recv activation -> forward -> recv grad ->
    backward -> send grad -> send activation`` (recv-greedy; see the module
    docstring for the deadlock argument).  Each stage keeps at most
    ``min(num_stages - stage_index, num_micro)`` activation graphs alive.
    """

    num_micro = len(micro_batches)
    plan = schedule_1f1b(stage.stage_index, stage.num_stages, num_micro)
    holders: list[torch.Tensor | None] = [None] * num_micro
    outputs: list[torch.Tensor | None] = [None] * num_micro
    losses: list[torch.Tensor | None] = [None] * num_micro

    def do_forward(micro_index: int, send_now: bool) -> None:
        holder, out = _forward_micro(stage, micro_batches[micro_index], device, timeline, micro_index)
        holders[micro_index] = holder
        outputs[micro_index] = out
        if stage.is_last:
            losses[micro_index] = stage.head_loss(
                out, micro_batches[micro_index]["labels"].to(device), micro_batches[micro_index]["scale"]
            )
        elif send_now:
            # Warmup: nothing to receive before the next forward, so the
            # output flies downstream immediately.
            _send_activation(out, stage.next_stage, timeline, micro_index)

    def do_backward(micro_index: int) -> None:
        if stage.is_last:
            _backward_micro(losses[micro_index], None, timeline, micro_index)
        else:
            grad = _recv_activation(outputs[micro_index].shape, outputs[micro_index].dtype,
                                    device, stage.next_stage, timeline, micro_index)
            _backward_micro(outputs[micro_index], grad, timeline, micro_index)
        if not stage.is_first:
            _send_activation(holders[micro_index].grad, stage.prev_stage, timeline, micro_index)

    # ---- warmup: forwards only ----
    for micro_index in range(plan["warmup"]):
        do_forward(micro_index, send_now=True)
    # ---- steady state: one forward, then one backward ----
    for offset in range(plan["steady"]):
        forward_index = plan["warmup"] + offset
        do_forward(forward_index, send_now=False)
        do_backward(offset)
        if not stage.is_last:
            # Deferred output send: issued after this iteration's grad recv
            # and backward so blocking transports cannot close a send/send
            # cycle with the next stage (NCCL is unaffected either way).
            _send_activation(outputs[forward_index], stage.next_stage, timeline, forward_index)
    # ---- cooldown: drain the remaining backwards ----
    for micro_index in range(plan["steady"], num_micro):
        do_backward(micro_index)

    # Per-rank loss shard: real summed loss on the last stage, all-zero
    # placeholder elsewhere (non-last stages have no loss; their backward is
    # seeded by the received input gradient).  ``input_grads`` are the
    # per-micro gradients at this stage's INPUT in micro order — exactly what
    # was sent to the previous stage; correctness scripts compare them against
    # the reference gradient at the stage boundary layer.
    loss_shard = float(sum(loss.detach() for loss in losses if loss is not None)) if stage.is_last else 0.0
    return {
        "local_loss_shard": loss_shard,
        "input_grads": [holder.grad.detach().clone() for holder in holders],
        "schedule": plan,
    }


def gpipe_step(stage: PipelineStage, micro_batches: list[dict], device: torch.device,
               timeline: StageTimeline) -> dict:
    """Chapter-24 baseline, kept verbatim: ALL forwards, then ALL backwards.

    Every micro-batch's graph stays alive through the fill phase — the
    activation-residency cost that ``onef1b_step`` (this chapter's variant)
    removes.  The fill phase's messages all flow towards the last stage and
    the drain phase's towards the first, each phase acyclic, so even strictly
    blocking transports cannot deadlock here.
    """

    holders: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []

    # ---- fill: continuous forwards ----
    for micro_index, micro in enumerate(micro_batches):
        holder, out = _forward_micro(stage, micro, device, timeline, micro_index)
        holders.append(holder)
        outputs.append(out)
        if stage.is_last:
            losses.append(stage.head_loss(out, micro["labels"].to(device), micro["scale"]))
        else:
            _send_activation(out, stage.next_stage, timeline, micro_index)

    # ---- drain: continuous backwards ----
    for micro_index in range(len(micro_batches)):
        if stage.is_last:
            _backward_micro(losses[micro_index], None, timeline, micro_index)
        else:
            grad = _recv_activation(outputs[micro_index].shape, outputs[micro_index].dtype,
                                    device, stage.next_stage, timeline, micro_index)
            _backward_micro(outputs[micro_index], grad, timeline, micro_index)
        if not stage.is_first:
            _send_activation(holders[micro_index].grad, stage.prev_stage, timeline, micro_index)

    loss_shard = float(sum(loss.detach() for loss in losses)) if stage.is_last else 0.0
    return {
        "local_loss_shard": loss_shard,
        "input_grads": [holder.grad.detach().clone() for holder in holders],
        "schedule": {"warmup": len(micro_batches), "steady": 0, "cooldown": len(micro_batches),
                     "in_flight_micros": float(len(micro_batches))},
    }


def exchange_tied_gradients(stage: PipelineStage, device: torch.device,
                            timeline: StageTimeline | None = None) -> dict:
    """P2P-sum the two partials of Qwen3's tied Embedding/LM Head weight.

    Unchanged from chapter 24: stage 0 owns the Embedding (lookup gradients),
    the last stage the LM Head (logit gradients); the tied reference update
    needs their SUM on BOTH copies so the two tensors stay bitwise identical
    across optimizer steps.  One tensor flies last -> first, one first -> last
    (in that order, so the blocking pairs cannot deadlock).
    """

    if stage.num_stages < 2:
        return {"tied_exchange_bytes": 0.0, "tied_exchange_count": 0.0}
    first_rank, last_rank = 0, stage.num_stages - 1
    moved_bytes = 0.0
    count = 0.0

    if stage.is_first:
        weight = stage.embed_tokens.weight
        grad = _recv_activation(tuple(weight.shape), weight.dtype, device, last_rank, timeline, None)
        moved_bytes += grad.numel() * grad.element_size()
        count += 1
        if weight.grad is None:
            weight.grad = grad.clone()
        else:
            weight.grad.add_(grad)
        _send_activation(weight.grad, last_rank, timeline, None)
        moved_bytes += weight.grad.numel() * weight.grad.element_size()
        count += 1
    elif stage.is_last:
        weight = stage.lm_head.weight
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)
        _send_activation(weight.grad, first_rank, timeline, None)
        moved_bytes += weight.grad.numel() * weight.grad.element_size()
        count += 1
        summed = _recv_activation(tuple(weight.shape), weight.dtype, device, first_rank, timeline, None)
        moved_bytes += summed.numel() * summed.element_size()
        count += 1
        weight.grad.copy_(summed)
    else:
        # Middle stages hold no tied weight; they only sync here so a
        # surrounding barrier stays collective-safe.
        pass
    return {"tied_exchange_bytes": moved_bytes, "tied_exchange_count": count}


def clip_grad_norm_pp(stage: PipelineStage, max_norm: float, world_size: int) -> torch.Tensor:
    """Exact global gradient norm across stages (one scalar AllReduce).

    Stages PARTITION the parameters, so the global squared norm is the sum of
    the per-stage sums — unlike TP, no ownership classes and no double-count
    risk.  Unchanged from chapter 24.
    """

    device = next(stage.parameters()).device
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in stage.parameters():
        if parameter.grad is not None:
            total_sq += parameter.grad.double().pow(2).sum()
    if world_size > 1:
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
    total_norm = torch.sqrt(total_sq)
    clip_coef = float(max_norm) / (float(total_norm.item()) + 1e-6)
    if clip_coef < 1.0:
        for parameter in stage.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coef)
    return total_norm


def pp_train_step(stage: PipelineStage, optimizer: torch.optim.Optimizer,
                  micro_batches: list[dict], device: torch.device, mode: str,
                  max_grad_norm: float) -> dict:
    """One optimizer step over all micro-batches in ``mode`` ('gpipe'|'onef1b').

    Order inside the step (identical for both modes except the schedule):
    zero_grad -> pipeline forwards/backwards (P2P in both phases) -> tied
    Embedding/LM Head gradient exchange (also P2P, timed in the timeline) ->
    exact global-norm clip -> AdamW.  ``global_loss`` is gathered once per
    step from the per-rank loss shards so every rank's metric record carries
    the same value.  Schedule fields (warmup/steady/cooldown and the
    theoretical in-flight count) travel with the metrics so the residency
    hypothesis can be checked against the measured peak memory.
    """

    if mode not in ("gpipe", "onef1b"):
        raise ValueError(f"unknown pipeline mode {mode!r}")
    timeline = StageTimeline(device)
    optimizer.zero_grad(set_to_none=True)
    timeline.start_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()

    if mode == "onef1b":
        result = onef1b_step(stage, micro_batches, device, timeline)
    else:
        result = gpipe_step(stage, micro_batches, device, timeline)
    tied = exchange_tied_gradients(stage, device, timeline)
    events = timeline.end_step()
    grad_norm = clip_grad_norm_pp(stage, max_grad_norm, stage.num_stages)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_ms = (time.perf_counter() - wall_start) * 1000

    # One tiny object gather per step keeps every rank's JSON self-contained.
    shards = [None] * stage.num_stages
    dist.all_gather_object(shards, result["local_loss_shard"])
    global_loss = float(sum(shards))

    p2p_events = [event for event in events if event["kind"].startswith("p2p")]
    metrics = {
        "step_time_ms": step_ms,
        "mode": mode,
        "stage_index": stage.stage_index,
        "num_stages": stage.num_stages,
        "stage_layer_range": f"{stage.layer_range[0]}:{stage.layer_range[1]}",
        "local_loss_shard": result["local_loss_shard"],
        "global_loss": global_loss,
        "gradient_norm_before_clip": float(grad_norm.item()),
        "p2p_count": float(len(p2p_events)),
        "p2p_bytes": float(sum(event["bytes"] for event in p2p_events)),
        "p2p_time_s": float(sum(event["end_s"] - event["start_s"] for event in p2p_events)),
        "tied_exchange_bytes": tied["tied_exchange_bytes"],
        "tied_exchange_count": tied["tied_exchange_count"],
        "micro_batches": float(len(micro_batches)),
        "warmup_micros": float(result["schedule"]["warmup"]),
        "steady_micros": float(result["schedule"]["steady"]),
        "cooldown_micros": float(result["schedule"]["cooldown"]),
        "in_flight_micros": float(result["schedule"]["in_flight_micros"]),
        "timeline": events,
        "bubble": compute_bubble(events),
        "physical_tokens": float(sum(m["physical_tokens"] for m in micro_batches)),
        "supervised_tokens": float(sum(m["supervised_tokens"] for m in micro_batches)),
    }
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "physical_tokens_per_second": metrics["physical_tokens"] / seconds,
        "supervised_tokens_per_second": metrics["supervised_tokens"] / seconds,
    })
    return metrics
