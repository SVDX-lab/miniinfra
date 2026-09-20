"""Chapter 41 core: Interleaved 1F1B over virtual (model-chunk) stages.

The only variable of this chapter: whether every physical rank owns ONE
contiguous pipeline stage (chapter 25's non-interleaved 1F1B, kept here as
``onef1b_step``) or SEVERAL non-contiguous VIRTUAL stages (``interleaved_1f1b_step``).
Physical GPU count, model, global batch and dtype are unchanged; what changes
is the stage table and the schedule over it.

PLACEMENT (the pipeline-parallel paper's interleaved layout): the model is
split into ``total_stages = num_physical * vpp`` virtual stages; virtual
stage ``s`` lives on physical rank ``s % num_physical`` as model chunk
``s // num_physical``.  Rank ``r`` therefore owns virtual stages
``{r, r + P, r + 2P, ...}`` — chunks whose LAYERS are non-contiguous, which
is what keeps every rank's compute balanced (e.g. 28 layers, P=4, vpp=2:
virtual stages 0..7 hold [4,4,4,4,3,3,3,3] layers, every rank 7).  The
divmod rule (earlier virtual stages take the extra layer) is recorded in the
split metadata instead of being hidden.

SCHEDULE (``interleaved_1f1b_step``): micro-batches are processed in groups
of ``num_physical`` (so ``num_micro % num_physical == 0`` is required).  The
iteration -> (chunk, micro-batch) maps follow the paper:

- forward  item ``i``: chunk ``(i % (P*vpp)) // P``,
  micro-batch ``(i // (P*vpp)) * P + (i % P)``;
- backward item ``i``: chunk ``vpp - 1 - (i % (P*vpp)) // P``, same
  micro-batch formula (chunks run backwards in reverse order).

Stage ``s`` first runs ``min(2*(P-1-r) + (vpp-1)*P, vpp*M)`` warmup forwards
(r = its physical rank), then alternates one forward with one backward, then
drains the remaining backwards.  Per physical rank the executed kind
sequence is still ``F×warmup + (F,B)×steady + B×cooldown``.

P2P TOPOLOGY: with this placement the predecessor/successor of a virtual
stage is owned by ``owner(s) +- 1 (mod P)`` — the neighbour link for
within-chunk edges and the WRAPPED pair ``P-1 <-> 0`` at every chunk
boundary, so peers are no longer only adjacent ranks.  The message count
follows the stage count: ``vpp*M`` activations plus as many gradients per
rank pair direction set over ``total_stages - 1 = P*vpp - 1`` edges, versus
``(P-1)*M`` over ``P-1`` edges for the non-interleaved arm — the same
``[b, s, h]`` payload flies more times per step, which is one of the costs
this chapter measures.  ``pass_activation`` / ``receive_activation`` handle
BOTH transports: remote stages use blocking ``dist.send``/``dist.recv`` (the
chapter-39/25 contract, payload and micro-batch order unchanged per pair and
direction); same-owner stages take a local mailbox handoff with no
communication — never exercised by the strided placement above (adjacent
virtual stages never share an owner), but kept so the placement table stays
the single source of truth, and unit-checked in the smoke test.

DEADLOCK NOTE (why sends are ASYNC here): the interleaved topology closes
communication cycles that the linear chapter-39/25 pipelines cannot form.
With strictly blocking transports, execution of the interleaved schedule on
P=2, vpp=2 deadlocks in a recv-recv cycle: rank A blocks receiving an
activation whose producer rank B deferred behind its own backward, while B
blocks receiving the very gradient that only A's backward can produce — the
chunk structure makes each rank's forward of its NEXT chunk depend on its
NEIGHBOUR's backward of the CURRENT one.  Megatron survives this on NCCL
because point-to-point ops enqueue asynchronously; on a strict transport the
honest fix is ``dist.isend``: sends are POSTED (never blocking the CPU
thread) and waited in one drain at the end of the step, so every blocking
wait is a recv whose chain terminates at the Embedding forward (no recv) or
the loss backward (no recv) — provably cycle-free.  Payloads, message order
per pair and direction, and the blocking ``dist.recv`` side are unchanged
from chapters 24/25; the timeline records each send from its posting to the
end-of-step drain, i.e. the true exposed communication.

TIED EMBEDDING/LM HEAD and the parameter update boundary are unchanged: the
Embedding lives on virtual stage 0 (rank 0, chunk 0), the LM Head on virtual
stage ``total_stages - 1`` (rank P-1, chunk vpp-1); after ALL backwards the
two copies exchange partial gradients over the PHYSICAL first/last ranks and
every rank clips with the exact global norm (one scalar AllReduce) before
its AdamW step.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from mini_batch import batch_loss
from qwen3_model import (
    Qwen3Config,
    Qwen3DecoderLayer,
    RMSNorm,
    RotaryEmbedding,
    resolve_model_directory,
)


def split_virtual_stages(num_hidden_layers: int, total_stages: int) -> tuple[list[tuple[int, int]], dict]:
    """Contiguous, near-even layer ranges per VIRTUAL stage.

    Same remainder rule as chapters 24/25 (earlier virtual stages take one
    extra layer: 28 layers / 8 stages -> [4,4,4,4,3,3,3,3]) so the imbalance
    is recorded, never hidden.  Under the strided placement a physical rank
    owns virtual stages ``{r, r+P, ...}``; with ``vpp = 2`` and ``remainder
    <= P`` every rank still ends up with the same layer count even though
    its chunks are non-contiguous.
    """

    if total_stages < 1:
        raise ValueError(f"total_stages must be >= 1, got {total_stages}")
    if total_stages > num_hidden_layers:
        raise ValueError(
            f"total_stages {total_stages} exceeds num_hidden_layers {num_hidden_layers}: a virtual stage would hold no layer"
        )
    base, remainder = divmod(num_hidden_layers, total_stages)
    counts = [base + (1 if stage < remainder else 0) for stage in range(total_stages)]
    ranges: list[tuple[int, int]] = []
    start = 0
    for count in counts:
        ranges.append((start, start + count))
        start += count
    if start != num_hidden_layers:
        raise AssertionError(f"layer split covers {start} of {num_hidden_layers} layers")
    return ranges, {"uneven": remainder != 0, "stage_layer_counts": counts, "remainder_layers": remainder}


def stage_owner(virtual_index: int, num_physical: int) -> int:
    """Physical rank of a virtual stage (the strided placement: ``s % P``)."""

    return virtual_index % num_physical


def stage_chunk(virtual_index: int, num_physical: int) -> int:
    """Local model-chunk id of a virtual stage on its owner (``s // P``)."""

    return virtual_index // num_physical


class _LocalMailbox:
    """Direct tensor handoff between virtual stages on the SAME physical rank.

    The strided placement never requests it (adjacent virtual stages never
    share an owner), but the transport layer supports it so the placement
    table stays the single source of truth; the smoke test round-trips it.
    """

    def __init__(self) -> None:
        self._slots: dict[tuple[int, int, int], torch.Tensor] = {}

    def put(self, tensor: torch.Tensor, src_virtual: int, dst_virtual: int, micro: int) -> None:
        key = (src_virtual, dst_virtual, micro)
        if key in self._slots:
            raise AssertionError(f"local mailbox slot {key} written twice")
        self._slots[key] = tensor

    def take(self, src_virtual: int, dst_virtual: int, micro: int) -> torch.Tensor:
        key = (src_virtual, dst_virtual, micro)
        if key not in self._slots:
            raise AssertionError(f"local mailbox slot {key} read before write")
        return self._slots.pop(key)


_MAILBOX = _LocalMailbox()


class StageTimeline:
    """Forward/backward/P2P event spans of one rank, in seconds from step start.

    Identical to chapters 24/25 (CUDA Event pairs on GPU, ``perf_counter`` on
    CPU); events additionally carry ``chunk`` / ``virtual_stage`` info so the
    interleaved timeline stays readable per virtual stage.
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
    """Per-rank busy/bubble ratios from timeline events (unchanged from 24/25).

    Busy time sums the compute spans (forward/backward); P2P spans are
    reported separately; local handoffs count as neither (zero-cost by
    construction).  The window is [first start, last end] of this rank's step.
    """

    if not events:
        return {"busy_s": 0.0, "window_s": 0.0, "bubble_ratio": 0.0, "p2p_s": 0.0}
    compute = [event for event in events if event["kind"] in ("forward", "backward")]
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


def pass_activation(tensor: torch.Tensor, src_virtual: int, dst_virtual: int,
                    num_physical: int, timeline: StageTimeline | None, micro: int | None,
                    pending_sends: list | None = None) -> None:
    """Hand an activation (or gradient) to the NEXT virtual stage's owner.

    Same owner -> local mailbox handoff (recorded as ``local_handoff``,
    zero communication); different owner -> ``dist.isend`` POSTED
    asynchronously (recorded as ``p2p_send`` from posting to the step-end
    drain).  Async sends are REQUIRED on strictly blocking transports: the
    interleaved topology closes recv-recv cycles that blocking sends cannot
    break (see the module docstring); on NCCL the enqueue is asynchronous
    either way.  ``pending_sends`` collects ``(work, timeline_handle)``
    pairs for the step-end drain; ``None`` falls back to a blocking
    ``dist.send`` for standalone callers (the tied-gradient exchange is a
    single ordered pair and cannot cycle).  Payloads are detached contiguous
    tensors, in micro-batch order per pair and direction, and must not be
    written until the drain completes — the step functions drain before any
    gradient is mutated (clip / optimizer).
    """

    payload = tensor.detach().contiguous()
    src_rank, dst_rank = stage_owner(src_virtual, num_physical), stage_owner(dst_virtual, num_physical)
    if src_rank == dst_rank:
        handle = None
        if timeline is not None:
            handle = timeline.begin("local_handoff", micro, src_stage=src_virtual, dst_stage=dst_virtual,
                                    bytes=float(payload.numel() * payload.element_size()))
        _MAILBOX.put(payload, src_virtual, dst_virtual, -1 if micro is None else micro)
        if handle is not None:
            timeline.end(handle)
        return
    handle = None
    if timeline is not None:
        handle = timeline.begin("p2p_send", micro, peer_stage=dst_rank, src_stage=src_virtual, dst_stage=dst_virtual,
                                bytes=float(payload.numel() * payload.element_size()))
    if pending_sends is None:
        dist.send(payload, dst=dst_rank)
        if handle is not None:
            timeline.end(handle)
    else:
        work = dist.isend(payload, dst=dst_rank)
        pending_sends.append((work, handle))


def wait_p2p_sends(pending_sends: list, timeline: StageTimeline | None) -> None:
    """Drain all posted async sends, closing their timeline spans."""

    for work, handle in pending_sends:
        work.wait()
        if handle is not None and timeline is not None:
            timeline.end(handle)
    pending_sends.clear()


def receive_activation(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device,
                       src_virtual: int, dst_virtual: int, num_physical: int,
                       timeline: StageTimeline | None, micro: int | None) -> torch.Tensor:
    """Receive an activation (or gradient) from the PREVIOUS virtual stage.

    Remote: blocking ``dist.recv`` into a fresh buffer (``p2p_recv`` span);
    local: the mailbox tensor produced by ``pass_activation`` on this rank
    (``local_handoff`` span).  The recv is always posted BEFORE the receiving
    stage's other communication in the same schedule iteration — the
    recv-greedy ordering that keeps strictly blocking transports deadlock
    free (see module docstring).
    """

    src_rank, dst_rank = stage_owner(src_virtual, num_physical), stage_owner(dst_virtual, num_physical)
    if src_rank == dst_rank:
        payload_bytes = 1
        for dimension in shape:
            payload_bytes *= dimension
        payload_bytes *= torch.empty((), dtype=dtype).element_size()
        handle = None
        if timeline is not None:
            handle = timeline.begin("local_handoff", micro, src_stage=src_virtual, dst_stage=dst_virtual,
                                    bytes=float(payload_bytes))
        tensor = _MAILBOX.take(src_virtual, dst_virtual, -1 if micro is None else micro)
        if handle is not None:
            timeline.end(handle)
        return tensor
    buffer = torch.empty(shape, dtype=dtype, device=device)
    handle = None
    if timeline is not None:
        handle = timeline.begin("p2p_recv", micro, peer_stage=src_rank, src_stage=src_virtual, dst_stage=dst_virtual,
                                bytes=float(buffer.numel() * buffer.element_size()))
    dist.recv(buffer, src=src_rank)
    if handle is not None:
        timeline.end(handle)
    return buffer


class VirtualStage(nn.Module):
    """One virtual stage: a chunk of decoder layers plus the boundary pieces.

    ``virtual_index`` runs over ``[0, total_stages)``; virtual stage 0 owns
    ``embed_tokens``, the last one owns ``norm`` and an UNTIED ``lm_head``
    (same tied-handling as chapters 24/25).  ``num_physical`` fixes the
    placement: this module knows its owner rank and local chunk id but owns
    no communication itself — the step functions drive ``pass_activation`` /
    ``receive_activation``.  State-dict keys are ``layers.{j}...`` with LOCAL
    layer indices; ``layer_range`` maps them to global layer numbers.
    """

    def __init__(self, config: Qwen3Config, virtual_index: int, total_stages: int,
                 num_physical: int) -> None:
        super().__init__()
        ranges, split_meta = split_virtual_stages(config.num_hidden_layers, total_stages)
        if num_physical < 1 or total_stages % num_physical != 0:
            raise ValueError(
                f"total_stages {total_stages} must be a positive multiple of num_physical {num_physical}"
            )
        self.config = config
        self.virtual_index = virtual_index
        self.total_stages = total_stages
        self.num_physical = num_physical
        self.owner_rank = stage_owner(virtual_index, num_physical)
        self.local_chunk = stage_chunk(virtual_index, num_physical)
        self.layer_range = ranges[virtual_index]
        self.split_meta = split_meta
        self.is_first = virtual_index == 0
        self.is_last = virtual_index == total_stages - 1
        layer_count = self.layer_range[1] - self.layer_range[0]
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(layer_count)])
        if self.is_first:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if self.is_last:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    @property
    def prev_virtual(self) -> int:
        if self.is_first:
            raise ValueError("virtual stage 0 has no previous stage")
        return self.virtual_index - 1

    @property
    def next_virtual(self) -> int:
        if self.is_last:
            raise ValueError("the last virtual stage has no next stage")
        return self.virtual_index + 1

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
        """Run this chunk's layers over the FULL sequence (PP never shards seq)."""

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
        """Final norm + LM Head + scaled token-mean loss (last virtual stage only)."""

        if not self.is_last:
            raise ValueError("only the last virtual stage can compute the loss")
        logits = self.lm_head(self.norm(hidden_states))
        return batch_loss(logits, labels, "token_mean") * scale


def load_pipeline_model(model_directory, device: torch.device, rank: int, num_physical: int,
                        vpp: int, dtype: torch.dtype | None = None) -> list[VirtualStage]:
    """Build THIS rank's ``vpp`` virtual stages, reading only their weights.

    Rank ``r`` owns virtual stages ``{r, r+P, ..., r+(vpp-1)P}`` (strided
    placement); only those layers are materialized from safetensors.  The
    Embedding is read by rank 0 (chunk 0), the final norm and the tied LM
    Head weight by rank ``P-1`` (its last chunk), mirroring chapters 24/25.
    """

    if vpp < 1:
        raise ValueError(f"vpp must be >= 1, got {vpp}")
    directory = str(resolve_model_directory(model_directory))
    config = Qwen3Config.from_model_directory(directory)
    total_stages = num_physical * vpp
    chunks = [VirtualStage(config, rank + chunk * num_physical, total_stages, num_physical)
              for chunk in range(vpp)]

    weight_path = Path(directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    from safetensors import safe_open

    wanted: set[int] = set()
    for chunk in chunks:
        wanted.update(range(chunk.layer_range[0], chunk.layer_range[1]))

    state: dict = {}
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        for key in keys:
            if key.startswith("model.layers."):
                layer_index = int(key.split(".")[2])
                if layer_index not in wanted:
                    continue
                chunk = next(item for item in chunks if item.layer_range[0] <= layer_index < item.layer_range[1])
                begin = chunk.layer_range[0]
                state[f"chunk{chunk.local_chunk}.layers." + str(layer_index - begin)
                      + key[len(f"model.layers.{layer_index}") :]] = handle.get_slice(key)[:]
                continue
            if key == "model.embed_tokens.weight":
                if rank == 0:
                    state["chunk0.embed_tokens.weight"] = handle.get_slice(key)[:]
                if rank == num_physical - 1 and "lm_head.weight" not in keys:
                    # Tied model: the last rank's LM Head shares the Embedding
                    # tensor; load it under the lm_head name.
                    state[f"chunk{vpp - 1}.lm_head.weight"] = handle.get_slice(key)[:]
                continue
            if key == "model.norm.weight":
                if rank == num_physical - 1:
                    state[f"chunk{vpp - 1}.norm.weight"] = handle.get_slice(key)[:]
                continue
            if key == "lm_head.weight":
                if rank == num_physical - 1:
                    state[f"chunk{vpp - 1}.lm_head.weight"] = handle.get_slice(key)[:]
                continue
            raise ValueError(f"未预期的权重键 {key!r}：本期 Stage 划分只覆盖 layers/embed/norm/lm_head")

    # Names above are prefixed per chunk; load through each chunk's own
    # submodule so state-dict keys stay ``layers.{j}...`` locally.
    for chunk in chunks:
        prefix = f"chunk{chunk.local_chunk}."
        chunk_state = {
            name[len(prefix):]: value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        load_result = chunk.load_state_dict(chunk_state, strict=False)
        missing_expected: set[str] = set()
        if not chunk.is_first:
            missing_expected.add("embed_tokens.weight")
        if not chunk.is_last:
            missing_expected.update({"norm.weight", "lm_head.weight"})
        unexpected_missing = set(load_result.missing_keys) - missing_expected
        if unexpected_missing or load_result.unexpected_keys:
            raise RuntimeError(
                f"虚拟 Stage {chunk.virtual_index} 权重名称不匹配，missing="
                + str(sorted(unexpected_missing))
                + ", unexpected="
                + str(sorted(load_result.unexpected_keys))
            )

    stage_dtype = dtype or torch.float32
    for chunk in chunks:
        rotary_inv_freq = chunk.rotary_emb.inv_freq.clone()
        chunk.to(device=device, dtype=stage_dtype)
        chunk.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    return chunks


def _forward_chunk(chunk: VirtualStage, micro: dict, device: torch.device,
                   timeline: StageTimeline, micro_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One micro-batch through THIS virtual stage; recv is posted first.

    Returns ``(input_holder, output)``; the holder's ``.grad`` is the input
    gradient handed back to the previous virtual stage.  The input recv (a
    blocking one for remote stages) is posted BEFORE any compute — the
    recv-greedy ordering; the output send is issued by the caller.
    """

    layers_label = f"{chunk.layer_range[0]}:{chunk.layer_range[1]}"
    if chunk.is_first:
        handle = timeline.begin("forward", micro_index, chunk=chunk.local_chunk, virtual_stage=chunk.virtual_index, layers=layers_label)
        holder = chunk.embed_tokens(micro["input_ids"].to(device))
        # The Embedding output is a NON-LEAF tensor (an op result on the
        # weight parameter); retain_grad() is what makes ``holder.grad`` —
        # this stage's boundary input gradient — populate during backward.
        # The Embedding weight's own gradient is unaffected.
        holder.retain_grad()
        out = chunk.forward_layers(holder, micro["attention_mask"].to(device))
        timeline.end(handle)
        return holder, out
    shape = (micro["batch_size"], micro["sequence_length"], chunk.config.hidden_size)
    holder_dtype = next(chunk.parameters()).dtype  # FP32 in this chapter; the payload dtype follows the stage
    holder = receive_activation(shape, holder_dtype, device, chunk.prev_virtual, chunk.virtual_index,
                                chunk.num_physical, timeline, micro_index)
    holder.requires_grad_(True)
    handle = timeline.begin("forward", micro_index, chunk=chunk.local_chunk, virtual_stage=chunk.virtual_index, layers=layers_label)
    out = chunk.forward_layers(holder, micro["attention_mask"].to(device))
    timeline.end(handle)
    return holder, out


def _backward_chunk(out: torch.Tensor, grad_output: torch.Tensor | None,
                    timeline: StageTimeline, micro_index: int, chunk: VirtualStage) -> None:
    handle = timeline.begin("backward", micro_index, chunk=chunk.local_chunk, virtual_stage=chunk.virtual_index)
    if grad_output is None:
        out.backward()  # scalar loss on the last virtual stage
    else:
        torch.autograd.backward(out, grad_tensors=grad_output)
    timeline.end(handle)


def interleaved_schedule(rank: int, num_physical: int, vpp: int, num_micro: int) -> dict:
    """Warmup/steady/cooldown split of the interleaved 1F1B schedule.

    Micro-batches are grouped by ``num_physical`` (the group size of the
    paper's schedule), so ``num_micro % num_physical == 0`` is REQUIRED —
    violated configurations raise instead of silently reordering.  Rank r
    runs ``min(2*(P-1-r) + (vpp-1)*P, vpp*M)`` warmup forward items, then
    alternates forward/backward items, then drains the backwards; the
    in-flight count (warmup graphs plus the steady forward's graph) is
    reported so the activation-residency cost of interleaving is explicit.
    """

    if vpp < 2:
        raise ValueError(f"interleaved schedule needs vpp >= 2, got {vpp}")
    if num_micro % num_physical != 0:
        raise ValueError(
            f"num_micro {num_micro} must be divisible by num_physical {num_physical} "
            "(the interleaved schedule processes micro-batches in groups of the pipeline size)"
        )
    total_items = vpp * num_micro
    warmup = min((num_physical - rank - 1) * 2 + (vpp - 1) * num_physical, total_items)
    steady = total_items - warmup
    return {
        "warmup": warmup,
        "steady": steady,
        "cooldown": warmup,
        "total_items": total_items,
        "in_flight_micros": warmup + (1 if steady > 0 else 0),
        "micro_group_size": num_physical,
    }


def interleaved_1f1b_step(chunks: list[VirtualStage], micro_batches: list[dict], device: torch.device,
                          timeline: StageTimeline) -> dict:
    """Interleaved 1F1B over this rank's virtual stages (the chapter variant).

    Iteration -> (chunk, micro) follows the paper's maps (``interleaved_schedule``);
    steady iterations run ``recv activation -> forward -> recv grad ->
    backward -> send grad -> send activation`` (recv-greedy; see module
    docstring).  Only the schedule and the placement differ from
    ``onef1b_step``: payloads, loss scaling and the update boundary are the
    chapter-39/25 contract.
    """

    num_chunks = len(chunks)
    if num_chunks < 2:
        raise ValueError("interleaved_1f1b_step needs at least 2 virtual stages per rank; use onef1b_step")
    rank = chunks[0].owner_rank
    num_physical = chunks[0].num_physical
    num_micro = len(micro_batches)
    plan = interleaved_schedule(rank, num_physical, num_chunks, num_micro)
    pending_sends: list = []  # async sends, drained at the end of the step

    holders: list[list[torch.Tensor | None]] = [[None] * num_micro for _ in range(num_chunks)]
    outputs: list[list[torch.Tensor | None]] = [[None] * num_micro for _ in range(num_chunks)]
    losses: list[list[torch.Tensor | None]] = [[None] * num_micro for _ in range(num_chunks)]

    def forward_item(index: int) -> tuple[int, int]:
        group = num_physical * num_chunks
        chunk = (index % group) // num_physical
        micro = (index // group) * num_physical + (index % num_physical)
        return chunk, micro

    def backward_item(index: int) -> tuple[int, int]:
        group = num_physical * num_chunks
        chunk = num_chunks - 1 - ((index % group) // num_physical)
        micro = (index // group) * num_physical + (index % num_physical)
        return chunk, micro

    def do_forward(chunk_id: int, micro_index: int) -> None:
        chunk = chunks[chunk_id]
        holder, out = _forward_chunk(chunk, micro_batches[micro_index], device, timeline, micro_index)
        holders[chunk_id][micro_index] = holder
        outputs[chunk_id][micro_index] = out
        if chunk.is_last:
            losses[chunk_id][micro_index] = chunk.head_loss(
                out, micro_batches[micro_index]["labels"].to(device), micro_batches[micro_index]["scale"]
            )
        else:
            # EVERY output is sent IMMEDIATELY (async, so posting never
            # blocks): the receiver's recv schedule interleaves its forward
            # activations with its backward gradients in a fixed per-pair
            # FIFO order, and deferring a send would silently reorder that
            # stream — the wrong tensor would be matched to the wrong slot.
            pass_activation(out, chunk.virtual_index, chunk.next_virtual, num_physical, timeline, micro_index,
                            pending_sends)

    def do_backward(chunk_id: int, micro_index: int) -> None:
        chunk = chunks[chunk_id]
        if chunk.is_last:
            _backward_chunk(losses[chunk_id][micro_index], None, timeline, micro_index, chunk)
        else:
            grad = receive_activation(outputs[chunk_id][micro_index].shape, outputs[chunk_id][micro_index].dtype,
                                      device, chunk.next_virtual, chunk.virtual_index, num_physical,
                                      timeline, micro_index)
            _backward_chunk(outputs[chunk_id][micro_index], grad, timeline, micro_index, chunk)
        if not chunk.is_first:
            pass_activation(holders[chunk_id][micro_index].grad, chunk.virtual_index, chunk.prev_virtual,
                            num_physical, timeline, micro_index, pending_sends)

    # ---- warmup: forward items only ----
    for index in range(plan["warmup"]):
        chunk_id, micro_index = forward_item(index)
        do_forward(chunk_id, micro_index)
    # ---- steady state: one forward item, then one backward item ----
    for offset in range(plan["steady"]):
        forward_chunk, forward_micro = forward_item(plan["warmup"] + offset)
        backward_chunk, backward_micro = backward_item(offset)
        do_forward(forward_chunk, forward_micro)
        do_backward(backward_chunk, backward_micro)
    # ---- cooldown: drain the remaining backward items ----
    for index in range(plan["steady"], plan["total_items"]):
        chunk_id, micro_index = backward_item(index)
        do_backward(chunk_id, micro_index)

    # All schedule sends must be COMPLETE before any gradient is mutated by
    # the clip / optimizer below (async payloads alias those tensors).
    wait_p2p_sends(pending_sends, timeline)

    loss_shard = 0.0
    if chunks[-1].is_last:
        loss_shard = float(sum(loss.detach() for row in losses for loss in row if loss is not None))
    return {
        "local_loss_shard": loss_shard,
        "input_grads": {
            f"chunk{chunk_id}_micro{micro_index}": holders[chunk_id][micro_index].grad.detach().clone()
            for chunk_id in range(num_chunks) for micro_index in range(num_micro)
            if not chunks[chunk_id].is_first
        },
        "schedule": plan,
    }


def onef1b_schedule(rank: int, num_physical: int, num_micro: int) -> dict:
    """The chapter-40 non-interleaved 1F1B split (one stage per rank)."""

    if num_micro < 1:
        raise ValueError(f"num_micro must be >= 1, got {num_micro}")
    warmup = min(num_physical - 1 - rank, num_micro)
    steady = num_micro - warmup
    return {
        "warmup": warmup,
        "steady": steady,
        "cooldown": warmup,
        "total_items": num_micro,
        "in_flight_micros": min(num_physical - rank, num_micro),
        "micro_group_size": 1,
    }


def onef1b_step(chunks: list[VirtualStage], micro_batches: list[dict], device: torch.device,
                timeline: StageTimeline) -> dict:
    """Non-interleaved 1F1B baseline: ONE virtual stage per physical rank.

    The chapter-40 schedule over the shared transport layer; requires exactly
    one chunk per rank (total_stages == num_physical).  The benchmark's
    ``onefoone_ppN`` arm runs this; the interleaved arm swaps in
    ``interleaved_1f1b_step`` with the same weights per rank.
    """

    if len(chunks) != 1:
        raise ValueError("onef1b_step needs exactly one virtual stage per rank")
    chunk = chunks[0]
    if chunk.total_stages != chunk.num_physical:
        raise ValueError(
            f"onef1b_step requires total_stages == num_physical (got {chunk.total_stages} vs {chunk.num_physical}); "
            "this is the non-interleaved baseline"
        )
    rank = chunk.owner_rank
    num_physical = chunk.num_physical
    num_micro = len(micro_batches)
    plan = onef1b_schedule(rank, num_physical, num_micro)
    pending_sends: list = []  # async sends, drained at the end of the step

    holders: list[torch.Tensor | None] = [None] * num_micro
    outputs: list[torch.Tensor | None] = [None] * num_micro
    losses: list[torch.Tensor | None] = [None] * num_micro

    def do_forward(micro_index: int) -> None:
        holder, out = _forward_chunk(chunk, micro_batches[micro_index], device, timeline, micro_index)
        holders[micro_index] = holder
        outputs[micro_index] = out
        if chunk.is_last:
            losses[micro_index] = chunk.head_loss(
                out, micro_batches[micro_index]["labels"].to(device), micro_batches[micro_index]["scale"]
            )
        else:
            pass_activation(out, chunk.virtual_index, chunk.next_virtual, num_physical, timeline, micro_index,
                            pending_sends)

    def do_backward(micro_index: int) -> None:
        if chunk.is_last:
            _backward_chunk(losses[micro_index], None, timeline, micro_index, chunk)
        else:
            grad = receive_activation(outputs[micro_index].shape, outputs[micro_index].dtype,
                                      device, chunk.next_virtual, chunk.virtual_index, num_physical,
                                      timeline, micro_index)
            _backward_chunk(outputs[micro_index], grad, timeline, micro_index, chunk)
        if not chunk.is_first:
            pass_activation(holders[micro_index].grad, chunk.virtual_index, chunk.prev_virtual,
                            num_physical, timeline, micro_index, pending_sends)

    for micro_index in range(plan["warmup"]):
        do_forward(micro_index)
    for offset in range(plan["steady"]):
        do_forward(plan["warmup"] + offset)
        do_backward(offset)
    for micro_index in range(plan["steady"], num_micro):
        do_backward(micro_index)

    # Async payloads alias the gradient tensors the clip / optimizer mutate.
    wait_p2p_sends(pending_sends, timeline)

    loss_shard = float(sum(loss.detach() for loss in losses if loss is not None)) if chunk.is_last else 0.0
    return {
        "local_loss_shard": loss_shard,
        "input_grads": {
            f"chunk0_micro{micro_index}": holders[micro_index].grad.detach().clone()
            for micro_index in range(num_micro) if not chunk.is_first
        },
        "schedule": plan,
    }


def exchange_tied_gradients(chunks: list[VirtualStage], device: torch.device,
                            timeline: StageTimeline | None = None) -> dict:
    """P2P-sum the two partials of Qwen3's tied Embedding/LM Head weight.

    The Embedding lives on rank 0 (chunk 0), the LM Head on rank ``P-1``
    (its last chunk); the tied reference update needs their SUM on BOTH
    copies.  One tensor flies last -> first, one first -> last (in that
    order, so the blocking pairs cannot deadlock) — the chapter-39/25
    exchange over PHYSICAL ranks, unchanged.
    """

    num_physical = chunks[0].num_physical
    if num_physical < 2:
        return {"tied_exchange_bytes": 0.0, "tied_exchange_count": 0.0}
    rank = chunks[0].owner_rank
    first_rank, last_rank = 0, num_physical - 1
    moved_bytes = 0.0
    count = 0.0

    if rank == first_rank:
        weight = chunks[0].embed_tokens.weight
        grad = receive_activation(tuple(weight.shape), weight.dtype, device,
                                  num_physical - 1, 0, num_physical, timeline, None)
        moved_bytes += grad.numel() * grad.element_size()
        count += 1
        if weight.grad is None:
            weight.grad = grad.clone()
        else:
            weight.grad.add_(grad)
        pass_activation(weight.grad, 0, num_physical - 1, num_physical, timeline, None)
        moved_bytes += weight.grad.numel() * weight.grad.element_size()
        count += 1
    elif rank == last_rank:
        weight = chunks[-1].lm_head.weight
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)
        pass_activation(weight.grad, chunks[-1].virtual_index, 0, num_physical, timeline, None)
        moved_bytes += weight.grad.numel() * weight.grad.element_size()
        count += 1
        summed = receive_activation(tuple(weight.shape), weight.dtype, device,
                                    0, chunks[-1].virtual_index, num_physical, timeline, None)
        moved_bytes += summed.numel() * summed.element_size()
        count += 1
        weight.grad.copy_(summed)
    else:
        # Middle ranks hold no tied weight; they only sync here so a
        # surrounding barrier stays collective-safe.
        pass
    return {"tied_exchange_bytes": moved_bytes, "tied_exchange_count": count}


def clip_grad_norm_chunks(chunks: list[VirtualStage], max_norm: float, world_size: int) -> torch.Tensor:
    """Exact global gradient norm across ALL virtual stages (one AllReduce).

    Virtual stages partition the parameters (across chunks AND ranks), so the
    global squared norm is the sum of the per-chunk sums on each rank, then
    one scalar AllReduce over the physical ranks — identical semantics to
    chapters 24/25.
    """

    device = next(chunks[0].parameters()).device
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    for chunk in chunks:
        for parameter in chunk.parameters():
            if parameter.grad is not None:
                total_sq += parameter.grad.double().pow(2).sum()
    if world_size > 1:
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
    total_norm = torch.sqrt(total_sq)
    clip_coef = float(max_norm) / (float(total_norm.item()) + 1e-6)
    if clip_coef < 1.0:
        for chunk in chunks:
            for parameter in chunk.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_coef)
    return total_norm


def pp_train_step(chunks: list[VirtualStage], optimizer: torch.optim.Optimizer,
                  micro_batches: list[dict], device: torch.device, mode: str,
                  max_grad_norm: float) -> dict:
    """One optimizer step over all micro-batches in ``mode``.

    ``mode`` selects only the schedule: ``'onef1b'`` (non-interleaved
    baseline, one virtual stage per rank) or ``'interleaved'`` (this
    chapter's variant, ``vpp >= 2``).  Order inside the step: zero_grad ->
    pipeline forwards/backwards (P2P or local handoffs) -> tied
    Embedding/LM-Head gradient exchange (physical first/last ranks) ->
    exact global-norm clip -> AdamW.  ``global_loss`` is gathered once per
    step from the per-rank loss shards so every rank's record carries the
    same value.
    """

    if mode not in ("onef1b", "interleaved"):
        raise ValueError(f"unknown pipeline mode {mode!r}")
    num_physical = chunks[0].num_physical
    timeline = StageTimeline(device)
    optimizer.zero_grad(set_to_none=True)
    timeline.start_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()

    if mode == "interleaved":
        result = interleaved_1f1b_step(chunks, micro_batches, device, timeline)
    else:
        result = onef1b_step(chunks, micro_batches, device, timeline)
    tied = exchange_tied_gradients(chunks, device, timeline)
    events = timeline.end_step()
    grad_norm = clip_grad_norm_chunks(chunks, max_grad_norm, num_physical)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_ms = (time.perf_counter() - wall_start) * 1000

    shards = [None] * num_physical
    dist.all_gather_object(shards, result["local_loss_shard"])
    global_loss = float(sum(shards))

    p2p_events = [event for event in events if event["kind"].startswith("p2p")]
    local_events = [event for event in events if event["kind"] == "local_handoff"]
    layer_ranges = [f"{chunk.layer_range[0]}:{chunk.layer_range[1]}" for chunk in chunks]
    metrics = {
        "step_time_ms": step_ms,
        "mode": mode,
        "owner_rank": chunks[0].owner_rank,
        "num_physical": num_physical,
        "vpp": len(chunks),
        "total_virtual_stages": chunks[0].total_stages,
        "chunk_layer_ranges": layer_ranges,
        "local_loss_shard": result["local_loss_shard"],
        "global_loss": global_loss,
        "gradient_norm_before_clip": float(grad_norm.item()),
        "p2p_count": float(len(p2p_events)),
        "p2p_bytes": float(sum(event["bytes"] for event in p2p_events)),
        "p2p_time_s": float(sum(event["end_s"] - event["start_s"] for event in p2p_events)),
        "local_handoff_count": float(len(local_events)),
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
