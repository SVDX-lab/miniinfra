"""Chapter 42 core: 3-D DP x TP x PP hybrid parallelism from verified parts.

This is the synthesis chapter: NO new parallel primitive is introduced.  The
parts and their origins are:

- TP layers: ``mlp_tp.py`` / ``attention_tp.py`` copied unchanged from
  chapter 21 (ParallelMLP + ParallelAttention, the f/g operators, the shared
  COMM_RECORDER).  The ONLY change is which group they get: this rank's TP
  group inside the 3-D layout instead of the whole world.
- PP schedule: the chapter-39 GPipe fill-drain (``split_layers`` boundary
  rule, ``StageTimeline``, ``compute_bubble``, the activation/input-gradient
  P2P contract, all-forwards-then-all-backwards).  NOT 1F1B (chapters 25/26).
- DP: the chapter-29 supervised-token loss weighting — every DP rank scales
  its local ``token_mean`` loss by ``supervised_local / supervised_global``
  and the post-backward AllReduce SUMS the partial gradients over the DP
  group.  Data is sharded ONLY along DP: the ranks of a TP group run the
  SAME micro-batch, and PP streams the DP rank's micro-batches in order.

RANK LAYOUT (the placement table).  With degrees (dp, pp, tp) the logical
rank decomposes as::

    rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank

so CONSECUTIVE ranks form one TP group, a TP2xPP2 block (pp_size*tp_size
consecutive ranks) is ONE full model replica (= one DP member), and DP groups
stride across replicas with the same (pp_rank, tp_rank).  With 2 nodes x 4
GPUs and degrees (DP=2, TP=2, PP=2): node 0 holds ranks 0-3 (DP replica 0),
node 1 holds ranks 4-7 (DP replica 1); TP groups {0,1},{2,3},{4,5},{6,7} and
PP groups {0,2},{1,3},{4,6},{5,7} stay INSIDE a node; DP groups {0,4},{1,5},
{2,6},{3,7} CROSS nodes — the only cross-node traffic of experiment D.

GRADIENT OWNERSHIP (composing the chapters, the load-bearing part).  Per
parameter exactly one DP-side action is needed, and inside a TP group the
classical three classes still hold:

1. TP shard weights (attn q/k/v/o, mlp gate/up/down): gradients are
   rank-local inside the TP group; across DP the replicas hold the SAME
   shard slice, so ONE AllReduce SUM over the DP group per parameter.
2. Residual-stream inputs: the "f" operator already sums them across the TP
   group during backward (once per parallel block) — no DP action (the DP
   contribution rides on the parameters' DP AllReduce above).
3. Replicated weights (norms, embedding, LM head, q/k norms after their
   in-backward TP hook): value-identical across the TP group — an extra TP
   AllReduce would be WRONG (x tp_size); across DP they carry data-partial
   gradients, so the same single DP AllReduce covers them.

The exact global gradient norm of the LOGICAL model therefore needs three
deduplications on top of the raw local squares: the TP shard class is summed
once over the TP group; the world AllReduce of the combined values counts
every logical gradient ``dp_size * tp_size`` times (the DP AllReduce has
already made the replicas bitwise identical); and the pp>=2 tied pair holds
the SAME full gradient on both stages, so only the Embedding copy counts.
See ``clip_grad_norm_hybrid`` for why a naive world AllReduce of raw squares
is wrong on all three counts.

TIED EMBEDDING/LM HEAD has one case distinction the earlier chapters did not
need: with pp_size == 1 the stage holds both ends, so the two weights are
TIED IN PLACE (one Parameter, gradient accumulates both contributions, no
exchange).  With pp_size >= 2 the chapter-39 pair exchange runs between the
first and last stage — pairwise across EQUAL tp_rank indices (the payloads
are TP-identical, so the t<->t pairs stay consistent and every replica of
the tied weight receives the same sum).

BOUNDARY P2P CONTRACT: stage k's output activation is post-"g" (the Row
parallel output AllReduce), i.e. FULL and identical across the stage's TP
ranks; the input gradient computed by stage k+1 has passed the first block's
"f" sum, i.e. is likewise full and TP-identical.  Every TP rank t therefore
exchanges the SAME-shape payloads with rank t of the adjacent stage.  The
per-pair byte count equals the pure-PP chapter's; the REDUNDANCY across the
TP group (tp_size identical copies in flight) is a real cost of this layout
and is recorded in the metrics instead of being hidden.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from mlp_tp import (
    COMM_RECORDER,
    ParallelMLP,
    split_local_parameters,
    tp_memory_ledger,
)
from mini_batch import batch_loss
from qwen3_model import (
    Qwen3Config,
    Qwen3DecoderLayer,
    RMSNorm,
    RotaryEmbedding,
    resolve_model_directory,
)


# ---------------------------------------------------------------------------
# 3-D topology: the placement table and the process groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelTopology:
    """Degrees of the 3-D layout plus the logical rank decomposition."""

    world_size: int
    tp_size: int
    pp_size: int
    dp_size: int

    def __post_init__(self) -> None:
        if self.tp_size * self.pp_size * self.dp_size != self.world_size:
            raise ValueError(
                f"tp {self.tp_size} * pp {self.pp_size} * dp {self.dp_size} != world {self.world_size}"
            )
        if min(self.tp_size, self.pp_size, self.dp_size) < 1:
            raise ValueError("all degrees must be >= 1")

    @classmethod
    def from_degrees(cls, dp_size: int, tp_size: int, pp_size: int) -> "ParallelTopology":
        return cls(dp_size=dp_size, tp_size=tp_size, pp_size=pp_size,
                   world_size=dp_size * tp_size * pp_size)

    def rank_of(self, dp_rank: int, pp_rank: int, tp_rank: int) -> int:
        """Logical rank of one (dp, pp, tp) coordinate — the placement table."""

        if not (0 <= dp_rank < self.dp_size and 0 <= pp_rank < self.pp_size and 0 <= tp_rank < self.tp_size):
            raise ValueError(
                f"coordinate (dp={dp_rank}, pp={pp_rank}, tp={tp_rank}) outside {self}"
            )
        return dp_rank * (self.pp_size * self.tp_size) + pp_rank * self.tp_size + tp_rank

    def decompose(self, rank: int) -> tuple[int, int, int]:
        """(dp_rank, pp_rank, tp_rank) of a logical rank."""

        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} outside world size {self.world_size}")
        dp_rank, remainder = divmod(rank, self.pp_size * self.tp_size)
        pp_rank, tp_rank = divmod(remainder, self.tp_size)
        return dp_rank, pp_rank, tp_rank

    def ranks_for(self, dp_rank: int | None = None, pp_rank: int | None = None,
                  tp_rank: int | None = None) -> list[int]:
        """All ranks matching the given coordinates (None = wildcard)."""

        return [
            self.rank_of(d, p, t)
            for d in range(self.dp_size)
            for p in range(self.pp_size)
            for t in range(self.tp_size)
            if (dp_rank is None or d == dp_rank)
            and (pp_rank is None or p == pp_rank)
            and (tp_rank is None or t == tp_rank)
        ]

    def group_tables(self) -> dict[str, list[list[int]]]:
        """Membership tables of the three group families (rank lists)."""

        tp_groups = [
            self.ranks_for(dp_rank=d, pp_rank=p)
            for d in range(self.dp_size) for p in range(self.pp_size)
        ]
        pp_groups = [
            self.ranks_for(dp_rank=d, tp_rank=t)
            for d in range(self.dp_size) for t in range(self.tp_size)
        ]
        dp_groups = [
            self.ranks_for(pp_rank=p, tp_rank=t)
            for p in range(self.pp_size) for t in range(self.tp_size)
        ]
        return {"tp": tp_groups, "pp": pp_groups, "dp": dp_groups}

    def describe(self) -> dict:
        return {
            "world_size": self.world_size,
            "dp_size": self.dp_size,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "rank_formula": "rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank",
            "rank_coordinates": {
                str(rank): {"dp": self.decompose(rank)[0], "pp": self.decompose(rank)[1], "tp": self.decompose(rank)[2]}
                for rank in range(self.world_size)
            },
            "groups": self.group_tables(),
        }


def build_parallel_groups(topology: ParallelTopology) -> dict:
    """Create the TP and DP process groups for THIS rank and verify membership.

    PP needs no group: its point-to-point sends/recvs address global ranks on
    the world group (chapter 24's contract).  TP and DP need real groups.
    Every rank re-derives its expected peers from the placement table,
    creates exactly one group per family, and asserts that the created group
    sizes match the topology.  Any mismatch raises — a wrong group here
    silently corrupts every gradient.
    """

    tables = topology.group_tables()
    rank = dist.get_rank()
    dp_rank, pp_rank, tp_rank = topology.decompose(rank)

    derived_tp = sorted(topology.rank_of(dp_rank, pp_rank, t) for t in range(topology.tp_size))
    derived_dp = sorted(topology.rank_of(d, pp_rank, tp_rank) for d in range(topology.dp_size))
    derived_pp = sorted(topology.rank_of(dp_rank, p, tp_rank) for p in range(topology.pp_size))
    table_tp = sorted(tables["tp"][dp_rank * topology.pp_size + pp_rank])
    table_dp = sorted(tables["dp"][pp_rank * topology.tp_size + tp_rank])
    if derived_tp != table_tp:
        raise AssertionError(f"rank {rank}: derived TP peers {derived_tp} disagree with the table {table_tp}")
    if derived_dp != table_dp:
        raise AssertionError(f"rank {rank}: derived DP peers {derived_dp} disagree with the table {table_dp}")
    if sorted(sum(tables["tp"], [])) != list(range(topology.world_size)):
        raise AssertionError("TP groups do not partition the world")
    if sorted(sum(tables["dp"], [])) != list(range(topology.world_size)):
        raise AssertionError("DP groups do not partition the world")
    if sorted(sum(tables["pp"], [])) != list(range(topology.world_size)):
        raise AssertionError("PP groups do not partition the world")

    tp_group = None
    dp_group = None
    # ``dist.new_group`` is collective across the whole world: EVERY rank must
    # create the groups of a family in the SAME order, keeping the one that
    # contains it.  Creating only "my" group would hang the other ranks.
    for group_ranks in tables["tp"]:
        group = dist.new_group(group_ranks)
        if rank in group_ranks:
            if tp_group is not None:
                raise AssertionError(f"rank {rank} appears in two TP groups")
            tp_group = group
    for group_ranks in tables["dp"]:
        group = dist.new_group(group_ranks)
        if rank in group_ranks:
            if dp_group is not None:
                raise AssertionError(f"rank {rank} appears in two DP groups")
            dp_group = group
    if tp_group is None or dp_group is None:
        raise AssertionError(f"rank {rank} found no TP/DP group for coordinate {(dp_rank, pp_rank, tp_rank)}")

    membership = {
        "rank": rank,
        "dp_rank": dp_rank,
        "pp_rank": pp_rank,
        "tp_rank": tp_rank,
        "tp_group_ranks": derived_tp,
        "pp_group_ranks": derived_pp,
        "dp_group_ranks": derived_dp,
        "tp_group_size": dist.get_world_size(group=tp_group),
        "dp_group_size": dist.get_world_size(group=dp_group),
    }
    if membership["tp_group_size"] != topology.tp_size or membership["dp_group_size"] != topology.dp_size:
        raise AssertionError(f"rank {rank}: created group sizes do not match the topology")
    return {"topology": topology, "tp_group": tp_group, "dp_group": dp_group, "membership": membership}


# ---------------------------------------------------------------------------
# PP stage over TP layers (chapter-39 stage + chapter-36 layer factories)
# ---------------------------------------------------------------------------


def split_layers(num_hidden_layers: int, num_stages: int) -> tuple[list[tuple[int, int]], dict]:
    """Contiguous, near-even layer ranges per stage (chapter 24's rule).

    Divisible splits are exact; otherwise EARLIER stages take one extra
    layer.  The remainder is recorded, never hidden.
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
    """Forward/backward/P2P event spans of one rank (chapter 24, unchanged).

    GPU spans use CUDA Event pairs resolved by one final synchronize; CPU
    spans use ``perf_counter``.  ``end_step`` returns plain dicts ready for
    the timeline JSON.
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
    """Per-rank busy/bubble ratios from one rank's timeline (chapter 24)."""

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


class HybridParallelStage(nn.Module):
    """One contiguous PP stage whose layers are TP-sharded (or replicated).

    Stage 0 owns ``embed_tokens``; the last stage owns ``norm`` and the LM
    Head.  TIED WEIGHTS: with ``num_stages == 1`` both ends live on the same
    module, so the LM Head is tied to the Embedding IN PLACE (one Parameter
    accumulates both gradient contributions — exactly the chapter-16
    reference semantics; no exchange exists to run).  With ``num_stages >= 2``
    the two copies stay separate and ``exchange_tied_gradients``
    P2P-sums their partials.  Forgetting this case distinction silently
    diverges the pp=1 arms from the tied reference after ONE step.

    With ``tp_size == 1`` the layers are the plain replicated modules (the
    chapter-39 path with zero TP collectives); with ``tp_size > 1`` every
    layer is built from ParallelAttention + ParallelMLP over ``tp_group``.
    """

    def __init__(self, config: Qwen3Config, stage_index: int, num_stages: int,
                 tp_rank: int, tp_size: int, tp_group=None) -> None:
        super().__init__()
        if not 0 <= stage_index < num_stages:
            raise ValueError(f"stage_index {stage_index} outside [0, {num_stages})")
        if tp_size < 1 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid (tp_rank={tp_rank}, tp_size={tp_size})")
        ranges, split_meta = split_layers(config.num_hidden_layers, num_stages)
        self.config = config
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.layer_range = ranges[stage_index]
        self.split_meta = split_meta
        self.is_first = stage_index == 0
        self.is_last = stage_index == num_stages - 1
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.tied_in_stage = self.is_first and self.is_last
        layer_count = self.layer_range[1] - self.layer_range[0]

        if tp_size == 1:
            self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(layer_count)])
        else:
            from attention_tp import ParallelAttention

            def make_layer(_layer_config):
                return Qwen3DecoderLayer(
                    config,
                    mlp_factory=lambda c: ParallelMLP(
                        c.hidden_size, c.intermediate_size, tp_rank, tp_size, group=tp_group,
                    ),
                    attn_factory=lambda c: ParallelAttention(
                        c.hidden_size, c.num_attention_heads, c.num_key_value_heads,
                        c.head_dim, c.rms_norm_eps, tp_rank, tp_size, group=tp_group,
                    ),
                )

            self.layers = nn.ModuleList([make_layer(config) for _ in range(layer_count)])

        if self.is_first:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if self.is_last:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.tied_in_stage:
            # pp=1: one Parameter serves lookup and logits (chapter-16 semantics).
            self.lm_head.weight = self.embed_tokens.weight
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

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


def load_hybrid_stage(model_directory, device: torch.device, stage_index: int, num_stages: int,
                      tp_rank: int, tp_size: int, tp_group=None,
                      dtype: torch.dtype | None = None) -> HybridParallelStage:
    """Build one stage and read ONLY its own weight window from safetensors.

    Combines the chapter-39 layer-window read with the chapter-36 TP shard
    windows: layers outside ``[begin, end)`` are never materialized, and
    inside the window the attn/MLP projections are read as this rank's
    ``[tp_rank*local, (tp_rank+1)*local)`` slice through the safetensors
    PySafeSlice API, so a rank never holds other ranks' shard bytes.
    """

    from attention_tp import (
        gqa_head_layout,
    )

    directory = str(resolve_model_directory(model_directory))
    config = Qwen3Config.from_model_directory(directory)
    if tp_size > 1:
        gqa_head_layout(tp_rank, tp_size, config.num_attention_heads, config.num_key_value_heads)
    stage = HybridParallelStage(config, stage_index, num_stages, tp_rank, tp_size, tp_group)

    weight_path = Path(directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    from safetensors import safe_open

    begin, end = stage.layer_range
    local_q = config.num_attention_heads // tp_size
    local_kv = config.num_key_value_heads // tp_size
    q_begin = tp_rank * local_q * config.head_dim
    kv_begin = tp_rank * local_kv * config.head_dim
    local_intermediate = config.intermediate_size // tp_size
    shard_begin = tp_rank * local_intermediate
    state: dict = {}
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        for key in keys:
            if key.startswith("model.layers."):
                layer_index = int(key.split(".")[2])
                if not begin <= layer_index < end:
                    continue
                suffix = key[len(f"model.layers.{layer_index}"):]
                if tp_size > 1 and suffix == ".self_attn.q_proj.weight":
                    state[f"layers.{layer_index - begin}.self_attn.q_proj.weight"] = handle.get_slice(key)[q_begin:q_begin + local_q * config.head_dim, :]
                elif tp_size > 1 and suffix == ".self_attn.k_proj.weight":
                    state[f"layers.{layer_index - begin}.self_attn.k_proj.weight"] = handle.get_slice(key)[kv_begin:kv_begin + local_kv * config.head_dim, :]
                elif tp_size > 1 and suffix == ".self_attn.v_proj.weight":
                    state[f"layers.{layer_index - begin}.self_attn.v_proj.weight"] = handle.get_slice(key)[kv_begin:kv_begin + local_kv * config.head_dim, :]
                elif tp_size > 1 and suffix == ".self_attn.o_proj.weight":
                    state[f"layers.{layer_index - begin}.self_attn.o_proj.weight"] = handle.get_slice(key)[:, q_begin:q_begin + local_q * config.head_dim]
                elif tp_size > 1 and suffix == ".mlp.gate_proj.weight":
                    state[f"layers.{layer_index - begin}.mlp.gate_proj.weight"] = handle.get_slice(key)[shard_begin:shard_begin + local_intermediate, :]
                elif tp_size > 1 and suffix == ".mlp.up_proj.weight":
                    state[f"layers.{layer_index - begin}.mlp.up_proj.weight"] = handle.get_slice(key)[shard_begin:shard_begin + local_intermediate, :]
                elif tp_size > 1 and suffix == ".mlp.down_proj.weight":
                    state[f"layers.{layer_index - begin}.mlp.down_proj.weight"] = handle.get_slice(key)[:, shard_begin:shard_begin + local_intermediate]
                else:
                    state[f"layers.{layer_index - begin}" + suffix] = handle.get_slice(key)[:]
                continue
            if key == "model.embed_tokens.weight":
                if stage.is_first:
                    state["embed_tokens.weight"] = handle.get_slice(key)[:]
                if stage.is_last and not stage.tied_in_stage and "lm_head.weight" not in keys:
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

    if stage.tied_in_stage:
        # The tied stage serves lookup and logits from ONE parameter: a
        # separate lm_head key would just re-copy the same values, so drop it
        # before loading (embed_tokens.weight already populated the tensor).
        state.pop("lm_head.weight", None)
    load_result = stage.load_state_dict(state, strict=False)
    missing_expected: set[str] = set()
    if not stage.is_first:
        missing_expected.add("embed_tokens.weight")
    if not stage.is_last:
        missing_expected.update({"norm.weight", "lm_head.weight"})
    elif stage.tied_in_stage:
        missing_expected.add("lm_head.weight")
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


# ---------------------------------------------------------------------------
# Communication pieces: P2P boundaries, tied exchange, DP AllReduce, clip
# ---------------------------------------------------------------------------


def _boundary_peer(topology: ParallelTopology, rank: int, stage_offset: int) -> int:
    """Global rank of the adjacent stage's owner with the SAME tp_rank."""

    dp_rank, pp_rank, tp_rank = topology.decompose(rank)
    return topology.rank_of(dp_rank, pp_rank + stage_offset, tp_rank)


def _send_activation(tensor: torch.Tensor, dst_rank: int, timeline: StageTimeline | None,
                     micro: int | None) -> None:
    payload = tensor.detach().contiguous()
    handle = None
    if timeline is not None:
        handle = timeline.begin("p2p_send", micro, peer_rank=dst_rank,
                                bytes=float(payload.numel() * payload.element_size()))
    dist.send(payload, dst=dst_rank)
    if handle is not None:
        timeline.end(handle)


def _recv_activation(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device,
                     src_rank: int, timeline: StageTimeline | None, micro: int | None) -> torch.Tensor:
    buffer = torch.empty(shape, dtype=dtype, device=device)
    handle = None
    if timeline is not None:
        handle = timeline.begin("p2p_recv", micro, peer_rank=src_rank,
                                bytes=float(buffer.numel() * buffer.element_size()))
    dist.recv(buffer, src=src_rank)
    if handle is not None:
        timeline.end(handle)
    return buffer


def _forward_micro(stage: HybridParallelStage, micro: dict, device: torch.device,
                   topology: ParallelTopology, rank: int, timeline: StageTimeline,
                   micro_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One micro-batch through THIS stage (chapter-39 contract, TP inside)."""

    if stage.is_first:
        handle = timeline.begin("forward", micro_index, layers=f"{stage.layer_range[0]}:{stage.layer_range[1]}")
        holder = stage.embed_tokens(micro["input_ids"].to(device))
        # Non-leaf embedding output: retain_grad() exposes the input gradient
        # for the P2P hand-back to the previous pipeline stage.
        holder.retain_grad()
        out = stage.forward_layers(holder, micro["attention_mask"].to(device))
        timeline.end(handle)
        return holder, out
    shape = (micro["batch_size"], micro["sequence_length"], stage.config.hidden_size)
    holder_dtype = next(stage.parameters()).dtype
    holder = _recv_activation(shape, holder_dtype, device,
                              _boundary_peer(topology, rank, -1), timeline, micro_index)
    holder.requires_grad_(True)
    handle = timeline.begin("forward", micro_index, layers=f"{stage.layer_range[0]}:{stage.layer_range[1]}")
    out = stage.forward_layers(holder, micro["attention_mask"].to(device))
    timeline.end(handle)
    return holder, out


def _backward_micro(out: torch.Tensor, grad_output: torch.Tensor | None,
                    timeline: StageTimeline, micro_index: int) -> None:
    handle = timeline.begin("backward", micro_index)
    if grad_output is None:
        out.backward()
    else:
        torch.autograd.backward(out, grad_tensors=grad_output)
    timeline.end(handle)


def gpipe_step(stage: HybridParallelStage, micro_batches: list[dict], device: torch.device,
               topology: ParallelTopology, rank: int, timeline: StageTimeline) -> dict:
    """GPipe fill-drain over the stage graph (chapter 24, TP collectives inside).

    ALL forwards, then ALL backwards.  Every TP rank of the stage exchanges
    boundary payloads with the SAME tp_rank of the adjacent stage; the TP
    f/g collectives run inside forward/backward and are timed by the shared
    COMM_RECORDER.  Only the last stage materializes the (scaled) loss; the
    per-rank loss shard of every other stage is an all-zero placeholder.
    """

    holders: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []

    # ---- fill: continuous forwards ----
    for micro_index, micro in enumerate(micro_batches):
        holder, out = _forward_micro(stage, micro, device, topology, rank, timeline, micro_index)
        holders.append(holder)
        outputs.append(out)
        if stage.is_last:
            losses.append(stage.head_loss(out, micro["labels"].to(device), micro["scale"]))
        else:
            _send_activation(out, _boundary_peer(topology, rank, +1), timeline, micro_index)

    # ---- drain: continuous backwards ----
    for micro_index in range(len(micro_batches)):
        if stage.is_last:
            _backward_micro(losses[micro_index], None, timeline, micro_index)
        else:
            grad = _recv_activation(outputs[micro_index].shape, outputs[micro_index].dtype,
                                    device, _boundary_peer(topology, rank, +1), timeline, micro_index)
            _backward_micro(outputs[micro_index], grad, timeline, micro_index)
        if not stage.is_first:
            _send_activation(holders[micro_index].grad, _boundary_peer(topology, rank, -1),
                             timeline, micro_index)

    loss_shard = float(sum(loss.detach() for loss in losses)) if stage.is_last else 0.0
    return {
        "local_loss_shard": loss_shard,
        "input_grads": [holder.grad.detach().clone() for holder in holders],
    }


def exchange_tied_gradients(stage: HybridParallelStage, topology: ParallelTopology, rank: int,
                            device: torch.device, timeline: StageTimeline | None = None) -> dict:
    """P2P-sum the tied Embedding/LM Head partials (pp >= 2 only).

    The chapter-39 exchange between the physical first and last stage, run
    pairwise across EQUAL tp_rank indices: stage 0's rank t exchanges with
    the last stage's rank t.  Both payloads are TP-identical by the boundary
    contract, so every pair moves the same bytes and both ends receive the
    same sum.  With ``tied_in_stage`` (pp == 1) there is nothing to exchange.
    """

    if stage.num_stages < 2 or stage.tied_in_stage:
        return {"tied_exchange_bytes": 0.0, "tied_exchange_count": 0.0}
    dp_rank, _, tp_rank = topology.decompose(rank)
    first_rank = topology.rank_of(dp_rank, 0, tp_rank)
    last_rank = topology.rank_of(dp_rank, topology.pp_size - 1, tp_rank)
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
        moved_bytes += weight.grad.numel() * weight.element_size()
        count += 1
    elif stage.is_last:
        weight = stage.lm_head.weight
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)
        _send_activation(weight.grad, first_rank, timeline, None)
        moved_bytes += weight.grad.numel() * weight.element_size()
        count += 1
        summed = _recv_activation(tuple(weight.shape), weight.dtype, device, first_rank, timeline, None)
        moved_bytes += summed.numel() * summed.element_size()
        count += 1
        weight.grad.copy_(summed)
    return {"tied_exchange_bytes": moved_bytes, "tied_exchange_count": count}


def allreduce_gradients_dp(stages: list[HybridParallelStage], dp_group, dp_size: int) -> tuple[float, float, int]:
    """SUM every parameter gradient over the DP group; (ms, bytes, count).

    One call covers BOTH ownership classes across DP: shard parameters of
    the replicas cover the same slice (their partials sum to the slice's
    full gradient), replicated parameters carry data-partial gradients.
    There is deliberately NO gradient AllReduce inside the TP group — the
    "f"/"g" operators and the q/k-norm hooks already made everything
    TP-correct during backward (an extra TP sum would multiply shard-free
    gradients by tp_size, i.e. corrupt them).
    """

    if dp_size <= 1:
        return 0.0, 0.0, 0
    parameters = [
        parameter for stage in stages for parameter in stage.parameters() if parameter.grad is not None
    ]
    total_bytes = float(sum(p.numel() * p.element_size() for p in parameters))
    if not parameters:
        return 0.0, 0.0, 0
    if parameters[0].is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for parameter in parameters:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=dp_group)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)), total_bytes, len(parameters)
    begin = time.perf_counter()
    for parameter in parameters:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=dp_group)
    return (time.perf_counter() - begin) * 1000, total_bytes, len(parameters)


def clip_grad_norm_hybrid(stages: list[HybridParallelStage], tp_group, tp_size: int,
                          max_norm: float, dp_size: int = 1) -> tuple[torch.Tensor, float, float]:
    """Exact logical-model gradient norm over the 3-D layout (two scalar AllReduces).

    The parameter set partitions across stages, but the TP group SHARDS one
    class and the DP replicas + tied pair REPLICATE the rest — three separate
    double-counting traps on the way to the norm of the ONE logical model:

    1. ``shard_sq`` (the TP shard classes) is summed ONCE over the TP group;
    2. ``combined = replicated_sq + shard_sq_full`` is then identical across
       the TP ranks of one (dp, pp) cell, and the DP AllReduce has already
       made every DP replica bitwise identical — so the WORLD AllReduce counts
       every logical gradient ``dp_size * tp_size`` times, and the exact
       squared norm is ``world_sum / (tp_size * dp_size)``.  Dividing by
       ``tp_size`` alone inflates the norm by ``sqrt(dp_size)``, which silently
       shrinks every dp>1 clip (and step) by that factor;
    3. the tied Embedding/LM Head is ONE logical parameter: after the pp>=2
       exchange both stages hold the SAME full tied gradient, so the last
       stage's LM Head copy is skipped and the first stage's Embedding counts
       it (pp == 1 ties them in place — nothing to skip).

    A world AllReduce of the RAW local squares (step 1 skipped) would count
    the replicated class tp_size times while the shard class needs the TP
    sum — the errors do not cancel.  Returns ``(total_norm,
    tp_scalar_ms, world_scalar_ms)``.
    """

    device = next(stages[0].parameters()).device
    replicated_sq = torch.zeros((), dtype=torch.float64, device=device)
    shard_sq = torch.zeros((), dtype=torch.float64, device=device)
    for stage in stages:
        replicated, shard = split_local_parameters(stage)
        for parameter in replicated:
            if parameter.grad is None:
                continue
            if stage.is_last and not stage.tied_in_stage and parameter is stage.lm_head.weight:
                # Tied pair counted ONCE: exchange_tied_gradients made this
                # copy equal the first stage's Embedding gradient, and the
                # logical tied weight has exactly one gradient.
                continue
            replicated_sq += parameter.grad.double().pow(2).sum()
        for parameter in shard:
            if parameter.grad is not None:
                shard_sq += parameter.grad.double().pow(2).sum()

    def timed_reduce(tensor: torch.Tensor, group) -> float:
        if group is not None and dist.get_world_size(group=group) == 1:
            return 0.0
        if tensor.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
            end.record()
            torch.cuda.synchronize()
            return float(start.elapsed_time(end))
        begin = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return (time.perf_counter() - begin) * 1000

    tp_scalar_ms = timed_reduce(shard_sq, tp_group if tp_size > 1 else None)
    combined = replicated_sq + shard_sq
    world_scalar_ms = timed_reduce(combined, None)  # world group: spans stages AND DP replicas
    combined = combined / (tp_size * dp_size)
    total_norm = torch.sqrt(combined)
    clip_coef = float(max_norm) / (float(total_norm.item()) + 1e-6)
    if clip_coef < 1.0:
        for stage in stages:
            for parameter in stage.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_coef)
    return total_norm, tp_scalar_ms, world_scalar_ms


# ---------------------------------------------------------------------------
# The full hybrid train step
# ---------------------------------------------------------------------------


def hybrid_train_step(stages: list[HybridParallelStage], optimizer: torch.optim.Optimizer,
                      micro_batches: list[dict], device: torch.device, groups: dict,
                      max_grad_norm: float) -> dict:
    """One optimizer step over all micro-batches of THIS DP rank.

    Order inside the step (each piece is a verified part):
    zero_grad -> GPipe fill-drain (PP P2P + TP f/g collectives) -> tied
    Embedding/LM-Head exchange (pp >= 2) -> DP gradient AllReduce -> exact
    global-norm clip (TP scalar + DP scalar) -> AdamW.  ``global_loss`` is
    the sum over DP replicas of the scaled local losses: the loss shards are
    gathered over the world and divided by tp_size (TP ranks of the last
    stage report identical values; non-last stages report 0).
    """

    topology: ParallelTopology = groups["topology"]
    tp_group = groups["tp_group"]
    dp_group = groups["dp_group"]
    rank = dist.get_rank()
    if len(stages) != 1:
        raise ValueError("chapter 27 keeps chapter 24's one-stage-per-rank layout")
    stage = stages[0]

    timeline = StageTimeline(device)
    optimizer.zero_grad(set_to_none=True)
    COMM_RECORDER.reset()
    timeline.start_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()

    result = gpipe_step(stage, micro_batches, device, topology, rank, timeline)
    tied = exchange_tied_gradients(stage, topology, rank, device, timeline)
    events = timeline.end_step()
    comm = COMM_RECORDER.resolve()

    dp_ms, dp_bytes, dp_count = allreduce_gradients_dp(stages, dp_group, topology.dp_size)
    grad_norm, tp_scalar_ms, world_scalar_ms = clip_grad_norm_hybrid(
        stages, tp_group, topology.tp_size, max_grad_norm, dp_size=topology.dp_size
    )
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_ms = (time.perf_counter() - wall_start) * 1000

    shards = [None] * topology.world_size
    dist.all_gather_object(shards, result["local_loss_shard"])
    global_loss = float(sum(shards)) / topology.tp_size

    dp_rank, pp_rank, tp_rank = topology.decompose(rank)
    p2p_events = [event for event in events if event["kind"].startswith("p2p")]
    forward_ms = sum(event["end_s"] - event["start_s"] for event in events if event["kind"] == "forward") * 1000
    backward_ms = sum(event["end_s"] - event["start_s"] for event in events if event["kind"] == "backward") * 1000
    tp_total_ms = comm["tp_allreduce_forward_ms"] + comm["tp_allreduce_backward_ms"] + tp_scalar_ms
    local_physical = float(sum(m["physical_tokens"] for m in micro_batches))
    local_supervised = float(sum(m["supervised_tokens"] for m in micro_batches))
    metrics = {
        "step_time_ms": step_ms,
        "schedule": "gpipe",
        "rank": rank,
        "dp_rank": dp_rank,
        "pp_rank": pp_rank,
        "tp_rank": tp_rank,
        "world_size": topology.world_size,
        "dp_size": topology.dp_size,
        "tp_size": topology.tp_size,
        "pp_size": topology.pp_size,
        "stage_index": stage.stage_index,
        "num_stages": stage.num_stages,
        "stage_layer_range": f"{stage.layer_range[0]}:{stage.layer_range[1]}",
        "stage_split": stage.split_meta,
        "is_last_stage": stage.is_last,
        "tied_in_stage": stage.tied_in_stage,
        "forward_time_ms": forward_ms,
        "backward_time_ms": backward_ms,
        "p2p_count": float(len(p2p_events)),
        "p2p_bytes": float(sum(event["bytes"] for event in p2p_events)),
        "p2p_time_s": float(sum(event["end_s"] - event["start_s"] for event in p2p_events)),
        "tp_allreduce_forward_ms": comm["tp_allreduce_forward_ms"],
        "tp_allreduce_backward_ms": comm["tp_allreduce_backward_ms"],
        "tp_allreduce_forward_count": comm["tp_allreduce_forward_count"],
        "tp_allreduce_backward_count": comm["tp_allreduce_backward_count"],
        "tp_scalar_sync_ms": tp_scalar_ms,
        "clip_world_scalar_ms": world_scalar_ms,
        "tp_total_ms": tp_total_ms,
        "dp_allreduce_ms": dp_ms,
        "dp_allreduce_bytes": dp_bytes,
        "dp_allreduce_count": float(dp_count),
        "dp_total_ms": dp_ms,
        "tied_exchange_bytes": tied["tied_exchange_bytes"],
        "tied_exchange_count": tied["tied_exchange_count"],
        "local_loss_shard": result["local_loss_shard"],
        "global_loss": global_loss,
        "gradient_norm_before_clip": float(grad_norm.item()),
        "local_physical_tokens": local_physical,
        "local_supervised_tokens": local_supervised,
        "global_physical_tokens": local_physical * topology.dp_size,
        "global_supervised_tokens": local_supervised * topology.dp_size,
        "micro_batches": float(len(micro_batches)),
        "timeline": events,
        "bubble": compute_bubble(events),
    }
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "physical_tokens_per_second": metrics["global_physical_tokens"] / seconds,
        "supervised_tokens_per_second": metrics["global_supervised_tokens"] / seconds,
    })
    return metrics


def state_ledger(stages: list[HybridParallelStage]) -> dict:
    """Byte ledger of what THIS rank owns (chapter-35 classes + PP partition).

    Parameters PARTITION across stages and DP replicas; inside a stage the
    TP group splits them into replicated (value-identical) and shard parts.
    Communication paths are enumerated with their triggers so the two-dim
    layouts' ownership is explicit per arm.
    """

    replicated_bytes = 0
    shard_bytes_local = 0
    shard_bytes_full = 0
    parameter_total = 0
    for stage in stages:
        ledger = tp_memory_ledger(stage, tp_size=stage.tp_size)
        replicated_bytes += int(ledger["replicated_parameter_bytes"])
        shard_bytes_local += int(ledger["shard_parameter_bytes_local"])
        shard_bytes_full += int(ledger["shard_parameter_bytes_full"])
        parameter_total += int(ledger["local_parameter_bytes"])
    return {
        "local_parameter_bytes": float(parameter_total),
        "replicated_parameter_bytes": float(replicated_bytes),
        "shard_parameter_bytes_local": float(shard_bytes_local),
        "shard_parameter_bytes_full": float(shard_bytes_full),
        "full_model_parameter_bytes": float(replicated_bytes + shard_bytes_full),
        "communication_paths": {
            "tp_allreduce": "per parallel block, inside forward/backward (f/g ops + q/k-norm hooks)",
            "pp_p2p": "boundary activations and input gradients, same tp_rank across adjacent stages",
            "dp_allreduce": "every parameter gradient once per step over the DP group",
            "tied_exchange": "first<->last stage pair per tp_rank (pp >= 2 only)",
        },
    }


ARMS: dict[str, dict[str, int]] = {
    # arm -> {tp, pp, dp}; world_size = tp * pp * dp.
    "a1_tp2": {"tp": 2, "pp": 1, "dp": 1},
    "a2_tp2pp2": {"tp": 2, "pp": 2, "dp": 1},
    "b2_tp2dp2": {"tp": 2, "pp": 1, "dp": 2},
    "c1_pp2": {"tp": 1, "pp": 2, "dp": 1},
    "c2_pp2dp2": {"tp": 1, "pp": 2, "dp": 2},
    "d_tp2pp2dp2": {"tp": 2, "pp": 2, "dp": 2},
}


def arm_topology(arm: str) -> ParallelTopology:
    """Topology of a named experiment arm; unknown arms raise."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose one of {sorted(ARMS)}")
    spec = ARMS[arm]
    return ParallelTopology.from_degrees(dp_size=spec["dp"], tp_size=spec["tp"], pp_size=spec["pp"])
