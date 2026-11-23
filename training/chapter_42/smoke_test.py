#!/usr/bin/env python3
"""CPU-only Chapter 42 checks: 3-D groups, membership, data flow, one step.

Runs a tiny Qwen3 (hidden 32, intermediate 64, vocab 64, 4 layers, 8 Q heads,
4 KV heads, head_dim 8, sequence 12) on CPU gloo, all rounds consuming the
SAME deterministic synthetic samples and the SAME single-process reference:

- Unit checks (no distribution): the placement table of every experiment arm
  (decomposition, TP/PP/DP group tables, world partition), the pp=1 tied
  stage construction, the DP data-shard partition, the layer split.
- Round A ``2 nodes x 4 GPUs`` topology: EIGHT gloo processes with
  TP=2, PP=2, DP=2 (experiment D).  Verifies the 3-D group membership per
  rank, the DP data sharding (disjoint shards, TP-replicated micro-batches),
  the full hybrid step against the reference (global loss, gradient norm,
  post-step full-model checksum per DP replica, tied Embedding/LM-Head
  consistency after the exchange), and the communication inventory
  (TP AllReduces inside forward/backward, PP P2P both directions, DP
  AllReduce once per parameter).
- Round B ``a1_tp2``: TWO processes, TP=2, PP=1, DP=1 — the pp=1 stage ties
  Embedding and LM Head IN PLACE (no exchange), so no P2P and no DP traffic
  may occur; the same reference checks apply on its own 4-sample batch.
- Round C ``c2_pp2dp2``: FOUR processes, TP=1, PP=2, DP=2 — the tp=1 path
  must show ZERO TP collectives while P2P and DP AllReduce stay active.

The reference is the exact degenerate form of the hybrid step (dp=tp=pp=1,
float64 norm, same AdamW): bars therefore cover FP32 reduction order only.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from assistant_loss import IGNORE_INDEX, EncodedConversation
from attention_tp import shard_attention_weight
from hybrid_parallel import (
    ARMS,
    HybridParallelStage,
    ParallelTopology,
    arm_topology,
    build_parallel_groups,
    hybrid_train_step,
    split_layers,
    state_ledger,
)
from hybrid_runtime import shard_micro_batches, micro_dicts
from mlp_tp import gather_tensor_parallel_parameters, shard_weight as shard_mlp_weight
from mini_batch import batch_loss, pad_samples
from qwen3_model import Qwen3Config, Qwen3ForCausalLM


HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 64
LAYERS = 4
NUM_Q_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 8
SEQ = 12
SAMPLES_PER_MICRO = 2
NUM_MICRO = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
REFERENCE_SEED = 23
LOSS_REL_BAR = 1e-5
PARAM_REL_BAR = 1e-4
NORM_REL_BAR = 1e-5


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=LAYERS,
        num_attention_heads=NUM_Q_HEADS, num_key_value_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )


def build_samples(count: int) -> list[EncodedConversation]:
    """Synthetic encoded conversations; identical on every process."""

    generator = torch.Generator().manual_seed(1000 + count)
    tokens = torch.randint(4, VOCAB, (count, SEQ), generator=generator)
    samples = []
    for row in range(count):
        ids = tokens[row].unsqueeze(0)
        labels = ids.clone()
        labels[:, :2] = IGNORE_INDEX
        samples.append(EncodedConversation(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=labels,
            assistant_loss_mask=labels != IGNORE_INDEX,
            source_line=row,
            rendered_text="",
            token_spans=(),
            loss_mode="assistant_only",
        ))
    return samples


def reference_micros(samples: list[EncodedConversation], num_micro_total: int) -> list[dict]:
    """The reference grouping: the whole global batch in micros of the same size."""

    batch = pad_samples(samples, pad_token_id=0)
    if batch.batch_size % num_micro_total != 0:
        raise ValueError("global batch must divide into reference micros")
    per_micro = batch.batch_size // num_micro_total
    supervised_global = float(batch.supervised_tokens)
    micros = []
    for index in range(num_micro_total):
        rows = slice(index * per_micro, (index + 1) * per_micro)
        micro = {
            "input_ids": batch.input_ids[rows],
            "attention_mask": batch.attention_mask[rows],
            "labels": batch.labels[rows],
            "batch_size": int(batch.input_ids[rows].shape[0]),
            "sequence_length": int(batch.input_ids.shape[1]),
            "physical_tokens": float(batch.input_ids[rows].numel()),
            "supervised_tokens": float((batch.labels[rows][:, 1:] != IGNORE_INDEX).sum()),
        }
        micro["scale"] = micro["supervised_tokens"] / supervised_global
        micros.append(micro)
    return micros


def reference_run(samples: list[EncodedConversation], num_micro_total: int) -> dict:
    """The degenerate form of the round: dp=tp=pp=1, float64 norm, same AdamW.

    Three outputs, each pinned to the weight state the round needs:

    - ``state``: ONE AdamW step on the fresh init (W0 -> W1) — exactly what
      the hybrid stages load;
    - ``loss`` / ``gradient_norm``: the weighted loss and the exact float64
      gradient norm evaluated at W1 — the diagnostics the hybrid step reports
      for the weights it STARTED from;
    - ``checksum``: after mirroring the hybrid's own step on W1 (fresh AdamW,
      clip, step) — the post-step state the round's checksum compares against.
    """

    torch.manual_seed(REFERENCE_SEED)
    model = Qwen3ForCausalLM(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=WEIGHT_DECAY, foreach=False, fused=False)
    micros = reference_micros(samples, num_micro_total)

    def loss_and_grad_norm() -> tuple[float, torch.Tensor]:
        """Weighted micro-summed loss plus the exact float64 gradient norm.

        One backward per micro (gradient accumulation), then the float64 norm
        that mirrors the hybrid clip with tp=dp=1 and a single stage.
        """

        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in micros:
            logits = model(micro["input_ids"], attention_mask=micro["attention_mask"])
            loss = batch_loss(logits, micro["labels"], "token_mean") * micro["scale"]
            loss.backward()
            total += float(loss.detach())
        total_sq = torch.zeros((), dtype=torch.float64)
        for parameter in model.parameters():
            if parameter.grad is not None:
                total_sq += parameter.grad.double().pow(2).sum()
        return total, torch.sqrt(total_sq)

    # The step from the fresh init (W0) produces the state dict the hybrid
    # rounds load.
    _, step_norm = loss_and_grad_norm()
    clip_coef = float(MAX_GRAD_NORM) / (float(step_norm.item()) + 1e-6)
    if clip_coef < 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coef)
    optimizer.step()

    # The hybrid round loads THIS state (W1), and its reported global_loss and
    # gradient norm describe the weights its step STARTED from.  Re-evaluate
    # both on W1; returning the pre-step forward values instead would compare
    # one AdamW step apart (a few percent on this tiny random init) and fail
    # the 1e-5 bars for a bookkeeping reason.
    total, grad_norm = loss_and_grad_norm()
    # W1 is what the rounds load and start their step from.
    handed_out_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    # The round's final comparison is the POST-STEP full-model checksum: the
    # hybrid steps from W1, so the reference must mirror that same degenerate
    # step (gradients at W1, float64 norm, clip, AdamW) and checksum its
    # result.  Comparing against W1 itself would differ by exactly one
    # optimizer step's worth of movement.  The mirror step uses a FRESH AdamW:
    # the hybrid's optimizer starts from zero moments on the loaded state.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=WEIGHT_DECAY, foreach=False, fused=False)
    _, w1_norm = loss_and_grad_norm()
    clip_coef = float(MAX_GRAD_NORM) / (float(w1_norm.item()) + 1e-6)
    if clip_coef < 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coef)
    optimizer.step()
    checksum = 0.0
    for value in model.state_dict().values():
        checksum += float(value.detach().double().pow(2).sum())
    return {
        "loss": total,
        "gradient_norm": float(grad_norm.item()),
        "checksum": checksum,
        # state_dict() counts the tied Embedding/LM Head twice, matching the
        # pp>=2 hybrid where the two copies exist as separate tensors.
        "state": handed_out_state,
    }


def stage_state_from_reference(reference_state: dict, stage: HybridParallelStage,
                               tp_rank: int, tp_size: int) -> dict:
    """This stage's (TP-sharded) slice of the reference full state dict."""

    begin, end = stage.layer_range
    state: dict = {}
    for key, value in reference_state.items():
        if key.startswith("model.layers."):
            parts = key.split(".")
            layer_index = int(parts[2])
            if not begin <= layer_index < end:
                continue
            rest = ".".join(parts[3:])
            local_key = f"layers.{layer_index - begin}.{rest}"
            if tp_size > 1 and rest.endswith("proj.weight") and (
                rest.startswith("self_attn.") or rest.startswith("mlp.")
            ):
                if rest.startswith("self_attn."):
                    shard = shard_attention_weight(rest, value, tp_rank, tp_size,
                                                   NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
                else:
                    shard = shard_mlp_weight(rest, value, tp_rank, tp_size)
                state[local_key] = shard.clone()
            else:
                state[local_key] = value.clone()
            continue
        if key == "model.embed_tokens.weight":
            if stage.is_first:
                state["embed_tokens.weight"] = value.clone()
            if stage.is_last and not stage.tied_in_stage and "lm_head.weight" not in reference_state:
                state["lm_head.weight"] = value.clone()
            continue
        if key == "model.norm.weight":
            if stage.is_last:
                state["norm.weight"] = value.clone()
            continue
        if key == "lm_head.weight":
            if stage.is_last:
                state["lm_head.weight"] = value.clone()
            continue
        raise ValueError(f"unexpected reference key {key!r}")
    return state


def checksum_state(state: dict) -> float:
    total = 0.0
    for value in state.values():
        total += float(value.detach().double().pow(2).sum())
    return total


# ---------------------------------------------------------------------------
# unit checks (no distribution)
# ---------------------------------------------------------------------------


def expect_arms_and_groups() -> None:
    for arm, spec in ARMS.items():
        topology = arm_topology(arm)
        assert topology.world_size == spec["tp"] * spec["pp"] * spec["dp"], arm
        tables = topology.group_tables()
        for family, groups in tables.items():
            flat = sorted(sum(groups, []))
            assert flat == list(range(topology.world_size)), (arm, family, groups)
    d = arm_topology("d_tp2pp2dp2")
    assert d.decompose(4) == (1, 0, 0) and d.decompose(3) == (0, 1, 1) and d.decompose(7) == (1, 1, 1)
    tables = d.group_tables()
    assert tables["tp"] == [[0, 1], [2, 3], [4, 5], [6, 7]], tables["tp"]
    assert tables["pp"] == [[0, 2], [1, 3], [4, 6], [5, 7]], tables["pp"]
    assert tables["dp"] == [[0, 4], [1, 5], [2, 6], [3, 7]], tables["dp"]
    b2 = arm_topology("b2_tp2dp2")
    assert b2.group_tables()["dp"] == [[0, 2], [1, 3]]
    c2 = arm_topology("c2_pp2dp2")
    assert c2.group_tables()["pp"] == [[0, 1], [2, 3]] and c2.group_tables()["dp"] == [[0, 2], [1, 3]]


def expect_tied_stage() -> None:
    stage = HybridParallelStage(tiny_config(), 0, 1, 0, 1)
    assert stage.tied_in_stage and stage.is_first and stage.is_last
    assert stage.lm_head.weight is stage.embed_tokens.weight
    ranges, meta = split_layers(LAYERS, 2)
    assert ranges == [(0, 2), (2, 4)] and not meta["uneven"]


def expect_data_partition() -> None:
    topology = ParallelTopology.from_degrees(dp_size=2, tp_size=2, pp_size=2)
    samples = build_samples(8)
    members = tuple(range(8))
    shard_a, accounting_a = shard_micro_batches(samples, members, topology, 0, NUM_MICRO, 0)
    shard_b, accounting_b = shard_micro_batches(samples, members, topology, 1, NUM_MICRO, 0)
    members_a = sorted(sum([list(batch.source_lines) for batch in shard_a], []))
    members_b = sorted(sum([list(batch.source_lines) for batch in shard_b], []))
    assert members_a == [0, 2, 4, 6] and members_b == [1, 3, 5, 7]
    assert sorted(members_a + members_b) == list(range(8))
    assert accounting_a["supervised_global"] == accounting_b["supervised_global"] > 0
    # TP ranks of one DP replica assemble identical micro-batch objects.
    again, _ = shard_micro_batches(samples, members, topology, 0, NUM_MICRO, 0)
    for left, right in zip(shard_a, again):
        assert torch.equal(left.input_ids, right.input_ids)


# ---------------------------------------------------------------------------
# distributed rounds
# ---------------------------------------------------------------------------


def _worker(rank: int, world_size: int, port: int, round_spec: dict) -> None:
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    torch.manual_seed(0)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        _run_round(rank, round_spec)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def _run_round(rank: int, spec: dict) -> None:
    config = tiny_config()
    topology = ParallelTopology.from_degrees(dp_size=spec["dp"], tp_size=spec["tp"], pp_size=spec["pp"])
    assert topology.world_size == spec["world_size"]
    groups = build_parallel_groups(topology)
    membership = groups["membership"]
    dp_rank, pp_rank, tp_rank = topology.decompose(rank)
    assert (membership["dp_rank"], membership["pp_rank"], membership["tp_rank"]) == (dp_rank, pp_rank, tp_rank)
    assert membership["tp_group_ranks"] == sorted(topology.ranks_for(dp_rank=dp_rank, pp_rank=pp_rank))
    assert membership["dp_group_ranks"] == sorted(topology.ranks_for(dp_rank=None, pp_rank=pp_rank, tp_rank=tp_rank))

    reference_state = spec["reference"]["state"]
    stage = HybridParallelStage(config, pp_rank, spec["pp"], tp_rank, spec["tp"], groups["tp_group"])
    stage.load_state_dict(stage_state_from_reference(reference_state, stage, tp_rank, spec["tp"]), strict=True)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=WEIGHT_DECAY, foreach=False, fused=False)

    samples = build_samples(spec["sample_count"])
    global_members = tuple(range(spec["sample_count"]))
    micros, accounting = shard_micro_batches(samples, global_members, topology, dp_rank, NUM_MICRO, 0)
    micro_ds = micro_dicts(micros, accounting["supervised_global"])

    # ---- data flow: DP shards partition the global batch; one shard per replica ----
    payloads = [None] * topology.world_size
    dist.all_gather_object(payloads, {"members": list(accounting["local_members"])})
    per_dp: dict[int, set] = {}
    for peer, item in enumerate(payloads):
        peer_dp = topology.decompose(peer)[0]
        per_dp.setdefault(peer_dp, set()).update(item["members"])
        same_members = item["members"] == list(accounting["local_members"])
        if peer_dp == dp_rank:
            assert same_members, (rank, peer)  # same replica -> same micro-batches
        else:
            assert not same_members, (rank, peer)  # other replicas -> other samples
            assert not set(item["members"]) & set(accounting["local_members"]), (rank, peer)
    covered = sorted(sum([sorted(items) for items in per_dp.values()], []))
    assert covered == list(range(spec["sample_count"])), covered

    metrics = hybrid_train_step([stage], optimizer, micro_ds, torch.device("cpu"), groups, MAX_GRAD_NORM)

    # ---- reference comparisons ----
    loss_error = abs(metrics["global_loss"] - spec["reference"]["loss"]) / abs(spec["reference"]["loss"])
    assert loss_error < LOSS_REL_BAR, (rank, metrics["global_loss"], spec["reference"]["loss"])
    norms = [None] * topology.world_size
    dist.all_gather_object(norms, metrics["gradient_norm_before_clip"])
    norm_error = max(abs(value - spec["reference"]["gradient_norm"]) / spec["reference"]["gradient_norm"] for value in norms)
    assert norm_error < NORM_REL_BAR, (rank, norms)

    # ---- communication inventory ----
    if spec["tp"] > 1:
        layers_per_stage = (LAYERS + spec["pp"] - 1) // spec["pp"]
        assert metrics["tp_allreduce_forward_count"] == layers_per_stage * NUM_MICRO * 2, metrics
        assert metrics["tp_allreduce_backward_count"] == layers_per_stage * NUM_MICRO * 4, metrics
    else:
        assert metrics["tp_allreduce_forward_count"] == 0 and metrics["tp_allreduce_backward_count"] == 0
    # p2p timeline = boundary activations/input-grads (2 per micro) PLUS the
    # tied Embedding/LM-Head exchange pair (send + recv, pp >= 2 only); the
    # README's P2P inventory includes the tied exchange, and the separate
    # tied_exchange_count below must match its contribution exactly.
    expected_p2p = float(2 * NUM_MICRO if spec["pp"] > 1 else 0) + metrics["tied_exchange_count"]
    assert metrics["p2p_count"] == expected_p2p, (rank, metrics["p2p_count"])
    assert metrics["tied_exchange_count"] == (0.0 if spec["pp"] == 1 else 2.0), metrics
    local_params = [parameter for parameter in stage.parameters() if parameter.grad is not None]
    assert metrics["dp_allreduce_count"] == (float(len(local_params)) if spec["dp"] > 1 else 0.0), metrics

    # ---- post-step full-model checksum per DP replica ----
    gathered_state = gather_tensor_parallel_parameters(stage, spec["tp"], group=groups["tp_group"])
    stage_checksum = checksum_state(gathered_state)
    if stage.tied_in_stage:
        # named_parameters() dedups the tied pair, so the gathered dict holds
        # the Embedding once; the reference checksum counts it twice (state
        # dict keys embed + lm_head).  Add the second copy back.
        stage_checksum += checksum_state({"lm_head": stage.lm_head.weight})
    del gathered_state
    checksums = [None] * topology.world_size
    dist.all_gather_object(checksums, {"stage": pp_rank, "tp": tp_rank,
                                       "checksum": stage_checksum, "dp": dp_rank})
    # Every TP rank of a stage reports the SAME TP-gathered stage state, so the
    # replica sum must take ONE rank per stage (tp == 0) — summing all ranks
    # would count every stage tp_size times.
    replica_checksum = sum(item["checksum"] for item in checksums
                           if item["dp"] == dp_rank and item["tp"] == 0)
    param_error = abs(replica_checksum - spec["reference"]["checksum"]) / spec["reference"]["checksum"]
    assert param_error < PARAM_REL_BAR, (rank, replica_checksum, spec["reference"]["checksum"])
    replicas = {item["dp"]: round(item["checksum"], 9) for item in checksums if item["stage"] == (spec["pp"] - 1)}
    if len(replicas) > 1:
        assert len(set(replicas.values())) == 1, replicas  # DP replicas identical

    # ---- tied consistency: embed (first stage) == lm_head (last stage) ----
    tied_checksum = None
    if spec["pp"] == 1:
        tied_checksum = checksum_state({"e": stage.embed_tokens.weight})
    elif stage.is_first:
        tied_checksum = checksum_state({"e": stage.embed_tokens.weight})
    elif stage.is_last:
        tied_checksum = checksum_state({"e": stage.lm_head.weight})
    if tied_checksum is not None:
        tied = [None] * topology.world_size
        dist.all_gather_object(tied, {"role": "embed" if pp_rank == 0 else "lm_head", "value": tied_checksum})
        values = {round(item["value"], 9) for item in tied if item["value"] is not None}
        assert len(values) == 1, values

    ledger = state_ledger([stage])
    if rank == 0:
        print(f"[{spec['name']}] loss_rel_error={loss_error:.2e} param_rel_error={param_error:.2e} "
              f"tp_fwd={metrics['tp_allreduce_forward_count']:.0f} p2p={metrics['p2p_count']:.0f} "
              f"dp_count={metrics['dp_allreduce_count']:.0f} "
              f"local_param_bytes={ledger['local_parameter_bytes']:.0f}")


def _run_round_with_spawn(name: str, dp: int, tp: int, pp: int, sample_count: int,
                          reference: dict) -> None:
    world_size = dp * tp * pp
    port = _free_port()
    spec = {
        "name": name, "dp": dp, "tp": tp, "pp": pp, "world_size": world_size,
        "sample_count": sample_count,
        "reference": {"loss": reference["loss"], "gradient_norm": reference["gradient_norm"],
                      "checksum": reference["checksum"], "state": reference["state"]},
    }
    mp.spawn(_worker, args=(world_size, port, spec), nprocs=world_size, join=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def main() -> None:
    expect_arms_and_groups()
    expect_tied_stage()
    expect_data_partition()
    print("unit checks passed: arms, 3-D group tables, tied pp=1 stage, DP data partition")

    samples8 = build_samples(8)
    reference_a = reference_run(samples8, num_micro_total=2 * NUM_MICRO)  # dp=2, micro=2
    # Round B's workers rebuild build_samples(4) (its own seed-1004 generator),
    # so the reference must consume THAT sample set — slicing build_samples(8)
    # would hand the reference tokens the round never sees.
    reference_b = reference_run(build_samples(1 * NUM_MICRO * SAMPLES_PER_MICRO),
                                num_micro_total=NUM_MICRO)

    # Round A: experiment D topology (2 nodes x 4 GPUs mapped onto 8 gloo ranks).
    _run_round_with_spawn("A_tp2pp2dp2", dp=2, tp=2, pp=2, sample_count=8, reference=reference_a)
    # Round B: the a1_tp2 arm shape (TP=2, PP=1, DP=1; tied-in-stage, no exchange).
    _run_round_with_spawn("B_tp2pp1dp1", dp=1, tp=2, pp=1, sample_count=4, reference=reference_b)
    # Round C: the c2_pp2dp2 arm shape (TP=1: zero TP collectives expected).
    _run_round_with_spawn("C_tp1pp2dp2", dp=2, tp=1, pp=2, sample_count=8, reference=reference_a)
    print("chapter 27 smoke passed: 3-D groups, membership, data flow, one step vs reference")


if __name__ == "__main__":
    main()
