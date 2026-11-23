#!/usr/bin/env python3
"""CPU-only Chapter 41 checks: virtual-stage placement, interleaved scheduling.

Runs a tiny Qwen3 (hidden 32, intermediate 64, vocab 64, 8 layers, 8 Q heads,
4 KV heads, head_dim 8, sequence 12) on CPU gloo groups over THREE rounds,
all consuming the SAME deterministic micro-batches (4 micro-batches of 2
samples) and the SAME replicated reference:

- Round A ``interleaved``: 2 physical ranks x 2 virtual stages (total
  stages 4, 2 layers each; rank r owns NON-contiguous virtual stages
  {r, r+2}).  Checks the divmod split rule, the strided placement table,
  the schedule formula (group size = P, warmup = min(2*(P-1-r) + (vpp-1)*P,
  vpp*M)), the interleaved step against the reference (loss, every weight
  gradient per chunk slice, stage-boundary input gradients per micro, tied
  Embedding/LM-Head exchange), the per-rank F/B kind pattern, the local
  mailbox transport, and one AdamW step through ``pp_train_step``.  Step-0
  gradients are stashed under GLOBAL layer names for the cross-round checks.
- Round B ``onef1b`` (baseline): 4 physical ranks x 1 stage — the SAME total
  stage count (4) as round A without interleaving.  Its gradients must equal
  the stashed interleaved gradients (only placement/scheduling differ) and
  its total P2P event count must match (same stage graph, different owners).
- Round C ``onef1b``: 2 physical ranks x 1 stage — the SAME physical GPUs as
  round A without virtual stages.  Gradients again equal the stash, and the
  total P2P event count is far SMALLER than interleaved's: the interleaved
  schedule buys its bubble reduction with more, smaller-stage P2P hops.
- Round D ``interleaved``: 4 physical ranks x 2 virtual stages (total
  stages 8, ONE layer per chunk) — exercises the wrapped P-1<->0 pairs and
  the deepest warmup (rank 0 runs all 8 forward items before its first
  backward).
- Round E ``interleaved``: 2 physical ranks x 2 virtual stages with EIGHT
  micro-batches — two complete micro-batch groups of size P.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from assistant_loss import IGNORE_INDEX, EncodedConversation
from interleaved import (
    StageTimeline,
    VirtualStage,
    _MAILBOX,
    compute_bubble,
    exchange_tied_gradients,
    interleaved_1f1b_step,
    interleaved_schedule,
    onef1b_step,
    pass_activation,
    pp_train_step,
    receive_activation,
    split_virtual_stages,
    stage_chunk,
    stage_owner,
)
from mini_batch import batch_loss, pad_samples
from pp_runtime import build_micro_batches, micro_dicts
from qwen3_model import Qwen3Config, Qwen3ForCausalLM


HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 64
LAYERS = 8
NUM_Q_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 8
SEQ = 12
SAMPLES_PER_MICRO = 2
NUM_MICRO = 4
STASH_PATH = Path("chapter41-smoke-stash.pt")


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=LAYERS,
        num_attention_heads=NUM_Q_HEADS, num_key_value_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )


def expect_split_rules() -> None:
    ranges, meta = split_virtual_stages(LAYERS, 4)
    assert [end - begin for begin, end in ranges] == [2, 2, 2, 2] and not meta["uneven"]
    ranges, meta = split_virtual_stages(28, 8)
    # The documented remainder rule: earlier virtual stages take the extra
    # layer (28 = 8*3 + 4 -> [4,4,4,4,3,3,3,3]); with vpp=2, P=4 every rank
    # still holds 7 layers even though its chunks are non-contiguous.
    assert meta["stage_layer_counts"] == [4, 4, 4, 4, 3, 3, 3, 3] and meta["remainder_layers"] == 4
    assert [sum(end - begin for begin, end in ranges[rank::4]) for rank in range(4)] == [7, 7, 7, 7]
    ranges, meta = split_virtual_stages(10, 4)
    assert meta["stage_layer_counts"] == [3, 3, 2, 2] and meta["uneven"]


def expect_placement() -> None:
    for virtual in range(8):
        assert stage_owner(virtual, 4) == virtual % 4
        assert stage_chunk(virtual, 4) == virtual // 4
    # Rank r owns the non-contiguous virtual stages {r, r+P, ...}.
    assert [stage_owner(v, 2) for v in range(4)] == [0, 1, 0, 1]
    assert [stage_chunk(v, 2) for v in range(4)] == [0, 0, 1, 1]


def expect_schedule_formula() -> None:
    # P=2, vpp=2, M=4: warmup 4 on rank 0, 2 on rank 1; groups of P micros.
    plan0 = interleaved_schedule(0, 2, 2, 4)
    plan1 = interleaved_schedule(1, 2, 2, 4)
    assert (plan0["warmup"], plan0["steady"], plan0["cooldown"]) == (4, 4, 4), plan0
    assert (plan1["warmup"], plan1["steady"], plan1["cooldown"]) == (2, 6, 2), plan1
    assert plan0["in_flight_micros"] == 5 and plan1["in_flight_micros"] == 3
    assert plan0["micro_group_size"] == 2
    # The group structure requires M % P == 0 — violated configs raise.
    try:
        interleaved_schedule(0, 2, 2, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("num_micro % num_physical != 0 must raise")
    try:
        interleaved_schedule(0, 2, 1, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("vpp < 2 must raise for the interleaved schedule")


def check_local_mailbox() -> None:
    """The same-owner transport: direct handoff, no communication."""

    tensor = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    pass_activation(tensor, src_virtual=0, dst_virtual=1, num_physical=1, timeline=None, micro=0)
    received = receive_activation((1, 2, 3), torch.float32, torch.device("cpu"),
                                  src_virtual=0, dst_virtual=1, num_physical=1, timeline=None, micro=0)
    assert torch.equal(received, tensor)
    try:
        _MAILBOX.take(0, 1, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("mailbox slot must be consumed by the first take")


def build_reference(generator: torch.Generator) -> Qwen3ForCausalLM:
    torch.manual_seed(23)
    return Qwen3ForCausalLM(tiny_config())


def build_chunks(config: Qwen3Config, rank: int, num_physical: int, vpp: int,
                 reference: Qwen3ForCausalLM) -> list[VirtualStage]:
    """Construct this rank's chunks and copy THEIR slices of the reference."""

    total_stages = num_physical * vpp
    chunks = [
        VirtualStage(config, rank + chunk * num_physical, total_stages, num_physical)
        for chunk in range(vpp)
    ]
    for chunk in chunks:
        begin, end = chunk.layer_range
        state = {}
        for local_index, global_index in enumerate(range(begin, end)):
            for name, value in reference.model.layers[global_index].state_dict().items():
                state[f"layers.{local_index}.{name}"] = value.clone()
        if chunk.is_first:
            state["embed_tokens.weight"] = reference.model.embed_tokens.weight.detach().clone()
        if chunk.is_last:
            state["norm.weight"] = reference.model.norm.weight.detach().clone()
            state["lm_head.weight"] = reference.model.embed_tokens.weight.detach().clone()
        chunk.load_state_dict(state, strict=True)
    return chunks


def make_micros(generator: torch.Generator, num_micro: int = NUM_MICRO) -> tuple[list[dict], dict]:
    tokens = torch.randint(4, VOCAB, (num_micro * SAMPLES_PER_MICRO, SEQ), generator=generator)
    labels = tokens.clone()
    labels[:, :2] = IGNORE_INDEX
    attention_mask = torch.ones_like(tokens)

    encoded = [
        EncodedConversation(
            input_ids=tokens[row].unsqueeze(0), attention_mask=attention_mask[row].unsqueeze(0),
            labels=labels[row].unsqueeze(0), assistant_loss_mask=torch.ones_like(labels[row].unsqueeze(0)),
            source_line=row, rendered_text="", token_spans=(), loss_mode="assistant_only",
        )
        for row in range(tokens.shape[0])
    ]
    full_batch = pad_samples(encoded, pad_token_id=0)
    supervised_global = float((full_batch.labels[:, 1:] != IGNORE_INDEX).sum())
    micro_batches = build_micro_batches(encoded, tuple(range(full_batch.batch_size)), num_micro, 0)
    return micro_dicts(micro_batches, supervised_global), {
        "input_ids": full_batch.input_ids,
        "labels": full_batch.labels,
        "attention_mask": full_batch.attention_mask,
    }


def reference_micro_trajectory(reference: Qwen3ForCausalLM, micros: list[dict],
                               boundary_layers: list[int]) -> tuple[float, dict, list]:
    captured: list[dict] = []
    handles = []
    for boundary in sorted(boundary_layers):
        def hook(module, grad_input, grad_output, _layer=boundary):
            captured.append({"layer": _layer, "grad": grad_input[0].detach().clone()})
        handles.append(reference.model.layers[boundary].register_full_backward_hook(hook))

    total = 0.0
    for micro in micros:
        logits = reference(micro["input_ids"], attention_mask=micro["attention_mask"])
        loss = batch_loss(logits, micro["labels"], "token_mean") * micro["scale"]
        loss.backward()
        total += float(loss.detach())
    for handle in handles:
        handle.remove()
    reference_grads = {name: parameter.grad.detach().clone() for name, parameter in reference.named_parameters()}
    return total, reference_grads, captured


def global_parameter_name(chunk: VirtualStage, local_name: str) -> str:
    begin, _ = chunk.layer_range
    if local_name.startswith("layers."):
        local_layer, rest = local_name.split(".", 1)[1].split(".", 1)
        return f"model.layers.{begin + int(local_layer)}.{rest}"
    if local_name in ("embed_tokens.weight", "lm_head.weight"):
        # Tied pair: kept under distinct keys so the stash holds both partials.
        return f"model.{local_name}"
    if local_name == "norm.weight":
        return "model.norm.weight"
    raise ValueError(f"unexpected chunk parameter {local_name}")


def snapshot_global_grads(chunks: list[VirtualStage]) -> dict:
    snapshot = {}
    for chunk in chunks:
        for name, parameter in chunk.named_parameters():
            if parameter.grad is not None:
                snapshot[global_parameter_name(chunk, name)] = parameter.grad.detach().clone()
    return snapshot


def check_chunk_grads_against_reference(chunks: list[VirtualStage], reference_grads: dict,
                                        context: str) -> None:
    for chunk in chunks:
        begin, end = chunk.layer_range
        for local_index, global_index in enumerate(range(begin, end)):
            for name, parameter in chunk.layers[local_index].named_parameters():
                reference_name = f"model.layers.{global_index}.{name}"
                torch.testing.assert_close(
                    parameter.grad, reference_grads[reference_name], rtol=1e-4, atol=1e-6,
                    msg=f"{context}: layer {global_index} '{name}' gradient must match the reference",
                )
        if chunk.is_first:
            torch.testing.assert_close(
                chunk.embed_tokens.weight.grad, reference_grads["model.embed_tokens.weight"],
                rtol=1e-4, atol=1e-6, msg=f"{context}: Embedding gradient must equal the tied reference gradient",
            )
        if chunk.is_last:
            torch.testing.assert_close(
                chunk.lm_head.weight.grad, reference_grads["model.embed_tokens.weight"],
                rtol=1e-4, atol=1e-6, msg=f"{context}: LM Head gradient must equal the tied reference gradient",
            )
            torch.testing.assert_close(
                chunk.norm.weight.grad, reference_grads["model.norm.weight"],
                rtol=1e-4, atol=1e-6, msg=f"{context}: final norm gradient must match the reference",
            )


def check_boundary_grads(input_grads: dict, chunks: list[VirtualStage], captures: list[dict],
                         context: str, num_micro: int) -> None:
    for chunk in chunks:
        if chunk.is_first:
            continue
        begin = chunk.layer_range[0]
        entries = [entry for entry in captures if entry["layer"] == begin]
        assert len(entries) == num_micro, (len(entries), num_micro)
        for micro_index in range(num_micro):
            grad = input_grads[f"chunk{chunk.local_chunk}_micro{micro_index}"]
            torch.testing.assert_close(
                grad, entries[micro_index]["grad"], rtol=1e-4, atol=1e-6,
                msg=f"{context}: virtual stage {chunk.virtual_index} boundary input gradient "
                    f"(micro {micro_index}) must match the reference at its first layer input",
            )


def check_timeline_pattern(events: list[dict], plan: dict) -> None:
    """Per-rank kind sequence F×w + (F,B)×R + B×w; micro order kept per chunk."""

    compute = [event for event in events if event["kind"] in ("forward", "backward")]
    kinds = [event["kind"] for event in compute]
    assert kinds == ["forward"] * plan["warmup"] + ["forward", "backward"] * plan["steady"] \
        + ["backward"] * plan["cooldown"], kinds
    forwards = [event for event in compute if event["kind"] == "forward"]
    backwards = [event for event in compute if event["kind"] == "backward"]
    assert len(forwards) == plan["warmup"] + plan["steady"]
    assert len(backwards) == plan["steady"] + plan["cooldown"]
    for chunk_id in sorted({event["chunk"] for event in compute}):
        chunk_forwards = [event["micro"] for event in forwards if event["chunk"] == chunk_id]
        chunk_backwards = [event["micro"] for event in backwards if event["chunk"] == chunk_id]
        assert chunk_forwards == sorted(chunk_forwards), (chunk_id, chunk_forwards)
        assert chunk_backwards == sorted(chunk_backwards), (chunk_id, chunk_backwards)


def _round_worker(rank: int, world_size: int, vpp: int, mode: str, num_micro: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(5)

    if rank == 0:
        expect_split_rules()
        expect_placement()
        expect_schedule_formula()
        check_local_mailbox()

    reference = build_reference(generator)
    micros, full_batch = make_micros(generator, num_micro)

    ref_logits = reference(full_batch["input_ids"], attention_mask=full_batch["attention_mask"])
    full_batch_loss = float(batch_loss(ref_logits, full_batch["labels"], "token_mean").detach())
    config = tiny_config()
    total_stages = world_size * vpp
    ranges, _ = split_virtual_stages(LAYERS, total_stages)
    boundary_layers = [begin for begin, _ in ranges[1:]]
    ref_loss, reference_grads, boundary_captures = reference_micro_trajectory(
        reference, micros, boundary_layers
    )
    assert abs(ref_loss - full_batch_loss) < 1e-5 * max(1.0, abs(full_batch_loss)), (ref_loss, full_batch_loss)

    chunks = build_chunks(config, rank, world_size, vpp, reference)

    # ---- step-0 grads through the schedule (timeline WITHOUT the tied
    # exchange, so the P2P census below counts only activation/grad traffic) ----
    timeline = StageTimeline(device)
    if mode == "interleaved":
        result = interleaved_1f1b_step(chunks, micros, device, timeline)
    else:
        result = onef1b_step(chunks, micros, device, timeline)
    events = timeline.end_step()
    exchange_tied_gradients(chunks, device)

    shards = [None] * world_size
    dist.all_gather_object(shards, result["local_loss_shard"])
    step_loss = float(sum(shards))
    assert abs(step_loss - ref_loss) < 1e-5 * max(1.0, abs(ref_loss)), (step_loss, ref_loss)

    check_chunk_grads_against_reference(chunks, reference_grads, mode)
    check_boundary_grads(result["input_grads"], chunks, boundary_captures, mode, num_micro)
    check_timeline_pattern(events, result["schedule"])
    bubble = compute_bubble(events)
    assert 0.0 <= bubble["bubble_ratio"] <= 1.0 and bubble["window_s"] > 0, bubble
    assert result["schedule"]["in_flight_micros"] <= result["schedule"]["total_items"]

    p2p_count = sum(1 for event in events if event["kind"].startswith("p2p"))
    local_count = sum(1 for event in events if event["kind"] == "local_handoff")
    # The strided placement keeps every inter-stage edge remote: no local
    # handoff may fire.  Each owned stage sends/receives M activations and
    # M gradients (send + recv = 2 events per message) unless it is the
    # first (no activation recv, no grad send) or the last (no activation
    # send, no grad recv) virtual stage.
    assert local_count == 0, local_count
    expected_events = sum(
        (0 if chunk.is_last else 2 * num_micro) + (0 if chunk.is_first else 2 * num_micro)
        for chunk in chunks
    )
    assert p2p_count == expected_events, (mode, p2p_count, expected_events)

    # ---- one AdamW step through pp_train_step, then compare to the
    # stepped reference (exact global-norm clip + AdamW on the tied model).
    # The pipeline stage starts from the PRE-step reference weights (W0) and
    # must land on the reference's post-step parameters (W1). ----
    reference_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.1, foreach=False, fused=False)
    reference_opt.zero_grad(set_to_none=True)
    for micro in micros:
        logits = reference(micro["input_ids"], attention_mask=micro["attention_mask"])
        (batch_loss(logits, micro["labels"], "token_mean") * micro["scale"]).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    pre_step_reference = copy.deepcopy(reference)
    reference_opt.step()
    stepped_named = dict(reference.named_parameters())

    step_chunks = build_chunks(config, rank, world_size, vpp, pre_step_reference)
    step_optimizer = torch.optim.AdamW(
        [parameter for chunk in step_chunks for parameter in chunk.parameters()],
        lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False,
    )
    metrics = pp_train_step(step_chunks, step_optimizer, micros, device, mode, max_grad_norm=1.0)
    assert abs(metrics["global_loss"] - ref_loss) < 1e-5 * max(1.0, abs(ref_loss)), metrics
    assert metrics["mode"] == mode and metrics["gradient_norm_before_clip"] > 0
    for chunk in step_chunks:
        begin, end = chunk.layer_range
        for local_index, global_index in enumerate(range(begin, end)):
            for name, parameter in chunk.layers[local_index].named_parameters():
                reference_parameter = stepped_named[f"model.layers.{global_index}.{name}"]
                difference = (parameter.detach() - reference_parameter.detach()).abs().max().item()
                scale = reference_parameter.detach().abs().max().item()
                assert difference / max(scale, 1e-9) < 5e-3, (mode, name, difference, scale)
    tied_reference = stepped_named["model.embed_tokens.weight"]
    for chunk in step_chunks:
        if chunk.is_first:
            difference = (chunk.embed_tokens.weight.detach() - tied_reference.detach()).abs().max().item()
            assert difference / tied_reference.detach().abs().max().item() < 5e-3, mode
        if chunk.is_last:
            difference = (chunk.lm_head.weight.detach() - tied_reference.detach()).abs().max().item()
            assert difference / tied_reference.detach().abs().max().item() < 5e-3, mode
            norm_parameter = stepped_named["model.norm.weight"]
            difference = (chunk.norm.weight.detach() - norm_parameter.detach()).abs().max().item()
            assert difference / max(norm_parameter.detach().abs().max().item(), 1e-9) < 5e-3, mode
    del step_chunks, step_optimizer

    # ---- stash the step-0 grads (round A), compare against them (B, C) ----
    own_grads = snapshot_global_grads(chunks)  # step-0 grads, global parameter names
    gathered_grads = [None] * world_size
    dist.all_gather_object(gathered_grads, own_grads)
    gathered_p2p = [None] * world_size
    dist.all_gather_object(gathered_p2p, p2p_count)
    total_p2p = sum(gathered_p2p)

    is_stash_writer = mode == "interleaved" and world_size == 2 and vpp == 2 and num_micro == NUM_MICRO
    uses_stash = num_micro == NUM_MICRO
    if rank == 0 and is_stash_writer:
        merged = {}
        for partial in gathered_grads:
            merged.update(partial)
        stash = {"grads": merged, "p2p_count_total": total_p2p}
        temporary = STASH_PATH.with_suffix(".pt.tmp")
        torch.save(stash, temporary)
        os.replace(temporary, STASH_PATH)
    dist.barrier()

    if uses_stash:
        stash = torch.load(STASH_PATH, map_location="cpu", weights_only=False)
        for name, grad in own_grads.items():
            torch.testing.assert_close(
                grad, stash["grads"][name], rtol=1e-6, atol=1e-7,
                msg=f"{mode} (P={world_size}, vpp={vpp}): parameter {name} gradient must match the "
                    "round-A interleaved schedule (only placement/scheduling differs)",
            )
        if is_stash_writer:
            pass
        elif world_size == 4 and vpp == 1:
            # Same TOTAL stage count as round A: identical stage graph, so the
            # total message count matches even though the owners differ.
            assert total_p2p == stash["p2p_count_total"], (total_p2p, stash["p2p_count_total"])
        elif world_size == 2 and vpp == 1:
            # Same PHYSICAL GPUs as round A, no virtual stages: interleaved
            # pays strictly more P2P messages for its shorter virtual stages.
            assert total_p2p < stash["p2p_count_total"], (total_p2p, stash["p2p_count_total"])

    if rank == 0:
        print(f"round {mode} P={world_size} vpp={vpp} M={num_micro}: schedule, loss/grads/tie/step vs "
              f"reference, cross-round parity, p2p events total {total_p2p}: OK", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    STASH_PATH.unlink(missing_ok=True)
    rounds = [
        (2, 2, "interleaved", NUM_MICRO),  # round A: 2 GPUs x 2 virtual stages (the variant)
        (4, 1, "onef1b", NUM_MICRO),       # round B: 4 GPUs x 1 stage (same total stages)
        (2, 1, "onef1b", NUM_MICRO),       # round C: 2 GPUs x 1 stage (same physical GPUs)
        (4, 2, "interleaved", NUM_MICRO),  # round D: 4 GPUs x 2 virtual stages (wrap pairs, 1 layer/chunk)
        (2, 2, "interleaved", 8),          # round E: two micro-batch groups (M = 2P)
    ]
    for world_size, vpp, mode, num_micro in rounds:
        port = _free_port()
        mp.spawn(_round_worker, args=(world_size, vpp, mode, num_micro, port), nprocs=world_size, join=True)
    STASH_PATH.unlink(missing_ok=True)
    print("chapter_41 smoke tests passed")


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


if __name__ == "__main__":
    main()
