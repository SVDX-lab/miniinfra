#!/usr/bin/env python3
"""CPU-only Chapter 40 checks: 1F1B scheduling, gradient parity, timelines.

Runs a tiny Qwen3 (hidden 32, intermediate 64, vocab 64, 4 layers, 8 Q heads,
4 KV heads, head_dim 8, sequence 12) on CPU gloo groups — a PP=2 round and an
uneven PP=3 round (4 layers -> [2,1,1]), each FOUR micro-batches of 2 samples:

1. ``split_layers`` rules (unchanged from chapter 24) and the ``schedule_1f1b``
   formula: warmup = min(stages - 1 - s, micros), in-flight =
   min(stages - s, micros) < micros wherever a steady phase exists.
2. GPipe fill-drain (the chapter-39 baseline, kept in this chapter) against
   the replicated reference: loss, every weight gradient, stage-boundary input
   gradients, tied Embedding/LM-Head exchange.
3. ``onef1b_step`` (the variant) produces the SAME loss shard and the SAME
   per-parameter gradients as ``gpipe_step`` on the same stage split and the
   same micro-batches — the two arms differ ONLY in scheduling — and both
   match the reference.  Every stage's boundary input gradients still equal
   the reference gradients at the stage's first layer input, per micro.
4. OneF1B timeline: the per-stage kind sequence equals
   ``forward×warmup + (forward,backward)×steady + backward×cooldown``,
   forwards and backwards each run in micro-batch order, the P2P census
   matches the stage position, and the theoretical in-flight count is
   reported per stage.
5. One AdamW step through ``pp_train_step`` (both modes) matches the
   reference step (exact global-norm clip + AdamW on the tied model).
"""

from __future__ import annotations

import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from assistant_loss import IGNORE_INDEX, EncodedConversation
from mini_batch import batch_loss, pad_samples
from pipeline_1f1b import (
    PipelineStage,
    StageTimeline,
    compute_bubble,
    exchange_tied_gradients,
    gpipe_step,
    onef1b_step,
    pp_train_step,
    schedule_1f1b,
    split_layers,
)
from pp_runtime import build_micro_batches, micro_dicts
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
NUM_MICRO = 4


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=LAYERS,
        num_attention_heads=NUM_Q_HEADS, num_key_value_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )


def expect_equal_split(num_layers: int, num_stages: int, expected_counts: list[int]) -> None:
    ranges, meta = split_layers(num_layers, num_stages)
    counts = [end - begin for begin, end in ranges]
    assert counts == expected_counts, (num_layers, num_stages, counts, expected_counts)
    assert meta["stage_layer_counts"] == expected_counts
    assert meta["uneven"] == (num_layers % num_stages != 0)
    assert ranges[0][0] == 0 and ranges[-1][1] == num_layers
    for (_, end_prev), (begin_next, _) in zip(ranges, ranges[1:]):
        assert end_prev == begin_next, "stage ranges must be contiguous"


def expect_schedule_formula() -> None:
    """The 1F1B schedule formula holds and shrinks the in-flight window."""

    for num_stages in (2, 3, 4):
        for num_micro in (1, 2, 4, 8):
            for stage_index in range(num_stages):
                plan = schedule_1f1b(stage_index, num_stages, num_micro)
                assert plan["warmup"] == min(num_stages - 1 - stage_index, num_micro)
                assert plan["steady"] == num_micro - plan["warmup"]
                assert plan["cooldown"] == plan["warmup"]
                assert plan["in_flight_micros"] == min(num_stages - stage_index, num_micro)
                assert plan["in_flight_micros"] <= num_micro
    # The residency hypothesis: with a steady phase the last stage holds far
    # fewer graphs than GPipe's num_micro (here 1 vs 4 on the last stage).
    assert schedule_1f1b(0, 4, 4)["in_flight_micros"] == 4
    assert schedule_1f1b(3, 4, 4)["in_flight_micros"] == 1


def build_reference(generator: torch.Generator) -> Qwen3ForCausalLM:
    torch.manual_seed(23)
    return Qwen3ForCausalLM(tiny_config())


def build_stage(config: Qwen3Config, stage_index: int, num_stages: int,
                reference: Qwen3ForCausalLM) -> PipelineStage:
    """Construct a stage and copy ITS slice of the reference weights."""

    stage = PipelineStage(config, stage_index, num_stages)
    begin, end = stage.layer_range
    state = {}
    for local_index, global_index in enumerate(range(begin, end)):
        for name, value in reference.model.layers[global_index].state_dict().items():
            state[f"layers.{local_index}.{name}"] = value.clone()
    if stage.is_first:
        state["embed_tokens.weight"] = reference.model.embed_tokens.weight.detach().clone()
    if stage.is_last:
        state["norm.weight"] = reference.model.norm.weight.detach().clone()
        state["lm_head.weight"] = reference.model.embed_tokens.weight.detach().clone()
    stage.load_state_dict(state, strict=True)
    return stage


def make_micros(generator: torch.Generator) -> tuple[list[dict], dict]:
    """One deterministic global batch split into micro-batch dicts, plus the
    full-batch tensors for the loss-consistency check."""

    tokens = torch.randint(4, VOCAB, (NUM_MICRO * SAMPLES_PER_MICRO, SEQ), generator=generator)
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
    micro_batches = build_micro_batches(encoded, tuple(range(full_batch.batch_size)), NUM_MICRO, 0)
    return micro_dicts(micro_batches, supervised_global), {
        "input_ids": full_batch.input_ids,
        "labels": full_batch.labels,
        "attention_mask": full_batch.attention_mask,
        "supervised_global": supervised_global,
    }


def snapshot_stage_grads(stage) -> dict:
    return {name: parameter.grad.detach().clone() for name, parameter in stage.named_parameters()}


def check_stage_grads_against_reference(stage, reference_grads: dict, context: str) -> None:
    begin, end = stage.layer_range
    for local_index, global_index in enumerate(range(begin, end)):
        for name, parameter in stage.layers[local_index].named_parameters():
            reference_name = f"model.layers.{global_index}.{name}"
            torch.testing.assert_close(
                parameter.grad, reference_grads[reference_name], rtol=1e-4, atol=1e-6,
                msg=f"{context}: layer {global_index} '{name}' gradient must match the reference",
            )
    if stage.is_first:
        torch.testing.assert_close(
            stage.embed_tokens.weight.grad, reference_grads["model.embed_tokens.weight"],
            rtol=1e-4, atol=1e-6, msg=f"{context}: Embedding gradient must equal the tied reference gradient",
        )
    if stage.is_last:
        torch.testing.assert_close(
            stage.lm_head.weight.grad, reference_grads["model.embed_tokens.weight"],
            rtol=1e-4, atol=1e-6, msg=f"{context}: LM Head gradient must equal the tied reference gradient",
        )
        torch.testing.assert_close(
            stage.norm.weight.grad, reference_grads["model.norm.weight"],
            rtol=1e-4, atol=1e-6, msg=f"{context}: final norm gradient must match the reference",
        )


def reference_micro_trajectory(reference: Qwen3ForCausalLM, micros: list[dict],
                               boundary_layers: list[int]) -> tuple[float, dict, list]:
    """Run the replicated model over the SAME micro-batches, accumulating
    gradients; capture the layer-input gradients at every stage boundary."""

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


def expected_kind_pattern(stage_index: int, num_stages: int, num_micro: int) -> list[str]:
    plan = schedule_1f1b(stage_index, num_stages, num_micro)
    return (
        ["forward"] * plan["warmup"]
        + ["forward", "backward"] * plan["steady"]
        + ["backward"] * plan["cooldown"]
    )


def check_onef1b_timeline(events: list[dict], stage_index: int, num_stages: int) -> None:
    """The executed 1F1B order: warmup Fs, alternating F/B, cooldown Bs."""

    compute = [event for event in events if event["kind"] in ("forward", "backward")]
    kinds = [event["kind"] for event in compute]
    assert kinds == expected_kind_pattern(stage_index, num_stages, NUM_MICRO), (stage_index, kinds)
    forwards = [event for event in compute if event["kind"] == "forward"]
    backwards = [event for event in compute if event["kind"] == "backward"]
    assert [event["micro"] for event in forwards] == list(range(NUM_MICRO)), stage_index
    assert [event["micro"] for event in backwards] == list(range(NUM_MICRO)), stage_index
    # Every micro's backward starts only after its own forward finished.
    forward_by_micro = {event["micro"]: event for event in forwards}
    for event in backwards:
        assert forward_by_micro[event["micro"]]["end_s"] <= event["start_s"] + 1e-9
    # The 1F1B signature on a mid/last stage: a backward lands between two
    # forwards (the steady alternation), unlike GPipe's F...F B...B blocks.
    plan = schedule_1f1b(stage_index, num_stages, NUM_MICRO)
    if plan["steady"] >= 2:
        anchored_forward_end = max(event["end_s"] for event in forwards[: plan["warmup"] + 1])
        first_backward_start = min(event["start_s"] for event in backwards)
        assert first_backward_start >= anchored_forward_end - 1e-9
        # A later forward still follows the first backward (alternation).
        assert first_backward_start < max(event["end_s"] for event in forwards) + 1e-9


def check_boundary_grads(result: dict, stage, captures: list[dict], context: str) -> None:
    if stage.is_first:
        return
    begin = stage.layer_range[0]
    entries = [entry for entry in captures if entry["layer"] == begin]
    assert len(entries) == NUM_MICRO, (len(entries), NUM_MICRO)
    for micro_index, grad in enumerate(result["input_grads"]):
        torch.testing.assert_close(
            grad, entries[micro_index]["grad"], rtol=1e-4, atol=1e-6,
            msg=f"{context}: stage {stage.stage_index} boundary input gradient (micro {micro_index}) "
                "must match the reference gradient at the stage's first layer input",
        )


def _round_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(5)

    if rank == 0:
        expect_equal_split(LAYERS, 2, [2, 2])
        expect_equal_split(LAYERS, 3, [2, 1, 1])
        expect_equal_split(10, 4, [3, 3, 2, 2])
        expect_schedule_formula()

    reference = build_reference(generator)
    micros, full_batch = make_micros(generator)

    # Consistency of the loss scale: the micro-summed objective equals the
    # full-batch token_mean (padding cannot change supervised contributions).
    ref_logits = reference(full_batch["input_ids"], attention_mask=full_batch["attention_mask"])
    full_batch_loss = float(batch_loss(ref_logits, full_batch["labels"], "token_mean").detach())
    ranges, _ = split_layers(LAYERS, world_size)
    boundary_layers = [begin for begin, _ in ranges[1:]]
    ref_loss, reference_grads, boundary_captures = reference_micro_trajectory(
        reference, micros, boundary_layers
    )
    assert abs(ref_loss - full_batch_loss) < 1e-5 * max(1.0, abs(full_batch_loss)), \
        (ref_loss, full_batch_loss)

    config = tiny_config()
    stage = build_stage(config, rank, world_size, reference)

    # ---- (2) GPipe fill-drain baseline: gradients only ----
    timeline = StageTimeline(device)
    gpipe_result = gpipe_step(stage, micros, device, timeline)
    exchange_tied_gradients(stage, device)
    gpipe_events = timeline.end_step()

    shards = [None] * world_size
    dist.all_gather_object(shards, gpipe_result["local_loss_shard"])
    gpipe_loss = float(sum(shards))
    assert abs(gpipe_loss - ref_loss) < 1e-5 * max(1.0, abs(ref_loss)), (gpipe_loss, ref_loss)

    check_stage_grads_against_reference(stage, reference_grads, "gpipe")
    check_boundary_grads(gpipe_result, stage, boundary_captures, "gpipe")
    # Fill-drain ordering of the baseline still holds: on the last stage all
    # forwards strictly precede all backwards.
    if stage.is_last:
        forward_end = max(e["end_s"] for e in gpipe_events if e["kind"] == "forward")
        backward_start = min(e["start_s"] for e in gpipe_events if e["kind"] == "backward")
        assert forward_end <= backward_start + 1e-9, \
            "GPipe on the last stage must finish ALL forwards before ANY backward"
    gpipe_grads = snapshot_stage_grads(stage)

    # ---- (3)(4) OneF1B: same grads, different schedule, checked timeline ----
    for parameter in stage.parameters():
        parameter.grad = None
    onef1b_timeline = StageTimeline(device)
    onef1b_result = onef1b_step(stage, micros, device, onef1b_timeline)
    onef1b_events = onef1b_timeline.end_step()
    tied = exchange_tied_gradients(stage, device)

    onef1b_shards = [None] * world_size
    dist.all_gather_object(onef1b_shards, onef1b_result["local_loss_shard"])
    onef1b_loss = float(sum(onef1b_shards))
    assert abs(onef1b_loss - gpipe_loss) < 1e-6 * max(1.0, abs(gpipe_loss)), (onef1b_loss, gpipe_loss)

    check_stage_grads_against_reference(stage, reference_grads, "onef1b")
    check_boundary_grads(onef1b_result, stage, boundary_captures, "onef1b")
    onef1b_grads = snapshot_stage_grads(stage)
    for name, grad in gpipe_grads.items():
        torch.testing.assert_close(
            onef1b_grads[name], grad, rtol=1e-6, atol=1e-7,
            msg=f"onef1b and gpipe must produce identical gradients for {name} (only scheduling differs)",
        )
    assert onef1b_result["schedule"]["in_flight_micros"] == min(world_size - rank, NUM_MICRO)
    assert tied["tied_exchange_count"] > 0 or (not stage.is_first and not stage.is_last)

    expected_sends = (0 if stage.is_last else NUM_MICRO) + (0 if stage.is_first else NUM_MICRO)
    expected_recvs = (0 if stage.is_first else NUM_MICRO) + (0 if stage.is_last else NUM_MICRO)
    kinds = [event["kind"] for event in onef1b_events]
    assert kinds.count("forward") == NUM_MICRO and kinds.count("backward") == NUM_MICRO, kinds
    assert kinds.count("p2p_send") == expected_sends, kinds
    assert kinds.count("p2p_recv") == expected_recvs, kinds
    bubble = compute_bubble(onef1b_events)
    assert 0.0 <= bubble["bubble_ratio"] <= 1.0 and bubble["window_s"] > 0, bubble
    check_onef1b_timeline(onef1b_events, rank, world_size)

    # ---- (5) one AdamW step through pp_train_step, both modes ----
    # The parity construction: the pipeline stage starts from the PRE-step
    # reference weights (W0), consumes the same micro-batch gradients, clips
    # with the same exact global norm and applies the same AdamW — its
    # post-step parameters must equal the reference's (W1).
    reference_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.1, foreach=False, fused=False)
    reference_opt.zero_grad(set_to_none=True)
    for micro in micros:
        logits = reference(micro["input_ids"], attention_mask=micro["attention_mask"])
        (batch_loss(logits, micro["labels"], "token_mean") * micro["scale"]).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    pre_step_reference = copy.deepcopy(reference)
    reference_opt.step()
    # `stepped_named` holds W1; the pipeline stage starts from the W0 copy.
    stepped_named = dict(reference.named_parameters())

    for mode in ("gpipe", "onef1b"):
        step_stage = build_stage(config, rank, world_size, pre_step_reference)
        optimizer = torch.optim.AdamW(step_stage.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.1, foreach=False, fused=False)
        metrics = pp_train_step(step_stage, optimizer, micros, device, mode, max_grad_norm=1.0)
        assert abs(metrics["global_loss"] - ref_loss) < 1e-5 * max(1.0, abs(ref_loss)), metrics
        assert metrics["gradient_norm_before_clip"] > 0 and metrics["step_time_ms"] >= 0
        assert metrics["mode"] == mode
        if mode == "onef1b":
            assert metrics["warmup_micros"] == min(world_size - 1 - rank, NUM_MICRO)
            assert metrics["cooldown_micros"] == metrics["warmup_micros"]
            assert metrics["steady_micros"] == NUM_MICRO - metrics["warmup_micros"]
            assert metrics["in_flight_micros"] == min(world_size - rank, NUM_MICRO)
        # Post-step comparison against the stepped reference, scale-relative
        # (first AdamW updates are sign-like: m/sqrt(v) amplifies FP32 noise
        # in near-zero gradients — the chapter-39 smoke uses the same bar).
        begin, end = step_stage.layer_range
        for local_index, global_index in enumerate(range(begin, end)):
            for name, parameter in step_stage.layers[local_index].named_parameters():
                reference_parameter = stepped_named[f"model.layers.{global_index}.{name}"]
                difference = (parameter.detach() - reference_parameter.detach()).abs().max().item()
                scale = reference_parameter.detach().abs().max().item()
                assert difference / max(scale, 1e-9) < 5e-3, (mode, name, difference, scale)
        tied_reference = stepped_named["model.embed_tokens.weight"]
        if step_stage.is_first:
            difference = (step_stage.embed_tokens.weight.detach() - tied_reference.detach()).abs().max().item()
            assert difference / tied_reference.detach().abs().max().item() < 5e-3, mode
        if step_stage.is_last:
            difference = (step_stage.lm_head.weight.detach() - tied_reference.detach()).abs().max().item()
            assert difference / tied_reference.detach().abs().max().item() < 5e-3, mode
            norm_parameter = stepped_named["model.norm.weight"]
            difference = (step_stage.norm.weight.detach() - norm_parameter.detach()).abs().max().item()
            assert difference / max(norm_parameter.detach().abs().max().item(), 1e-9) < 5e-3, mode
        del optimizer, step_stage

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"pp={world_size}: splits, schedule formula, gpipe/onef1b gradient parity vs reference, "
              f"boundary grads, tie exchange, AdamW step parity, 1F1B timeline pattern: OK")


def main() -> None:
    for world_size in (2, 3):
        port = _free_port()
        mp.spawn(_round_worker, args=(world_size, port), nprocs=world_size, join=True)
    print("chapter_40 smoke tests passed")


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


if __name__ == "__main__":
    main()
