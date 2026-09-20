#!/usr/bin/env python3
"""CPU-only Chapter 39 checks: layer splits, Fill-Drain scheduling, gradients.

Runs a tiny Qwen3 (hidden 32, intermediate 64, vocab 64, 4 layers, 8 Q heads,
4 KV heads, head_dim 8, sequence 12) on CPU gloo groups — a PP=2 round and an
uneven PP=3 round (4 layers -> [2,1,1]), each 2 micro-batches of 2 samples:

1. ``split_layers``: even splits (4/2 -> [2,2], 4/4 -> [1,1,1,1]) and the
   remainder rule (4/3 -> [2,1,1], 10/4 -> [3,3,2,2]; earlier stages take the
   extra layer).
2. GPipe fill-drain against the replicated reference (which consumes the SAME
   micro-batches, accumulating gradients): global token-mean loss assembled
   from the per-micro scales, EVERY weight gradient (stage-local layers to
   their reference slices; Embedding and LM Head both equal the tied reference
   gradient after the P2P exchange), and the stage-boundary input gradients
   (stage k's received gradients == the reference gradient at layer
   ``begin(k)``'s input, per micro-batch).
3. ``serial_step`` (baseline) produces the SAME gradients as ``gpipe_step``
   on the same stage split and micro-batches — the two arms differ ONLY in
   scheduling, which is the chapter's single variable.
4. One AdamW step through ``pp_train_step`` (both modes) matches the
   reference step (exact global-norm clip + AdamW on the tied model).
5. Timeline: every micro-batch has forward/backward events, the P2P census
   matches the stage position, the LAST stage's GPipe fill strictly precedes
   its drain (scheduling actually happened), and bubble ratios are finite.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from assistant_loss import IGNORE_INDEX, EncodedConversation
from mini_batch import batch_loss, pad_samples
from pipeline_parallel import (
    StageTimeline,
    compute_bubble,
    exchange_tied_gradients,
    gpipe_step,
    pp_train_step,
    serial_step,
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
NUM_MICRO = 2


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


def build_reference(generator: torch.Generator) -> Qwen3ForCausalLM:
    torch.manual_seed(23)
    return Qwen3ForCausalLM(tiny_config())


def build_stage(config: Qwen3Config, stage_index: int, num_stages: int,
                reference: Qwen3ForCausalLM) -> "PipelineStage":
    """Construct a stage and copy ITS slice of the reference weights."""

    from pipeline_parallel import PipelineStage

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


def check_stage_grads_against_reference(stage, reference: Qwen3ForCausalLM,
                                        reference_grads: dict, context: str) -> None:
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


def reference_micro_trajectory(reference: Qwen3ForCausalLM, micros: list[dict]) -> tuple[float, dict, list]:
    """Run the replicated model over the SAME micro-batches, accumulating
    gradients; capture the layer-input gradients at every stage boundary."""

    boundaries: set[int] = set()
    for num_stages in (2, 3):
        if num_stages == world_size_of_round:
            ranges, _ = split_layers(LAYERS, num_stages)
            boundaries.update(begin for begin, _ in ranges[1:])
    captured: list[dict] = []

    handles = []
    for boundary in sorted(boundaries):
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


world_size_of_round = 2  # set by the worker before the trajectory runs


def _round_worker(rank: int, world_size: int, port: int) -> None:
    global world_size_of_round
    world_size_of_round = world_size
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(5)

    if rank == 0:
        expect_equal_split(LAYERS, 2, [2, 2])
        expect_equal_split(LAYERS, 4, [1, 1, 1, 1])
        expect_equal_split(LAYERS, 3, [2, 1, 1])
        expect_equal_split(10, 4, [3, 3, 2, 2])

    reference = build_reference(generator)
    micros, full_batch = make_micros(generator)

    # Consistency of the loss scale: the micro-summed objective equals the
    # full-batch token_mean (padding cannot change supervised contributions).
    ref_logits = reference(full_batch["input_ids"], attention_mask=full_batch["attention_mask"])
    full_batch_loss = float(batch_loss(ref_logits, full_batch["labels"], "token_mean").detach())
    ref_loss, reference_grads, boundary_captures = reference_micro_trajectory(reference, micros)
    assert abs(ref_loss - full_batch_loss) < 1e-5 * max(1.0, abs(full_batch_loss)), \
        (ref_loss, full_batch_loss)

    config = tiny_config()
    stage = build_stage(config, rank, world_size, reference)

    # ---- (2) GPipe fill-drain: gradients only ----
    timeline = StageTimeline(device)
    gpipe_result = gpipe_step(stage, micros, device, timeline)
    tied = exchange_tied_gradients(stage, device)
    gpipe_events = timeline.end_step()

    shards = [None] * world_size
    dist.all_gather_object(shards, gpipe_result["local_loss_shard"])
    gpipe_loss = float(sum(shards))
    assert abs(gpipe_loss - ref_loss) < 1e-5 * max(1.0, abs(ref_loss)), (gpipe_loss, ref_loss)

    check_stage_grads_against_reference(stage, reference, reference_grads, "gpipe")

    # Boundary input gradients: stage k's per-micro received gradient equals
    # the reference gradient at its first layer's input.
    if not stage.is_first:
        begin = stage.layer_range[0]
        captures = [entry for entry in boundary_captures if entry["layer"] == begin]
        assert len(captures) == NUM_MICRO, (len(captures), NUM_MICRO)
        for micro_index, grad in enumerate(gpipe_result["input_grads"]):
            torch.testing.assert_close(
                grad, captures[micro_index]["grad"], rtol=1e-4, atol=1e-6,
                msg=f"gpipe: stage {stage.stage_index} boundary input gradient (micro {micro_index}) "
                    "must match the reference gradient at the stage's first layer input",
            )

    # ---- (4) timeline sanity + fill-drain ordering ----
    expected_sends = (0 if stage.is_last else NUM_MICRO) + (0 if stage.is_first else NUM_MICRO)
    expected_recvs = (0 if stage.is_first else NUM_MICRO) + (0 if stage.is_last else NUM_MICRO)
    kinds = [event["kind"] for event in gpipe_events]
    assert kinds.count("forward") == NUM_MICRO and kinds.count("backward") == NUM_MICRO, kinds
    assert kinds.count("p2p_send") == expected_sends, kinds
    assert kinds.count("p2p_recv") == expected_recvs, kinds
    bubble = compute_bubble(gpipe_events)
    assert 0.0 <= bubble["bubble_ratio"] <= 1.0 and bubble["window_s"] > 0, bubble
    if stage.is_last:
        forward_end = max(e["end_s"] for e in gpipe_events if e["kind"] == "forward")
        backward_start = min(e["start_s"] for e in gpipe_events if e["kind"] == "backward")
        assert forward_end <= backward_start + 1e-9, \
            "GPipe on the last stage must finish ALL forwards before ANY backward"
    if tied["tied_exchange_count"] > 0 and not stage.is_first and not stage.is_last:
        raise AssertionError("middle stages must not move tied-gradient bytes")

    gpipe_grads = snapshot_stage_grads(stage)

    # ---- (3) serial baseline: SAME grads, different schedule ----
    for parameter in stage.parameters():
        parameter.grad = None
    serial_timeline = StageTimeline(device)
    serial_result = serial_step(stage, micros, device, serial_timeline)
    serial_events = serial_timeline.end_step()
    exchange_tied_gradients(stage, device)
    check_stage_grads_against_reference(stage, reference, reference_grads, "serial")
    serial_grads = snapshot_stage_grads(stage)
    for name, grad in gpipe_grads.items():
        torch.testing.assert_close(
            serial_grads[name], grad, rtol=1e-6, atol=1e-7,
            msg=f"serial and gpipe must produce identical gradients for {name} (only scheduling differs)",
        )
    serial_shards = [None] * world_size
    dist.all_gather_object(serial_shards, serial_result["local_loss_shard"])
    assert abs(float(sum(serial_shards)) - gpipe_loss) < 1e-6, (serial_shards, gpipe_loss)

    # serial timeline keeps at most one micro-batch of activations alive per
    # stage: the last stage interleaves forward/backward per micro-batch.
    if stage.is_last:
        for micro_index in range(NUM_MICRO):
            forward = next(e for e in serial_events if e["kind"] == "forward" and e["micro"] == micro_index)
            backward = next(e for e in serial_events if e["kind"] == "backward" and e["micro"] == micro_index)
            assert forward["end_s"] <= backward["start_s"] + 1e-9

    # ---- (5) one AdamW step through pp_train_step, both modes ----
    # pp_train_step must REPLAY the reference's single AdamW step: its forward
    # loss is compared against ref_loss (measured on the PRE-step weights) and
    # its update must land on the stepped reference below.  The stage models
    # are therefore built from a pre-step snapshot, not from the stepped
    # module (a forward on already-stepped weights would measure a different
    # loss and take a SECOND step).
    pre_step_reference = build_reference(generator)
    pre_step_reference.load_state_dict(reference.state_dict())

    reference_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.1, foreach=False, fused=False)
    reference_opt.zero_grad(set_to_none=True)
    for micro in micros:
        logits = reference(micro["input_ids"], attention_mask=micro["attention_mask"])
        (batch_loss(logits, micro["labels"], "token_mean") * micro["scale"]).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    reference_opt.step()
    # `reference` now holds the stepped tied weights every stage must match.
    stepped_named = dict(reference.named_parameters())

    for mode in ("gpipe", "serial"):
        step_stage = build_stage(config, rank, world_size, pre_step_reference)
        optimizer = torch.optim.AdamW(step_stage.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.1, foreach=False, fused=False)
        metrics = pp_train_step(step_stage, optimizer, micros, device, mode, max_grad_norm=1.0)
        assert abs(metrics["global_loss"] - ref_loss) < 1e-5 * max(1.0, abs(ref_loss)), metrics
        assert metrics["gradient_norm_before_clip"] > 0 and metrics["step_time_ms"] >= 0
        # Post-step comparison against the stepped reference, scale-relative
        # (first AdamW updates are sign-like: m/sqrt(v) amplifies FP32 noise
        # in near-zero gradients — the chapter-36 smoke uses the same bar).
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
        print(f"pp={world_size}: splits, gpipe/serial gradient parity vs reference, boundary grads, "
              f"tie exchange, AdamW step parity, timeline ordering: OK")


def main() -> None:
    for world_size in (2, 3):
        port = _free_port()
        mp.spawn(_round_worker, args=(world_size, port), nprocs=world_size, join=True)
    print("chapter_39 smoke tests passed")


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


if __name__ == "__main__":
    main()
