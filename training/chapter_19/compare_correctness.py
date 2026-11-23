#!/usr/bin/env python3
"""Chapter 19 correctness: sampler partition, membership invariance, matched updates.

Checks, in order:

1. Plan level (CPU): both samplers partition the fixed 512-sample corpus, a
   window equal to the batch size degenerates to the random baseline, plans
   replay exactly from the seed, and supervised token accounting is unchanged
   by regrouping.
2. Row invariance (CPU): a sample's tokens and labels inside a padded batch are
   bitwise independent of which batch it was assigned to.
3. Twin step (GPU): the same member set built through the baseline path and the
   bucketed path yields bitwise-identical tensors, loss, and one AdamW update.
4. Serial anchor (GPU): a bucketed batch matches per-sample accumulation on a
   real Qwen3 FP32 model within the Chapter 18 tolerances.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment_utils import environment_record, write_json
from length_bucket import (
    bucketed_plan,
    membership_multiset,
    project_plan_tokens,
    random_plan,
)
from mini_batch import PaddedBatch, batch_loss, pad_samples
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
    move_batch,
    train_batch_step,
)


PROBE_PARAMETER = "model.layers.0.self_attn.q_norm.weight"
LOSS_ABSOLUTE_TOLERANCE = 2e-4
GRADIENT_RELATIVE_L2_TOLERANCE = 2e-3
UPDATE_ABSOLUTE_TOLERANCE = 2e-6


def _norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(
        float(parameter.grad.double().square().sum())
        for parameter in model.parameters() if parameter.grad is not None
    ))


def serial_reference_backward(model: torch.nn.Module, batch: PaddedBatch, device: torch.device) -> float:
    """Per-sample Forward/Backward weighted into one token_mean group gradient."""

    counts = torch.tensor(batch.supervised_tokens_per_sample, device=device)
    total = float(counts.sum().item())
    weighted = 0.0
    for row, length in enumerate(batch.lengths):
        logits = model(
            batch.input_ids[row : row + 1, :length],
            attention_mask=batch.attention_mask[row : row + 1, :length],
        )
        sample_mean = batch_loss(logits, batch.labels[row : row + 1, :length], "token_mean")
        (sample_mean * (counts[row] / counts.sum())).backward()
        weighted += float(sample_mean.detach().item()) * batch.supervised_tokens_per_sample[row]
    return weighted / total


def serial_anchor_run(
    batch: PaddedBatch,
    device: torch.device,
    serial: bool,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
) -> dict[str, object]:
    seed_everything(seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    host_batch = batch
    batch = move_batch(host_batch, device)
    optimizer.zero_grad(set_to_none=True)
    if serial:
        loss = serial_reference_backward(model, batch, device)
        # recompute probe logits row by row without autograd to keep memory flat
        with torch.no_grad():
            probe_logits = []
            for row, length in enumerate(host_batch.lengths):
                logits = model(
                    batch.input_ids[row : row + 1, :length],
                    attention_mask=batch.attention_mask[row : row + 1, :length],
                )
                probe_logits.append(logits[0, :, :32].detach().float().cpu())
    else:
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss_tensor = batch_loss(logits, batch.labels, "token_mean")
        loss_tensor.backward()
        loss = float(loss_tensor.detach().item())
        probe_logits = [
            logits[index, :length, :32].detach().float().cpu()
            for index, length in enumerate(host_batch.lengths)
        ]
    parameter = dict(model.named_parameters())[PROBE_PARAMETER]
    gradient = parameter.grad.detach().float().cpu().clone()
    gradient_norm = _norm(model)
    before = parameter.detach().float().cpu().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    update = parameter.detach().float().cpu() - before
    probe_sum = float(parameter.detach().double().sum().cpu())
    result = {
        "path": "serial" if serial else "batched",
        "loss": loss,
        "gradient_norm": gradient_norm,
        "probe_gradient": gradient.tolist(),
        "probe_update": update.tolist(),
        "probe_parameter_sum_after_step": probe_sum,
        "logits_probe": [tensor.tolist() for tensor in probe_logits],
    }
    del optimizer, model, batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _flatten(values: list) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64).flatten()


def anchor_comparison(serial: dict[str, object], batched: dict[str, object]) -> dict[str, float]:
    sg = _flatten(serial["probe_gradient"]); bg = _flatten(batched["probe_gradient"])
    su = _flatten(serial["probe_update"]); bu = _flatten(batched["probe_update"])
    sl = torch.cat([_flatten(record) for record in serial["logits_probe"]])
    bl = torch.cat([_flatten(record) for record in batched["logits_probe"]])
    return {
        "loss_absolute_error": abs(float(serial["loss"]) - float(batched["loss"])),
        "loss_relative_error": abs(float(serial["loss"]) - float(batched["loss"])) / max(abs(float(serial["loss"])), 1e-12),
        "logits_max_absolute_error": float((sl - bl).abs().max()),
        "logits_mean_absolute_error": float((sl - bl).abs().mean()),
        "probe_gradient_max_absolute_error": float((sg - bg).abs().max()),
        "probe_gradient_relative_l2_error": float(torch.linalg.vector_norm(sg - bg) / torch.linalg.vector_norm(sg)),
        "probe_gradient_cosine_similarity": float(F.cosine_similarity(sg, bg, dim=0)),
        "gradient_norm_absolute_error": abs(float(serial["gradient_norm"]) - float(batched["gradient_norm"])),
        "probe_update_max_absolute_error": float((su - bu).abs().max()),
        "probe_parameter_sum_absolute_error": abs(float(serial["probe_parameter_sum_after_step"]) - float(batched["probe_parameter_sum_after_step"])),
    }


def twin_step_check(samples, members: tuple[int, ...], pad_token_id: int, device: torch.device, seed: int) -> dict[str, object]:
    """The same member set built twice must produce one bitwise-identical step.

    The two executions share no live state: the first model and its optimizer are
    freed (after copying the stepped parameters to host memory) before the second
    execution starts, so the comparison is exact but never holds two full
    training states at once.
    """

    first = build_batch(samples, members, pad_token_id)
    second = build_batch(samples, members, pad_token_id)
    tensors_equal = all(
        torch.equal(getattr(first, name), getattr(second, name))
        for name in ("input_ids", "attention_mask", "labels")
    )
    seed_everything(seed)
    model_a = load_tokenizer_and_model(device)[1]
    optimizer_a = build_optimizer(model_a, 2e-5, 0.1)
    metrics_a = train_batch_step(model_a, optimizer_a, first, device, 1.0, collect_telemetry=False)
    parameters_a = {name: tensor.detach().cpu().clone() for name, tensor in model_a.named_parameters()}
    del optimizer_a, model_a
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed)
    model_b = load_tokenizer_and_model(device)[1]
    optimizer_b = build_optimizer(model_b, 2e-5, 0.1)
    metrics_b = train_batch_step(model_b, optimizer_b, second, device, 1.0, collect_telemetry=False)
    parameters_b = dict(model_b.named_parameters())
    bitwise_parameters = all(
        torch.equal(parameters_a[name], parameters_b[name].detach().cpu()) for name in parameters_a
    )
    result = {
        "members": list(members),
        "source_lines": list(first.source_lines),
        "lengths": list(first.lengths),
        "padded_tensors_bitwise_equal": tensors_equal,
        "loss_bitwise_equal": metrics_a["loss"] == metrics_b["loss"],
        "loss": metrics_a["loss"],
        "parameters_bitwise_equal_after_step": bitwise_parameters,
        "supervised_tokens": first.supervised_tokens,
    }
    del optimizer_b, model_b
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter19_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bucket-window", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    records = load_jsonl_records(args.data)
    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    samples = encode_samples(tokenizer, records, args.max_length)
    lengths = [sample.non_padding_tokens for sample in samples]
    pad_token_id = tokenizer.pad_token_id
    encoded_supervised = sum(sample.supervised_tokens for sample in samples)

    plans = {
        "random_epoch1": random_plan(len(samples), args.batch_size, args.seed, 1),
        "random_epoch2": random_plan(len(samples), args.batch_size, args.seed, 2),
        "bucketed_epoch1": bucketed_plan(lengths, args.batch_size, args.bucket_window, args.seed, 1),
        "bucketed_epoch2": bucketed_plan(lengths, args.batch_size, args.bucket_window, args.seed, 2),
        "degenerate_epoch1": bucketed_plan(lengths, args.batch_size, args.batch_size, args.seed, 1),
    }
    plan_checks: dict[str, object] = {}
    for name, plan in plans.items():
        covered = sorted(index for batch in plan.batches for index in batch)
        plan_checks[name] = {
            "is_partition_of_corpus": covered == list(range(len(samples))),
            "num_batches": plan.num_batches,
            "partial_batches": plan.partial_batches,
            "supervised_tokens_total": sum(
                samples[index].supervised_tokens for batch in plan.batches for index in batch
            ),
        }
        if plan_checks[name]["supervised_tokens_total"] != encoded_supervised:
            raise AssertionError(f"{name} changes the epoch's supervised token total")
    plan_checks["window_equals_batch_matches_random"] = (
        membership_multiset(plans["degenerate_epoch1"]) == membership_multiset(plans["random_epoch1"])
    )
    replay = bucketed_plan(lengths, args.batch_size, args.bucket_window, args.seed, 1)
    plan_checks["bucketed_plan_replays_exactly"] = replay.batches == plans["bucketed_epoch1"].batches
    if not plan_checks["window_equals_batch_matches_random"] or not plan_checks["bucketed_plan_replays_exactly"]:
        raise AssertionError("sampler degeneration or replay check failed")

    row_invariance: dict[str, bool] = {}
    reference_batches: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name in ("random_epoch1", "bucketed_epoch1"):
        all_rows_match = True
        for batch_indices in plans[name].batches:
            group = pad_samples([samples[index] for index in batch_indices], pad_token_id)
            for row, index in enumerate(batch_indices):
                expected = samples[index]
                span = slice(0, lengths[index])
                same = (
                    torch.equal(group.input_ids[row, span], expected.input_ids[0, span])
                    and torch.equal(group.labels[row, span], expected.labels[0, span])
                    and group.supervised_tokens_per_sample[row] == expected.supervised_tokens
                )
                all_rows_match = all_rows_match and same
                if name == "bucketed_epoch1":
                    reference_batches[index] = (group.input_ids[row].clone(), group.labels[row].clone(), group.attention_mask[row].clone())
        row_invariance[name] = all_rows_match
    if not all(row_invariance.values()):
        raise AssertionError("a sample's row changed when its batch companions changed")

    device = torch.device("cuda:0")
    shortest_members = min(
        plans["bucketed_epoch1"].batches,
        key=lambda batch_indices: max(lengths[index] for index in batch_indices),
    )
    twin = twin_step_check(samples, shortest_members, pad_token_id, device, args.seed)
    if not (twin["padded_tensors_bitwise_equal"] and twin["loss_bitwise_equal"] and twin["parameters_bitwise_equal_after_step"]):
        raise AssertionError(f"twin step diverged: {twin}")

    anchor_batch = None
    lightest = min(
        plans["bucketed_epoch1"].batches,
        key=lambda batch_indices: max(lengths[index] for index in batch_indices),
    )
    anchor_batch = build_batch(samples, lightest, pad_token_id)
    if max(anchor_batch.lengths) > 384:
        print(f"warning: lightest bucketed batch still has max length {max(anchor_batch.lengths)}")
    serial_result = serial_anchor_run(
        anchor_batch, device, True, args.seed, args.learning_rate, args.weight_decay, args.max_grad_norm,
    )
    batched_result = serial_anchor_run(
        anchor_batch, device, False, args.seed, args.learning_rate, args.weight_decay, args.max_grad_norm,
    )
    metrics = anchor_comparison(serial_result, batched_result)
    if (
        metrics["loss_absolute_error"] > LOSS_ABSOLUTE_TOLERANCE
        or metrics["probe_gradient_relative_l2_error"] > GRADIENT_RELATIVE_L2_TOLERANCE
        or metrics["probe_update_max_absolute_error"] > UPDATE_ABSOLUTE_TOLERANCE
    ):
        raise AssertionError(f"serial anchor exceeds FP32 tolerance: {metrics}")

    projection = project_plan_tokens(lengths, plans["bucketed_epoch1"])
    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "seed": args.seed, "batch_size": args.batch_size, "bucket_window": args.bucket_window,
        "max_length": args.max_length, "num_samples": len(samples),
        "corpus": {
            "lengths_p50": sorted(lengths)[len(lengths) // 2],
            "lengths_min": min(lengths), "lengths_max": max(lengths),
            "physical_tokens_if_concatenated": sum(lengths),
            "supervised_tokens": encoded_supervised,
        },
        "plan_checks": plan_checks,
        "row_invariance": row_invariance,
        "twin_step": twin,
        "anchor_batch": {
            "source_lines": list(anchor_batch.source_lines), "lengths": list(anchor_batch.lengths),
            "physical_tokens": anchor_batch.physical_tokens, "supervised_tokens": anchor_batch.supervised_tokens,
            "padding_ratio": anchor_batch.padding_ratio,
            "projected_epoch_padding_ratio_bucketed_window": projection.padding_ratio,
        },
        "serial_anchor": {"serial": serial_result, "batched": batched_result, "comparison": metrics},
        "tolerances": {
            "loss_absolute": LOSS_ABSOLUTE_TOLERANCE,
            "probe_gradient_relative_l2": GRADIENT_RELATIVE_L2_TOLERANCE,
            "probe_update_absolute": UPDATE_ABSOLUTE_TOLERANCE,
        },
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_19 correctness checks passed")


if __name__ == "__main__":
    main()
