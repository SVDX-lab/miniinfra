#!/usr/bin/env python3
"""Chapter 20 correctness: pack equivalence, isolation proof, negative control.

Layers, in order:

1. Plan level (CPU): greedy packing covers the corpus without splitting or
   overflowing, replays exactly, and reports utilization.
2. Equivalence (GPU): every sample inside a pack matches its independent run
   within FP32 kernel tolerance, for logits and loss.
3. Isolation (GPU): replacing an earlier sample's content leaves later samples'
   packed logits bitwise unchanged.
4. Negative control (GPU): continuous positions plus plain causal attention
   keep the first sample intact but move later samples far away - the check
   itself can detect missing isolation.
5. Gradients (GPU): the pack objective equals the weighted independent sum for
   gradients and one AdamW update, within the Chapter 18 tolerances.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment_utils import environment_record, write_json
from mini_batch import batch_loss
from sequence_packing import build_block_diagonal_mask, build_pack, greedy_packing_plan
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
)


PROBE_PARAMETER = "model.layers.0.self_attn.q_norm.weight"
LOGITS_ABSOLUTE_TOLERANCE = 2e-4
LOSS_ABSOLUTE_TOLERANCE = 2e-4
GRADIENT_RELATIVE_L2_TOLERANCE = 2e-3
UPDATE_ABSOLUTE_TOLERANCE = 2e-6
NEGATIVE_CONTROL_DIVERGENCE_MIN = 1e-2


def _norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(
        float(parameter.grad.double().square().sum())
        for parameter in model.parameters() if parameter.grad is not None
    ))


def gradient_run(
    pack,
    samples,
    members,
    device: torch.device,
    mode: str,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
) -> dict[str, object]:
    """One forward/backward/clip/step on a fresh model per mode."""

    seed_everything(seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    optimizer.zero_grad(set_to_none=True)
    if mode == "pack":
        mask = build_block_diagonal_mask(
            pack.sample_spans, pack.capacity, torch.float32, device,
            total_length=pack.non_padding_tokens,
        )
        logits = model(
            pack.input_ids.to(device),
            attention_mask=pack.attention_mask.to(device),
            position_ids=pack.position_ids.to(device),
            attention_mask_4d=mask,
        )
        loss_tensor = batch_loss(logits, pack.labels.to(device), "token_mean")
        loss_tensor.backward()
        loss = float(loss_tensor.detach().item())
    elif mode == "independent":
        total_supervised = sum(samples[index].supervised_tokens for index in members)
        loss = 0.0
        for index in members:
            sample = samples[index]
            length = sample.non_padding_tokens
            sample_logits = model(sample.input_ids[:, :length].to(device))
            sample_loss = batch_loss(sample_logits, sample.labels[:, :length].to(device), "token_mean")
            weight = sample.supervised_tokens / total_supervised
            (sample_loss * weight).backward()
            loss += float(sample_loss.detach().item()) * weight
    else:
        raise ValueError(f"unsupported mode: {mode}")
    parameter = dict(model.named_parameters())[PROBE_PARAMETER]
    gradient = parameter.grad.detach().float().cpu().clone()
    gradient_norm = _norm(model)
    before = parameter.detach().float().cpu().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    update = parameter.detach().float().cpu() - before
    probe_sum = float(parameter.detach().double().sum().cpu())
    result = {
        "mode": mode,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "probe_gradient": gradient.tolist(),
        "probe_update": update.tolist(),
        "probe_parameter_sum_after_step": probe_sum,
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _flatten(values: list) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64).flatten()


def gradient_comparison(independent: dict, packed: dict) -> dict[str, float]:
    ig = _flatten(independent["probe_gradient"]); pg = _flatten(packed["probe_gradient"])
    iu = _flatten(independent["probe_update"]); pu = _flatten(packed["probe_update"])
    return {
        "loss_absolute_error": abs(float(independent["loss"]) - float(packed["loss"])),
        "probe_gradient_relative_l2_error": float(torch.linalg.vector_norm(ig - pg) / torch.linalg.vector_norm(ig)),
        "probe_gradient_cosine_similarity": float(F.cosine_similarity(ig, pg, dim=0)),
        "gradient_norm_absolute_error": abs(float(independent["gradient_norm"]) - float(packed["gradient_norm"])),
        "probe_update_max_absolute_error": float((iu - pu).abs().max()),
        "probe_parameter_sum_absolute_error": abs(float(independent["probe_parameter_sum_after_step"]) - float(packed["probe_parameter_sum_after_step"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter20_train.jsonl"))
    parser.add_argument("--pack-capacity", type=int, default=512)
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
    encoded_supervised = sum(sample.supervised_tokens for sample in samples)
    capacity = args.pack_capacity
    pad_token_id = tokenizer.pad_token_id

    plan = greedy_packing_plan(lengths, capacity, args.seed, 1)
    covered = sorted(index for pack in plan.packs for index in pack)
    plan_checks = {
        "covers_corpus_exactly_once": covered == list(range(len(samples))),
        "capacity_respected": all(
            sum(lengths[index] for index in pack) <= capacity for pack in plan.packs
        ),
        "num_packs": plan.num_packs,
        "samples_per_pack_p50": sorted(plan.samples_per_pack())[plan.num_packs // 2],
        "max_samples_per_pack": max(plan.samples_per_pack()),
        "single_sample_packs": sum(1 for pack in plan.packs if len(pack) == 1),
        "utilization": plan.utilization(lengths),
        "supervised_tokens_total": sum(
            samples[index].supervised_tokens for pack in plan.packs for index in pack
        ),
    }
    if plan_checks["supervised_tokens_total"] != encoded_supervised:
        raise AssertionError("packing changes the epoch's supervised token total")
    replay = greedy_packing_plan(lengths, capacity, args.seed, 1)
    plan_checks["replays_exactly"] = replay.packs == plan.packs
    if not (plan_checks["covers_corpus_exactly_once"] and plan_checks["replays_exactly"]):
        raise AssertionError("packing plan checks failed")

    multi_packs = [pack for pack in plan.packs if len(pack) >= 2]
    if multi_packs:
        multi_packs.sort(key=lambda pack: max(lengths[index] for index in pack))
        test_packs = multi_packs[:3]
    else:
        # The natural corpus is nearly uniform at the cap: 496 of 512 samples
        # are exactly 512 tokens and the short tail is too heavy for two
        # random shorts to share a pack.  Construct deterministic two-sample
        # test packs from the shortest samples instead; packing two samples is
        # exactly the mechanism under test (cross-sample isolation, boundary
        # labels, position reset), while the natural plan stays as the
        # benchmark reality.
        shorts = sorted(range(len(samples)), key=lambda index: lengths[index])[:16]
        candidates = [
            (shorts[0], shorts[1]),
            (shorts[2], shorts[3]),
            (shorts[1], shorts[2]),
            (shorts[0], shorts[2]),
            (shorts[0], shorts[3]),
        ]
        test_packs = [
            pair for pair in candidates
            if sum(lengths[index] for index in pair) <= capacity
        ][:3]
        plan_checks["constructed_test_packs"] = True
        plan_checks["constructed_from_shortest"] = len(shorts)
    if not test_packs:
        raise AssertionError("could not construct a multi-sample pack to test")

    device = torch.device("cuda:0")
    equivalence_model = load_tokenizer_and_model(device)[1]
    pack_reports: list[dict[str, object]] = []
    for pack_number, members in enumerate(test_packs):
        pack = build_pack(samples, members, capacity, pad_token_id)
        mask = build_block_diagonal_mask(
            pack.sample_spans, capacity, torch.float32, device,
            total_length=pack.non_padding_tokens,
        )
        with torch.no_grad():
            packed_out = equivalence_model(
                pack.input_ids.to(device),
                attention_mask=pack.attention_mask.to(device),
                position_ids=pack.position_ids.to(device),
                attention_mask_4d=mask,
            )
            independent_out = {
                index: equivalence_model(
                    samples[index].input_ids[:, : samples[index].non_padding_tokens].to(device)
                )
                for index in members
            }
            sample_errors = []
            for (start, end), index in zip(pack.sample_spans, members):
                reference = independent_out[index][0].float().cpu()
                observed = packed_out[0, start:end, :].float().cpu()
                error = {
                    "source_line": int(samples[index].source_line),
                    "length": lengths[index],
                    "logits_max_absolute_error": float((observed - reference).abs().max()),
                    "logits_mean_absolute_error": float((observed - reference).abs().mean()),
                }
                sample_errors.append(error)
                if error["logits_max_absolute_error"] > LOGITS_ABSOLUTE_TOLERANCE:
                    raise AssertionError(f"pack sample {index} exceeds logits tolerance: {error}")

            pack_loss_value = float(batch_loss(packed_out, pack.labels.to(device), "token_mean").item())
            total_supervised = sum(samples[index].supervised_tokens for index in members)
            independent_loss_value = 0.0
            for index in members:
                sample = samples[index]
                length = sample.non_padding_tokens
                sample_loss = batch_loss(
                    independent_out[index], sample.labels[:, :length].to(device), "token_mean"
                )
                independent_loss_value += float(sample_loss.item()) * sample.supervised_tokens / total_supervised

            isolation_report = None
            negative_report = None
            if pack_number == 0:
                swapped_input = pack.input_ids.clone()
                first_start, first_end = pack.sample_spans[0]
                swapped_input[0, first_start:first_end] = torch.randint(
                    1, 151643, (1, first_end - first_start)
                )
                swapped_out = equivalence_model(
                    swapped_input.to(device),
                    attention_mask=pack.attention_mask.to(device),
                    position_ids=pack.position_ids.to(device),
                    attention_mask_4d=mask,
                )
                later_bitwise_equal = all(
                    torch.equal(swapped_out[0, start:end, :], packed_out[0, start:end, :])
                    for (start, end) in pack.sample_spans[1:]
                )
                isolation_report = {
                    "swapped_sample_span": [first_start, first_end],
                    "later_samples_bitwise_equal": later_bitwise_equal,
                }
                if not later_bitwise_equal:
                    raise AssertionError("block-diagonal isolation failed the content-swap test")

                naive_out = equivalence_model(
                    pack.input_ids.to(device),
                    attention_mask=pack.attention_mask.to(device),
                )
                first_span = pack.sample_spans[0]
                first_error = float(
                    (naive_out[0, first_span[0] : first_span[1], :] - independent_out[members[0]]).abs().max()
                )
                divergences = []
                for (start, end), index in list(zip(pack.sample_spans, members))[1:]:
                    divergence = float(
                        (naive_out[0, start:end, :] - independent_out[index]).abs().max()
                    )
                    divergences.append({"source_line": int(samples[index].source_line), "max_absolute_divergence": divergence})
                    if divergence <= NEGATIVE_CONTROL_DIVERGENCE_MIN:
                        raise AssertionError(
                            f"negative control failed to diverge at sample {index}: {divergence}"
                        )
                negative_report = {
                    "first_sample_max_absolute_error": first_error,
                    "later_sample_divergences": divergences,
                }
                if first_error > LOGITS_ABSOLUTE_TOLERANCE:
                    raise AssertionError(f"negative control moved the first sample: {first_error}")

        loss_absolute_error = abs(pack_loss_value - independent_loss_value)
        if loss_absolute_error > LOSS_ABSOLUTE_TOLERANCE:
            raise AssertionError(f"pack loss exceeds tolerance: {loss_absolute_error}")
        pack_reports.append({
            "members": list(members),
            "source_lines": list(pack.source_lines),
            "lengths": list(pack.lengths),
            "supervised_tokens": pack.supervised_tokens,
            "utilization": pack.non_padding_tokens / pack.capacity,
            "sample_errors": sample_errors,
            "loss_pack": pack_loss_value,
            "loss_independent_weighted": independent_loss_value,
            "loss_absolute_error": loss_absolute_error,
            "isolation": isolation_report,
            "negative_control": negative_report,
        })
        del packed_out, independent_out
        gc.collect()
        torch.cuda.empty_cache()
    del equivalence_model
    gc.collect()
    torch.cuda.empty_cache()

    lightest = test_packs[0]
    lightest_pack = build_pack(samples, lightest, capacity, pad_token_id)
    independent_run = gradient_run(
        lightest_pack, samples, lightest, device, "independent",
        args.seed, args.learning_rate, args.weight_decay, args.max_grad_norm,
    )
    packed_run = gradient_run(
        lightest_pack, samples, lightest, device, "pack",
        args.seed, args.learning_rate, args.weight_decay, args.max_grad_norm,
    )
    metrics = gradient_comparison(independent_run, packed_run)
    if (
        metrics["loss_absolute_error"] > LOSS_ABSOLUTE_TOLERANCE
        or metrics["probe_gradient_relative_l2_error"] > GRADIENT_RELATIVE_L2_TOLERANCE
        or metrics["probe_update_max_absolute_error"] > UPDATE_ABSOLUTE_TOLERANCE
    ):
        raise AssertionError(f"pack gradients exceed FP32 tolerance: {metrics}")

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "seed": args.seed, "pack_capacity": capacity, "max_length": args.max_length,
        "num_samples": len(samples),
        "corpus": {
            "lengths_p50": sorted(lengths)[len(lengths) // 2],
            "lengths_min": min(lengths), "lengths_max": max(lengths),
            "supervised_tokens": encoded_supervised,
        },
        "plan_checks": plan_checks,
        "pack_reports": pack_reports,
        "gradient_anchor": {
            "members": list(lightest),
            "lengths": list(lightest_pack.lengths),
            "supervised_tokens": lightest_pack.supervised_tokens,
            "independent": independent_run,
            "packed": packed_run,
            "comparison": metrics,
        },
        "tolerances": {
            "logits_absolute": LOGITS_ABSOLUTE_TOLERANCE,
            "loss_absolute": LOSS_ABSOLUTE_TOLERANCE,
            "probe_gradient_relative_l2": GRADIENT_RELATIVE_L2_TOLERANCE,
            "probe_update_absolute": UPDATE_ABSOLUTE_TOLERANCE,
            "negative_control_divergence_min": NEGATIVE_CONTROL_DIVERGENCE_MIN,
        },
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_20 correctness checks passed")


if __name__ == "__main__":
    main()
