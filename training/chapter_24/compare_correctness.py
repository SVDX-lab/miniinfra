#!/usr/bin/env python3
"""Chapter 24 correctness: accumulation equivalence and capacity boundaries.

Layers (all FP32, eager attention, real weights):

1. Gradient parity: one optimizer step's worth of gradients computed as a big
   batch versus 2x2 and 4x1 accumulation on identical members; per-parameter
   error, cosine similarity and global norm ratios.
2. Parameter parity after one AdamW step on the same effective batch.
3. Short-trajectory comparison: 8 optimizer steps big(4) versus 2x2.
4. Capacity probe: ascending big-batch sizes until CUDA OOM, while the 4x4
   accumulation path must run B_eff=16 within budget.
5. Padding accounting: what the big batch pads versus what the micro split pads.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from experiment_utils import environment_record, write_json
from gradient_accumulation import accumulated_step
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
    make_batch_plan,
    make_step_plan,
    split_effective_batch,
    to_device_batch,
    train_batch_step,
)


SELECTED_PARAMETERS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.13.mlp.down_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
    "model.norm.weight",
)


def fresh(args, device):
    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    return model, optimizer


def free_gpu_cache():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def move_to_device(batch, device):
    return to_device_batch(batch, device)


def gradient_parity(args, samples, device) -> dict:
    grads = {}
    norms = {}
    for arm, micro_size in (("big4", None), ("acc_2x2", 2), ("acc_4x1", 1)):
        model, optimizer = fresh(args, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        members = make_step_plan(len(samples), 4, args.seed)[0]
        if micro_size is None:
            batch = move_to_device(build_batch(samples, members, args.pad_token_id), device)
            from mini_batch import batch_loss

            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            loss = batch_loss(logits, batch.labels, "token_mean")
            loss.backward()
        else:
            micros = [move_to_device(b, device) for b in split_effective_batch(samples, members, micro_size, args.pad_token_id)]
            accumulated_step(model, optimizer, micros, device, float("inf"))
        # 选定参数的梯度立刻搬到 CPU：三条臂的克隆与后续双精度余弦都不占 GPU。
        selected = {
            name: parameter.grad.detach().clone().cpu()
            for name, parameter in model.named_parameters()
            if name in SELECTED_PARAMETERS
        }
        grads[arm] = selected
        norms[arm] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item())
        if micro_size is None:
            del logits, loss, batch
        else:
            del micros
        del model, optimizer
        free_gpu_cache()
        print(f"parity arm={arm}: allocated after release = {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    report = {"global_norm": norms, "selected_parameters": {}}
    reference = grads["big4"]
    for arm in ("acc_2x2", "acc_4x1"):
        arm_report = {}
        for name in SELECTED_PARAMETERS:
            base = reference[name]
            other = grads[arm][name]
            error = (base - other).abs()
            relative = error / base.abs().clamp_min(1e-12)
            # 逐元素相对误差被近零元素主导；significant 取 1e-2 × 峰值，
            # 过滤后反映真实的求和顺序差异（FP32 GEMM 形状不同 → ~1e-5 量级）。
            significant = base.abs() > (1e-2 * base.abs().max())
            relative_significant = error[significant] / base.abs()[significant]
            cosine = torch.nn.functional.cosine_similarity(
                base.flatten().double(), other.flatten().double(), dim=0
            )
            arm_report[name] = {
                "max_abs_error": float(error.max()),
                "grad_norm_ratio": float(other.norm() / base.norm().clamp_min(1e-12)),
                "max_rel_error": float(relative.max()),
                "max_rel_error_significant": float(relative_significant.max()),
                "cosine_similarity": float(cosine),
            }
            # 绝对误差随参数梯度量级变化，只记录不设阈；把关用无量纲指标：
            # 方向（余弦）与显著带（1e-2 × 峰值以上）相对误差。漏乘/错加缩放
            # 的真错误是 O(1) 量级，远高于这里的容差。
            if (
                float(cosine) < 0.9999999
                or float(relative_significant.max()) > 1e-2
            ):
                raise AssertionError(f"accumulation gradient mismatch at {name} ({arm}): {arm_report[name]}")
        report["selected_parameters"][arm] = arm_report
    return report


def one_step_parity(args, samples, device) -> dict:
    endpoints = {}
    for arm, micro_size in (("big4", None), ("acc_4x1", 1)):
        model, optimizer = fresh(args, device)
        init = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        members = make_step_plan(len(samples), 4, args.seed)[0]
        if micro_size is None:
            batch = build_batch(samples, members, args.pad_token_id)
            train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
        else:
            micros = [move_to_device(b, device) for b in split_effective_batch(samples, members, micro_size, args.pad_token_id)]
            accumulated_step(model, optimizer, micros, device, args.max_grad_norm)
        endpoints[arm] = (
            init,
            {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()},
        )
        if micro_size is None:
            del batch
        else:
            del micros
        del model, optimizer
        free_gpu_cache()

    init, big_end = endpoints["big4"]
    _init2, acc_end = endpoints["acc_4x1"]
    report = {}
    for name in SELECTED_PARAMETERS:
        base_delta = big_end[name] - init[name]
        acc_delta = acc_end[name] - init[name]
        cosine = torch.nn.functional.cosine_similarity(
            base_delta.flatten().double(), acc_delta.flatten().double(), dim=0
        )
        report[name] = {
            "param_max_abs_error": float((big_end[name] - acc_end[name]).abs().max()),
            "delta_cosine_similarity": float(cosine),
            "delta_norm_ratio": float(acc_delta.norm() / base_delta.norm().clamp_min(1e-12)),
        }
        if float(cosine) < 0.99999:
            raise AssertionError(f"one-step delta direction mismatch at {name}")
    return report


def short_trajectory(args, samples, device) -> dict:
    trajectories = {}
    memory = {}
    for arm, micro_size in (("big4", None), ("acc_2x2", 2)):
        model, optimizer = fresh(args, device)
        step_plan = make_step_plan(len(samples), 4, args.seed)[: 8 + args.warmup]
        steps = []
        torch.cuda.reset_peak_memory_stats(device)
        for members in step_plan:
            if micro_size is None:
                batch = build_batch(samples, members, args.pad_token_id)
                metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
                steps.append({"loss": metrics["loss"], "gradient_norm": metrics["gradient_norm_before_clip"]})
            else:
                micros = [move_to_device(b, device) for b in split_effective_batch(samples, members, micro_size, args.pad_token_id)]
                metrics = accumulated_step(model, optimizer, micros, device, args.max_grad_norm)
                steps.append({"loss": metrics["loss"], "gradient_norm": metrics["gradient_norm_before_clip"]})
        trajectories[arm] = steps[args.warmup :]
        memory[arm] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del model, optimizer
        free_gpu_cache()

    losses_a = [step["loss"] for step in trajectories["big4"]]
    losses_b = [step["loss"] for step in trajectories["acc_2x2"]]
    relative = [abs(a - b) / abs(a) for a, b in zip(losses_a, losses_b)]
    norms_a = [step["gradient_norm"] for step in trajectories["big4"]]
    norms_b = [step["gradient_norm"] for step in trajectories["acc_2x2"]]
    norm_relative = [abs(a - b) / abs(a) for a, b in zip(norms_a, norms_b)]
    summary = {
        "steps": len(losses_a),
        "loss_max_rel_error": max(relative),
        "gradient_norm_max_rel_error": max(norm_relative),
        "big4_first_loss": losses_a[0],
        "acc_2x2_first_loss": losses_b[0],
        "memory": memory,
    }
    if max(relative) > 1e-4:
        raise AssertionError(f"trajectory loss diverged beyond FP32 tolerance: {summary}")
    return summary


def big_batch_capacity_probe(args, samples, device) -> list[dict]:
    """Ascending big-batch sizes; stop at the first CUDA OOM."""

    probe = []
    for size in (4, 5, 6, 8, 12, 16):
        model, optimizer = fresh(args, device)
        plan = make_batch_plan(len(samples), size, args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        entry = {"batch_size": size}
        try:
            batch = move_to_device(build_batch(samples, plan[0], args.pad_token_id), device)
            train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
            entry.update({
                "status": "success",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            })
        except torch.cuda.OutOfMemoryError as error:
            entry.update({
                "status": "oom",
                "allocated_at_failure_bytes": torch.cuda.memory_allocated(),
                "reserved_at_failure_bytes": torch.cuda.memory_reserved(),
                "message": str(error).splitlines()[0][:200],
            })
        finally:
            del model, optimizer
            free_gpu_cache()
        probe.append(entry)
        print(f"big-batch probe size={size}: {entry['status']}", flush=True)
        if entry["status"] == "oom":
            break
    return probe


def accumulation_capacity(args, samples, device) -> dict:
    """B_eff=16 via 4x4 accumulation must fit and be measured."""

    model, optimizer = fresh(args, device)
    step_plan = make_step_plan(len(samples), 16, args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    members = step_plan[0]
    micros = [move_to_device(b, device) for b in split_effective_batch(samples, members, 4, args.pad_token_id)]
    metrics = accumulated_step(model, optimizer, micros, device, args.max_grad_norm, collect_telemetry=False)
    result = {
        "status": "success",
        "effective_batch": int(metrics["effective_batch_size"]),
        "supervised_tokens": int(metrics["supervised_tokens"]),
        "loss": metrics["loss"],
        "padding_ratio": metrics["padding_ratio"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del model, optimizer, micros
    free_gpu_cache()
    return result


def padding_accounting(args, samples, device) -> dict:
    step_plan = make_step_plan(len(samples), 16, args.seed)
    members = step_plan[0]
    big = build_batch(samples, members, args.pad_token_id)
    micros = split_effective_batch(samples, members, 4, args.pad_token_id)
    return {
        "big_batch": {
            "physical_tokens": big.physical_tokens,
            "padding_tokens": big.padding_tokens,
            "padding_ratio": big.padding_ratio,
            "batch_max_length": max(big.lengths),
        },
        "micro_4x4": {
            "physical_tokens": sum(b.physical_tokens for b in micros),
            "padding_tokens": sum(b.padding_tokens for b in micros),
            "padding_ratio": sum(b.padding_tokens for b in micros) / sum(b.physical_tokens for b in micros),
            "batch_max_length": max(max(b.lengths) for b in micros),
        },
        "supervised_tokens_equal": big.supervised_tokens == sum(b.supervised_tokens for b in micros),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter24_train.jsonl"))
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them for BOTH arms so the
        # accumulation-vs-big-batch comparison stays on identical samples.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
    # The step planners hardcode groups of 4 and 16; drop trailing samples so
    # the filtered corpus satisfies both divisibility constraints.
    usable = len(samples) - len(samples) % 16
    dropped_for_divisibility = len(samples) - usable
    samples = samples[:usable]
    print(
        f"encoded {usable} samples, filtered_no_supervision={filtered_no_supervision}, "
        f"dropped_for_divisibility={dropped_for_divisibility}"
    )

    parity = gradient_parity(args, samples, device)
    one_step = one_step_parity(args, samples, device)
    trajectory = short_trajectory(args, samples, device)
    big_probe = big_batch_capacity_probe(args, samples, device)
    acc_capacity = accumulation_capacity(args, samples, device)
    padding = padding_accounting(args, samples, device)
    if not padding["supervised_tokens_equal"]:
        raise AssertionError("supervised tokens differ between big batch and micro split")

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32",
        "tf32": False,
        "attention_implementation": "eager",
        "seed": args.seed,
        "num_samples": len(samples),
        "max_length": args.max_length,
        "gradient_parity": parity,
        "one_step_parity": one_step,
        "short_trajectory": trajectory,
        "big_batch_capacity_probe": big_probe,
        "accumulation_capacity_b16_4x4": acc_capacity,
        "padding_accounting": padding,
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_24 correctness checks passed")


if __name__ == "__main__":
    main()
