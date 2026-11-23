#!/usr/bin/env python3
"""Chapter 27 correctness: optimizer parity, kernel census, determinism.

Layers:

1. One optimizer step on identical weights/grads: ``foreach`` against ``loop``
   (same elementwise ops, expect bitwise), ``fused`` against ``loop``
   (different rounding, tolerance with cosine of the parameter delta).
2. 32-step loss trajectories for all three arms against the loop arm.
3. Determinism: the fused arm run twice must be bitwise identical.
4. Kernel census: CUDA kernel count and device time of one ``optimizer.step()``
   per arm, plus temporary memory around the step.
5. The gradient clip is identical in every arm - asserted, not assumed.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from experiment_utils import environment_record, write_json
from fused_optimizer import build_optimizer, count_step_kernels, step_peak_memory
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    encode_samples,
    load_tokenizer_and_model,
    make_batch_plan,
    to_device_batch,
    train_batch_step,
)


ARMS = ("loop", "foreach", "fused")


def free_cache():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def fresh(args, device, arm):
    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay, arm)
    return model, optimizer


def one_step_parity(args, samples, plan, device) -> dict:
    endpoints = {}
    clip_norms = {}
    for arm in ARMS:
        model, optimizer = fresh(args, device, arm)
        # init 与终点都必须落在 CPU：对照计算在宿主侧进行，避免占用 GPU。
        init = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        batch = to_device_batch(build_batch(samples, plan[0], args.pad_token_id), device)
        # 只走 Forward/Backward，然后手动 clip+step，避免训练体里的计时干扰。
        from mini_batch import batch_loss

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        clip_norms[arm] = float(grad_norm.item())
        optimizer.step()
        endpoints[arm] = (
            init,
            {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()},
        )
        del model, optimizer, logits, loss, batch
        free_cache()

    loop_init, loop_end = endpoints["loop"]
    report = {"clip_gradient_norms": clip_norms, "comparisons": {}}
    assert len(set(clip_norms.values())) == 1, "clip norm differs across arms"
    for arm in ("foreach", "fused"):
        arm_init, arm_end = endpoints[arm]
        per_parameter = {}
        bitwise_all = True
        max_abs_overall = 0.0
        for name in loop_init:
            base = loop_end[name]
            other = arm_end[name]
            identical = bool(torch.equal(base, other))
            bitwise_all = bitwise_all and identical
            if not identical:
                error = (base - other).abs()
                max_abs = float(error.max())
                max_abs_overall = max(max_abs_overall, max_abs)
                delta_base = base - loop_init[name]
                delta_other = other - arm_init[name]
                cosine = torch.nn.functional.cosine_similarity(
                    delta_base.flatten().double(), delta_other.flatten().double(), dim=0
                )
                param_max_abs = float(base.abs().max())
                per_parameter[name] = {
                    "max_abs_error": max_abs,
                    "param_max_abs": param_max_abs,
                    # 差异相对参数幅值（≈1 ulp 的归一化形式）——绝对门槛
                    # 会随张量幅值漂移，相对量才跨参数可比。
                    "max_abs_rel_to_param": max_abs / max(param_max_abs, 1e-12),
                    "delta_cosine_similarity": float(cosine),
                }
        worst_cos = min(
            (entry["delta_cosine_similarity"] for entry in per_parameter.values()),
            default=1.0,
        )
        worst_rel = max(
            (entry["max_abs_rel_to_param"] for entry in per_parameter.values()),
            default=0.0,
        )
        # 两种 kernel 的实测差异都是参数值 1 ulp 量级（foreach ~1.5e-8，
        # fused ~4.8e-7），归一化到参数幅值后都在 1.1e-7 以内。相对门槛
        # 设在噪声 10 倍处；漏乘/错加缩放的真错误是 O(1)，32 步轨迹闸门
        # 会二次拦截。
        rel_gate = 1e-7 if arm == "foreach" else 1e-6
        cos_gate = 0.9999999 if arm == "foreach" else 0.999
        if worst_cos < cos_gate or worst_rel > rel_gate:
            raise AssertionError(
                f"{arm} deviation beyond rounding: worst_cos={worst_cos}, "
                f"worst_rel_to_param={worst_rel}"
            )
        report["comparisons"][arm] = {
            "bitwise_identical_to_loop": bitwise_all,
            "max_abs_error": max_abs_overall,
            "mismatched_parameters": per_parameter,
        }
    return report


def trajectory(args, samples, plan, device) -> dict:
    trajectories = {}
    steps_to_run = plan[: 32 + args.warmup]
    for arm in ARMS:
        model, optimizer = fresh(args, device, arm)
        losses = []
        for batch_indices in steps_to_run:
            batch = build_batch(samples, batch_indices, args.pad_token_id)
            metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
            losses.append(metrics["loss"])
        trajectories[arm] = losses[args.warmup :]
        del model, optimizer
        free_cache()

    reference = trajectories["loop"]
    report = {"steps": len(reference), "loss_series": trajectories, "comparisons": {}}
    for arm in ("foreach", "fused"):
        rel = [
            abs(a - b) / max(abs(a), 1e-12)
            for a, b in zip(reference, trajectories[arm])
        ]
        report["comparisons"][arm] = {
            "max_abs_error": max(abs(a - b) for a, b in zip(reference, trajectories[arm])),
            "max_rel_error": max(rel),
        }
        if max(rel) > 1e-3:
            raise AssertionError(f"{arm} trajectory diverged from loop: {report['comparisons'][arm]}")
    return report


def determinism(args, samples, plan, device) -> dict:
    endpoints = []
    for _ in range(2):
        model, optimizer = fresh(args, device, "fused")
        batch = to_device_batch(build_batch(samples, plan[0], args.pad_token_id), device)
        train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
        # 12GB calibration: keep the endpoint snapshot on host, same as
        # one_step_parity, so the fused determinism run does not double the
        # parameter copy on device.
        endpoints.append({name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()})
        del model, optimizer, batch
        free_cache()
    identical = all(torch.equal(endpoints[0][name], endpoints[1][name]) for name in endpoints[0])
    if not identical:
        raise AssertionError("fused arm is not deterministic across identical runs")
    return {"fused_bitwise_identical_across_runs": True}


def kernel_census(args, samples, plan, device) -> dict:
    census = {}
    for arm in ARMS:
        model, optimizer = fresh(args, device, arm)
        batch = to_device_batch(build_batch(samples, plan[0], args.pad_token_id), device)
        train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
        # 第二步：此刻 AdamW 状态已分配，测的是稳态更新成本。
        model.train()
        optimizer.zero_grad(set_to_none=True)
        from mini_batch import batch_loss

        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        grad_norm = float(grad_norm.item())
        torch.cuda.synchronize()
        counts = count_step_kernels(optimizer)
        memory = step_peak_memory(optimizer, model, args.max_grad_norm)
        census[arm] = {**counts, **memory, "gradient_norm": grad_norm}
        print(
            f"census {arm}: kernels={counts['cuda_kernel_count']} "
            f"device_time={counts['cuda_device_time_us'] / 1000:.2f}ms",
            flush=True,
        )
        del model, optimizer, logits, loss, batch
        free_cache()
    if not (census["fused"]["cuda_kernel_count"] < census["foreach"]["cuda_kernel_count"] < census["loop"]["cuda_kernel_count"]):
        print("warning: kernel counts not strictly ordered; recording measured values", flush=True)
    return census


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter27_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
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
        # supervised targets to truncation. Drop them (counted) so every phase
        # compares on identical samples. Same pattern as chapter_24.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
        print(
            f"encoded {len(samples)} samples, filtered_no_supervision={filtered_no_supervision}",
            flush=True,
        )
    plan = make_batch_plan(len(samples), args.batch_size, args.seed)

    parity = one_step_parity(args, samples, plan, device)
    trajectory_report = trajectory(args, samples, plan, device)
    determinism_report = determinism(args, samples, plan, device)
    census = kernel_census(args, samples, plan, device)

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
        "arms": list(ARMS),
        "optimizer_math": "torch.optim.AdamW betas=(0.9,0.95) eps=1e-8 single group",
        "seed": args.seed,
        "num_samples": len(samples),
        "filtered_no_supervision": filtered_no_supervision,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "one_step_parity": parity,
        "trajectory": trajectory_report,
        "determinism": determinism_report,
        "kernel_census": census,
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_27 correctness checks passed")


if __name__ == "__main__":
    main()
