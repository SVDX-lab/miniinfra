#!/usr/bin/env python3
"""第 01 期三后端正确性对照。

在指定后端（cpu / cuda / musa）上执行四条对照链并写结果 JSON：
1. RMSNorm 组合表达式 vs NumPy FP32 真值：18 组合同 Shape 逐组误差；
2. dtype 误差来源分解：同一输入上（a）FP32 组合表达式 vs NumPy 真值
   （torch-vs-NumPy 实现差）、（b）FP16 组合表达式 vs FP32 组合表达式
   （dtype 差），两条链分开归因；
3. 确定性：同一算子连续 3 次运行的位级一致性；
4. 模型真值对照：FP16 手写 Qwen3 在本后端上 Greedy 解码 + 首 Token
   Logits + 逐层探针 vs CPU FP32 真值（truth/ 目录）。

用法示例：
    python compare_correctness.py --backend cuda --model-dir models/qwen3-0.6b
    python compare_correctness.py --backend musa --skip-model   # 无模型时
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiment_utils import (
    comparison_metrics,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from qwen3_model import load_handwritten_model, resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from rmsnorm_reference import RMS_NORM_EPS, composite_rmsnorm_torch

FIXTURES_DEFAULT = Path(__file__).resolve().parent / "fixtures" / "correctness.npz"
TRUTH_DEFAULT = Path(__file__).resolve().parent / "truth"
DEFAULT_PROMPT = "用一句话解释 RMSNorm 的作用。"


def load_fixtures(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    with path.with_name("manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return arrays, manifest


def rmsnorm_fixture_chain(arguments, device: torch.device, arrays, manifest) -> dict:
    dtype = torch.float16 if arguments.dtype == "fp16" else torch.float32
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden_size}"
        x_fp32 = torch.from_numpy(arrays[f"x_{key}"])
        expected = torch.from_numpy(arrays[f"y_{key}"])
        if hidden_size not in weight_cache:
            weight_cache[hidden_size] = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(device=device, dtype=dtype)
        x = x_fp32.to(device=device, dtype=dtype)
        result = composite_rmsnorm_torch(x, weight_cache[hidden_size], RMS_NORM_EPS)
        metrics = comparison_metrics(result, expected)
        metrics.update({"rows": rows, "hidden_size": hidden_size, "dtype": arguments.dtype})
        groups.append(metrics)
    worst = max(groups, key=lambda item: item["max_abs_error"])
    print(
        f"[rmsnorm/{arguments.backend}] {len(groups)} 组 | 最差组 {worst['rows']}x{worst['hidden_size']}"
        f" max_abs={worst['max_abs_error']:.3e} cos={worst['cosine_similarity']:.8f}"
    )
    return {"dtype": arguments.dtype, "groups": groups, "worst_group": worst}


def dtype_error_chain(device: torch.device, arrays) -> dict:
    """误差来源分解：FP32 组合表达式 vs NumPy 真值；FP16 vs FP32 组合表达式。"""
    rows, hidden_size = 128, 1024
    x_fp32 = torch.from_numpy(arrays[f"x_{rows}x{hidden_size}"])
    expected = torch.from_numpy(arrays[f"y_{rows}x{hidden_size}"])
    weight_fp32 = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(device=device, dtype=torch.float32)

    fp32_on_device = composite_rmsnorm_torch(
        x_fp32.to(device=device), weight_fp32, RMS_NORM_EPS
    )
    torch_vs_numpy = comparison_metrics(fp32_on_device, expected)

    fp16 = composite_rmsnorm_torch(
        x_fp32.to(device=device, dtype=torch.float16),
        weight_fp32.to(dtype=torch.float16),
        RMS_NORM_EPS,
    )
    fp16_vs_fp32 = comparison_metrics(fp16, fp32_on_device.cpu())

    print(
        f"[dtype-error] FP32组合 vs NumPy真值 max_abs={torch_vs_numpy['max_abs_error']:.3e} | "
        f"FP16组合 vs FP32组合 max_abs={fp16_vs_fp32['max_abs_error']:.3e}"
    )
    return {
        "shape": {"rows": rows, "hidden_size": hidden_size},
        "torch_fp32_vs_numpy_truth": torch_vs_numpy,
        "fp16_composite_vs_fp32_composite": fp16_vs_fp32,
    }


def determinism_chain(device: torch.device, arrays) -> dict:
    rows, hidden_size = 128, 1024
    x = torch.from_numpy(arrays[f"x_{rows}x{hidden_size}"]).to(device=device, dtype=torch.float16)
    weight = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(device=device, dtype=torch.float16)
    runs = [
        composite_rmsnorm_torch(x, weight, RMS_NORM_EPS).detach().cpu().contiguous()
        for _ in range(3)
    ]
    bitwise_identical = all(
        torch.equal(runs[0], run) and runs[0].numpy().tobytes() == run.numpy().tobytes()
        for run in runs[1:]
    )
    print(f"[determinism] 3 次运行位级一致: {bitwise_identical}")
    return {"runs": 3, "bitwise_identical": bool(bitwise_identical)}


def model_truth_chain(arguments, device: torch.device) -> dict | None:
    truth_dir = Path(arguments.truth_dir)
    if arguments.skip_model or not all(
        (truth_dir / name).is_file() for name in ("truth-manifest.json", "probes.npz")
    ):
        print("[model] 跳过模型真值对照（--skip-model 或真值文件未完整生成）")
        return None

    with (truth_dir / "truth-manifest.json").open("r", encoding="utf-8") as file:
        truth = json.load(file)
    probes_npz = np.load(truth_dir / "probes.npz")

    model_directory = resolve_model_directory(arguments.model_dir)
    tokenizer = Qwen3Tokenizer(model_directory)
    # 模型对照 dtype：CPU 后端走 FP32（自洽校验，应近零误差），
    # GPU 后端走 FP16（合同规定的公共性能路径）。
    model_dtype = torch.float32 if arguments.backend == "cpu" else torch.float16
    model = load_handwritten_model(model_directory, device, dtype=model_dtype).eval()

    prompt_ids = truth["prompt_token_ids"]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Prompt 前向（带探针）与首 Token Logits。
    with torch.no_grad():
        logits, probes = model(input_ids, probe=True)
    backend_first_logits = logits[0, -1, :].float().cpu()
    first_token_logits_truth = torch.from_numpy(probes_npz["first_token_logits"])

    top5_backend = torch.topk(backend_first_logits, 5).indices.tolist()
    top5_truth = [entry["id"] for entry in truth["first_token_top5"]]
    first_token_metrics = comparison_metrics(backend_first_logits, first_token_logits_truth)
    first_token_metrics["top5_backend"] = top5_backend
    first_token_metrics["top5_truth"] = top5_truth
    first_token_metrics["top5_overlap"] = len(set(top5_backend) & set(top5_truth))

    # 探针逐层误差（FP16 后端探针 vs CPU FP32 真值探针）。
    layer_metrics = {}
    for name in sorted(probes_npz.files):
        if name == "first_token_logits":
            continue
        truth_probe = torch.from_numpy(probes_npz[name])
        backend_probe = probes[name][0].float().cpu()
        layer_metrics[name] = comparison_metrics(backend_probe, truth_probe)

    # Greedy 解码对照。
    generated, finished = model.greedy_generate(
        input_ids,
        max_new_tokens=truth["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id,
    )
    backend_ids = generated[0].tolist()[len(prompt_ids):]
    truth_ids = truth["generated_token_ids"]
    first_divergence = None
    for index, (a, b) in enumerate(zip(backend_ids, truth_ids)):
        if a != b:
            first_divergence = index
            break
    if first_divergence is None and len(backend_ids) != len(truth_ids):
        first_divergence = min(len(backend_ids), len(truth_ids))

    greedy_result = {
        "backend_tokens": backend_ids,
        "truth_tokens": truth_ids,
        "token_count_match": len(backend_ids) == len(truth_ids),
        "all_match": backend_ids == truth_ids,
        "first_divergence_index": first_divergence,
        "finished_by_eos": finished,
    }
    worst_layer = max(layer_metrics.items(), key=lambda item: item[1]["max_abs_error"])
    print(
        f"[model/{arguments.backend}] greedy 全一致={greedy_result['all_match']} "
        f"首分叉={first_divergence} | 最差探针 {worst_layer[0]} "
        f"max_abs={worst_layer[1]['max_abs_error']:.3e} cos={worst_layer[1]['cosine_similarity']:.6f}"
    )
    return {
        "model_dtype": "float32" if arguments.backend == "cpu" else "float16",
        "first_token_logits": first_token_metrics,
        "greedy": greedy_result,
        "probe_layers": layer_metrics,
        "worst_probe_layer": {"name": worst_layer[0], **worst_layer[1]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--dtype", default=None, choices=["fp32", "fp16"], help="默认：CPU 用 fp32，GPU 用 fp16")
    parser.add_argument("--fixtures", default=str(FIXTURES_DEFAULT))
    parser.add_argument("--truth-dir", default=str(TRUTH_DEFAULT))
    parser.add_argument("--model-dir", default="models/qwen3-0.6b")
    parser.add_argument("--skip-model", action="store_true", help="只跑算子链，跳过模型真值对照")
    arguments = parser.parse_args()

    if arguments.dtype is None:
        arguments.dtype = "fp32" if arguments.backend == "cpu" else "fp16"

    device = resolve_device(arguments.backend)
    arrays, manifest = load_fixtures(Path(arguments.fixtures))

    results = {
        "schema_version": "kernel-chapter01.correctness.v1",
        "backend": arguments.backend,
        "rmsnorm_fixtures": rmsnorm_fixture_chain(arguments, device, arrays, manifest),
        "dtype_error_source": dtype_error_chain(device, arrays),
        "determinism": determinism_chain(device, arrays),
        "model_truth": model_truth_chain(arguments, device),
        "environment": device_environment_record(device),
    }
    output = Path("results") / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
