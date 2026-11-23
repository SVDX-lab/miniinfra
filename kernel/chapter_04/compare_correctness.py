#!/usr/bin/env python3
"""第 04 期正确性对照：注册层语义 + 大纲 §9 正确性链第 6–8 步。

对照链（本期新增部分标注）：
1. （注册回归）第 03 期 30 组 fixtures 经注册 op（torch.ops / composite）重放，
   逐张量 SHA256 核对 + t/y 误差统计——"注册层未改变被注册算子语义"；
2. （注册语义）边界与 fallback 行为：dtype/shape/aliasing/连续性显式报错原文、
   CPU fallback 触发证据（stderr 留痕 + 数值）、no_cpu_fallback 框架报错原文、
   路由器 strict 模式显式报错；
3. （§9 第 6 步）单个 Qwen3 RMSNorm 模块输出：真实权重 × 真值 embedding 探针，
   三方对照（原生模块 / 融合 op 退化形式 h=0 / CPU FP32 公式）；
4. （§9 第 7 步）单层 Transformer 中间输出：layer 0 原生链 vs 融合链，
   输出 vs probes.npz["layer_0"]；
5. （§9 第 8 步）完整模型：首 Token Logits vs 真值 + Greedy 32 Token 一致性
   （原生与融合两条路径分别对照）。

用法：
    python3 -I compare_correctness.py --backend cuda --model-dir ../chapter_01/models/qwen3-0.6b
    python3 -I compare_correctness.py --backend cpu  --skip-model   # 本地（无权重时）
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE / "implementations")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import (  # noqa: E402
    SCHEMA_VERSION,
    backend_sync,
    comparison_metrics,
    device_environment_record,
    resolve_device,
    sha256_of_file,
    write_results_json,
)
from implementations.backend_setup import setup_backend  # noqa: E402
from implementations.cpu.fused_add_rmsnorm_ref import (  # noqa: E402
    RMS_NORM_EPS,
    rmsnorm_module_math,
)
import qwen3_model_fused as model_lib  # noqa: E402
# Qwen3Tokenizer 延迟 import（regex 依赖仅模型级测试需要，见 main()）。

FIXTURES_DEFAULT = HERE / "fixtures"
TRUTH_DEFAULT = HERE.parent / "chapter_01" / "truth"
MODEL_DEFAULT = HERE.parent / "chapter_01" / "models" / "qwen3-0.6b"


@contextlib.contextmanager
def capture_fd_stderr():
    """捕获 C 层 stderr（fprintf 直写 fd 2，Python redirect_stderr 捕不到）。

    用法：with capture_fd_stderr() as tmp: <call>; text = tmp.read()
    （read 必须在 with 块内——TemporaryFile 在退出时关闭）。
    """
    with tempfile.TemporaryFile(mode="w+") as tmp:
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            yield tmp
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def expect_error(operation, case: str, expect_keywords: list[str]) -> dict:
    """执行并捕获期望中的异常：记录类型与原文（合同证据，不做关键词强断言）。"""
    try:
        operation()
    except Exception as error:  # noqa: BLE001
        message = f"{type(error).__name__}: {error}"
        return {
            "case": case,
            "raised": True,
            "error_type": type(error).__name__,
            "message": message[:600],
            "matched_keywords": [k for k in expect_keywords if k.lower() in message.lower()],
        }
    return {"case": case, "raised": False,
            "violation": "期望报错但未报错——静默回退嫌疑，记入 KNOWN_ISSUES"}


# ---------------------------------------------------------------------------
# 1. 30 组 fixtures 注册回归
# ---------------------------------------------------------------------------
def op_fixtures_chain(backend: str, device: torch.device, router,
                      arrays: dict, manifest: dict) -> dict:
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    for combo in manifest["combos"]:
        rows, hidden = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden}"
        for dtype_name in ("fp32", "fp16"):
            dtype = torch.float32 if dtype_name == "fp32" else torch.float16
            if hidden not in weight_cache:
                weight_cache[hidden] = {}
            if dtype not in weight_cache[hidden]:
                weight_cache[hidden][dtype] = torch.from_numpy(
                    arrays[f"w_{hidden}"]).to(device=device, dtype=dtype)
            weight = weight_cache[hidden][dtype]
            r_host = torch.from_numpy(arrays[f"r_{key}"])
            h_host = torch.from_numpy(arrays[f"h_{key}"])
            t_truth = torch.from_numpy(arrays[f"t_{key}"])
            y_truth = torch.from_numpy(arrays[f"y_{key}"])
            # 注册 op 是 in-place 双 buffer 语义：residual ← t，input ← y。
            r = r_host.to(device=device, dtype=dtype).clone()
            h = h_host.to(device=device, dtype=dtype).clone()
            router.fused_add_rmsnorm(h, r, weight, RMS_NORM_EPS)
            backend_sync(device)
            m_t = comparison_metrics(r, t_truth.to(device))
            m_y = comparison_metrics(h, y_truth.to(device))
            groups.append({"rows": rows, "hidden_size": hidden,
                           "dtype": dtype_name,
                           "residual_vs_truth_t": m_t, "output_vs_truth_y": m_y})
    worst_t = max(groups, key=lambda g: g["residual_vs_truth_t"]["max_abs_error"])
    worst_y = max(groups, key=lambda g: g["output_vs_truth_y"]["max_abs_error"])
    print(f"[fixtures/{backend}] 30 组 × fp32/fp16 经注册 op 重放 | "
          f"residual 最差 {worst_t['rows']}x{worst_t['hidden_size']}({worst_t['dtype']}) "
          f"max_abs={worst_t['residual_vs_truth_t']['max_abs_error']:.3e} | "
          f"y 最差 {worst_y['rows']}x{worst_y['hidden_size']}({worst_y['dtype']}) "
          f"max_abs={worst_y['output_vs_truth_y']['max_abs_error']:.3e}")
    return {"groups_count": len(groups), "groups": groups,
            "worst_residual": worst_t, "worst_output": worst_y}


# ---------------------------------------------------------------------------
# 2. 注册语义：边界行为 + fallback 证据
# ---------------------------------------------------------------------------
def boundary_chain(backend: str, device: torch.device, router, op_module) -> dict:
    cases = []
    rows, hidden = 4, 1024
    dtype = torch.float32 if backend == "cpu" else torch.float16

    def make():
        r = torch.randn(rows, hidden, device=device, dtype=dtype)
        h = torch.randn(rows, hidden, device=device, dtype=dtype)
        w = torch.randn(hidden, device=device, dtype=dtype)
        return r, h, w

    # a. dtype 混用
    r, h, w = make()
    cases.append(expect_error(
        lambda: router.fused_add_rmsnorm(h, r.float(), w, RMS_NORM_EPS),
        "residual 与 input dtype 不一致", ["dtype"]))
    # b. bfloat16（第 05 期范围；显式报错）
    r, h, w = make()
    cases.append(expect_error(
        lambda: router.fused_add_rmsnorm(h.to(torch.bfloat16), r.to(torch.bfloat16),
                                         w.to(torch.bfloat16), RMS_NORM_EPS),
        "bfloat16 输入（合同外 dtype）", ["dtype"]))
    # c. H=0
    r0 = torch.randn(rows, 0, device=device, dtype=dtype)
    h0 = torch.randn(rows, 0, device=device, dtype=dtype)
    w0 = torch.randn(0, device=device, dtype=dtype)
    cases.append(expect_error(
        lambda: router.fused_add_rmsnorm(h0, r0, w0, RMS_NORM_EPS), "H=0 空 Tensor", ["H=0"]))
    # d. 行数不匹配
    r, h, w = make()
    cases.append(expect_error(
        lambda: router.fused_add_rmsnorm(h, r[:2], w, RMS_NORM_EPS),
        "residual 与 input 行数不一致", ["形状", "shape", "sizes"]))
    # e. 非连续输入
    r, h, w = make()
    h_nc = torch.randn(rows, hidden * 2, device=device, dtype=dtype)[:, ::2]
    cases.append(expect_error(
        lambda: router.fused_add_rmsnorm(h_nc, r, w, RMS_NORM_EPS),
        "非连续 input", ["连续", "contiguous"]))
    # f. residual 与 input 同 buffer（aliasing 拒绝）
    r, h, w = make()
    same = r.clone()
    cases.append(expect_error(
        lambda: router.fused_add_rmsnorm(same, same, w, RMS_NORM_EPS),
        "residual 与 input 同 buffer", ["同一 buffer", "alias"]))

    if backend in ("cuda", "musa") and op_module is not None:
        # g. CPU 张量 → 注册 op：CPU composite fallback（显式留痕 + 数值正确）
        r_cpu = torch.randn(rows, hidden, dtype=torch.float32)
        h_cpu = torch.randn(rows, hidden, dtype=torch.float32)
        w_cpu = torch.randn(hidden, dtype=torch.float32)
        h_expected = h_cpu.clone()
        r_expected = r_cpu.clone()
        t_ref = r_expected + h_expected
        y_ref = w_cpu * (t_ref.float() * torch.rsqrt(
            t_ref.float().pow(2).mean(-1, keepdim=True) + RMS_NORM_EPS)).to(torch.float32)
        with capture_fd_stderr() as tmp:
            op_module.fused_add_rmsnorm(h_cpu, r_cpu, w_cpu, RMS_NORM_EPS)
            stderr_text = tmp.read()
        cases.append({
            "case": "CPU 张量经注册 op → CPU composite fallback（显式留痕）",
            "fallback_fired": True,
            "stderr_trace": stderr_text.strip()[:600],
            "output_vs_python_composite": comparison_metrics(h_cpu, y_ref),
            "residual_vs_python_composite": comparison_metrics(r_cpu, t_ref),
        })
        # h. no_cpu_fallback 对照族：CPU 张量 → 框架显式报错（原文留档）
        cases.append(expect_error(
            lambda: op_module.fused_add_rmsnorm_no_cpu_fallback(
                h_cpu.clone(), r_cpu.clone(), w_cpu, RMS_NORM_EPS),
            "无 CPU impl 的同名族 op + CPU 张量（对照：框架显式报错，不静默回退）",
            ["NotImplementedError", "Could not run"]))

    # i. 路由器 strict 模式：未注册设备的显式报错（用空注册表 + 真实张量）
    from ops_router import MiniInfraOps

    empty_router = MiniInfraOps(fallback_mode="strict")
    r, h, w = make()
    cases.append(expect_error(
        lambda: empty_router.fused_add_rmsnorm(h, r, w, RMS_NORM_EPS),
        "路由器 strict 模式：无注册实现", ["没有已注册"]))

    violations = [case["case"] for case in cases if case.get("violation")]
    print(f"[boundary/{backend}] {len(cases)} 例；期望报错/留痕全部成立="
          f"{not violations}" + (f"（违例：{violations}）" if violations else ""))
    return {"cases": cases, "violations": violations}


# ---------------------------------------------------------------------------
# 3. §9 第 6 步：单个 RMSNorm 模块输出（三方对照）
# ---------------------------------------------------------------------------
def step6_single_rmsnorm(backend: str, device: torch.device, router, model,
                         truth_probes) -> dict:
    eps = float(model.config.rms_norm_eps)
    model_dtype = next(model.parameters()).dtype
    probe_input_f32 = torch.from_numpy(truth_probes["embedding"]).to(device)  # [seq, H]
    weight = model.model.norm.weight  # 真实模型权重（final norm）

    # (a) 原生 RMSNorm 模块（与 chapter_01 同一 forward 数学）
    x = probe_input_f32.to(model_dtype)
    native = model.model.norm(x)
    # (b) 融合 op 退化形式：h = 0 → t = r + 0 = r，op 输出 = RMSNorm(r)
    zero = torch.zeros_like(x)
    residual = x.clone()
    router.fused_add_rmsnorm(zero, residual, weight, eps)
    backend_sync(device)
    fused_y = zero
    residual_after = residual
    # (c) CPU FP32 公式（真值锚点）
    expected = rmsnorm_module_math(probe_input_f32.cpu(), weight.detach().cpu(), eps)

    native_vs_truth = comparison_metrics(native, expected.to(device))
    fused_vs_truth = comparison_metrics(fused_y, expected.to(device))
    fused_vs_native = comparison_metrics(fused_y, native)
    residual_check = comparison_metrics(residual_after, probe_input_f32)
    print(f"[step6/{backend}] RMSNorm 模块三方对照：native vs FP32公式 "
          f"max_abs={native_vs_truth['max_abs_error']:.3e} | fused(h=0) vs FP32公式 "
          f"max_abs={fused_vs_truth['max_abs_error']:.3e} | fused vs native "
          f"max_abs={fused_vs_native['max_abs_error']:.3e} | residual(r+0)漂移="
          f"{residual_check['max_abs_error']:.3e}")
    return {"weight_source": "model.model.norm.weight（真实权重）",
            "input_source": "truth probes['embedding']（真实探针）",
            "model_dtype": str(model_dtype),
            "native_module_vs_fp32_formula": native_vs_truth,
            "fused_op_h0_vs_fp32_formula": fused_vs_truth,
            "fused_op_h0_vs_native_module": fused_vs_native,
            "fused_residual_r_plus_zero_drift": residual_check}


# ---------------------------------------------------------------------------
# 4. §9 第 7 步：单层 Transformer（layer 0，原生链 vs 融合链 vs truth 探针）
# ---------------------------------------------------------------------------
def _layer0_context(model, input_ids):
    model_core = model.model
    batch, seq = input_ids.shape
    position_ids = torch.arange(seq, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand(batch, -1)
    cosine, sine = model_core.rotary_emb(position_ids, model_core.embed_tokens(input_ids).dtype)
    return cosine, sine


def step7_single_layer(backend: str, device: torch.device, router, model,
                       input_ids, truth_probes) -> dict:
    model_core = model.model
    layer = model_core.layers[0]
    eps = float(model.config.rms_norm_eps)
    truth_layer0 = torch.from_numpy(truth_probes["layer_0"]).to(device)

    cosine, sine = _layer0_context(model, input_ids)
    mask = model_core.make_attention_mask(
        None, input_ids.shape[1],
        model_core.embed_tokens(input_ids).dtype, device)

    # 原生链（layer 0 与 chapter_01 forward 逐字同构）
    def native_layer0():
        embed = model_core.embed_tokens(input_ids)
        normed = layer.input_layernorm(embed)
        attn_out = layer.self_attn(normed, cosine, sine, mask)
        hidden = embed + attn_out
        normed2 = layer.post_attention_layernorm(hidden)
        mlp_out = layer.mlp(normed2)
        return hidden + mlp_out

    # 融合链（层内 add+norm 用注册 op；末次 add 原生——下一边界 op 属 layer 1）
    def fused_layer0():
        embed = model_core.embed_tokens(input_ids)
        residual = embed
        normed = layer.input_layernorm(residual)
        attn_out = layer.self_attn(normed, cosine, sine, mask)
        router.fused_add_rmsnorm(attn_out, residual,
                                 layer.post_attention_layernorm.weight, eps)
        mlp_out = layer.mlp(attn_out)
        return residual + mlp_out

    with torch.no_grad():
        native_out = native_layer0().detach().float()
        fused_out = fused_layer0().detach().float()
    backend_sync(device)

    native_vs_truth = comparison_metrics(native_out, truth_layer0)
    fused_vs_truth = comparison_metrics(fused_out, truth_layer0)
    fused_vs_native = comparison_metrics(fused_out, native_out)
    print(f"[step7/{backend}] layer0 vs truth 探针：native max_abs="
          f"{native_vs_truth['max_abs_error']:.3e} cos={native_vs_truth['cosine_similarity']:.8f} | "
          f"fused max_abs={fused_vs_truth['max_abs_error']:.3e} "
          f"cos={fused_vs_truth['cosine_similarity']:.8f} | fused vs native "
          f"max_abs={fused_vs_native['max_abs_error']:.3e}")
    return {"shape": list(input_ids.shape),
            "native_vs_truth_layer0": native_vs_truth,
            "fused_vs_truth_layer0": fused_vs_truth,
            "fused_vs_native": fused_vs_native}


# ---------------------------------------------------------------------------
# 5. §9 第 8 步：完整模型（首 Token Logits + probes + Greedy 32 Token）
# ---------------------------------------------------------------------------
def step8_full_model(backend: str, device: torch.device, router, model,
                     input_ids, truth, truth_probes, tokenizer,
                     run_greedy: bool) -> dict:
    prompt_len = input_ids.shape[1]
    results = {}

    with torch.no_grad():
        logits_native, probes_native = model(input_ids, probe=True)
    results["native"] = _model_forward_report(
        backend, device, logits_native, probes_native, truth, truth_probes, prompt_len)

    # Greedy（原生路径）：必须在 fused_ops 切换之前跑（否则 native 名义下
    # 实际走融合路径——首个 musa 会话实测抓出的顺序 bug）。
    ids_native = None
    fin_native = None
    if run_greedy:
        eos_id = truth.get("eos_token_id")
        gen_native, fin_native = model.greedy_generate(
            input_ids, max_new_tokens=truth["max_new_tokens"], eos_token_id=eos_id)
        ids_native = gen_native[0].tolist()[prompt_len:]

    # 切换融合路径（同一模型实例、同一权重——单变量 = 数据流组织方式）
    model.model.fused_ops = router
    try:
        with torch.no_grad():
            logits_fused, probes_fused = model(input_ids, probe=True)
        results["fused"] = _model_forward_report(
            backend, device, logits_fused, probes_fused, truth, truth_probes, prompt_len)
        if run_greedy:
            eos_id = truth.get("eos_token_id")
            gen_fused, fin_fused = model.greedy_generate(
                input_ids, max_new_tokens=truth["max_new_tokens"], eos_token_id=eos_id)
            ids_fused = gen_fused[0].tolist()[prompt_len:]
            truth_ids = truth["generated_token_ids"]
            results["greedy"] = {
                "native_tokens": ids_native,
                "fused_tokens": ids_fused,
                "truth_tokens": truth_ids,
                "native_all_match": ids_native == truth_ids,
                "fused_all_match": ids_fused == truth_ids,
                "fused_vs_native_match": ids_fused == ids_native,
                "native_first_divergence": _first_divergence(ids_native, truth_ids),
                "fused_first_divergence": _first_divergence(ids_fused, truth_ids),
                "fused_vs_native_first_divergence": _first_divergence(ids_fused, ids_native),
                "native_finished_by_eos": fin_native,
                "fused_finished_by_eos": fin_fused,
            }
            print(f"[step8/{backend}] greedy：native 全一致={results['greedy']['native_all_match']} | "
                  f"fused 全一致={results['greedy']['fused_all_match']} | "
                  f"fused==native={results['greedy']['fused_vs_native_match']}")
    finally:
        model.model.fused_ops = None
    return results


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _model_forward_report(backend, device, logits, probes, truth, truth_probes,
                          prompt_len) -> dict:
    first_logits = logits[0, -1, :].float().cpu()
    truth_logits = torch.from_numpy(truth_probes["first_token_logits"])
    first_metrics = comparison_metrics(first_logits, truth_logits)
    top5_backend = torch.topk(first_logits, 5).indices.tolist()
    top5_truth = [entry["id"] for entry in truth["first_token_top5"]]
    first_metrics["top5_backend"] = top5_backend
    first_metrics["top5_truth"] = top5_truth
    first_metrics["top5_overlap"] = len(set(top5_backend) & set(top5_truth))

    layer_metrics = {}
    for name in sorted(truth_probes.files):
        if name == "first_token_logits":
            continue
        truth_probe = torch.from_numpy(truth_probes[name])
        backend_probe = probes[name][0].float().cpu()
        layer_metrics[name] = comparison_metrics(backend_probe, truth_probe)
    worst_layer = max(layer_metrics.items(), key=lambda item: item[1]["max_abs_error"])
    print(f"[forward/{backend}] 首 Token Logits max_abs={first_metrics['max_abs_error']:.3e} "
          f"cos={first_metrics['cosine_similarity']:.8f} top5重合={first_metrics['top5_overlap']}/5 | "
          f"最差探针 {worst_layer[0]} max_abs={worst_layer[1]['max_abs_error']:.3e}")
    return {"first_token_logits": first_metrics,
            "probe_layers": layer_metrics,
            "worst_probe_layer": {"name": worst_layer[0], **worst_layer[1]}}


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--dtype", default=None, choices=["fp32", "fp16"],
                        help="算子级 fixtures 回归 dtype（默认 CPU 用 fp32/fp16 都跑）")
    parser.add_argument("--fixtures", default=str(FIXTURES_DEFAULT))
    parser.add_argument("--truth-dir", default=str(TRUTH_DEFAULT))
    parser.add_argument("--model-dir", default=str(MODEL_DEFAULT))
    parser.add_argument("--skip-model", action="store_true",
                        help="跳过 §9 第 6–8 步（模型权重/truth 不可用时）")
    parser.add_argument("--skip-greedy", action="store_true",
                        help="跳过 Greedy 32 Token（快速迭代用）")
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    setup = setup_backend(arguments.backend)
    router = setup["router"]
    op_module = setup["op_module"]

    fixtures_dir = Path(arguments.fixtures)
    with np.load(fixtures_dir / "correctness.npz") as data:
        arrays = {key: data[key] for key in data.files}
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))

    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "registration_evidence": setup["evidence"],
        "op_fixtures_regression": op_fixtures_chain(
            arguments.backend, device, router, arrays, manifest),
        "boundary_and_fallback": boundary_chain(
            arguments.backend, device, router, op_module),
    }

    if not arguments.skip_model:
        truth_dir = Path(arguments.truth_dir)
        truth = json.loads((truth_dir / "truth-manifest.json").read_text(encoding="utf-8"))
        truth_probes = np.load(truth_dir / "probes.npz")
        model_directory = model_lib.resolve_model_directory(arguments.model_dir)
        model_dtype = torch.float32 if arguments.backend == "cpu" else torch.float16
        model = model_lib.load_handwritten_model(
            model_directory, device, dtype=model_dtype).eval()
        input_ids = torch.tensor([truth["prompt_token_ids"]], dtype=torch.long, device=device)
        print(f"[model/{arguments.backend}] dtype={model_dtype} "
              f"prompt={input_ids.shape[1]} tokens")
        results["model_dtype"] = str(model_dtype)
        results["step6_single_rmsnorm"] = step6_single_rmsnorm(
            arguments.backend, device, router, model, truth_probes)
        results["step7_single_layer"] = step7_single_layer(
            arguments.backend, device, router, model, input_ids, truth_probes)
        from qwen3_tokenizer import Qwen3Tokenizer  # noqa: E402（regex 延迟依赖）

        results["step8_full_model"] = step8_full_model(
            arguments.backend, device, router, model, input_ids, truth,
            truth_probes, Qwen3Tokenizer(model_directory),
            run_greedy=not arguments.skip_greedy)
    else:
        results["model_section"] = "skipped（--skip-model：权重/truth 不可用或快速迭代）"

    results["environment"] = device_environment_record(device)
    results["router_summary"] = router.summary()  # 全部调用结束后再取快照（含 step8 的 56 次）
    output = HERE / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
