#!/usr/bin/env python3
"""第 09 期正确性比较：图回放输出 vs eager 输出（同设备同权重，eager 即参照）。

本期无 CPU 列（期首声明）——Graph 是设备侧调用组织概念，正确性口径因此
不是"vs CPU 真值"而是"图回放 ≡ eager"（同 kernel 同序，期望位级一致；
对账失败即捕获语义破坏，退出码 1）。四组对账：

  A. launch-bound 链（K=8@1024 元素）graph vs eager（位级）；
  B. normal-shape RMSNorm composite 链 @2048×1024 graph vs eager（位级）；
  C. 注册融合 op @2048×1024 graph vs eager（位级；in-place 复位静态 buffer 后对账）；
  D. decode 步 @seq46：graph_native/graph_fused vs 各自 eager 的 logits max_abs
     与 next-token 一致性 + 真值 expected_next sanity。

输出：results/correctness-results-{backend}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE / "implementations")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import (  # noqa: E402
    GraphCaptureCompat,
    backend_module,
    backend_sync,
    comparison_metrics,
    resolve_device,
    write_results_json,
)

SCHEMA_VERSION = "kernel-chapter10.correctness.v1"


def capture(operation, device, warmup=3, compat=None):
    module = backend_module(device)
    side_stream = module.Stream()
    current = module.current_stream()
    side_stream.wait_stream(current)
    with module.stream(side_stream):
        for _ in range(warmup):
            operation()
    current.wait_stream(side_stream)
    backend_sync(device)
    reference = operation()
    backend_sync(device)
    graph = module.CUDAGraph()
    if compat is not None and compat.supported:
        with compat.capture(graph):
            captured = operation()
    else:
        with module.graph(graph):
            captured = operation()
    backend_sync(device)
    graph.replay()
    backend_sync(device)
    return graph, captured, reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    parser.add_argument("--model-dir", default=str(HERE.parent / "chapter_01" / "models" / "qwen3-0.6b"))
    parser.add_argument("--truth-dir", default=str(HERE.parent / "chapter_01" / "truth"))
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    module = backend_module(device)
    module.empty_cache()
    results: dict = {"schema_version": SCHEMA_VERSION, "backend": arguments.backend,
                     "reference_convention": "graph_replay_vs_eager_same_device（无 CPU 列声明见 README）"}
    all_pass = True

    # A. launch-bound 链
    x = torch.arange(1024, device=device, dtype=torch.float16).clamp(-100, 100)
    t_eager = x.clone()
    for _ in range(8):
        t_eager.add_(1.0)
    ref_a = t_eager.clone()

    side_stream = module.Stream()
    current = module.current_stream()
    side_stream.wait_stream(current)
    t_warm = x.clone()
    with module.stream(side_stream):
        for _ in range(3):
            t_warm.add_(1.0)
    current.wait_stream(side_stream)
    backend_sync(device)
    graph_a = module.CUDAGraph()
    t_static = x.clone()
    with module.graph(graph_a):
        for _ in range(8):
            t_static.add_(1.0)
    backend_sync(device)
    graph_a.replay()
    backend_sync(device)
    entry_a = {
        "bitwise_equal": bool(torch.equal(t_static, ref_a)),
        "max_abs_error": float((t_static.float() - ref_a.float()).abs().max().item()),
    }
    entry_a["pass"] = entry_a["bitwise_equal"]
    all_pass &= entry_a["pass"]
    results["A_launch_bound_chain"] = entry_a
    del graph_a

    # B. RMSNorm composite 链 @2048×1024（fixtures 冻结输入）
    fixture_path = HERE / "fixtures" / "spot-rmsnorm-2048x1024.pt"
    if fixture_path.is_file():
        payload = torch.load(fixture_path, weights_only=False)
        r = payload["r"].to(device=device)
        h = payload["h"].to(device=device)
        w = payload["w"].to(device=device)
    else:
        generator = torch.Generator().manual_seed(20261121 + 2048 * 100000 + 1024)
        r = torch.randn(2048, 1024, generator=generator).to(device=device, dtype=torch.float16)
        h = torch.randn(2048, 1024, generator=generator).to(device=device, dtype=torch.float16)
        w = torch.randn(1024, generator=generator).to(device=device, dtype=torch.float16)

    def composite():
        t = r + h
        float_states = t.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + 1e-6)
        return w * normalized.to(t.dtype)

    graph_b, captured_b, ref_b = capture(composite, device)
    entry_b = comparison_metrics(captured_b, ref_b)
    entry_b["bitwise_equal"] = bool(torch.equal(captured_b, ref_b))
    entry_b["pass"] = entry_b["bitwise_equal"] or entry_b["max_abs_error"] < 1e-2
    all_pass &= entry_b["pass"]
    results["B_rmsnorm_composite_graph"] = entry_b
    del graph_b

    # C. 注册融合 op（in-place：静态 buffer 复位后回放对账）
    from implementations.backend_setup import setup_backend
    setup = setup_backend(arguments.backend)
    router = setup["router"]
    compat = GraphCaptureCompat(device, setup.get("pybind_module"))
    results["capture_mode"] = ("explicit_stream(musa_v2)" if compat.explicit_stream_mode
                               else "torch_graph_context")
    eps = 1e-6
    rs, hs = r.clone(), h.clone()
    rc, hc = r.clone(), h.clone()
    router.fused_add_rmsnorm(hc, rc, w, eps)
    ref_c = hc.clone()
    backend_sync(device)
    graph_c = module.CUDAGraph()
    if compat.supported:
        with compat.capture(graph_c):
            router.fused_add_rmsnorm(hs, rs, w, eps)
    else:
        with module.graph(graph_c):
            router.fused_add_rmsnorm(hs, rs, w, eps)
    backend_sync(device)
    rs.copy_(r)
    hs.copy_(h)
    graph_c.replay()
    backend_sync(device)
    entry_c = comparison_metrics(hs, ref_c)
    entry_c["bitwise_equal"] = bool(torch.equal(hs, ref_c))
    entry_c["pass"] = entry_c["bitwise_equal"] or entry_c["max_abs_error"] < 1e-2
    all_pass &= entry_c["pass"]
    results["C_fused_op_graph"] = entry_c
    del graph_c

    # D. decode 步 @seq46
    sys.path.insert(0, str(HERE / "implementations"))
    import qwen3_model_fused as model_lib
    model = model_lib.load_handwritten_model(arguments.model_dir, device, dtype=torch.float16).eval()
    truth = json.loads((Path(arguments.truth_dir) / "truth-manifest.json").read_text(encoding="utf-8"))
    generated_index = 26
    seq = len(truth["prompt_token_ids"]) + generated_index
    ids = (truth["prompt_token_ids"] + truth["generated_token_ids"])[:seq]
    static_ids = torch.tensor([ids], dtype=torch.long, device=device)
    expected_next = truth["generated_token_ids"][generated_index]

    def decode_step():
        logits = model(static_ids)
        return logits, logits[:, -1, :].argmax(dim=-1, keepdim=True)

    entry_d: dict = {"seq": seq, "expected_next_token": expected_next}
    model.model.fused_ops = None
    graph_n, captured_n, ref_n = capture(decode_step, device, warmup=3, compat=compat)
    logits_n, next_n = ref_n
    logits_gn, next_gn = captured_n
    entry_d["graph_native_vs_eager_native"] = {
        **comparison_metrics(logits_gn, logits_n),
        "bitwise_equal": bool(torch.equal(logits_gn, logits_n)),
        "next_token_equal": int(next_gn[0].item()) == int(next_n[0].item()),
        "next_token_matches_truth": int(next_n[0].item()) == expected_next,
    }
    entry_d["pass_native"] = (entry_d["graph_native_vs_eager_native"]["bitwise_equal"]
                              and entry_d["graph_native_vs_eager_native"]["next_token_matches_truth"])
    del graph_n

    model.model.fused_ops = router
    graph_f, captured_f, ref_f = capture(decode_step, device, warmup=3, compat=compat)
    logits_f, next_f = ref_f
    logits_gf, next_gf = captured_f
    entry_d["graph_fused_vs_eager_fused"] = {
        **comparison_metrics(logits_gf, logits_f),
        "bitwise_equal": bool(torch.equal(logits_gf, logits_f)),
        "next_token_equal": int(next_gf[0].item()) == int(next_f[0].item()),
    }
    entry_d["eager_fused_vs_eager_native"] = {
        **comparison_metrics(logits_f, logits_n),
        "note": "ch05 融合语义容差（fp16 归一化两次舍入 vs 单次舍入）",
    }
    entry_d["pass_fused"] = (entry_d["graph_fused_vs_eager_fused"]["bitwise_equal"]
                             and entry_d["graph_fused_vs_eager_fused"]["next_token_equal"])
    del graph_f
    model.model.fused_ops = None

    all_pass &= entry_d["pass_native"] and entry_d["pass_fused"]
    results["D_decode_step"] = entry_d
    results["all_pass"] = bool(all_pass)

    output = Path("results") / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    print(f"[correctness/{arguments.backend}] all_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
