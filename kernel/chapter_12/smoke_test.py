#!/usr/bin/env python3
"""第 12 期边界行为验证（smoke test，本地 CPU 可跑，不依赖模型权重）。

验证面（全部为"显式"语义的行为学证据）：
  A. 后端注册表与显式选择（未知后端报错 / 不可用后端报错 / 注册表全集）；
  B. dispatch 前置条件（setup 未调用即报错——不做隐式初始化）；
  C. fused op 合同违规显式报错（bf16 / H=0 / 非连续 / 别名 / 设备不符）；
  D. strict 注册表：未注册设备显式报错（列出已注册集合）；
     explicit_fallback / silent 两模式对照（第 04 期语义原样保留）；
  E. 平台无关路径功能正确性：rmsnorm dispatch = 第 01 期公式；
     fused op vs composite 数学一致；小模型 native/fused Greedy 同 ids；
  F. fixtures 重放（seeded 输入 vs 期望输出）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE / "implementations" / "runtime")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import sha256_of_tensor, write_results_json  # noqa: E402
from implementations.runtime import (  # noqa: E402
    BackendNotAvailableError,
    CapabilityError,
    create_backend,
    list_backends,
)
from implementations.runtime.backend import ExplicitFallbackError  # noqa: E402

RESULTS: list[dict[str, object]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append({"check": name, "passed": bool(condition), "detail": detail[:400]})
    marker = "PASS" if condition else "FAIL"
    print(f"[smoke/{marker}] {name}" + (f" | {detail[:200]}" if detail else ""))


def expect_raises(name: str, expected_type: type, function, needle: str | None = None):
    """断言 function 抛出 expected_type（错误文本留档）。"""
    try:
        function()
    except expected_type as error:  # noqa: BLE001
        text = str(error)
        ok = (needle is None) or (needle in text)
        check(name, ok, f"{type(error).__name__}: {text[:220]}")
        return text
    except Exception as error:  # noqa: BLE001
        check(name, False, f"错误类型不符：{type(error).__name__}: {error}")
        return ""
    check(name, False, "未抛错（预期显式报错）")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "results" / "smoke-results-cpu.json"))
    arguments = parser.parse_args()

    print(f"[smoke] torch {torch.__version__}")

    # A. 注册表与显式选择
    check("A1 注册表全集 = cpu/cuda/musa", list_backends() == ["cpu", "cuda", "musa"],
          str(list_backends()))
    expect_raises("A2 未知后端显式报错", ValueError,
                  lambda: create_backend("tpu"), needle="已注册")
    probe_results = {}
    for backend_name in ("cpu", "cuda", "musa"):
        backend = create_backend(backend_name)
        available, reason = backend.probe_availability()
        probe_results[backend_name] = {"available": available, "reason": reason}
    check("A3 cpu 后端可用", probe_results["cpu"]["available"])
    for backend_name in ("cuda", "musa"):
        if not probe_results[backend_name]["available"]:
            backend = create_backend(backend_name)
            expect_raises(
                f"A4 不可用后端 {backend_name} 显式报错（不静默回退）",
                BackendNotAvailableError, backend.ensure_available,
                needle="静默 CPU fallback",
            )
    print(f"[smoke] 可用性探测：{json.dumps(probe_results, ensure_ascii=False)}")

    # B. dispatch 前置条件
    cpu = create_backend("cpu")
    x = torch.randn(2, 4)
    r = torch.randn(2, 4)
    w = torch.ones(4)
    expect_raises("B1 setup 未调用时 dispatch 显式报错", CapabilityError,
                  lambda: cpu.fused_add_rmsnorm(x, r, w, 1e-6), needle="setup")

    # C/D. 合同与注册表显式语义（setup 后）
    evidence = cpu.setup()
    check("C0 cpu setup 返回注册证据", "composite" in evidence["fused_op_path"],
          evidence["fused_op_path"])

    bf16_input = torch.randn(2, 4, dtype=torch.bfloat16)
    bf16_res = torch.randn(2, 4, dtype=torch.bfloat16)
    bf16_w = torch.ones(4, dtype=torch.bfloat16)
    expect_raises("C1 bf16 进融合 op 显式报错（合同 dtype 集合）", CapabilityError,
                  lambda: cpu.fused_add_rmsnorm(bf16_input, bf16_res, bf16_w, 1e-6),
                  needle="不静默回退")

    h0 = torch.randn(0, 4)
    r0 = torch.randn(0, 4)
    # 注：空张量 data_ptr()==0，Python ref 实现的别名规则先于 H=0 规则触发
    # （两条都是显式 ValueError——顺序差异如实留档，见 EXPERIMENT_LOG）。
    expect_raises("C2 H=0 空张量显式报错（别名规则先行）", ValueError,
                  lambda: cpu.fused_add_rmsnorm(h0, r0, w, 1e-6))

    nc = torch.randn(4, 2).t()  # shape (2,4)，非连续
    ncr = torch.randn(2, 4)     # shape (2,4)，连续
    expect_raises("C3 非连续输入显式报错", ValueError,
                  lambda: cpu.fused_add_rmsnorm(nc, ncr, torch.ones(4), 1e-6),
                  needle="连续")

    same = torch.randn(2, 4)
    expect_raises("C4 input/residual 同 buffer 显式报错", ValueError,
                  lambda: cpu.fused_add_rmsnorm(same, same, w, 1e-6), needle="同一 buffer")

    meta = torch.zeros(2, 4, device="meta")
    meta_r = torch.zeros(2, 4, device="meta")
    expect_raises("C5 异设备张量显式报错（strict，不改道）", CapabilityError,
                  lambda: cpu.fused_add_rmsnorm(meta, meta_r, w, 1e-6), needle="strict")

    expect_raises("D1 未注册设备 strict 报错（列出已注册集合）", RuntimeError,
                  lambda: cpu.fused_ops_protocol().resolve(meta), needle="已注册")

    # D2/D3. explicit_fallback / silent 对照（第 04 期语义原样；用记录型
    # callable 隔离 op 合同，只考察路由模式语义）。
    from implementations.runtime.ops_router import MiniInfraOps

    calls: list[tuple] = []

    def recording_impl(input_tensor, residual, weight, eps):
        calls.append((input_tensor.device.type, weight.shape[0]))

    fb_router = MiniInfraOps(fallback_mode="explicit_fallback")
    fb_router.register("cpu", recording_impl, label="recording", is_fallback=True)
    stderr_capture = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture):
        fb_router.fused_add_rmsnorm(meta, meta_r, w, 1e-6)
    check("D2 explicit_fallback 留痕+计数", fb_router.fallback_counts.get("meta") == 1
          and "显式降级" in stderr_capture.getvalue() and len(calls) == 1,
          stderr_capture.getvalue().strip()[:180])

    silent_router = MiniInfraOps(fallback_mode="silent")
    silent_router.register("cpu", recording_impl, label="recording", is_fallback=True)
    quiet = io.StringIO()
    calls.clear()
    with contextlib.redirect_stderr(quiet):
        silent_router.fused_add_rmsnorm(meta, meta_r, w, 1e-6)
    check("D3 silent 模式无任何留痕（反模式对照组）",
          silent_router.fallback_counts == {} and quiet.getvalue() == ""
          and silent_router.call_counts.get("cpu") == 1 and len(calls) == 1,
          f"stderr={quiet.getvalue()!r} counts={silent_router.summary()}")

    from implementations.runtime.fused_add_rmsnorm_ref import (
        composite_fallback_fused_add_rmsnorm,
    )

    expect_raises("D4 重复注册拒绝（显式注册表语义）", ValueError,
                  lambda: cpu.fused_ops_protocol().register(
                      "cpu", composite_fallback_fused_add_rmsnorm, "dup"),
                  needle="拒绝覆盖")

    # E. 平台无关路径功能正确性
    torch.manual_seed(0)
    xe = torch.randn(4, 16)
    we = torch.randn(16)
    y_dispatch = cpu.rmsnorm(xe, we, 1e-6)
    float_states = xe.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    y_formula = we * (float_states * torch.rsqrt(variance + 1e-6)).to(xe.dtype)
    check("E1 rmsnorm dispatch = 第 01 期公式（位级）",
          torch.equal(y_dispatch, y_formula),
          f"max_abs={(y_dispatch - y_formula).abs().max().item():.3e}")

    h2 = torch.randn(4, 16)
    r2 = torch.randn(4, 16)
    w2 = torch.randn(16)
    h_op = h2.clone()
    r_op = r2.clone()
    cpu.fused_add_rmsnorm(h_op, r_op, w2, 1e-6)
    t_ref, y_ref = r2 + h2, None
    fs = t_ref.float()
    y_ref = w2 * (fs * torch.rsqrt(fs.pow(2).mean(-1, keepdim=True) + 1e-6)).to(t_ref.dtype)
    check("E2 fused op residual 语义（r+h 位级）", torch.equal(r_op, t_ref))
    check("E3 fused op 输出 vs composite（位级，CPU FP32）", torch.equal(h_op, y_ref))

    # 小模型 native/fused Greedy 一致
    from implementations.runtime.qwen3_model_fused import (
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    config = Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6,
        rope_theta=10000.0, torch_dtype="float32", tie_word_embeddings=True,
    )
    torch.manual_seed(7)
    native_model = Qwen3ForCausalLM(config)
    native_model.eval()
    torch.manual_seed(7)
    fused_model = Qwen3ForCausalLM(config, fused_ops=cpu.fused_ops_protocol())
    fused_model.eval()
    ids = torch.randint(0, 64, (1, 5), generator=torch.Generator().manual_seed(3))
    out_native, done_n = native_model.greedy_generate(ids, max_new_tokens=6, eos_token_id=None)
    out_fused, done_f = fused_model.greedy_generate(ids, max_new_tokens=6, eos_token_id=None)
    check("E4 小模型 native/fused Greedy 全同 ids",
          out_native.tolist() == out_fused.tolist() and not done_n and not done_f,
          f"ids={out_native[0].tolist()}")

    # F. fixtures 重放
    fixtures_path = HERE / "fixtures" / "probes.npz"
    manifest_path = HERE / "fixtures" / "manifest.json"
    if fixtures_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        arrays = np.load(fixtures_path)  # type: ignore[assignment]
        mismatches = [
            key for key, record in manifest["tensors"].items()
            if sha256_of_tensor(torch.from_numpy(arrays[key])) != record["sha256"]
        ]
        check("F1 fixtures SHA256 清单全对账", not mismatches, f"mismatch={mismatches[:3]}")
        fp32_in = torch.from_numpy(arrays["op_probe_float32_input"])
        fp32_r = torch.from_numpy(arrays["op_probe_float32_residual"])
        fp32_w = torch.from_numpy(arrays["op_probe_float32_weight"])
        expected_y = torch.from_numpy(arrays["op_probe_fp32_expected_output"])
        got_r = fp32_r.clone()
        got_h = fp32_in.clone()
        cpu.fused_add_rmsnorm(got_h, got_r, fp32_w, 1e-6)
        check("F2 fixture 重放输出 vs 期望（位级）",
              torch.equal(got_h, expected_y) and torch.equal(
                  got_r, torch.from_numpy(arrays["op_probe_fp32_expected_residual"])),
              f"max_abs={(got_h.float() - expected_y.float()).abs().max().item():.3e}")
    else:
        check("F1 fixtures 存在", False, "先运行 generate_fixtures.py")

    passed = sum(1 for item in RESULTS if item["passed"])
    payload = {
        "schema_version": "kernel-chapter13.smoke.v1",
        "torch": torch.__version__,
        "availability_probe": probe_results,
        "passed": passed,
        "total": len(RESULTS),
        "checks": RESULTS,
    }
    write_results_json(Path(arguments.out), payload)
    print(f"[smoke] {passed}/{len(RESULTS)} PASS")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    import numpy as np  # noqa: F401  (F 段使用)

    raise SystemExit(main())
