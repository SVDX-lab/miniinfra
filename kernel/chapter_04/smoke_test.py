#!/usr/bin/env python3
"""第 04 期冒烟测试（本地无 GPU 即可跑）。

覆盖：
1. CPU composite fallback 实现的手算例 / in-place 语义 / h=0 退化 / NaN 语义；
2. 合同边界显式报错（dtype/形状/H=0/非连续/aliasing）；
3. ops_router 三模式语义（strict 显式报错 / explicit_fallback 留痕降级 /
   silent 对照组）与 fallback 计数；
4. 可切换模型端到端管线（随机小配置 Qwen3，同一套权重）：native 与 fused
   前向在 CPU FP32 下位级一致（composite 数学同构）、probe 键集合一致、
   Greedy 解码 token 一致；
5. 注册 schema 静态自检（OP_SCHEMA 与实现签名一致声明）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE / "implementations")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from implementations.cpu.fused_add_rmsnorm_ref import (  # noqa: E402
    RMS_NORM_EPS,
    composite_fallback_fused_add_rmsnorm,
    rmsnorm_module_math,
)
from implementations.ops_router import MiniInfraOps  # noqa: E402
import qwen3_model_fused as model_lib  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f"（{detail}）" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name} —— {detail}")


def expect_value_error(name: str, operation) -> None:
    try:
        operation()
        check(name, False, "期望 ValueError，未抛出（静默回退嫌疑）")
    except ValueError as error:
        check(name, True, str(error)[:80])
    except Exception as error:  # noqa: BLE001
        check(name, False, f"异常类型 {type(error).__name__}: {error}"[:120])


def main() -> int:
    torch.manual_seed(20260501)

    print("== 1. 手算例（2×4，纸面可核） ==")
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, -2.0]])
    h = torch.tensor([[1.0, 0.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    w = torch.ones(4)
    r_buf = r.clone()
    h_buf = h.clone()
    composite_fallback_fused_add_rmsnorm(h_buf, r_buf, w, RMS_NORM_EPS)
    t_expected = r + h
    y_expected = rmsnorm_module_math(t_expected, w)
    check("input buffer = RMSNorm(r+h)*w", torch.equal(h_buf, y_expected),
          f"max_abs={(h_buf - y_expected).abs().max().item():.2e}")
    check("residual buffer = r+h", torch.equal(r_buf, t_expected))
    check("外部引用不受影响（out-of-place 对照）", torch.equal(r, torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, -2.0]])))

    print("== 2. h=0 退化（= 纯 RMSNorm） ==")
    x = torch.randn(3, 16)
    weight = torch.randn(16)
    x_norm_ref = rmsnorm_module_math(x, weight)
    zero = torch.zeros_like(x)
    composite_fallback_fused_add_rmsnorm(zero, x.clone(), weight, RMS_NORM_EPS)
    check("fused(h=0) == RMSNorm 模块数学", torch.equal(zero, x_norm_ref))

    print("== 3. NaN 传播语义 ==")
    r = torch.randn(2, 8)
    h = torch.randn(2, 8)
    h[0, 3] = float("nan")
    w = torch.ones(8)
    r_buf, h_buf = r.clone(), h.clone()
    composite_fallback_fused_add_rmsnorm(h_buf, r_buf, w, RMS_NORM_EPS)
    check("residual 仅 NaN 位置 NaN", bool(torch.isnan(r_buf[0, 3]) and not torch.isnan(r_buf[0, :3]).any()
                                          and not torch.isnan(r_buf[1]).any()))
    check("该行 y 全 NaN、其他行不受影响", bool(torch.isnan(h_buf[0]).all() and not torch.isnan(h_buf[1]).any()))

    print("== 4. 合同边界显式报错 ==")
    r = torch.randn(4, 16)
    h = torch.randn(4, 16)
    w = torch.randn(16)
    h16 = h.to(torch.float16)
    expect_value_error("dtype 混用", lambda: composite_fallback_fused_add_rmsnorm(h16, r, w, RMS_NORM_EPS))
    expect_value_error("bfloat16 拒绝", lambda: composite_fallback_fused_add_rmsnorm(
        h.to(torch.bfloat16), r.to(torch.bfloat16), w.to(torch.bfloat16), RMS_NORM_EPS))
    expect_value_error("H=0", lambda: composite_fallback_fused_add_rmsnorm(
        torch.randn(4, 0), torch.randn(4, 0), torch.randn(0), RMS_NORM_EPS))
    expect_value_error("行数不匹配", lambda: composite_fallback_fused_add_rmsnorm(h, r[:2], w, RMS_NORM_EPS))
    expect_value_error("非连续 input", lambda: composite_fallback_fused_add_rmsnorm(
        torch.randn(4, 32)[:, ::2], r, w, RMS_NORM_EPS))
    expect_value_error("residual/input 同 buffer", lambda: composite_fallback_fused_add_rmsnorm(r, r, w, RMS_NORM_EPS))
    expect_value_error("weight 与 input 共用 buffer", lambda: composite_fallback_fused_add_rmsnorm(
        h, r, h[0], RMS_NORM_EPS))

    print("== 5. ops_router 三模式 ==")
    recorder = MiniInfraOps(fallback_mode="explicit_fallback")
    recorder.register("cpu", lambda i, res, wt, e: None, label="stub.cpu", is_fallback=True)
    stub = SimpleNamespace(device=SimpleNamespace(type="npu"))
    registration = recorder.resolve(stub)
    check("explicit_fallback：未注册设备显式降级到 CPU", registration.label == "stub.cpu")
    check("fallback 计数留痕", recorder.fallback_counts.get("npu") == 1)
    strict = MiniInfraOps(fallback_mode="strict")
    try:
        strict.resolve(stub)
        check("strict 模式显式报错", False, "未抛出")
    except RuntimeError as error:
        check("strict 模式显式报错", "没有已注册" in str(error), str(error)[:80])
    silent = MiniInfraOps(fallback_mode="silent")
    silent.register("cpu", lambda i, res, wt, e: None, label="stub.cpu.silent")
    check("silent 对照组：无留痕降级", (silent.resolve(stub).label == "stub.cpu.silent"
                                        and silent.fallback_counts == {}))
    dup = MiniInfraOps(fallback_mode="strict")
    dup.register("cpu", lambda *a: None, label="a")
    try:
        dup.register("cpu", lambda *a: None, label="b")
        check("重复注册拒绝", False, "未抛出")
    except ValueError:
        check("重复注册拒绝", True)

    print("== 6. 可切换模型端到端管线（随机小配置，同一权重） ==")

    class TinyConfig:
        vocab_size = 64
        hidden_size = 32
        intermediate_size = 64
        num_hidden_layers = 2
        num_attention_heads = 4
        num_key_value_heads = 2
        head_dim = 8
        rms_norm_eps = 1e-6
        rope_theta = 10000.0
        torch_dtype = "float32"
        tie_word_embeddings = True

        def validate(self):
            pass

    config = TinyConfig()
    router = MiniInfraOps(fallback_mode="strict")
    router.register("cpu", composite_fallback_fused_add_rmsnorm,
                    label="cpu.composite_fallback", is_fallback=True)
    model_native = model_lib.Qwen3ForCausalLM(config, fused_ops=None)
    model_fused = model_lib.Qwen3ForCausalLM(config, fused_ops=router)
    model_fused.load_state_dict(model_native.state_dict())  # 同一套权重
    model_native.eval()
    model_fused.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 7))

    with torch.no_grad():
        logits_native, probes_native = model_native(input_ids, probe=True)
        logits_fused, probes_fused = model_fused(input_ids, probe=True)
    check("native/fused FP32 前向位级一致（composite 数学同构）",
          torch.equal(logits_native, logits_fused),
          f"max_abs={(logits_native - logits_fused).abs().max().item():.2e}")
    check("probe 键集合一致", set(probes_native) == set(probes_fused))
    probe_diff = max((probes_native[k] - probes_fused[k]).abs().max().item()
                     for k in probes_native)
    check("probe 数值一致", probe_diff == 0.0, f"max_diff={probe_diff:.2e}")

    gen_native, _ = model_native.greedy_generate(input_ids, max_new_tokens=8)
    gen_fused, _ = model_fused.greedy_generate(input_ids, max_new_tokens=8)
    check("Greedy 8 Token 一致（小模型）", torch.equal(gen_native, gen_fused),
          f"native={gen_native[0].tolist()[-8:]} fused={gen_fused[0].tolist()[-8:]}")

    print("== 7. op 调用数与 fallback 痕迹 ==")
    router2 = MiniInfraOps(fallback_mode="strict")
    router2.register("cpu", composite_fallback_fused_add_rmsnorm,
                     label="cpu.composite_fallback", is_fallback=True)
    model_fused.model.fused_ops = router2
    with torch.no_grad():
        model_fused(input_ids)
    # op 次数 = layers*2：layer0 只含层内 op（input_layernorm 无前置 add，原生），
    # 其余每层 2 次（跨层边界 + 层内），终局 op 覆盖最后 add + final norm。
    check("op 调用计数 = layers*2", router2.call_counts.get("cpu") ==
          config.num_hidden_layers * 2, str(router2.call_counts))
    check("fallback 计数为 0（cpu 是原生注册项而非降级）", router2.fallback_counts == {})
    model_fused.model.fused_ops = None

    print(f"\n== 冒烟结果：{PASS} PASS / {FAIL} FAIL ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
