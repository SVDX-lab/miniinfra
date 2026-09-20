#!/usr/bin/env python3
"""CPU-only Chapter 27 checks: AdamW math reference, arm construction."""

from __future__ import annotations

import torch

from fused_optimizer import build_optimizer, reference_adamw_step


def check_reference_math_matches_torch_loop() -> None:
    """The hand-written update must reproduce torch.optim.AdamW (loop)."""

    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(64, 32))
    gradient_source = torch.randn(64, 32)
    learning_rate, beta1, beta2, eps, weight_decay = 2e-5, 0.9, 0.95, 1e-8, 0.1

    optimizer = build_optimizer([parameter], learning_rate, weight_decay, "loop")
    initial = parameter.detach().clone()
    # 走满一步：先反向拿到真实梯度，再 step，然后用手写公式从同一初值重放。
    (parameter * gradient_source).sum().backward()
    optimizer.step()
    torch_result = parameter.detach().clone()

    # AdamW 的权重衰减是解耦的，在优化器内部应用；损失梯度只含 g 本身。
    # 参照函数接收“上一步状态”，首步重放应传零状态。
    hand = reference_adamw_step(
        initial, gradient_source,
        torch.zeros_like(initial), torch.zeros_like(initial), 1,
        learning_rate, beta1, beta2, eps, weight_decay,
    )
    error = (torch_result - hand).abs().max()
    # (1 - lr*wd)*p 与 p - lr*wd*p 在 fp32 中相差 1 ulp 量级，1e-6 是
    # 排除公式错误后仍能看到的真实舍入底噪。
    assert float(error) < 1e-6, f"reference math diverges: {float(error)}"


def check_arms_build_with_expected_flags() -> None:
    model = torch.nn.Linear(8, 8)
    loop = build_optimizer(model, 2e-5, 0.1, "loop")
    foreach = build_optimizer(model, 2e-5, 0.1, "foreach")
    assert loop.defaults["foreach"] is False and loop.defaults["fused"] is False
    assert foreach.defaults["foreach"] is True and foreach.defaults["fused"] is False
    for arm in ("loop", "foreach"):
        optimizer = build_optimizer(model, 2e-5, 0.1, arm)
        assert optimizer.defaults["betas"] == (0.9, 0.95)
        assert optimizer.defaults["eps"] == 1e-8
        del optimizer
    try:
        build_optimizer(model, 2e-5, 0.1, "unknown")
    except ValueError:
        return
    raise AssertionError("unknown arm must raise")


def check_loop_and_foreach_agree_on_cpu() -> None:
    """foreach batches the same elementwise ops; results should match loop."""

    torch.manual_seed(1)
    model_a = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16))
    model_b = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16))
    model_b.load_state_dict(model_a.state_dict())
    batch = torch.randn(8, 16)
    target = torch.randn(8, 16)
    loss_fn = lambda output: ((output - target) ** 2).mean()

    optimizer_a = build_optimizer(model_a, 1e-3, 0.1, "loop")
    optimizer_b = build_optimizer(model_b, 1e-3, 0.1, "foreach")
    for _ in range(4):
        optimizer_a.zero_grad(set_to_none=True)
        optimizer_b.zero_grad(set_to_none=True)
        loss_a = loss_fn(model_a(batch))
        loss_b = loss_fn(model_b(batch))
        assert loss_a.item() == loss_b.item()
        loss_a.backward()
        loss_b.backward()
        optimizer_a.step()
        optimizer_b.step()
    for (name, parameter_a), (_other, parameter_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert torch.allclose(parameter_a, parameter_b, rtol=0, atol=0), f"foreach diverged at {name}"


def main() -> None:
    check_reference_math_matches_torch_loop()
    check_arms_build_with_expected_flags()
    check_loop_and_foreach_agree_on_cpu()
    print("chapter_27 smoke tests passed")


if __name__ == "__main__":
    main()
