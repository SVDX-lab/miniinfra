"""第 04 期显式 Dispatch/Fallback 路由层（模型可切换版的接线盒）。

本期思想的最小可运行化：自定义算子进入模型前必须回答三个问题——
1. **进入条件**：这个算子在哪些 device/dtype/shape 组合上有真实现？
   （registered 表 = 显式白名单）
2. **fallback 如何发生**：没注册的后端是静默滑进别的实现，还是显式报错/
   显式带日志地降级？（三种模式：strict / explicit_fallback / silent 对照）
3. **谁在用**：fallback 计数全程留痕，事后可归因。

三个世界的接线方式：
- CUDA 服务器：torch.ops.mini_infra_ops.fused_add_rmsnorm（TORCH_LIBRARY
  注册，CUDA dispatch key + CPU composite fallback impl 同库注册）；
- MUSA 服务器：torch.ops 同名入口（PrivateUse1 dispatch key；路径不通时
  降级为 musa_extension pybind 函数 + 本层显式路由，差异如实记录）；
- 本地 CPU：composite_fallback_fused_add_rmsnorm（本目录 cpu/ 实现）。

模型文件（qwen3_model_fused.py）只依赖本层的 MiniInfraOps 协议，不直接
import torch.ops——保证三后端共用同一份模型代码。
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from typing import Callable

import torch

# 统一算子签名（协议）：op(input, residual, weight, eps) -> None，双 buffer 原地改写。
OpCallable = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], None]

FALLBACK_MODES = ("strict", "explicit_fallback", "silent")


@dataclass
class Registration:
    device_type: str          # "cpu" / "cuda" / "musa"
    callable: OpCallable      # op(input, residual, weight, eps) -> None
    label: str                # 实现标签（进结果 JSON）
    is_fallback: bool = False  # 显式声明：这是降级实现（而非原生 Kernel）


@dataclass
class MiniInfraOps:
    """显式注册表 + 三模式 fallback 策略。非线程安全细节用锁保护计数器。"""

    fallback_mode: str = "strict"          # strict（默认，显式报错）/ explicit_fallback / silent
    registrations: dict[str, Registration] = field(default_factory=dict)
    fallback_counts: dict[str, int] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(self, device_type: str, function: OpCallable, label: str,
                 is_fallback: bool = False) -> None:
        key = device_type.lower()
        if key in self.registrations:
            raise ValueError(f"device '{key}' 已注册为 "
                             f"'{self.registrations[key].label}'，拒绝覆盖（显式注册表语义）")
        self.registrations[key] = Registration(key, function, label, is_fallback)

    # ------------------------------------------------------------------
    # 解析（dispatch 进入条件的显式化）
    # ------------------------------------------------------------------
    def resolve(self, tensor: torch.Tensor) -> Registration:
        key = tensor.device.type
        registration = self.registrations.get(key)
        if registration is not None:
            return registration

        if self.fallback_mode == "strict":
            # 显式报错路径：列出已注册集合，绝不静默改道。
            raise RuntimeError(
                f"[mini_infra_ops] device '{key}' 没有已注册的 fused_add_rmsnorm 实现"
                f"（已注册：{sorted(self.registrations)}；fallback_mode=strict 拒绝静默回退）。"
                f"显式修复方式：register('{key}', ...) 或 fallback_mode='explicit_fallback'"
            )
        if self.fallback_mode == "explicit_fallback":
            target = self.registrations.get("cpu")
            if target is None or not target.is_fallback:
                raise RuntimeError(
                    "[mini_infra_ops] fallback_mode='explicit_fallback' 要求先显式注册一个 "
                    "is_fallback=True 的 CPU 降级实现；当前 registrations="
                    f"{sorted(self.registrations)}"
                )
            with self._lock:
                self.fallback_counts[key] = self.fallback_counts.get(key, 0) + 1
            print(f"[mini_infra_ops][fallback] device '{key}' 无注册实现 → 显式降级到 "
                  f"CPU composite fallback（第 {self.fallback_counts[key]} 次）",
                  file=sys.stderr, flush=True)
            return target
        if self.fallback_mode == "silent":
            # 对照组：静默滑到 CPU（如有）且不留任何痕迹——这正是要暴露的反模式。
            target = self.registrations.get("cpu")
            if target is None:
                raise RuntimeError("[mini_infra_ops] silent 模式需要已注册 CPU 实现")
            return target
        raise ValueError(f"未知 fallback_mode: {self.fallback_mode}（可选 {FALLBACK_MODES}）")

    # ------------------------------------------------------------------
    # 调用入口（模型文件用）
    # ------------------------------------------------------------------
    def fused_add_rmsnorm(self, input_tensor: torch.Tensor, residual: torch.Tensor,
                          weight: torch.Tensor, eps: float) -> None:
        registration = self.resolve(input_tensor)
        with self._lock:
            self.call_counts[registration.device_type] = \
                self.call_counts.get(registration.device_type, 0) + 1
        registration.callable(input_tensor, residual, weight, eps)

    def summary(self) -> dict:
        return {
            "fallback_mode": self.fallback_mode,
            "registered": {k: {"label": v.label, "is_fallback": v.is_fallback}
                           for k, v in self.registrations.items()},
            "call_counts": dict(self.call_counts),
            "fallback_counts": dict(self.fallback_counts),
        }


# ----------------------------------------------------------------------
# 后端接线工厂
# ----------------------------------------------------------------------
def build_cpu_router() -> MiniInfraOps:
    """本地 CPU 路由：composite fallback 实现（显式注册，is_fallback=True）。"""
    from implementations.cpu.fused_add_rmsnorm_ref import (
        composite_fallback_fused_add_rmsnorm,
    )

    router = MiniInfraOps(fallback_mode="strict")
    router.register("cpu", composite_fallback_fused_add_rmsnorm,
                    label="cpu.composite_fallback", is_fallback=True)
    return router


def build_gpu_router(device_type: str, op_module) -> MiniInfraOps:
    """GPU 服务器路由：torch.ops 注册入口（框架 dispatch 真正路由）。

    op_module = 已加载的注册扩展（torch.ops.mini_infra_ops 或 musa pybind 模块），
    提供 .fused_add_rmsnorm(input, residual, weight, eps)。
    注意：GPU 服务器上 CPU 张量也会被送进 torch.ops 入口——CUDA 库内注册了
    CPU composite impl（显式 fallback），框架 dispatch 会路由到它；这层
    router 仍以显式注册表记录意图，并同样留痕。
    """
    router = MiniInfraOps(fallback_mode="strict")

    def _torch_op(input_tensor, residual, weight, eps):
        op_module.fused_add_rmsnorm(input_tensor, residual, weight, eps)

    for key in (device_type, "cpu"):
        router.register(key, _torch_op, label=f"torch.ops.mini_infra_ops::{key}")
    return router


def build_gpu_router_musa_pybind(device_type: str, pybind_module) -> MiniInfraOps:
    """MUSA 替代路径：musa_extension pybind 函数（无 dispatch key）+ 显式路由。

    与 CUDA 的差异（如实记录进对照表）：PrivateUse1 TORCH_LIBRARY 注册失败时，
    框架级 dispatch 不存在，设备路由由本层 Python 显式完成——"进入框架的
    dispatch"在 MUSA 上退化为"进入显式注册的 Python 路由"。
    """
    router = MiniInfraOps(fallback_mode="strict")

    def _pybind_op(input_tensor, residual, weight, eps):
        pybind_module.fused_add_rmsnorm(input_tensor, residual, weight, eps)

    router.register(device_type, _pybind_op, label=f"musa_ext_pybind::{device_type}")
    return router
