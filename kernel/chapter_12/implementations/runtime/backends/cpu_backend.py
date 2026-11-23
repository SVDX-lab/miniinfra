"""第 12 期 CPU Backend 适配（本地 WSL2 / 任意 torch CPU 环境）。

实现组合（复用声明）：
  - fused op：第 04 期 CPU composite 参考实现（fused_add_rmsnorm_ref.py
    原样数学：residual.add_ + FP32 方差/rsqrt/缩放 + copy_ 回写），
    经第 04 期 MiniInfraOps 显式注册表（strict）路由——注册表语义与
    GPU 后端完全一致，只是 callable 不同；
  - rmsnorm / greedy_decode / 能力查询：基类平台无关实现；
  - 遥测：CPU 无设备 kernel 概念——launch 数恒记 N/A（显式声明为平台
    边界，不是 0），per-op 计时走 torch.profiler CPU 事件 / perf_counter。

平台化部位（本文件内全部）：无——CPU 是 torch 的"原生"后端，无需设备
guard、无流管理、无扩展构建。这一事实本身就是抽象边界的对照证据
（GPU 后端的 platform-forced 部位在 CPU 全部消失）。
"""

from __future__ import annotations

import platform
from typing import Any

import torch

from ..backend import Backend


class CPUBackend(Backend):
    """CPU 后端：composite 表达式实现融合 op（显式注册，非 fallback 语义——
    在 CPU 后端上它就是唯一正确实现）。"""

    name = "cpu"

    def probe_availability(self) -> tuple[bool, str | None]:
        # torch CPU 张量能力是硬前提；torch import 失败时进程早已无法运行。
        try:
            torch.zeros(1)
            return True, None
        except Exception as error:  # noqa: BLE001
            return False, f"torch CPU 张量创建失败: {type(error).__name__}: {error}"

    def resolve_device(self) -> torch.device:
        available, reason = self.probe_availability()
        if not available:
            raise RuntimeError(reason)
        return torch.device("cpu")

    def device_probe(self) -> dict[str, Any]:
        import os

        return {
            "available": True,
            "name": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "os_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "total_memory_bytes": None,   # 统一 schema：CPU 不填显存字段
        }

    def sync_api_name(self) -> str:
        return "none (cpu synchronous)"

    def telemetry_signature(self) -> dict[str, Any]:
        return {
            "profiler_activities": ["CPU"],
            "runtime_launch_event": None,
            "device_kernel_duration": False,
            "launch_count_semantics": "N/A（CPU 无设备 kernel 发射概念；显式平台边界）",
            "event_api": None,
        }

    def setup(self) -> dict[str, Any]:
        """装配 CPU composite fused op（第 04 期参考实现 + strict 注册表）。"""
        from ..fused_add_rmsnorm_ref import composite_fallback_fused_add_rmsnorm
        from ..ops_router import MiniInfraOps

        router = MiniInfraOps(fallback_mode="strict")
        # is_fallback=False：在 CPU 后端上 composite 不是降级路径而是唯一实现
        # （第 04 期 GPU 服务器上同一函数以 is_fallback=True 注册——语义随
        # "它对当前后端是不是原生"变化，这正是显式注册表的价值）。
        router.register("cpu", composite_fallback_fused_add_rmsnorm,
                        label="cpu.composite_expression", is_fallback=False)
        self._router = router
        self._op_evidence = {
            "fused_op_path": "python_registry(cpu.composite_expression, ch05 ref impl)",
            "dispatch_key": "python-level",
            "fused_op_available": True,
            "notes": [
                "CPU 后端无扩展构建环节（对照 GPU 后端的 cpp_extension/musa_extension 平台化部位）",
            ],
        }
        self._setup_done = True
        return dict(self._op_evidence)
