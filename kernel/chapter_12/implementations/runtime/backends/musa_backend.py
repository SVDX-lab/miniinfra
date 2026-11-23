"""第 12 期 MUSA Backend 适配（MTT S4000 列；torch_musa PrivateUse1）。

实现组合（复用声明）：
  - fused op：第 04 期 MUSA 注册 TU（TORCH_LIBRARY(mini_infra_ops) +
    PrivateUse1 impl + CPU composite fallback impl），经
    implementations/musa/build.py 构建（torch_musa.utils.musa_extension.load，
    mcc；.mu 后缀、单编译单元、默认流变体——M-07/08/09/10/28 已封案路径）；
    本目录源文件与第 04 期逐字相同（kernel 零改动）；
  - 模型侧经 MiniInfraOps（strict）注册表路由，签名与 CPU/CUDA 后端一致。

平台化部位（与 cuda_backend 逐项对照——抽象边界清单素材）：
  - 设备探测：torch_musa import 本身就是平台分支（CUDA 无此步）；
  - 同步：torch.musa.synchronize；
  - 事件计时：torch.musa.Event；
  - 扩展构建：torch_musa.utils.musa_extension.load（mcc，.mu 复制、
    单编译单元、流变体宏）——比 CUDA 侧多三道机械适配；
  - dispatch key：PrivateUse1（名字不同、机制同构，M-27 正向封案）；
  - 遥测 activity：ProfilerActivity.PrivateUse1（若存在）；
    launch 运行时事件：musaLaunchKernel（CPU 侧事件）；
    设备侧 kernel 时长：不可观察（全 0，M-04/M-30 平台边界）——
    这是"遥测可观察性差异"的运行时层体现。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from ..backend import Backend

GPU_IMPL_DIR = Path(__file__).resolve().parent.parent.parent / "musa"


def _import_torch_musa():
    """torch_musa import 探测（MUSA 特有的平台分支：插件即平台）。"""
    try:
        import torch_musa  # noqa: F401
        return True, None
    except ImportError as error:
        return False, f"torch_musa import 失败: {error}"


class MUSABackend(Backend):
    name = "musa"

    def probe_availability(self) -> tuple[bool, str | None]:
        ok, reason = _import_torch_musa()
        if not ok:
            return False, reason
        if not hasattr(torch, "musa"):
            return False, "import torch_musa 后 torch.musa 仍不存在"
        if not torch.musa.is_available():
            return False, "torch.musa.is_available() 为 False（无 MUSA 设备或驱动不可用）"
        return True, None

    def resolve_device(self) -> torch.device:
        # 注意：不得调用 ensure_available()（基类 ensure_available 在探测后
        # 会再调 resolve_device——互调即无穷递归，KI-13-06）。直接探测一次。
        available, reason = self.probe_availability()
        if not available:
            from ..backend import BackendNotAvailableError

            raise BackendNotAvailableError(
                f"[backend:musa] 不可用：{reason}（显式报错，不静默回退）"
            )
        index = 0 if self.device_index is None else self.device_index
        return torch.device("musa", index)

    def device_probe(self) -> dict[str, Any]:
        available, reason = self.probe_availability()
        if not available:
            return {"available": False, "reason": reason}
        properties = torch.musa.get_device_properties(
            torch.device("musa", 0 if self.device_index is None else self.device_index)
        )
        return {
            "available": True,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
            "driver": self._safe_command(["mthreads-gmi"]),
        }

    def sync_api_name(self) -> str:
        return "torch.musa.synchronize"

    def telemetry_signature(self) -> dict[str, Any]:
        from torch.profiler import ProfilerActivity

        activity = "PrivateUse1" if hasattr(ProfilerActivity, "PrivateUse1") else None
        return {
            "profiler_activities": ["CPU"] + ([activity] if activity else []),
            "runtime_launch_event": "musaLaunchKernel",
            "device_kernel_duration": False,   # M-04/M-30：设备时长全 0（运行时层体现）
            "launch_count_semantics": "仅 CPU 侧 musaLaunchKernel 事件（计数口径成立，时长口径缺位）",
            "event_api": "torch.musa.Event(enable_timing=True)",
        }

    def setup(self) -> dict[str, Any]:
        """构建/加载第 04 期 MUSA 注册扩展（PrivateUse1 impl）。"""
        self.ensure_available()
        if str(GPU_IMPL_DIR) not in sys.path:
            sys.path.insert(0, str(GPU_IMPL_DIR))
        import build as musa_build  # noqa: E402  (implementations/musa/build.py)

        pybind_module, namespace, variant = musa_build.get_ops()
        from ..ops_router import MiniInfraOps

        router = MiniInfraOps(fallback_mode="strict")

        def _torch_op(input_tensor, residual, weight, eps):
            namespace.fused_add_rmsnorm(input_tensor, residual, weight, eps)

        # strict 注册表：只注册 musa 设备入口（CPU 张量显式报错，不静默改道；
        # 对照实验：直接调 torch.ops 时框架会路由到库内 CPU composite impl——
        # capability_matrix.py 采集这条框架级行为差异）。
        router.register("musa", _torch_op, label="torch.ops.mini_infra_ops::musa",
                        is_fallback=False)
        self._router = router
        self._op_evidence = {
            "fused_op_path": pybind_module.registration_info(),
            "dispatch_key": "PrivateUse1",
            "stream_variant": variant,
            "fused_op_available": True,
            "notes": [
                "扩展构建：torch_musa.utils.musa_extension.load（mcc，.mu 复制+单编译单元+流变体宏）"
                "——比 CUDA 侧多三道机械适配（M-07/08/09）",
                "c10/musa 流头不可与官方头同译（M-10/M-28）——默认流变体为可用路径",
            ],
        }
        self._setup_done = True
        return dict(self._op_evidence)

    @staticmethod
    def _safe_command(arguments: list[str]) -> str | None:
        import subprocess

        try:
            result = subprocess.run(arguments, check=False, capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
