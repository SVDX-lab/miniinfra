"""第 12 期 CUDA Backend 适配（RTX 4090 列；当前 4090 故障抢修中——接口就绪）。

实现组合（复用声明）：
  - fused op：第 04 期 TORCH_LIBRARY 注册扩展 mini_infra_ops（CUDA dispatch
    key + CPU composite fallback impl 同库），经 implementations/cuda/build.py
    构建（torch.utils.cpp_extension.load，nvcc）；本目录源文件与第 04 期逐字
    相同（kernel 零改动，协议"不发明新 Kernel"）；
  - 模型侧经 MiniInfraOps（strict）注册表路由，签名与 CPU/MUSA 后端一致。

平台化部位（与 musa_backend 逐项对照——抽象边界清单素材）：
  - 设备探测：torch.cuda.is_available / get_device_properties；
  - 同步：torch.cuda.synchronize；
  - 事件计时：torch.cuda.Event；
  - 扩展构建：torch.utils.cpp_extension.load（nvcc，-gencode sm_89）；
  - dispatch key：CUDA；遥测 activity：ProfilerActivity.CUDA；
    launch 运行时事件：cudaLaunchKernel；设备侧 kernel 时长：可观察。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from ..backend import Backend

# implementations/cuda（第 04 期构建器，逐字复用）。
GPU_IMPL_DIR = Path(__file__).resolve().parent.parent.parent / "cuda"


class CUDABackend(Backend):
    name = "cuda"

    def probe_availability(self) -> tuple[bool, str | None]:
        if not hasattr(torch, "cuda"):
            return False, "torch 无 cuda 模块"
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() 为 False（无 GPU 或驱动不可用）"
        if torch.cuda.device_count() < 1:
            return False, "torch.cuda.device_count() == 0"
        return True, None

    def resolve_device(self) -> torch.device:
        # 注意：不得调用 ensure_available()（基类 ensure_available 在探测后
        # 会再调 resolve_device——互调即无穷递归，KI-13-06）。直接探测一次。
        available, reason = self.probe_availability()
        if not available:
            from ..backend import BackendNotAvailableError

            raise BackendNotAvailableError(
                f"[backend:cuda] 不可用：{reason}（显式报错，不静默回退）"
            )
        index = 0 if self.device_index is None else self.device_index
        return torch.device("cuda", index)

    def device_probe(self) -> dict[str, Any]:
        available, reason = self.probe_availability()
        if not available:
            return {"available": False, "reason": reason}
        properties = torch.cuda.get_device_properties(
            torch.device("cuda", 0 if self.device_index is None else self.device_index)
        )
        return {
            "available": True,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
            "driver_cuda_version": self._safe_command(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
        }

    def sync_api_name(self) -> str:
        return "torch.cuda.synchronize"

    def telemetry_signature(self) -> dict[str, Any]:
        return {
            "profiler_activities": ["CPU", "CUDA"],
            "runtime_launch_event": "cudaLaunchKernel",
            "device_kernel_duration": True,
            "launch_count_semantics": "kernel 事件 + cudaLaunchKernel 双口径",
            "event_api": "torch.cuda.Event(enable_timing=True)",
        }

    def setup(self) -> dict[str, Any]:
        """构建/加载第 04 期 TORCH_LIBRARY 注册扩展（CUDA impl 为主）。"""
        self.ensure_available()
        if str(GPU_IMPL_DIR) not in sys.path:
            sys.path.insert(0, str(GPU_IMPL_DIR))
        import build as cuda_build  # noqa: E402  (implementations/cuda/build.py)

        pybind_module, namespace = cuda_build.get_ops()
        from ..ops_router import MiniInfraOps

        router = MiniInfraOps(fallback_mode="strict")

        def _torch_op(input_tensor, residual, weight, eps):
            namespace.fused_add_rmsnorm(input_tensor, residual, weight, eps)

        # strict 注册表：只注册 cuda 设备入口。CPU 张量进入本后端会在
        # Backend.fused_add_rmsnorm 的设备检查处显式报错——即使扩展库内
        # 注册了框架级 CPU composite impl，runtime 层也不静默改道。
        router.register("cuda", _torch_op, label="torch.ops.mini_infra_ops::cuda",
                        is_fallback=False)
        self._router = router
        self._op_evidence = {
            "fused_op_path": pybind_module.registration_info(),
            "dispatch_key": "CUDA",
            "fused_op_available": True,
            "notes": [
                "扩展构建：torch.utils.cpp_extension.load（nvcc）——构建系统为平台化部位",
                "no_cpu_fallback 对照族同库注册（能力缺失显式报错的框架级证据源）",
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
