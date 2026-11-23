"""第 04 期后端接线（build.py 的消费入口 + ops_router 组装）。

三个世界的接线方式（scripts 只调用 setup_backend，不感知构建细节）：
  cpu  ：纯 Python composite fallback + 显式注册表（本地可跑）；
  cuda ：implementations/cuda/build.py（cpp_extension + TORCH_LIBRARY 注册，
         CUDA impl + CPU composite fallback impl 同库）；
  musa ：implementations/musa/build.py（torch_musa 扩展注册路径验证，
         PrivateUse1 impl + CPU fallback impl；路径不通时 pybind 显式路由）。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTER_ROOT = HERE.parent
for _path in (str(CHAPTER_ROOT), str(HERE / "cpu")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from ops_router import MiniInfraOps, build_cpu_router  # noqa: E402


def setup_backend(backend: str) -> dict:
    """装配后端，返回 {router, op_module, evidence}。

    evidence 携带注册路径证据（registration_info 原文、构建变体、
    dispatch key 名等），由调用方写进结果 JSON。
    """
    backend = backend.lower()
    if backend == "cpu":
        router = build_cpu_router()
        return {"router": router, "op_module": None,
                "evidence": {"registration_path": "python_registry(composite_fallback)",
                             "dispatch_key": "python-level"}}

    if backend == "cuda":
        sys.path.insert(0, str(HERE / "cuda"))
        import build as cuda_build  # noqa: E402

        pybind_module, namespace = cuda_build.get_ops()
        cuda_build.build_ctypes_lib()
        router = MiniInfraOps(fallback_mode="strict")

        def _op(input_tensor, residual, weight, eps):
            namespace.fused_add_rmsnorm(input_tensor, residual, weight, eps)

        for key in ("cuda", "cpu"):
            router.register(key, _op, label=f"torch.ops.mini_infra_ops({key})")
        return {"router": router, "op_module": namespace,
                "pybind_module": pybind_module,
                "evidence": {"registration_path": "TORCH_LIBRARY+cpp_extension",
                             "dispatch_key": "CUDA",
                             "registration_info": pybind_module.registration_info()}}

    if backend == "musa":
        sys.path.insert(0, str(HERE / "musa"))
        import build as musa_build  # noqa: E402

        pybind_module, namespace, variant = musa_build.get_ops()
        musa_build.build_ctypes_lib()
        router = MiniInfraOps(fallback_mode="strict")

        def _op(input_tensor, residual, weight, eps):
            namespace.fused_add_rmsnorm(input_tensor, residual, weight, eps)

        for key in ("musa", "cpu"):
            router.register(key, _op, label=f"torch.ops.mini_infra_ops({key})")
        return {"router": router, "op_module": namespace,
                "pybind_module": pybind_module,
                "evidence": {"registration_path": "TORCH_LIBRARY+musa_extension",
                             "dispatch_key": "PrivateUse1",
                             "stream_variant": variant,
                             "registration_info": pybind_module.registration_info()}}

    raise ValueError(f"未知后端: {backend}（可选 cpu/cuda/musa）")
