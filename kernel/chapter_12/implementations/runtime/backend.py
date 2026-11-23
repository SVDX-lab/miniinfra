"""第 12 期统一 Backend 抽象（大纲行 844–846 的可运行化）。

思想：把三个世界的实现收进一个入口——能力查询、显式设备选择、算子
Dispatch、错误报告；**程序只选择已安装且通过能力检查的后端，不做静默
CPU fallback**（接第 04 期 ops_router 的 explicit 语义：能力不足 = 显式
报错，绝不改道）。

抽象的四根支柱（本文件逐一给出统一接口，三份适配各自实现）：
  1. 能力查询（query_capabilities）：dtype 支持 / 算子可用性 / 设备探测，
     返回可序列化的 CapabilityReport——"这个后端能做什么"是数据不是猜测；
  2. 显式设备选择（resolve_device / ensure_available）：后端不可用即
     BackendNotAvailableError，绝不滑进 CPU；
  3. 算子 Dispatch（rmsnorm / fused_add_rmsnorm / greedy_decode）：
     同一签名三后端实现——CPU composite 表达式；CUDA/MUSA 第 04 期
     torch.ops.mini_infra_ops 注册融合 op（同一份 Python 前端）；
  4. 错误报告（CapabilityError 家族 + 错误文本留档）：一切能力缺失都
     显式抛错，错误原文进结果 JSON（平台边界证据）。

复用声明（大纲"复用第 01–10 期已验证实现，不发明新 Kernel"）：
  - 融合 op：第 04 期 TORCH_LIBRARY 注册扩展（CUDA key / PrivateUse1 key）
    与 CPU composite fallback 参考实现（fused_add_rmsnorm_ref.py 原样）；
  - 显式注册表语义：第 04 期 ops_router.MiniInfraOps（strict 模式）作为
    fused op 的路由载体（call_counts 全程留痕）；
  - 模型与 Greedy 解码：第 04 期 qwen3_model_fused.py 原样（native/fused
    可切换），第 01 期 tokenizer 原样。

抽象边界（ EXPERIMENT_LOG §抽象边界清单的代码级依据）：
  - 平台无关（本文件 + qwen3_runtime 的公共路径）：能力协议数据结构、
    dispatch 签名、Greedy 解码循环、合同校验语义；
  - 被迫平台化（三个 backend 子文件）：设备探测 API（torch.cuda /
    torch.musa）、同步原语、扩展构建与加载（cpp_extension /
    musa_extension）、dispatch key 名、事件 API（torch.cuda.Event /
    torch.musa.Event）、遥测 activity 名。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any

import torch

# 能力矩阵探测的 dtype 全集（第 05 期口径：fp32/fp16/bf16）。
PROBE_DTYPES: tuple[str, ...] = ("float32", "float16", "bfloat16")
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# 融合 op 的合同 dtype（第 04 期合同原文：dtype ∈ {float32, float16}；
# bfloat16 属第 05 期主题——对 fused op 是"显式报错"而非"静默回退"）。
FUSED_OP_CONTRACT_DTYPES = ("float32", "float16")

PROBE_SHAPE = (8, 1024)          # 能力探测统一 shape（rows, H）
PROBE_EPS = 1e-6


# ---------------------------------------------------------------------------
# 错误报告家族：能力缺失的显式异常（绝不静默改道）
# ---------------------------------------------------------------------------
class CapabilityError(RuntimeError):
    """能力缺失类错误的基类（错误文本即平台边界证据，进结果 JSON）。"""


class BackendNotAvailableError(CapabilityError):
    """后端在当前进程/机器不可用（如无 GPU、缺 torch_musa）。

    正确用法是程序捕获后选择下一个已安装后端——而不是被静默路由到 CPU。
    """


class CapabilityMissingError(CapabilityError):
    """后端存在但请求的能力缺失（dtype / 算子 / 布局不在合同内）。"""


class ExplicitFallbackError(CapabilityError):
    """strict 模式下触发了未注册路径（第 04 期 ops_router 语义的透传）。"""


# ---------------------------------------------------------------------------
# 能力查询的数据结构（可 JSON 序列化——能力是数据，不是隐式行为）
# ---------------------------------------------------------------------------
@dataclass
class DtypeCapability:
    """单个 dtype 在本后端的四项探测结果。"""

    dtype: str
    tensor_create: bool = False
    tensor_create_error: str | None = None
    linear_ok: bool | None = None            # 小 matmul 可用且数值有限
    linear_error: str | None = None
    composite_rmsnorm_ok: bool | None = None  # 组合表达式 RMSNorm 可用
    composite_rmsnorm_error: str | None = None
    fused_op_ok: bool | None = None           # 融合 op 接受该 dtype（合同 {fp32,fp16}）
    fused_op_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityReport:
    """后端能力快照（results JSON 直接消费）。"""

    backend: str
    device_type: str
    available: bool
    available_error: str | None = None
    device_name: str | None = None
    device_probe: dict[str, Any] = field(default_factory=dict)
    torch_version: str | None = None
    dtype_capabilities: dict[str, DtypeCapability] = field(default_factory=dict)
    fused_op_available: bool = False
    fused_op_path: str | None = None      # 注册路径证据（registration_info 原文等）
    dispatch_key: str | None = None       # python-level / CUDA / PrivateUse1
    sync_api: str = "none (cpu synchronous)"
    telemetry: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dtype_capabilities"] = {
            key: value.to_dict() for key, value in self.dtype_capabilities.items()
        }
        return payload


def _truncate_error(error: BaseException, limit: int = 240) -> str:
    return f"{type(error).__name__}: {str(error)}"[:limit]


# ---------------------------------------------------------------------------
# Backend 抽象基类
# ---------------------------------------------------------------------------
class Backend(ABC):
    """统一后端接口。构造必须零副作用（无 GPU 的机器上也能构造 + 探测）；
    资源加载/扩展构建集中在 setup()，显式调用。"""

    name: str = "abstract"

    def __init__(self, device_index: int | None = None):
        self.device_index = device_index
        self._router = None            # MiniInfraOps（setup 后就绪）
        self._op_evidence: dict[str, Any] = {}
        self._setup_done = False

    # ------------------------------------------------------------------
    # 1) 可用性与显式设备选择
    # ------------------------------------------------------------------
    @abstractmethod
    def probe_availability(self) -> tuple[bool, str | None]:
        """静态探测（不分配显存/不建扩展）：(available, 原因)。"""

    @abstractmethod
    def resolve_device(self) -> torch.device:
        """显式设备选择。不可用时抛 BackendNotAvailableError（不静默回退）。"""

    def ensure_available(self) -> torch.device:
        """探测 + 设备解析一步走；不可用即显式报错（含已安装后端提示）。"""
        available, reason = self.probe_availability()
        if not available:
            raise BackendNotAvailableError(
                f"[backend:{self.name}] 不可用：{reason}。"
                f"显式修复：选择已安装后端（可用性用 create_backend+probe 逐个查询），"
                f"本 runtime 不做静默 CPU fallback。"
            )
        return self.resolve_device()

    # ------------------------------------------------------------------
    # 2) 能力查询
    # ------------------------------------------------------------------
    @abstractmethod
    def device_probe(self) -> dict[str, Any]:
        """设备静态信息（名称/显存/计算单元/驱动），不可用返回 {'available': False}。"""

    @abstractmethod
    def telemetry_signature(self) -> dict[str, Any]:
        """遥测能力签名：profiler activity 名、launch 运行时事件名、
        设备侧 kernel 时长是否可观察（MUSA = False，M-04/M-30 口径）。"""

    def query_capabilities(self, deep: bool = True) -> CapabilityReport:
        """能力矩阵：静态探测 + 逐 dtype 动态探测（deep=True 时分配真实张量）。"""
        available, reason = self.probe_availability()
        report = CapabilityReport(
            backend=self.name,
            device_type=self.name,
            available=available,
            available_error=reason,
            torch_version=torch.__version__,
            sync_api=self.sync_api_name(),
            telemetry=self.telemetry_signature(),
        )
        if available:
            report.device_probe = self.device_probe()
            report.device_name = report.device_probe.get("name")
        if self._op_evidence:
            report.fused_op_path = self._op_evidence.get("fused_op_path")
            report.dispatch_key = self._op_evidence.get("dispatch_key")
            report.fused_op_available = self._op_evidence.get("fused_op_available", False)
            report.notes.extend(self._op_evidence.get("notes", []))

        if deep and available:
            device = self.resolve_device()
            for dtype_name in PROBE_DTYPES:
                report.dtype_capabilities[dtype_name] = self._probe_dtype(
                    device, DTYPE_MAP[dtype_name], dtype_name
                )
            fp32 = report.dtype_capabilities.get("float32")
            report.fused_op_available = bool(
                self._op_evidence.get("fused_op_available", False)
            ) and bool(fp32 and fp32.fused_op_ok)
        return report

    def _probe_dtype(self, device: torch.device, dtype: torch.dtype, name: str) -> DtypeCapability:
        """单 dtype 四项探测：建张量 → 线性层 → 组合 RMSNorm → 融合 op。"""
        capability = DtypeCapability(dtype=name)
        try:
            x = torch.zeros(PROBE_SHAPE, device=device, dtype=dtype)
            weight = torch.eye(PROBE_SHAPE[1], device=device, dtype=dtype)
            capability.tensor_create = True
        except Exception as error:  # noqa: BLE001
            capability.tensor_create_error = _truncate_error(error)
            return capability

        # 小 matmul（线性层代表 GEMM 路径；只判可执行 + 有限，不比数值）。
        try:
            result = x @ weight
            if bool(torch.isnan(result).any()) or bool(torch.isinf(result).any()):
                raise CapabilityMissingError("matmul 输出含 NaN/Inf")
            capability.linear_ok = True
        except CapabilityMissingError as error:
            capability.linear_ok = False
            capability.linear_error = _truncate_error(error)
        except Exception as error:  # noqa: BLE001
            capability.linear_ok = False
            capability.linear_error = _truncate_error(error)

        # 组合表达式 RMSNorm（平台无关路径的代表）。
        try:
            norm_weight = torch.ones(PROBE_SHAPE[1], device=device, dtype=dtype)
            float_states = x.float()
            variance = float_states.pow(2).mean(dim=-1, keepdim=True)
            normalized = float_states * torch.rsqrt(variance + PROBE_EPS)
            scaled = (normalized.to(dtype) * norm_weight).sum()
            if bool(torch.isnan(scaled)) or bool(torch.isinf(scaled)):
                raise CapabilityMissingError("composite RMSNorm 输出含 NaN/Inf")
            capability.composite_rmsnorm_ok = True
        except CapabilityMissingError as error:
            capability.composite_rmsnorm_ok = False
            capability.composite_rmsnorm_error = _truncate_error(error)
        except Exception as error:  # noqa: BLE001
            capability.composite_rmsnorm_ok = False
            capability.composite_rmsnorm_error = _truncate_error(error)

        # 融合 op（合同 dtype {fp32, fp16}；bf16 预期显式报错并留档）。
        if not self._setup_done:
            capability.fused_op_ok = None
            capability.fused_op_error = "setup() 未调用——融合 op 可用性需显式装配后探测"
            return capability
        try:
            residual = torch.ones(PROBE_SHAPE, device=device, dtype=dtype)
            w = torch.ones(PROBE_SHAPE[1], device=device, dtype=dtype)
            self.fused_add_rmsnorm(x, residual, w, PROBE_EPS)
            capability.fused_op_ok = True
        except Exception as error:  # noqa: BLE001
            capability.fused_op_ok = False
            capability.fused_op_error = _truncate_error(error)
        return capability

    # ------------------------------------------------------------------
    # 3) setup（扩展构建/加载；CPU 为无操作）与 op Dispatch
    # ------------------------------------------------------------------
    @abstractmethod
    def setup(self) -> dict[str, Any]:
        """加载融合 op 实现（GPU：构建/加载第 04 期注册扩展；CPU：composite）。
        返回 evidence dict（注册路径证据，进结果 JSON）。"""

    def _require_setup(self) -> None:
        if not self._setup_done:
            raise CapabilityError(
                f"[backend:{self.name}] setup() 未调用——op dispatch 前必须显式完成装配"
                f"（不做隐式初始化）"
            )

    def fused_ops_protocol(self):
        """返回模型可用的 fused_ops 协议对象（第 04 期 MiniInfraOps，strict）。"""
        self._require_setup()
        return self._router

    @abstractmethod
    def sync_api_name(self) -> str:
        """同步原语名（能力表一列）。"""

    def sync(self) -> None:
        """后端同步（遥测与计时边界）。CPU 为空操作。"""

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """RMSNorm dispatch：三后端同一签名，组合表达式实现（平台无关路径）。

        第 01 期手写 RMSNorm 模块的前向数学；作为对照，融合 op 只覆盖
        "相邻 Add+Norm 对"，独立 RMSNorm（q/k norm、layer 0 input norm）走本路径。
        """
        input_dtype = x.dtype
        float_states = x.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + eps)
        return weight * normalized.to(input_dtype)

    def fused_add_rmsnorm(self, input_tensor: torch.Tensor, residual: torch.Tensor,
                          weight: torch.Tensor, eps: float) -> None:
        """融合 op dispatch（strict：能力缺失显式报错，绝不静默回退）。"""
        self._require_setup()
        expected = self.resolve_device()
        if input_tensor.device.type != expected.type:
            raise CapabilityMissingError(
                f"[backend:{self.name}] fused_add_rmsnorm 收到 device="
                f"{input_tensor.device.type} 张量，本后端只接 {expected.type}。"
                f"strict 模式拒绝静默改道（第 04 期 explicit 语义）。"
            )
        if input_tensor.dtype not in (torch.float32, torch.float16):
            raise CapabilityMissingError(
                f"[backend:{self.name}] fused op 合同 dtype ∈ {FUSED_OP_CONTRACT_DTYPES}，"
                f"实际 {input_tensor.dtype}（显式报错，不静默回退）"
            )
        self._router.fused_add_rmsnorm(input_tensor, residual, weight, eps)

    # ------------------------------------------------------------------
    # 4) Greedy 解码（平台无关循环——三后端共用同一份实现，抽象边界的
    #    "无关侧"代表；边界只通过 backend.sync() 露出）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(self, model, input_ids: torch.Tensor, max_new_tokens: int,
                      eos_token_id: int | None = None) -> dict[str, Any]:
        """统一 Greedy 解码：每步重算整段序列（第 01 期无优化边界）。

        返回 tokens/finished + per-step 计时（同步边界由 backend.sync 统一）。
        """
        import time

        generated = input_ids
        step_times_ms: list[float] = []
        finished = False
        for _ in range(max_new_tokens):
            self.sync()
            start = time.perf_counter()
            logits = model(generated)
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            self.sync()
            step_times_ms.append((time.perf_counter() - start) * 1000.0)
            generated = torch.cat((generated, next_token_id), dim=1)
            if eos_token_id is not None and bool((next_token_id == eos_token_id).all()):
                finished = True
                break
        return {
            "generated_ids": generated[0].tolist(),
            "finished_by_eos": finished,
            "step_times_ms": [round(value, 2) for value in step_times_ms],
            "steps": len(step_times_ms),
            "mean_step_ms": round(sum(step_times_ms) / len(step_times_ms), 2)
            if step_times_ms else None,
        }

    # ------------------------------------------------------------------
    # 5) 遥测
    # ------------------------------------------------------------------
    def count_kernel_launches(self, operation, iterations: int = 1) -> dict[str, Any]:
        """torch profiler 统计 operation 的设备发射数（第 03/04 期已验证实现）。"""
        from experiment_utils import count_kernel_launches as _count

        return _count(operation, iterations, self.resolve_device())


# ---------------------------------------------------------------------------
# 工厂：显式后端选择入口（不可用后端也能构造——探测与报错是行为的一部分）
# ---------------------------------------------------------------------------
_BACKEND_CLASSES: dict[str, str] = {
    "cpu": "backends.cpu_backend:CPUBackend",
    "cuda": "backends.cuda_backend:CUDABackend",
    "musa": "backends.musa_backend:MUSABackend",
}


def list_backends() -> list[str]:
    """已知后端注册表（显式列表；不探测可用性）。"""
    return sorted(_BACKEND_CLASSES)


def create_backend(name: str, device_index: int | None = None) -> Backend:
    """按名构造后端对象（零副作用）。未知名字显式报错。

    可用性判断推迟到 probe_availability()/ensure_available()——构造失败与
    能力缺失是两类错误，分开报告。
    """
    key = name.lower()
    if key not in _BACKEND_CLASSES:
        raise ValueError(
            f"未知后端 '{name}'（已注册：{list_backends()}）；"
            f"runtime 不做最近似后端的静默猜测"
        )
    module_path, class_name = _BACKEND_CLASSES[key].split(":")
    import importlib

    if __package__:
        module = importlib.import_module(f"{__package__}.{module_path}")
    else:
        # 顶层 import 模式（runtime 目录直接在 sys.path）：退回顶层模块名。
        import sys as _sys
        from pathlib import Path as _Path

        here = str(_Path(__file__).resolve().parent)
        if here not in _sys.path:
            _sys.path.insert(0, here)
        module = importlib.import_module(module_path)
    return getattr(module, class_name)(device_index=device_index)
