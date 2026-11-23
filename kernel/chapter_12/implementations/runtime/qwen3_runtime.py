"""第 12 期统一推理入口：load → backend 选择 → generate → 输出 + 遥测。

同一份 Python 前端跑三后端（大纲行 846"同一份 Python 前端 + 多份平台实现"
的直接实现）：本文件不含任何设备分支——设备差异全部收敛在 Backend 适配层
（backends/*.py），入口层只消费 Backend 协议（能力查询 / sync / op dispatch）。

遥测（大纲实验 4）：
  - per-op 计时：一个代表前向步的 torch.profiler op 表（top N CPU 时间）+
    整段 decode 的 per-step wall time（backend.sync 统一同步边界）；
  - kernel launch 数：count_kernel_launches（第 03/04 期已验证实现），
    CPU = N/A（平台边界）、CUDA = kernel 事件 + cudaLaunchKernel 双口径、
    MUSA = 仅 musaLaunchKernel CPU 侧事件且设备时长全 0（M-04/M-30 的
    运行时层体现）。
"""

from __future__ import annotations

import site
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .backend import Backend, create_backend

# 第 01 期真值锚点（跨期共享，只读）。
CHAPTER_ROOT = Path(__file__).resolve().parent.parent.parent
TRUTH_MANIFEST = CHAPTER_ROOT.parent / "chapter_01" / "truth" / "truth-manifest.json"
TRUTH_PROBES = CHAPTER_ROOT.parent / "chapter_01" / "truth" / "probes.npz"
DEFAULT_MODEL_DIR = CHAPTER_ROOT.parent / "chapter_01" / "models" / "qwen3-0.6b"

# 模型 dtype 约定（与第 01/04 期口径一致）：CPU 列 FP32（真值锚点）、
# GPU 列 FP16（第 01 期 GPU 列口径）。
DEFAULT_DTYPE_BY_BACKEND = {"cpu": torch.float32, "cuda": torch.float16, "musa": torch.float16}


def bootstrap_compat() -> dict[str, str]:
    """本地兼容层注入（幂等）：venv 缺 safetensors → 本地 shim；缺 regex →
    用户 site-packages（第 26 期已验证方案）。返回注入记录（进结果 JSON）。"""
    applied: dict[str, str] = {}
    here = Path(__file__).resolve().parent
    try:
        import safetensors  # noqa: F401
    except ModuleNotFoundError:
        compat = str(here / "safetensors_compat")
        if compat not in sys.path:
            sys.path.insert(0, compat)
        applied["safetensors"] = f"local shim: {compat}"
    try:
        import regex  # noqa: F401
    except ModuleNotFoundError:
        candidate = site.getusersitepackages()
        if Path(candidate).exists() and candidate not in sys.path:
            sys.path.append(candidate)
            applied["regex"] = f"user site-packages: {candidate}"
    return applied


@dataclass
class GenerateResult:
    """generate() 的结构化输出：token 与遥测分离（对照实验读数互不污染）。"""

    generated_ids: list[int]
    prompt_token_ids: list[int]
    finished_by_eos: bool
    backend: str
    dtype: str
    fused_mode: str                     # native / fused
    total_wall_ms: float
    step_times_ms: list[float] = field(default_factory=list)
    op_call_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "dtype": self.dtype, "fused_mode": self.fused_mode,
            "prompt_token_ids": self.prompt_token_ids,
            "generated_ids": self.generated_ids,
            "generated_count": len(self.generated_ids) - len(self.prompt_token_ids),
            "finished_by_eos": self.finished_by_eos,
            "total_wall_ms": round(self.total_wall_ms, 1),
            "steps": len(self.step_times_ms),
            "step_times_ms": self.step_times_ms,
            "mean_step_ms": round(sum(self.step_times_ms) / len(self.step_times_ms), 1)
            if self.step_times_ms else None,
            "op_call_counts": self.op_call_counts,
        }


class Qwen3Runtime:
    """统一入口：一个类，三个世界。"""

    def __init__(self, backend: Backend, model_dir: str | Path,
                 dtype: torch.dtype | None = None, fused: bool = True):
        self.backend = backend
        self.model_dir = Path(model_dir)
        self.fused = fused
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.load_evidence: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 构造管线（显式：不可用即抛错，绝不滑进 CPU）
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, backend_name: str, model_dir: str | Path = DEFAULT_MODEL_DIR,
               dtype: torch.dtype | None = None, fused: bool = True,
               device_index: int | None = None) -> "Qwen3Runtime":
        backend = create_backend(backend_name, device_index=device_index)
        backend.ensure_available()          # BackendNotAvailableError 显式报错点
        if fused:
            # GPU：构建/加载第 04 期注册扩展；native 模式不触碰融合 op 装配
            # （第 01 期路径对扩展零依赖——复用边界的对照证据）。
            backend.setup()
        return cls(backend, model_dir, dtype=dtype, fused=fused)

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load(self) -> dict[str, Any]:
        """tokenizer + 权重加载（平台无关代码路径；设备选择经 backend）。"""
        bootstrap_compat()
        here = Path(__file__).resolve().parent
        for path in (str(here),):
            if path not in sys.path:
                sys.path.insert(0, path)

        from qwen3_model_fused import load_handwritten_model
        from qwen3_tokenizer import Qwen3Tokenizer

        if self.dtype is None:
            self.dtype = DEFAULT_DTYPE_BY_BACKEND[self.backend.name]
        device = self.backend.resolve_device()

        self.tokenizer = Qwen3Tokenizer(self.model_dir)
        self.model = load_handwritten_model(
            str(self.model_dir), device, dtype=self.dtype,
            fused_ops=self.backend.fused_ops_protocol() if self.fused else None,
        )
        self.model.eval()
        self.load_evidence = {
            "model_dir": str(self.model_dir),
            "dtype": str(self.dtype).replace("torch.", ""),
            "fused_mode": "fused" if self.fused else "native",
            "device": str(device),
            "op_evidence": self.backend._op_evidence,
        }
        return dict(self.load_evidence)

    # ------------------------------------------------------------------
    # encode（tokenizer = 平台无关抽象的代表）
    # ------------------------------------------------------------------
    def encode_chat_prompt(self, text: str) -> list[int]:
        return self.tokenizer.encode_chat_prompt(text)

    def input_ids_tensor(self, token_ids: list[int]) -> torch.Tensor:
        device = self.backend.resolve_device()
        return torch.tensor([token_ids], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # generate（Greedy 循环走 Backend.greedy_decode——三后端同一份循环）
    # ------------------------------------------------------------------
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32,
                 eos_token_id: int | None = None) -> GenerateResult:
        if self.model is None:
            raise RuntimeError("runtime 未加载：先调用 load()")
        start = time.perf_counter()
        outcome = self.backend.greedy_decode(
            self.model, input_ids, max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        total_ms = (time.perf_counter() - start) * 1000.0
        router = self.backend.fused_ops_protocol() if self.fused else None
        return GenerateResult(
            generated_ids=outcome["generated_ids"],
            prompt_token_ids=input_ids[0].tolist(),
            finished_by_eos=outcome["finished_by_eos"],
            backend=self.backend.name,
            dtype=str(self.dtype).replace("torch.", ""),
            fused_mode="fused" if self.fused else "native",
            total_wall_ms=total_ms,
            step_times_ms=outcome["step_times_ms"],
            op_call_counts=dict(router.call_counts) if router is not None else {},
        )

    def first_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(input_ids)
        return logits[0, -1, :].float().cpu()

    # ------------------------------------------------------------------
    # 遥测（实验 4）：per-op 计时 + launch 数
    # ------------------------------------------------------------------
    def profile_one_forward(self, input_ids: torch.Tensor, top_n: int = 12) -> dict[str, Any]:
        """一个代表前向步的 per-op 表 + 发射数证据（遥测可观察性对照）。"""
        from experiment_utils import backend_sync, count_kernel_launches

        device = self.backend.resolve_device()
        signature = self.backend.telemetry_signature()

        launch_stats = self.backend.count_kernel_launches(
            lambda: self.model(input_ids), iterations=1
        )

        # per-op 表（profiler，含设备时长列的可观察性检查）。
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        elif device.type == "musa" and hasattr(ProfilerActivity, "PrivateUse1"):
            activities.append(ProfilerActivity.PrivateUse1)

        op_rows: list[dict[str, Any]] = []
        device_duration_nonzero = None
        backend_sync(device)
        try:
            with profile(activities=activities) as prof:
                self.model(input_ids)
                backend_sync(device)
            for event in sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total):
                if event.key in ("ProfilerStep#",) or event.count == 0:
                    continue
                op_rows.append({
                    "op": event.key[:80],
                    "count": event.count,
                    "self_cpu_time_us_total": round(event.self_cpu_time_total, 1),
                    "self_device_time_us_total": round(event.self_device_time_total, 1),
                })
                if len(op_rows) >= top_n:
                    break
            device_duration_nonzero = any(
                row["self_device_time_us_total"] > 0.0 for row in op_rows
            )
        except Exception as error:  # noqa: BLE001
            op_rows = [{"error": f"{type(error).__name__}: {error}"[:240]}]

        return {
            "backend": self.backend.name,
            "telemetry_signature": signature,
            "launch_stats": launch_stats,
            "per_op_top": op_rows,
            "device_kernel_duration_observable": {
                "signature_claim": signature["device_kernel_duration"],
                "measured_nonzero": device_duration_nonzero,
            },
        }


def load_truth() -> dict[str, Any]:
    """第 01 期真值 manifest（共享锚点，只读）。"""
    import json

    with TRUTH_MANIFEST.open("r", encoding="utf-8") as file:
        return json.load(file)
