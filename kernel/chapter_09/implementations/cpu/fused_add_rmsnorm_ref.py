"""第 04 期 CPU 实现：PyTorch composite fallback（显式注册的 fallback 语义载体）。

第 04 期核心问题（大纲 §13 行 700–716）：自定义 Kernel 只有被框架 dispatch
机制认识，才能安全替换模型里的原生算子。三个世界给出三种答案：

- **CPU（本文件）**：没有 Kernel 时"还能跑"的 fallback 语义载体。对照物是
  "静默回退"——原生组合表达式（r+h; RMSNorm）在任何后端都被框架静默地
  逐算子执行，调用方无法感知"这里没有专用 Kernel"；本实现的对应答案是
  **显式**：它以独立可注册的算子单元存在（functions 表 + 合同校验 +
  fallback 计数/日志），谁在什么时候用它，全部留痕（见 ops_router.py）。

语义（与第 03 期融合 kernel / vLLM fused_add_rms_norm 完全同构，in-place）：

    residual ← residual + input          （T dtype 一次舍入）
    input    ← weight * RMSNorm(residual)（平方和/均值 FP32，输出写回 input buffer）

即 vLLM 式双 buffer 原地改写、零新分配：调用后 input buffer 内容 =
归一化输出 y，residual buffer 内容 = 更新后的 residual t。Transformer
residual 流里这两个值正是下一步要用的两份张量。

显式报错路径（合同 error_behavior）：dtype 不在 {float32, float16}、
H=0、形状不匹配、非连续输入、设备不一致——全部 ValueError，绝不静默改道。
"""

from __future__ import annotations

import torch

RMS_NORM_EPS = 1e-6

OP_NAME = "fused_add_rmsnorm"
OP_SCHEMA = (
    "fused_add_rmsnorm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, "
    "float eps) -> ()"
)
OP_SEMANTICS = (
    "residual ← residual + input（T 一次舍入）；"
    "input ← RMSNorm(residual) * weight（平方和/均值 FP32）；双 buffer 原地改写，零新分配"
)


def _validate(input_tensor: torch.Tensor, residual: torch.Tensor,
              weight: torch.Tensor) -> None:
    """合同校验：与 CUDA/MUSA 注册实现的 TORCH_CHECK 逐条同构。"""
    # 形状合同（vLLM 同款）：同形状、末维 H、任意 leading 维度（模型激活
    # [batch, seq, H]，内部展平 rows = numel/H）。
    if input_tensor.dim() < 1:
        raise ValueError("input 至少一维，实际 dim=" + str(input_tensor.dim()))
    if residual.shape != input_tensor.shape:
        raise ValueError("residual 与 input 形状必须一致："
                         + str(tuple(residual.shape)) + " vs "
                         + str(tuple(input_tensor.shape)))
    if weight.dim() != 1 or weight.numel() != input_tensor.shape[-1]:
        raise ValueError("weight 必须是 [H] 且与 input 的末维一致：w.numel()="
                         + str(weight.numel()) + " input.size(-1)="
                         + str(input_tensor.shape[-1]))
    if input_tensor.dtype not in (torch.float32, torch.float16):
        raise ValueError("本期合同 dtype ∈ {float32, float16}，实际 "
                         + str(input_tensor.dtype))
    if input_tensor.dtype != residual.dtype or input_tensor.dtype != weight.dtype:
        raise ValueError("input/residual/weight dtype 必须一致")
    if input_tensor.device != residual.device or input_tensor.device != weight.device:
        raise ValueError("input/residual/weight 必须在同一设备")
    if input_tensor.shape[-1] == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错（不支持静默处理）")
    if input_tensor.data_ptr() == residual.data_ptr():
        raise ValueError("residual 与 input 不能是同一 buffer（op 会分别改写两者）")
    if (input_tensor.data_ptr() == weight.data_ptr()
            or residual.data_ptr() == weight.data_ptr()):
        raise ValueError("weight 不能与 input/residual 共用 buffer")
    if not (input_tensor.is_contiguous() and residual.is_contiguous()
            and weight.is_contiguous()):
        raise ValueError("合同：本期只接受连续 Tensor（非连续布局是第 06 期主题）")


def composite_fallback_fused_add_rmsnorm(
    input_tensor: torch.Tensor, residual: torch.Tensor,
    weight: torch.Tensor, eps: float = RMS_NORM_EPS
) -> None:
    """CPU composite fallback 实现（真值锚点 + 显式 fallback 载体）。

    数学 = 第 03 期 composite_separated_add_rmsnorm_torch 的 in-place 版：
      1. residual.add_(input)                     —— t = r + h 写回 residual；
      2. FP32 平方和/均值 + rsqrt + 权重缩放       —— 与第 01 期手写 RMSNorm
         模块（qwen3_model.py RMSNorm.forward）同一表达式；
      3. input.copy_(y)                           —— y 写回 input buffer。

    融合 kernel 的"单次输出舍入"在这里表现为第 01 期组合表达式的两次舍入
    （normalized.to(T) 后再乘 weight）——注册层不改数学，差异由正确性链
    容差吸收（第 01 期容差依据）。
    """
    _validate(input_tensor, residual, weight)
    if not (0.0 < float(eps) < 1.0):
        raise ValueError("eps 合同范围 (0,1)，实际 " + str(eps))
    # Pass 1（对应融合 kernel Pass 1 的加法段）：t = r + h 原位写回 residual。
    residual.add_(input_tensor)
    # Pass 2（对应融合 kernel Pass 2 的归一化段）：FP32 方差 + rsqrt + 缩放。
    float_states = residual.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    y = weight * normalized.to(residual.dtype)
    input_tensor.copy_(y)
    return None


def rmsnorm_module_math(hidden_states: torch.Tensor, weight: torch.Tensor,
                        eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """第 01 期手写 RMSNorm 模块的前向数学（原样引用，正确性链第 6 步锚点）：

    qwen3_model.py::RMSNorm.forward：
        float_states = hidden.float()
        variance = float_states.pow(2).mean(-1, keepdim=True)
        normalized = float_states * rsqrt(variance + eps)
        return weight * normalized.to(input_dtype)
    """
    input_dtype = hidden_states.dtype
    float_states = hidden_states.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)
