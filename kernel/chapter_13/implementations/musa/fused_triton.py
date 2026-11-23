"""第 13 期 CUDA Triton 融合 Residual Add + RMSNorm —— 对照第 03 期手写融合。

第 03 期手写融合语义（chapter_03/implementations/cuda/fused_rmsnorm_kernels.cu）：
  单 Kernel：Pass 1 读 r、读 h，t = r + h 在寄存器完成，t 先按 dtype T 舍入
  一次（t_T = T(t_f32)——分离路径"物化中间 Tensor"的舍入语义），更新后的
  residual 原位写回 r；t_T 的 FP32 值缓存进 shared 并累加平方和；Pass 2
  y = t_T * rsqrt(mean(t_T^2) + eps) * w。

Triton 映射：
  - t_f32 = r + h（FP32）→ t_T = t_f32.to(T) → t = t_T.to(fp32)：
    与手写版"先舍入一次再用舍入值"逐位一致（fp32 路径 T=fp32 无变化）；
  - 原位写回 r（in-place residual，vLLM fused_add_rmsnorm 同语义）；
  - 平方和块内树形归约 FP32；y 一次舍入到 T。
  - BLOCK = next_power_of_2(H) >= H（H=1024 → 1024）；ch04 主形状单遍即可。

本文件同时提供 Triton 分离路径（torch add + Triton RMSNorm），用于同会话内
测"DSL 融合收益"，再与 ch04 手写融合的实测数字（延迟/加速比）对照。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_rmsnorm_kernel(
    r_ptr,   # *T [rows, H]，in-place residual buffer
    h_ptr,   # *T [rows, H]
    w_ptr,   # *T [H]
    y_ptr,   # *T [rows, H]
    h, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_off = row * h
    cols = tl.arange(0, BLOCK)
    mask = cols < h
    r = tl.load(r_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    hval = tl.load(h_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    t_f32 = r + hval
    # 关键语义：t 先按 dtype T 舍入一次（= 分离路径物化 t 的舍入），随后
    # 一切（原位写回、平方和、scale、y）都用舍入后的 t —— 与 ch04 手写
    # 融合 kernel 逐位同语义（分离/融合 y 位级一致是 ch04 的实验结论）。
    t_t = t_f32.to(r_ptr.dtype.element_ty)
    t = t_t.to(tl.float32)
    tl.store(r_ptr + row_off + cols, t_t, mask=mask)  # in-place residual
    ssq = tl.sum(t * t, axis=0)
    scale = tl.math.rsqrt(ssq / h + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = t * scale * w
    tl.store(y_ptr + row_off + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(
    r: torch.Tensor,
    h_input: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    num_warps: int = 4,
) -> torch.Tensor:
    """融合入口（in-place：调用后 r buffer 内容 = T(r+h)；y 独立输出）。

    返回 (y, r_updated)。调用方需保留原 r 时先克隆（ch04 合同同款说明）。
    """
    _validate_fused(r, h_input, w)
    rows, hidden = r.shape
    if rows == 0:
        return torch.empty_like(r), r
    y = torch.empty_like(r)
    block = max(1, triton.next_power_of_2(hidden))
    fused_add_rmsnorm_kernel[(rows,)](
        r, h_input, w, y, hidden, eps, BLOCK=block, num_warps=num_warps
    )
    return y, r


def triton_separated(r: torch.Tensor, h_input: torch.Tensor, w: torch.Tensor, eps: float):
    """Triton 分离路径：torch.add（1 kernel）+ Triton RMSNorm（1 kernel）。

    同会话内对照"DSL 融合收益"；与 ch04 手写分离（add_kernel + rmsnorm kernel）
    结构同构但第 1 步用框架 kernel —— 中间 t 物化在显存。
    """
    from rmsnorm_triton import rmsnorm  # noqa: PLC0415

    t = torch.add(r, h_input)  # dtype T 舍入（torch add 的物化语义）
    y = rmsnorm(t, w, eps)
    return y, t


def _validate_fused(r: torch.Tensor, h_input: torch.Tensor, w: torch.Tensor) -> None:
    if r.ndim != 2 or h_input.ndim != 2:
        raise ValueError("r/h 必须是 [rows, H] 二维 Tensor")
    if r.shape != h_input.shape:
        raise ValueError("r 与 h 形状必须一致")
    if w.ndim != 1 or w.shape[0] != r.shape[1]:
        raise ValueError("weight 必须是 [H] 一维 Tensor 且与 H 一致")
    if r.shape[1] == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错")
    if not (r.is_contiguous() and h_input.is_contiguous() and w.is_contiguous()):
        raise ValueError("合同：本期只接受连续输入")
    if r.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"合同：本期 dtype 集合 {{fp16, fp32}}，收到 {r.dtype}")
