"""第 14 期被编译对象（CPU 列）：交给 torch.compile 的三档 Python 程序。

第 14 期核心问题（大纲 §13 第 14 期，行 870–887）：编译器（Inductor）的
自动融合规则是什么、它什么时候融合/放弃；它的融合结果离手写（第 03 期）
和 DSL 手写（第 13 期）差多少。

被编译对象 = 第 02/03/01 期的"手写对象"在框架层的对应物：

1. rmsnorm_expr —— RMSNorm 组合表达式（第 01 期 RMSNorm.forward 的数学、
   第 02 期 rmsnorm kernel 的框架层对应物）。eager 下框架对它发射
   pow/mean/rsqrt/mul 等多个 kernel（ch04 实测 torch 组合链 9 kernel/迭代），
   编译器的第一项工作就是把这些逐元素/归约算子融合进一个 C++ 循环；
2. add_rmsnorm_separated —— 分离 add+rmsnorm 段（第 03 期手写分离路径
   的框架层对应物：t = r + h 后接 RMSNorm，返回 (t, y)，中间 t 物化）。
   编译器面临的抉择：add + 归一化是否合并成一个 kernel（对应第 03 期
   手写融合 fused_add_rmsnorm_kernel）；
3. qwen3_segment_forward —— 小 Qwen3 前向段（embed → 2×DecoderLayer →
   final norm，复用第 01 期模型代码与 Qwen3-0.6B 的 hidden 口径）。
   编译器在整个段上的融合/放弃决策（matmul 不融合、逐元素链融合、
   attention 内 softmax 等的图级处理）。

数值语义与第 02/03 期真值锚点一致：eps=1e-6（Qwen3 config.json），平方
和均值固定 FP32 累加；本模块全部函数都是纯函数（无 in-place 参数突变），
in-place/functionalization 行为单独在 smoke_test 探测（见第 03 期合同
residual 更新语义的编译器对应面）。
"""

from __future__ import annotations

import torch

RMS_NORM_EPS = 1e-6


def rmsnorm_expr(hidden: torch.Tensor, weight: torch.Tensor, eps: float = RMS_NORM_EPS):
    """RMSNorm 组合表达式（第 01 期 RMSNorm.forward 数学，纯函数版）。

    y = w * (x.float() * rsqrt(mean(x.float()^2) + eps)).to(x.dtype)
    eager 发射链：to/pow/mean/rsqrt/mul/to/mul（ch04 profiler 口径下是
    组合链的成员）。
    合同（ch04 §H=0）：空 Tensor 显式报错——校验写在被编译函数体内，
    dynamo 以 trace 期 raise / guard 特化两种方式保留该语义（smoke 实测）。
    """
    if hidden.shape[-1] == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错（不支持静默处理）")
    input_dtype = hidden.dtype
    float_states = hidden.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def add_rmsnorm_separated(
    residual: torch.Tensor, hidden: torch.Tensor, weight: torch.Tensor,
    eps: float = RMS_NORM_EPS,
):
    """分离 add+rmsnorm 段（第 03 期手写分离路径的框架层对应物）。

    返回 (t, y)：t = r + h（物化中间 Tensor，对应手写分离路径的 add
    kernel 输出），y = rmsnorm(t)。编译器要回答的问题：add 能不能
    与后面的归一化融进同一个 kernel（即自动合成第 03 期的
    fused_add_rmsnorm_kernel）。
    """
    if residual.shape[-1] == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错（不支持静默处理）")
    t = residual + hidden
    y = rmsnorm_expr(t, weight, eps)
    return t, y


def add_rmsnorm_inplace_style(
    residual: torch.Tensor, hidden: torch.Tensor, weight: torch.Tensor,
    eps: float = RMS_NORM_EPS,
):
    """residual 原位更新语义的编译器版（第 03 期合同 in-place 语义）。

    与 add_rmsnorm_separated 的差异：对调用方传入的 residual 做 r.add_(h)
    原位突变（vLLM fused_add_rms_norm 的调用语义）。torch.compile 需要
    functionalization：把输入突变重写成"纯函数 + 返回更新值"。返回
    (residual_updated, y)；调用后传入 buffer 内容 = r + h。
    （此对象主要用于 codegen 阅读/functionalization 探测，性能主表不含它。）
    """
    residual.add_(hidden)
    y = rmsnorm_expr(residual, weight, eps)
    return residual, y
