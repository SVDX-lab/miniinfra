"""第 14 期 compile 变体构建器（CPU 列；CUDA 列脚本复用同一构建逻辑）。

变体矩阵（按 CPU 适用性选择，mode 对照为 CUDA 待补列保留 reduce-overhead）：

| 变体名 | 构建 | 作用 |
|---|---|---|
| eager | 原函数 | 基线（框架逐算子发射） |
| dynamo-eager | backend="eager" | 只做 dynamo 图捕获不做代码生成——隔离"图开销 vs 代码生成收益" |
| aot-eager | backend="aot_eager" | AOTAutograd 跟踪 + 无 inductor——隔离 AOT 层开销 |
| inductor | torch.compile(fn)（默认 mode） | 本期主角：Inductor 默认代码生成 |
| inductor-mat | mode="max-autotune-no-cudagraphs" | Inductor + 自动调优（cudagraphs 与 CPU 无关，按任务书用 no-cudagraphs 形态） |

冷编译控制：每次构建 compiled 变体前执行 torch._dynamo.reset() 并关闭
fx_graph_cache / autotune_local_cache，使"首调耗时"是诚实的冷编译时间
（缓存命中会让 compile time 失真；稳态延迟测量不受影响——稳态在首调之后）。
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

VARIANTS_CPU = ("eager", "dynamo-eager", "aot-eager", "inductor", "inductor-mat")

# qwen3 段只跑子集（max-autotune 在整段上会逐 kernel 调优、耗时失控，且
# CPU 上其收益面已被前两档覆盖；如实记录取舍）。
VARIANTS_QWEN = ("eager", "dynamo-eager", "aot-eager", "inductor")


def disable_codegen_caches() -> None:
    """关闭 inductor 持久化缓存，保证首调=冷编译。"""
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.fx_graph_remote_cache = False
    try:
        torch._inductor.config.autotune_local_cache = False
    except (AttributeError, TypeError):
        pass


def reset_compilers() -> None:
    """dynamo 状态清零（guards/缓存/计数），变体之间互不污染。"""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()


def make_callable(object_name: str, variant: str, module: Any | None = None) -> Callable:
    """把 (对象名, 变体名) 变成可调用对象。

    qwen3 段编译的是 nn.Module（module 参数传入），其余对象是纯函数。
    """
    import implementations.cpu.compiled_objects as objects

    if object_name == "qwen3":
        if module is None:
            raise ValueError("qwen3 对象需要传入构建好的 nn.Module")
        target = module
    elif object_name == "rmsnorm":
        target = objects.rmsnorm_expr
    elif object_name == "add_rmsnorm":
        target = objects.add_rmsnorm_separated
    else:
        raise ValueError(f"未知对象: {object_name}")

    if variant == "eager":
        return target
    if variant == "dynamo-eager":
        return torch.compile(target, backend="eager", fullgraph=True)
    if variant == "aot-eager":
        return torch.compile(target, backend="aot_eager", fullgraph=True)
    if variant == "inductor":
        return torch.compile(target)
    if variant == "inductor-mat":
        return torch.compile(target, mode="max-autotune-no-cudagraphs")
    raise ValueError(f"未知变体: {variant}")


def timed_first_call(callable_fn: Callable, *args) -> tuple[float, Any]:
    """首调墙钟（毫秒）——dynamo 图捕获 + 代码生成 + kernel 编译 + 执行。

    返回 (elapsed_ms, output)。调用方须已 reset 并关闭缓存，否则这是
    "缓存命中时间"而非冷编译时间。
    """
    started = time.perf_counter()
    output = callable_fn(*args)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, output
