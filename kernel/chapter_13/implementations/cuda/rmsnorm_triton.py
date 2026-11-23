"""第 13 期 CUDA Triton RMSNorm —— DSL 路径（对照第 02 期手写版语义）。

第 02 期手写语义（本期必须逐条对齐，见 chapter_02/implementations/cuda/rmsnorm_kernel.cu）：
  1. 每 block 一行；平方和与均值固定 FP32 累加；
  2. scale = rsqrt(mean(x^2) + eps)，eps 在根号内（全零行输出全零，无 NaN）；
  3. 归一化与权重乘法在 FP32 完成，写回时一次舍入到 T；
  4. 非整 Tile（H=127/129 等）块内 stride 循环覆盖（手写）；Triton 用 mask 覆盖。

DSL 语义映射（本文件）：
  - "每 block 一行" -> 每 program 一行（grid = (rows,)，无 grid-stride：
    Triton 的 launch 粒度由 host 决定，行数即 grid 大小，上限 2^31-1）；
  - "FP32 累加" -> tl.load(...).to(tl.float32) 后 tl.sum(x*x)（块内树形归约，
    归约顺序由编译器决定 —— 与手写 warp 蝶形的对照素材）；
  - "rsqrtf" -> tl.math.rsqrt（NVIDIA 下降到 rsqrt.approx.f32，同手写近似级别）；
  - "一次舍入到 T" -> tl.store 时 y.to(dtype) 单次转换；
  - "块内 stride 循环"（手写非整 Tile 方案）-> Triton 两种等价物：
      (a) 单遍 kernel（rmsnorm_fwd）：BLOCK = next_power_of_2(H) >= H，
          tl.arange(0, BLOCK) + mask —— vLLM/Liger 的生产写法；
      (b) 双遍循环 kernel（rmsnorm_fwd_looped）：for off in range(0, H, BLOCK)，
          BLOCK 可以 < H —— 与第 02 期 TILE 扫描 {64..1024} 同一搜索空间，
          Pass 2 重读 x（手写版用 shared 缓存避免重读，DSL 侧不可控 ——
          这正是"换走了多少控制权"的数据点）。

文件内包含：
  - rmsnorm_fwd_kernel        单遍 @triton.jit（非 autotune，BLOCK/num_warps 显式传）；
  - rmsnorm_fwd_looped        双遍循环 @triton.jit（非 autotune，BLOCK 可 < H）；
  - rmsnorm(...)              host 入口（显式参数版，合同语义检查在这里）；
  - rmsnorm_autotuned(...)    @triton.autotune 包装（同一 kernel 本体），
                              配置空间与第 02 期手动 Tile 扫描对齐。

本文件不依赖手写 .cu（对照实验在 benchmark.py 里按数字引用第 02 期结果）。
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

TRITON_VERSION = triton.__version__


# ---------------------------------------------------------------------------
# Kernel 1：单遍 RMSNorm（BLOCK >= H，生产写法；BLOCK/num_warps 由 host 显式给）
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr,          # *T [rows, H]
    w_ptr,          # *T [H]
    y_ptr,          # *T [rows, H]
    h,              # 运行时 H（int32）
    eps,            # float
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < h
    row_off = row * h
    # Pass 1：读 x -> FP32，块内树形归约平方和（一次 load，行驻留寄存器）
    x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    ssq = tl.sum(x * x, axis=0)
    # scale = 1/sqrt(mean(x^2) + eps)（eps 在根号内；rsqrt 近似级别同手写 rsqrtf）
    scale = tl.math.rsqrt(ssq / h + eps)
    # Pass 2：w 读为 FP32，y = x*scale*w 在 FP32 完成，store 时一次舍入到 T
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * scale * w
    tl.store(y_ptr + row_off + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2：双遍循环 RMSNorm（BLOCK 可 < H；与第 02 期 TILE 扫描同一搜索空间）
#   Pass 1 分块累加平方和；Pass 2 分块重读 x 写 y。
#   与手写版的差异点（论文素材）：手写版把 x 行缓存在 shared memory（H<=2048
#   免重读）；Triton 无显式 shared 控制，BLOCK<H 时 Pass 2 从 global/L2 重读 x。
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_fwd_looped(
    x_ptr, w_ptr, y_ptr,
    h, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_off = row * h
    ssq = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, h, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < h
        x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        ssq += x * x
    total = tl.sum(ssq, axis=0)
    scale = tl.math.rsqrt(total / h + eps)
    for off in range(0, h, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < h
        x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * scale * w
        tl.store(y_ptr + row_off + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Kernel 3：autotune 版（同一 kernel 本体 rmsnorm_fwd_looped）。
# 配置空间设计（对照第 02 期手动 Tile 扫描 tiles={64,128,256,512,1024}）：
#   - parity 空间（与 ch03 同一 5 个 Tile，num_warps 固定 4）；
#   - wide 空间（+ num_warps {1,2,4,8}，Triton 额外维度）。
# autotune 的 key=['h']：H 变化时重搜（与 ch03 "每 H 手动扫" 同语义）。
# ---------------------------------------------------------------------------
PARITY_CONFIGS = [
    triton.Config({"BLOCK": 64}, num_warps=4),
    triton.Config({"BLOCK": 128}, num_warps=4),
    triton.Config({"BLOCK": 256}, num_warps=4),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
]

WIDE_CONFIGS = [
    triton.Config({"BLOCK": b}, num_warps=nw)
    for b in (64, 128, 256, 512, 1024, 2048)
    for nw in (1, 2, 4, 8)
]

SP_CONFIGS = [
    triton.Config({"BLOCK": b}, num_warps=nw)
    for b in (1024, 2048)
    for nw in (2, 4, 8, 16)
]


def _make_autotuned(kernel, configs):
    """为指定 kernel 本体构造 autotune 包装（key=['h']）。

    直接包装已有 @triton.jit 函数（triton.autotune 是装饰器工厂，对任意
    JITFunction 可用）；BLOCK 只出现在 config.kwargs 里，launch 时不传。
    """
    return triton.autotune(configs=list(configs), key=["h"])(kernel)


# ---------------------------------------------------------------------------
# host 入口
# ---------------------------------------------------------------------------
def _validate(x: torch.Tensor, w: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError("x 必须是 [rows, H] 二维 Tensor")
    if w.ndim != 1 or w.shape[0] != x.shape[1]:
        raise ValueError("weight 必须是 [H] 一维 Tensor 且与 x 的 H 一致")
    if x.shape[1] == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错（不支持静默处理）")
    if not x.is_contiguous() or not w.is_contiguous():
        raise ValueError("合同：本期只接受连续输入（非连续显式报错，第 06 期主题）")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"合同：本期 dtype 集合 {{fp16, fp32}}，收到 {x.dtype}")


def next_pow2(n: int) -> int:
    return max(1, triton.next_power_of_2(int(n)))


def rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    block_size: int | None = None,
    num_warps: int = 4,
    looped: bool = False,
) -> torch.Tensor:
    """显式参数入口（非 autotune）。

    block_size=None 时走单遍 kernel，BLOCK = next_power_of_2(H)（生产默认）；
    block_size 显式给出时走 looped kernel（允许 BLOCK < H，与 ch03 TILE 语义
    对齐）——当 BLOCK >= H 时 looped kernel 退化为单次循环体。
    返回新 Tensor y（无原位语义，同 ch03 合同）。
    """
    _validate(x, w)
    rows, h = x.shape
    if rows == 0:
        return torch.empty_like(x)
    y = torch.empty_like(x)
    if block_size is None and not looped:
        block = next_pow2(h)
        kernel = rmsnorm_fwd_kernel
    else:
        block = block_size if block_size is not None else next_pow2(h)
        kernel = rmsnorm_fwd_looped
    grid = (rows,)
    kernel[grid](x, w, y, h, eps, BLOCK=block, num_warps=num_warps)
    return y


def rmsnorm_timed(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    block_size: int | None = None,
    num_warps: int = 4,
    looped: bool = False,
) -> tuple[torch.Tensor, float]:
    """同 rmsnorm，但返回 (y, 首次调用墙钟秒)——编译/缓存成本测量用。"""
    _validate(x, w)
    rows, h = x.shape
    y = torch.empty_like(x)
    if block_size is None and not looped:
        block = next_pow2(h)
        kernel = rmsnorm_fwd_kernel
    else:
        block = block_size if block_size is not None else next_pow2(h)
        kernel = rmsnorm_fwd_looped
    grid = (rows,)
    start = time.perf_counter()
    kernel[grid](x, w, y, h, eps, BLOCK=block, num_warps=num_warps)
    torch.cuda.synchronize()
    return y, time.perf_counter() - start


class AutotuneRMSNorm:
    """autotune 版入口：持有 autotune JITFunction 与其配置空间名。

    kernel_kind: 'parity'（ch03 5 配置）/'wide'（24 配置，含 BLOCK=2048）
    """

    def __init__(self, kernel_kind: str = "parity"):
        if kernel_kind == "parity":
            self.configs = PARITY_CONFIGS
        elif kernel_kind == "wide":
            self.configs = WIDE_CONFIGS
        else:
            raise ValueError(kernel_kind)
        self.fn = _make_autotuned(rmsnorm_fwd_looped, self.configs)
        self.kernel_kind = kernel_kind
        self.cache_read_attempts: list = []

    def __call__(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        _validate(x, w)
        rows, h = x.shape
        if rows == 0:
            return torch.empty_like(x)
        y = torch.empty_like(x)
        self.fn[(rows,)](x, w, y, h, eps)  # BLOCK 由 autotune 配置注入，launch 不传
        return y

    def best_config(self):
        """autotune 选中的配置（版本容忍，3.7 实测：Autotuner.best_config 直出）。"""
        fn = self.fn
        attempts = []
        # Triton 3.7 实测：Autotuner.best_config 属性直出（含 num_warps/num_stages）
        direct = getattr(fn, "best_config", None)
        if direct is not None:
            self.cache_read_attempts = [{"best_config_attr": "hit"}]
            return direct
        for attr in ("cache", "autotune_cache", "device_caches"):
            try:
                caches = getattr(fn, attr, None)
                if not caches:
                    attempts.append({attr: "missing"})
                    continue
                attempts.append({attr: "present"})
                values = caches.values()
                for entry in values:  # 3.7: cache 为 flat dict[key_tuple -> Config]
                    if isinstance(entry, triton.Config):
                        self.cache_read_attempts = attempts
                        return entry
                    if hasattr(entry, "values"):
                        for inner in entry.values():
                            if isinstance(inner, triton.Config):
                                self.cache_read_attempts = attempts
                                return inner
            except Exception as error:  # noqa: BLE001
                attempts.append({attr: f"error: {type(error).__name__}: {error}"})
        self.cache_read_attempts = attempts
        return None

    def search_timings(self):
        """3.7 实测：Autotuner.configs_timings = {Config: [每次 rep 的 ms]}，
        bench_time = 搜索墙钟秒——autotune 搜索的逐配置透明证据。"""
        fn = self.fn
        out = {"bench_time_s": getattr(fn, "bench_time", None)}
        timings = getattr(fn, "configs_timings", None)
        if timings:
            rows = []
            for config, samples in timings.items():
                kwargs = getattr(config, "kwargs", {})
                rows.append({
                    "config": f"B={kwargs.get('BLOCK')},nw={getattr(config, 'num_warps', '?')}",
                    "samples_ms": [round(s, 4) for s in samples] if samples else None,
                    "min_ms": round(min(samples), 4) if samples else None,
                })
            out["per_config"] = rows
        return out


def fused_add_rmsnorm(
    r: torch.Tensor,
    h_input: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    num_warps: int = 4,
) -> torch.Tensor:
    """融合 add+rmsnorm（见 fused_triton.py，此处转引以便 benchmark 单点 import）。"""
    from fused_triton import fused_add_rmsnorm as _impl  # noqa: PLC0415

    return _impl(r, h_input, w, eps, num_warps=num_warps)
