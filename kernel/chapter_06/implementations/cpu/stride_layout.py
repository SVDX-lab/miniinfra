"""第 06 期 CPU 教学参考实现——stride（步长）与物理布局：逻辑索引 → 物理地址映射。

第 06 期的核心问题：张量的逻辑形状和物理布局是两回事；合并访存（coalescing）
依赖物理布局。本模块把"布局"这件事从框架黑盒里拆出来，用纯 Python + NumPy
显式演示：

1. StridedTensor —— 最小张量抽象：buffer（一维物理存储）+ shape + strides
   （元素为单位），physical_offset(i, j) = offset0 + i*stride0 + j*stride1
   就是全部秘密。行主序 / 列主序 / 转置视图 / 行切片视图 都是同一块 buffer
   的不同 (shape, strides) 解释；
2. naive_rmsnorm_strided —— 按物理地址显式寻址的 RMSNorm 串行实现（FP32
   累加，与第 02 期真值锚点同一数学），即"策略 C：kernel 内 stride 寻址"
   的 CPU 对应物——它能吃任意布局，但内层访问模式由 stride 决定；
3. 三种合同策略的 CPU 演示：
   - policy_a_reject：is_contiguous 检查，非连续显式报错（torch .is_contiguous
     语义的本地复刻， contiguous 检查函数 is_contiguous_view）；
   - policy_b_autocopy：内部 ascontiguousarray（= torch .contiguous() 的
     NumPy 对应物）后计算——隐性拷贝在这里发生；
   - policy_c_strided：直接 stride 寻址（naive_rmsnorm_strided）。
4. verify_stride_semantics —— 用 NumPy as_strided 验证我们的 offset 公式与
   NumPy/torch 的 stride 语义逐元素一致（教学公式的正确性锚点）。

RMSNorm 数学定义（eps=1e-6，与第 01/02 期一致）：
    y_i = w_i * x_i / sqrt( mean_j(x_j^2) + eps )
平方和 FP32 累加（自然顺序——与第 02 期 CPU 串行真值同口径）。
"""

from __future__ import annotations

import numpy as np

RMS_NORM_EPS = 1e-6


# ---------------------------------------------------------------------------
# 最小张量抽象：shape + strides + 一维物理 buffer
# ---------------------------------------------------------------------------
class StridedTensor:
    """教学用最小张量：逻辑形状 (shape) + 步长 (strides，元素单位) + 物理 buffer。

    物理地址（元素下标）= base_offset + sum(idx[k] * strides[k])。
    buffer 是一维 np.ndarray，长度只需覆盖 max 会被触达的物理下标。
    """

    def __init__(self, buffer: np.ndarray, shape: tuple[int, ...], strides: tuple[int, ...],
                 base_offset: int = 0):
        if len(shape) != len(strides):
            raise ValueError("shape 与 strides 维度必须一致")
        self.buffer = np.asarray(buffer)
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.base_offset = base_offset

    # ---- 物理地址映射（本期核心教学内容：这一个函数就是 stride 的全部）----
    def physical_offset(self, *index: int) -> int:
        if len(index) != len(self.shape):
            raise ValueError(f"需要 {len(self.shape)} 维索引，收到 {len(index)} 维")
        offset = self.base_offset
        for k, (i, s) in enumerate(zip(index, self.strides)):
            if not (0 <= i < self.shape[k]):
                raise IndexError(f"索引 {index} 越界于 shape {self.shape}")
            offset += i * s
        return offset

    def physical_read(self, *index: int) -> np.float32:
        return np.float32(self.buffer[self.physical_offset(*index)])

    def physical_write(self, value: np.float32, *index: int) -> None:
        self.buffer[self.physical_offset(*index)] = np.float32(value)

    # ---- 布局判定（torch.Tensor.is_contiguous 的教学复刻）----
    def is_contiguous(self) -> bool:
        """行主序连续 = stride 严格等于"去掉本维后的元素数"（size-1 步长为 1）。

        torch 的精确语义：第 k 维 stride == prod(shape[k+1:])（对 size=1 的维
        不约束；2D 下即 strides == (shape[1], 1)）。
        """
        expected = 1
        for dim_size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if dim_size != 1 and stride != expected:
                return False
            expected *= dim_size
        return True

    # ---- 工厂：四种典型布局，同一份逻辑数据的不同物理解释 ----
    @classmethod
    def row_major(cls, array2d: np.ndarray) -> "StridedTensor":
        """行主序（C order）：stride=(H, 1)——第 02 期 kernel 假设的布局。"""
        rows, hidden = array2d.shape
        return cls(np.ascontiguousarray(array2d).reshape(-1), (rows, hidden), (hidden, 1))

    @classmethod
    def column_major(cls, array2d: np.ndarray) -> "StridedTensor":
        """列主序（Fortran order）：stride=(1, rows)——转置视图的物理本质。"""
        rows, hidden = array2d.shape
        return cls(np.asfortranarray(array2d).reshape(-1, order="F"),
                   (rows, hidden), (1, rows))

    @classmethod
    def transposed_view(cls, array2d: np.ndarray) -> "StridedTensor":
        """转置视图：base 为 [H, rows] 行主序，逻辑 [rows, H]、stride=(1, rows)。

        这是本期实验的"转置后输入"构造：x = base[H, rows].t()——零拷贝，
        逻辑上仍是 [rows, H]，物理上每一行的相邻元素在内存中相距 rows 个元素。
        """
        rows, hidden = array2d.shape
        base = np.ascontiguousarray(np.asarray(array2d).T)  # [H, rows] 行主序
        return cls(base.reshape(-1), (rows, hidden), (1, rows))

    @classmethod
    def row_sliced_view(cls, array2d: np.ndarray, pad: int = 8) -> "StridedTensor":
        """行切片视图：base 为 [rows, H+pad] 行主序，逻辑 [rows, H]、
        stride=(H+pad, 1)——最后一维仍连续（行内合并访存保留），只有行间跳步。
        大模型里的 slice（如 x = cache[:, :H]）就是这种布局。"""
        rows, hidden = array2d.shape
        big = np.zeros((rows, hidden + pad), dtype=np.float32)
        big[:, :hidden] = array2d
        return cls(big.reshape(-1), (rows, hidden), (hidden + pad, 1))

    # ---- 交叉核对：用 NumPy 官方 stride 语义重建同一视图 ----
    def to_numpy_via_as_strided(self) -> np.ndarray:
        from numpy.lib.stride_tricks import as_strided

        byte_strides = tuple(s * self.buffer.itemsize for s in self.strides)
        view = as_strided(self.buffer[self.base_offset:], shape=self.shape,
                          strides=byte_strides)
        return np.ascontiguousarray(view)

    def to_numpy_copy(self) -> np.ndarray:
        return self.to_numpy_via_as_strided().copy()


# ---------------------------------------------------------------------------
# 按物理地址显式寻址的 RMSNorm（策略 C 的 CPU 对应物；真值同第 02 期）
# ---------------------------------------------------------------------------
def naive_rmsnorm_strided(x: StridedTensor, weight: np.ndarray,
                          eps: float = RMS_NORM_EPS) -> StridedTensor:
    """stride 寻址 RMSNorm：物理地址显式计算，两遍扫描（FP32 自然顺序累加）。

    与第 02 期 naive_rmsnorm_serial 的唯一差别：元素访问不经过 NumPy 切片，
    而是手写 offset 公式。输出 y 恒为行主序（合同：输出布局连续）。
    """
    if x.shape[1] != weight.shape[0]:
        raise ValueError("w 必须与 x 的 H 一致")
    rows, hidden = x.shape
    if hidden == 0:
        raise ValueError("合同：H=0 为空 Tensor，显式报错（不支持静默处理）")
    y_buffer = np.zeros(rows * hidden, dtype=np.float32)
    y = StridedTensor(y_buffer, (rows, hidden), (hidden, 1))
    f32 = np.float32
    for i in range(rows):
        # 第一遍：按物理地址读 x，FP32 平方和自然顺序累加。
        acc = f32(0.0)
        for j in range(hidden):
            v = x.physical_read(i, j)          # buf[base + i*s0 + j*s1]
            acc = f32(acc + f32(v * v))
        scale = f32(1.0) / f32(np.sqrt(f32(acc / f32(hidden)) + f32(eps)))
        # 第二遍：写 y（行主序物理地址）。
        for j in range(hidden):
            w_j = f32(weight[j])
            y.physical_write(f32(f32(x.physical_read(i, j) * scale) * w_j), i, j)
    return y


def policy_a_reject(x: StridedTensor, weight: np.ndarray,
                    eps: float = RMS_NORM_EPS) -> StridedTensor:
    """策略 A（显式报错）：is_contiguous 检查，非连续输入拒绝服务。

    对应 CUDA 版 TORCH_CHECK(x.is_contiguous(), ...)——合同显式拒绝，
    把布局问题在入口暴露，绝不静默处理。
    """
    if not x.is_contiguous():
        raise ValueError(
            "合同：本 Kernel 只接受连续输入（is_contiguous=False，stride="
            f"{x.strides} shape={x.shape}）。非连续布局处理策略见第 06 期合同。"
        )
    return naive_rmsnorm_strided(x, weight, eps)


def policy_b_autocopy(x: StridedTensor, weight: np.ndarray,
                      eps: float = RMS_NORM_EPS) -> StridedTensor:
    """策略 B（自动拷贝）：内部物化一份行主序副本再计算。

    np.ascontiguousarray = torch .contiguous() 的对应物：布局转换拷贝
    （读一遍 + 写一遍）发生在这里——调用方看不见，这就是"隐性拷贝成本"。
    """
    if not x.is_contiguous():
        x = StridedTensor(np.ascontiguousarray(x.to_numpy_via_as_strided()).reshape(-1),
                          x.shape,
                          (x.shape[1], 1))
    return naive_rmsnorm_strided(x, weight, eps)


def policy_c_strided(x: StridedTensor, weight: np.ndarray,
                     eps: float = RMS_NORM_EPS) -> StridedTensor:
    """策略 C（stride 寻址）：Kernel 直接吃任意布局（naive_rmsnorm_strided）。

    零拷贝，但内层访存模式由 stride 决定——在 GPU 上这决定合并访存能否
    成立（本期实测主题）；在 CPU 串行版上代价是每元素一次乘加寻址 + 缓存
    局部性随 stride 恶化。
    """
    return naive_rmsnorm_strided(x, weight, eps)


# ---------------------------------------------------------------------------
# 语义验证与布局演示
# ---------------------------------------------------------------------------
def verify_stride_semantics(rows: int = 5, hidden: int = 7) -> dict:
    """我们的 offset 公式 vs NumPy as_strided 官方语义：逐元素一致。

    同时验证四种布局的逻辑内容互相相等（同一逻辑数据、不同物理布局）——
    即"张量的逻辑形状和物理布局是两回事"的最小实证。
    """
    rng = np.random.default_rng(20260701)
    x = rng.standard_normal((rows, hidden), dtype=np.float32)
    layouts = {
        "row_major": StridedTensor.row_major(x),
        "column_major": StridedTensor.column_major(x),
        "transposed_view": StridedTensor.transposed_view(x),
        "row_sliced_view": StridedTensor.row_sliced_view(x, pad=3),
    }
    report = {"layouts": {}, "offset_formula_vs_numpy_as_strided": {}, "all_equal": True}
    logical = x
    for name, st in layouts.items():
        rebuilt = st.to_numpy_via_as_strided()
        equal = bool(np.array_equal(rebuilt, logical))
        report["offset_formula_vs_numpy_as_strided"][name] = equal
        report["layouts"][name] = {
            "shape": st.shape, "strides": st.strides,
            "is_contiguous": st.is_contiguous(),
            "buffer_len": int(st.buffer.size),
        }
        report["all_equal"] = report["all_equal"] and equal
    # 手工抽核 3 个物理地址（纸面可核）。
    rm = layouts["row_major"]
    report["manual_offsets"] = {
        "(0,0)": rm.physical_offset(0, 0),      # 0
        "(1,0)": rm.physical_offset(1, 0),      # H
        "(2,3)": rm.physical_offset(2, 3),      # 2H + 3
    }
    cm = layouts["column_major"]
    report["manual_offsets_colmajor"] = {
        "(0,0)": cm.physical_offset(0, 0),      # 0
        "(1,0)": cm.physical_offset(1, 0),      # 1
        "(2,3)": cm.physical_offset(2, 3),      # 2 + 3*rows
    }
    return report


def layout_demonstration(rows: int = 3, hidden: int = 5) -> dict:
    """教学演示：同一逻辑矩阵 [rows,H] 在四种布局下的物理下标表。

    返回逻辑位置 (i,j) → 各布局物理下标的映射表（打印进 EXPERIMENT_LOG）。
    """
    rng = np.random.default_rng(20260702)
    x = rng.standard_normal((rows, hidden), dtype=np.float32)
    layouts = {
        "row_major": StridedTensor.row_major(x),
        "column_major": StridedTensor.column_major(x),
        "transposed_view": StridedTensor.transposed_view(x),
        "row_sliced_view": StridedTensor.row_sliced_view(x, pad=2),
    }
    table = {}
    for name, st in layouts.items():
        table[name] = {
            "strides": st.strides,
            "offsets": {f"({i},{j})": st.physical_offset(i, j)
                        for i in range(rows) for j in range(hidden)},
        }
    return table


# ---------------------------------------------------------------------------
# torch 对照（可选：本地 venv torch 2.6.0+cpu 直接验证 .contiguous()/stride）
# ---------------------------------------------------------------------------
def torch_stride_probe() -> dict:
    """torch CPU 的 stride/.contiguous() 行为探针（三平台同脚本对照用）。"""
    import torch

    rows, hidden = 8, 1024
    x = torch.randn(rows, hidden)
    tv = x.t().contiguous().t()          # 转置视图：逻辑 [rows,H]、stride=(1,rows)
    rs = torch.randn(rows, hidden + 8)[:, :hidden]   # 行切片视图
    expanded = torch.randn(1, hidden).expand(rows, hidden)  # expand：stride0=0
    probe = {
        "torch_version": torch.__version__,
        "contiguous": {"shape": list(x.shape), "stride": list(x.stride()),
                       "is_contiguous": x.is_contiguous()},
        "transposed_view": {"shape": list(tv.shape), "stride": list(tv.stride()),
                            "is_contiguous": tv.is_contiguous(),
                            "same_values_as_x": bool(torch.equal(tv, x))},
        "row_sliced_view": {"shape": list(rs.shape), "stride": list(rs.stride()),
                            "is_contiguous": rs.is_contiguous(),
                            "is_view_of_base": rs._base is not None},
        "expanded": {"shape": list(expanded.shape), "stride": list(expanded.stride()),
                     "is_contiguous": expanded.is_contiguous()},
        "transpose_back_trap": {
            # 教学陷阱：x.t().contiguous() 对"转置视图"是零成本（x.t() 本身连续），
            # 它并不会把 [rows,H] 变回行主序——返回的是 self 的别名。
            "xt_is_contiguous": tv.t().is_contiguous(),
            "xt_contiguous_returns_self": tv.t().contiguous().data_ptr() == tv.t().data_ptr(),
            "xt_contiguous_t_stride": list(tv.t().contiguous().t().stride()),
        },
        "contiguous_results": {
            "tv_contig_stride": list(tv.contiguous().stride()),
            "tv_contig_is_contiguous": tv.contiguous().is_contiguous(),
            "tv_contig_equal_x": bool(torch.equal(tv.contiguous(), x)),
        },
    }
    return probe
