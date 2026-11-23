"""第 07 期 CPU 求和顺序实验室 —— 浮点结合律破缺的定量展示。

第 07 期思想：浮点加法不满足结合律 → 归约顺序是"实现的一部分"。
本模块把同一行向量的平方和按六种顺序 + 三种累加 dtype 全矩阵展开：

1. serial_l2r      —— 左到右自然顺序（CPU 串行参照 = 真值锚点口径）；
2. serial_r2l      —— 右到左（教学对照：同一"串行"，反向也位级不同）；
3. pairwise_tree   —— 分治两两合并（warp 蝶形归约的数学模型，顺序固定）；
4. blocked(k)      —— 每 k 元素一块，块内串行、块间串行
                      （GPU "warp 部分和 + 跨 warp 固定顺序合并"的模型）；
5. numpy_pairwise  —— NumPy sum（ pairwise 实现，顺序由库决定）；
6. shuffled(seed)  —— 随机到达顺序串行累加（atomicAdd 到同一累加器的
                      一阶模型：增量到达顺序近似均匀随机置换，声明见
                      EXPERIMENT_LOG；对 P 个种子统计漂移分布）。

累加 dtype 三档（与第 05 期 dtype_sim 的语义一致）：
- fp32：每步 round-to-nearest-even 到 FP32（numpy float32 运算天然如此）；
- fp16：每步舍入到 binary16（ep06 卡死区：ulp(acc) > 增量/2 后冻结）；
- fp64：FP64 累加（顺序敏感性仍在，但 ulp 小 2^29 倍，FP32 数据下
  实际差异淹没在 1e-16 —— 教学对照"顺序不敏感是相对的"）。

对 RMSNorm 的意义：平方和的归约顺序漂移经 rsqrt 放大（漂移一半）进入
scale，再乘到每个输出元素 —— 确定性声明（合同条目）必须写明归约组织。
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# 位精确累加模拟器基础（fp16 舍入用 numpy binary16；fp32 用 numpy float32）
# ---------------------------------------------------------------------------

_ACC_DTYPES = {
    "fp32": np.float32,
    "fp16": np.float16,
    "fp64": np.float64,
}


def _round(acc_value, acc_dtype) -> float:
    """把中间值舍入回累加器 dtype（fp16 卡死区在这里发生：RNE 冻结）。"""
    return acc_dtype(acc_value)


def sum_serial(x: np.ndarray, acc_dtype_name: str = "fp32",
               start: int = 0, stop: int | None = None, stride: int = 1) -> float:
    """左到右（或指定区间/步长）串行累加 x[i]^2，每步舍入到累加器 dtype。"""
    dt = _ACC_DTYPES[acc_dtype_name]
    stop = x.shape[0] if stop is None else stop
    acc = dt(0.0)
    for j in range(start, stop, stride):
        acc = dt(acc + dt(x[j] * x[j]))
    return float(acc)


def sum_r2l(x: np.ndarray, acc_dtype_name: str = "fp32") -> float:
    """右到左串行累加（教学对照）。"""
    dt = _ACC_DTYPES[acc_dtype_name]
    acc = dt(0.0)
    for j in range(x.shape[0] - 1, -1, -1):
        acc = dt(acc + dt(x[j] * x[j]))
    return float(acc)


def sum_pairwise(x: np.ndarray, acc_dtype_name: str = "fp32") -> float:
    """分治 pairwise 归约（偶数长度对半分，奇数时尾元素并入右半）。

    这是 GPU warp 蝶形归约（固定 XOR 伙伴树）与 NumPy pairwise 的共同
    数学模型：树的形状固定 => 确定性；但树的顺序 != 串行顺序 => 位级不同。
    """
    dt = _ACC_DTYPES[acc_dtype_name]
    n = x.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return float(dt(x[0] * x[0]))
    mid = n // 2
    left = sum_pairwise(x[:mid], acc_dtype_name)
    right = sum_pairwise(x[mid:], acc_dtype_name)
    return float(dt(dt(left) + dt(right)))


def sum_blocked(x: np.ndarray, block: int, acc_dtype_name: str = "fp32") -> float:
    """每 block 元素块内串行、块间串行（warp 部分和 + 固定顺序合并的模型）。

    与 ep06 exp_accumulation 的"块内 FP32 + 块间 fp16"混合组织不同：
    这里块内块外同 dtype，纯粹展示"分块改变顺序"这一件事。
    """
    dt = _ACC_DTYPES[acc_dtype_name]
    acc = dt(0.0)
    for start in range(0, x.shape[0], block):
        inner = dt(0.0)
        stop = min(start + block, x.shape[0])
        for j in range(start, stop):
            inner = dt(inner + dt(x[j] * x[j]))
        acc = dt(acc + inner)
    return float(acc)


def sum_numpy(x: np.ndarray, acc_dtype_name: str = "fp32") -> float:
    """NumPy sum（pairwise 实现，顺序由库内部决定，用户不可指定）。"""
    dt = _ACC_DTYPES[acc_dtype_name]
    return float(np.sum((x * x).astype(dt), dtype=dt))


def sum_shuffled(x: np.ndarray, seed: int, acc_dtype_name: str = "fp32") -> float:
    """atomicAdd 乱序到达的一阶模型：增量按随机置换串行累加。

    模型声明（诚实记录）：真实 atomicAdd 的到达顺序由 warp 调度/访存延迟
    决定，不是均匀随机置换（同 warp 内近似 lane 序、跨 warp/block 由竞争
    决定）；本模型是一阶近似，用于回答"顺序不确定会漂移到哪"——与设备
    实测的 100 次分布互相印证（设备分布是"真实抽样"，本模型是"理论采样"）。
    """
    dt = _ACC_DTYPES[acc_dtype_name]
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    acc = dt(0.0)
    for j in order:
        acc = dt(acc + dt(x[j] * x[j]))
    return float(acc)


def sum_fsum(x: np.ndarray) -> float:
    """math.fsum 精确和（Shewchuk 算法，无累积舍入）—— FP64 语义锚点。

    与 float64 串行的区别：fsum 对任意顺序给出同一个正确舍入的结果，
    即"顺序无关"的参照系（误差 = 一次最终舍入）。
    """
    return math.fsum(float(v) * float(v) for v in x)


ORDER_FUNCTIONS = {
    "serial_l2r": sum_serial,
    "serial_r2l": sum_r2l,
    "pairwise_tree": sum_pairwise,
    "numpy_pairwise": sum_numpy,
    "fsum_exact": lambda x, **kw: sum_fsum(x),
}


# ---------------------------------------------------------------------------
# RMSNorm 级：把平方和的顺序差异传播到输出（scale = rsqrt(mean+eps)）
# ---------------------------------------------------------------------------

def rmsnorm_with_sqsum(x: np.ndarray, weight: np.ndarray, sqsum: float, h: int,
                       eps: float = 1e-6) -> np.ndarray:
    """给定（任意顺序算出的）平方和，完成 RMSNorm 其余步骤（FP32）。

    把"归约顺序漂移"单独隔离：scale 差 = 0.5 × 相对平方和差（rsqrt 的
    一阶展开），乘到整行输出。
    """
    scale = np.float32(1.0) / np.float32(np.sqrt(np.float32(sqsum / np.float32(h)) + np.float32(eps)))
    return (np.float32(scale) * x * weight).astype(np.float32)


def shuffle_drift_distribution(x: np.ndarray, reference_sqsum: float, seeds: int,
                               acc_dtype_name: str = "fp32",
                               seed_base: int = 20260801) -> dict:
    """P 个随机到达顺序的漂移分布（atomic 乱序的一阶模型统计量）。

    返回 min/max/mean/std 相对误差 + 卡死计数（结果 == 冻结值判据：
    与前一种子完全同值即可能冻结——严格判据用 fp16 的 ulp 关系）。
    """
    rel_errors = []
    values = []
    for i in range(seeds):
        v = sum_shuffled(x, seed_base + i, acc_dtype_name)
        values.append(v)
        rel_errors.append(abs(v - reference_sqsum) / reference_sqsum if reference_sqsum else math.inf)
    return {
        "seeds": seeds,
        "acc_dtype": acc_dtype_name,
        "rel_error_min": min(rel_errors),
        "rel_error_max": max(rel_errors),
        "rel_error_mean": float(np.mean(rel_errors)),
        "rel_error_std": float(np.std(rel_errors)),
        "unique_values": len(set(values)),
        "sqsum_min": min(values),
        "sqsum_max": max(values),
    }


def ulp_of(acc_dtype_name: str, value: float) -> float:
    """给定累加器 dtype 与当前值，返回 ulp(acc)（教学/卡死区判据用）。"""
    if acc_dtype_name == "fp16":
        return float(np.spacing(np.float16(value)))
    if acc_dtype_name == "fp32":
        return float(np.spacing(np.float32(value)))
    return float(np.spacing(np.float64(value)))
