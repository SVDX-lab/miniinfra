#!/usr/bin/env python3
"""第 19 期 Smoke Test（不占 GPU，本地/服务器 CPU 通用）。

覆盖：
1. 手算极小例 4×8·8×4 vs 真值实现（位级整数一致，真值 = ep18 同源 CPU 朴素实现）；
2. 真值实现两层（triple loop vs naive rows）一致性与确定性；
3. tile 配置表自洽：线程数 ≤1024、warpspec 线程数 = 计算槽 + 32、
   smem 公式与双缓冲预算 ≤ MUSA 静态上限（M-43）与 CUDA 48KB 静态上限；
4. shape 交叉点实验的网格设计自检（K 深度扫描的 steps 数、快路条件判定）；
5. cp.async 布局对齐约束的纸面校验：+1 padding 布局 4B 拷贝对齐、
   +4 padding 布局 16B 拷贝行对齐（基准形状全整除 → 快路生效）；
6. roofline 手算数（GEMM AI，与 ch02/ep18 同式）。

无 CPU 性能列（期首声明：设备侧数据供给组织概念，CPU 不参与计时口径）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "implementations" / "cpu"))

from naive_gemm import (  # noqa: E402
    TINY_A,
    TINY_B,
    matmul_naive_rows,
    matmul_triple_loop,
)
from async_gemm_configs import (  # noqa: E402
    CONFIGS,
    MUSA_STATIC_SMEM_LIMIT,
    SCALAR_LEVELS,
    V16_CONFIGS,
    block_dims,
    grid_dims,
    kernel_supports_config,
    smem_bytes,
    smem_bytes_double,
)
from roofline_gemm import (  # noqa: E402
    gemm_arithmetic_intensity,
    theoretical_bytes,
    theoretical_flops,
)

PASS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise SystemExit(f"Smoke Test 失败于: {name}")
    PASS += 1


def test_truth_reference() -> None:
    # 手算核对（与 ep18 同一 tiny 例，真值链单源）
    triple = matmul_triple_loop(TINY_A, TINY_B)
    naive = matmul_naive_rows(TINY_A, TINY_B)
    check("tiny 例 triple_loop == naive_rows（整数域位级）",
          np.array_equal(triple, naive) and triple.dtype.kind in "fi")
    expected00 = int((TINY_A[0] @ TINY_B[:, 0].astype(TINY_A.dtype)))
    check("tiny 例 C[0,0] 与 numpy 内积一致", int(triple[0, 0]) == expected00,
          f"C[0,0]={triple[0, 0]}")
    rng = np.random.default_rng(7)
    A = rng.standard_normal((17, 13), dtype=np.float32)
    B = rng.standard_normal((13, 11), dtype=np.float32)
    check("ragged 例 triple vs naive（FP32 容差 1e-4）",
          np.allclose(matmul_triple_loop(A, B), matmul_naive_rows(A, B), atol=1e-4))


def test_config_table() -> None:
    from async_gemm_configs import WARPSPEC_EXCLUDED
    for c in CONFIGS:
        bx, by = block_dims(c)
        threads = bx * by
        check(f"{c['name']} 线程数 ≤1024", threads <= 1024, f"{threads}")
        if c["name"] in WARPSPEC_EXCLUDED:
            check(f"{c['name']} warpspec 排除（+32 超 1024 上限）",
                  threads + 32 > 1024, f"{threads + 32}")
        else:
            check(f"{c['name']} warpspec 线程数 ≤1024",
                  threads + 32 <= 1024, f"{threads + 32}")
        smem1 = smem_bytes(c)
        smem2 = smem_bytes_double(c)
        check(f"{c['name']} 单缓冲 smem 与 kernel 公式一致",
              smem1 == (c["tm"] * (c["tk"] + 1) + c["tk"] * (c["tn"] + 1)) * 4, f"{smem1}B")
        check(f"{c['name']} 双缓冲 smem ≤ CUDA 静态 48KB", smem2 <= 48 * 1024, f"{smem2}B")
        check(f"{c['name']} 双缓冲 smem ≤ MUSA 静态上限 {MUSA_STATIC_SMEM_LIMIT}",
              smem2 <= MUSA_STATIC_SMEM_LIMIT, f"{smem2}B")
        # wmma_pipe 组合核 smem（64x64x16 固定）
    wmma_smem = 2 * (64 * 24 + 16 * 72) * 2 + 64 * 64 * 4
    check("wmma_pipe smem ≤ CUDA 静态 48KB", wmma_smem <= 48 * 1024, f"{wmma_smem}B")
    for name in V16_CONFIGS:
        c = next(x for x in CONFIGS if x["name"] == name)
        check(f"v16 {name} +4 padding 行距 16B 对齐（A/B 行）",
              (c["tk"] + 4) % 4 == 0 and (c["tn"] + 4) % 4 == 0, f"{c['tk'] + 4}/{c['tn'] + 4} floats")


def test_fast_path_and_steps() -> None:
    # 快路条件：M/N/K 全为 tile 倍数才走 cp.async 快路；基准形状全整除
    for n in (256, 512, 1024):
        for c in CONFIGS:
            full = all(n % c[dim] == 0 for dim in ("tm", "tn", "tk"))
            check(f"{n}³ 走 {c['name']} cp.async 快路", full)
            gx, gy = grid_dims(c, n, n)
            check(f"{n}³ {c['name']} 网格无量化损耗", gx * c["tn"] == n and gy * c["tm"] == n)
    # K 深度扫描：steps 数（每步一个流水级）
    c = next(x for x in CONFIGS if x["name"] == "32x32x8_reg4x4")
    steps = {k: -(-k // c["tk"]) for k in (8, 64, 256, 1024)}
    check("K 深度扫描 steps 递增（8/64/256/1024 → 1/8/32/128）",
          steps == {8: 1, 64: 8, 256: 32, 1024: 128}, str(steps))
    # K=8 < TK=16 的组合（16x16x16 tile, K=8）不能走快路
    c16 = next(x for x in CONFIGS if x["name"] == "16x16x16_reg4x4")
    check("K=8 非 16 倍数 → 16x16x16 走同步回退路径", 8 % c16["tk"] != 0)


def test_roofline_numbers() -> None:
    ai256 = gemm_arithmetic_intensity(256, 256, 256, 4)
    ai1024 = gemm_arithmetic_intensity(1024, 1024, 1024, 4)
    check("GEMM AI 256³ ≈ 42.7 FLOP/B", abs(ai256 - 42.7) < 0.2, f"{ai256}")
    check("GEMM AI 1024³ ≈ 170.7 FLOP/B", abs(ai1024 - 170.7) < 0.2, f"{ai1024}")
    check("理论 FLOPs 1024³ = 2.147e9",
          theoretical_flops(1024, 1024, 1024) == 2 * 1024**3)
    check("理论字节 1024³ FP32 = 12 MB",
          theoretical_bytes(1024, 1024, 1024, 4) == 3 * 1024 * 1024 * 4)


def test_level_semantics_documents() -> None:
    # 三级递进主对照表 + 期首声明确认（无 CPU 性能列）
    check("三级递进主对照为 sync→cpasync1→cpasync2",
          SCALAR_LEVELS == ["sync", "cpasync1", "cpasync2"], str(SCALAR_LEVELS))
    check("broken 核只挂 reg4x4 演示配置",
          all(kernel_supports_config(k, "32x32x8_reg4x4")
              and not kernel_supports_config(k, "16x16x16_reg2x2")
              for k in ("cpasync2v16", "sync2_broken", "cpasync2_broken")))


def main() -> int:
    test_truth_reference()
    test_config_table()
    test_fast_path_and_steps()
    test_roofline_numbers()
    test_level_semantics_documents()
    print(f"\nSmoke Test 全部通过（{PASS} checks）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
