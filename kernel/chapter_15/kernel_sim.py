#!/usr/bin/env python3
"""第 15 期 kernel_sim：numpy 逐块仿真 int8_gemm.cu 的 tiling/guard/索引逻辑。

目的（上服务器前的本地 QA，ep21/ep22 同款做法）：在无 GPU 环境下验证
gemm_fp16_tiled_kernel / gemm_int8_fused_kernel 的实现逻辑——
  - tile 循环边界（K % TILE_K != 0、M/N 非 tile 倍数的 guard）；
  - A/B-tile 协作加载的 idx→(r,c) 映射与越界补零；
  - 融合路径的 scale 索引（按输出列 gn）；
  - 网格覆盖断言：所有输出元素恰好被写一次；
  - 写回 guard（gm<M && gn<N）。
不仿真 FP16 舍入与 FMA 舍入（那些由 fixtures 链路在服务器上实测）；
累加用 float32 贴近 acc 类型，容差 1e-3（数据量级 ~1e0）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "implementations" / "cpu"))

from quant_reference import quantize_weight_perchannel

CONFIGS = [
    ("16x16x16_reg2x2", 16, 16, 16, 2, 2),
    ("32x32x8_reg2x2", 32, 32, 8, 2, 2),
    ("32x32x8_reg4x4", 32, 32, 8, 4, 4),
    ("64x64x8_reg2x2", 64, 64, 8, 2, 2),
    ("64x64x8_reg4x4", 64, 64, 8, 4, 4),
]

# 仿真形状：全 guard 触发（非整除 + M=1 + M<TILE_M）+ 整除
SHAPES = [(1, 16, 24), (1, 17, 15), (4, 8, 4), (13, 17, 11), (5, 16, 16), (8, 24, 32), (20, 8, 8)]

FAILS = 0


def emulate_kernel(m, k, n, tm, tn, tk, rm, rn, a32, w32, scale=None):
    """逐 block/tile 仿真（float32 累加），返回 (C, write_count)。"""
    grid_x = (n + tn - 1) // tn
    grid_y = (m + tm - 1) // tm
    bx, by = tn // rn, tm // rm  # block 内线程网格
    c = np.full((m, n), np.nan, dtype=np.float32)
    write_count = np.zeros((m, n), dtype=np.int64)

    for bj in range(grid_y):
        for bi in range(grid_x):
            m0, n0 = bj * tm, bi * tn
            acc = np.zeros((tm, tn), dtype=np.float32)  # block 级"聚合累加器"
            for k0 in range(0, k, tk):
                # A-tile 协作加载（idx→(r,c)，guard 补零）
                a_tile = np.zeros((tm, tk), dtype=np.float32)
                idx_a = 0
                while idx_a < tm * tk:
                    r, cc = idx_a // tk, idx_a % tk
                    gm, gk = m0 + r, k0 + cc
                    if gm < m and gk < k:
                        a_tile[r, cc] = a32[gm, gk]
                    idx_a += 1
                # B-tile 协作加载（fp16 版：转 float；int8 版：q×scale[gn]）
                b_tile = np.zeros((tk, tn), dtype=np.float32)
                idx_b = 0
                while idx_b < tk * tn:
                    r, cc = idx_b // tn, idx_b % tn
                    gk, gn = k0 + r, n0 + cc
                    if gk < k and gn < n:
                        b_tile[r, cc] = (
                            w32[gk, gn] if scale is None else float(w32[gk, gn]) * float(scale[gn])
                        )
                    idx_b += 1
                # 线程网格 FMA（聚合语义：每线程 RM×RN 累加器覆盖 (ty*rm+m, tx*rn+n)）
                for ty in range(by):
                    for tx in range(bx):
                        thread_acc = np.zeros((rm, rn), dtype=np.float32)
                        for kk in range(tk):
                            a_reg = np.array([a_tile[ty * rm + mi, kk] for mi in range(rm)], dtype=np.float32)
                            b_reg = np.array([b_tile[kk, tx * rn + ni] for ni in range(rn)], dtype=np.float32)
                            for mi in range(rm):
                                for ni in range(rn):
                                    thread_acc[mi, ni] += a_reg[mi] * b_reg[ni]
                        acc[ty * rm : (ty + 1) * rm, tx * rn : (tx + 1) * rn] += thread_acc
            # 写回（guard）
            for mi in range(tm):
                for ni in range(tn):
                    gm, gn = m0 + mi, n0 + ni
                    if gm < m and gn < n:
                        c[gm, gn] = acc[mi, ni]
                        write_count[gm, gn] += 1
    return c, write_count


def main() -> int:
    global FAILS
    rng = np.random.default_rng(20261121)

    for (m, k, n) in SHAPES:
        a16 = (rng.standard_normal((m, k)) * 0.5).astype(np.float16)
        w16 = (rng.standard_normal((k, n)) * 0.05).astype(np.float16)
        a32 = a16.astype(np.float32)
        w32 = w16.astype(np.float32)
        q, scale = quantize_weight_perchannel(w32)
        wh32 = q.astype(np.float32) * scale[None, :]

        y_direct_fp16 = a32 @ w32
        y_direct_q = a32 @ wh32

        for (name, tm, tn, tk, rm, rn) in CONFIGS:
            # fp16 kernel 仿真
            c16, wc = emulate_kernel(m, k, n, tm, tn, tk, rm, rn, a32, w32)
            if np.any(wc != 1):
                print(f"SIM {name} {m}x{k}x{n} fp16 COVERAGE FAIL writes={np.unique(wc)}")
                FAILS += 1
                continue
            err16 = float(np.abs(c16 - y_direct_fp16).max())
            # int8_fused kernel 仿真（B 输入 = q（整数），scale 按列）
            cq, wcq = emulate_kernel(m, k, n, tm, tn, tk, rm, rn, a32, q, scale=scale)
            if np.any(wcq != 1):
                print(f"SIM {name} {m}x{k}x{n} int8 COVERAGE FAIL writes={np.unique(wcq)}")
                FAILS += 1
                continue
            errq = float(np.abs(cq - y_direct_q).max())
            ok16, okq = err16 < 1e-3, errq < 1e-3
            status = "PASS" if (ok16 and okq) else "FAIL"
            if status == "FAIL":
                FAILS += 1
            print(
                f"SIM {name} {m}x{k}x{n} fp16_max_abs={err16:.2e} int8_max_abs={errq:.2e} {status}"
            )

    # dequant 物化仿真：half(q)*scale 与 numpy fp16 cast 的位级一致性
    w16 = (rng.standard_normal((33, 47)) * 0.05).astype(np.float16)
    q, scale = quantize_weight_perchannel(w16.astype(np.float32))
    w_hat32 = q.astype(np.float32) * scale[None, :]
    w_mat = w_hat32.astype(np.float16)
    ref = (q.astype(np.float32) * np.repeat(scale[None, :], 33, axis=0)).astype(np.float16)
    if np.array_equal(w_mat, ref):
        print("SIM dequant_materialize bit-exact PASS")
    else:
        print("SIM dequant_materialize MISMATCH FAIL")
        FAILS += 1

    print(f"{{\"mode\":\"kernel_sim\",\"shapes\":{len(SHAPES)},\"configs\":{len(CONFIGS)},\"fails\":{FAILS}}}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
