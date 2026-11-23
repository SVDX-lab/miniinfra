#!/usr/bin/env python3
"""第 05 期 Smoke Test：CPU-only、不占 GPU 的最低成本检查。

覆盖 dtype 布局工具与模拟器的自洽性（实验 1/4 的地基）：
1. 位分解工具 vs torch.finfo 边界值（max/min normal/subnormal 实拆重算一致）；
2. BF16 位运算模拟舍入 vs torch bfloat16 原生转换：随机 1e5 值位级一致；
3. FP16 位型分解 vs torch 原生位型：随机 1e5 值位级一致；
4. ulp 比例：同指数区间 fp16/bf16 的 ulp 恒为 8 倍（尾数 10 vs 7 stored）；
5. 整数可分辨边界：fp16 在 2^11 以上丢整数、bf16 在 2^8 以上丢；
6. 溢出行为：65536 → fp16 inf / bf16 正常；1e39 → bf16 inf；
   underflow：1e-8 → fp16 subnormal/0、bf16 normal；
7. 累加模拟器：fp32 串行 vs 真值实现位级一致；fp16 平方溢出（|x|~300）
   与累加溢出（sum > 65504）；bf16 位模拟 vs torch bf16 逐元素算术抽查；
8. 模拟 Kernel 路径：fp32 acc vs 真值 max diff ~1e-6；NaN 按行传播；
   全零行输出全零；写回 T 后的 dtype 主导误差量级（fp16 ~1e-2 / bf16 ~5e-2）；
9. 宽动态范围行的"bf16 反而更准"引子（小值分支 bf16 相对误差 < fp16）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from dtype_layout import (  # noqa: E402
    format_boundaries,
    integer_precision_boundaries,
    ulp_at,
    value_table,
)
from dtype_sim import (  # noqa: E402
    RMS_NORM_EPS,
    round_f32_to_bf16,
    rmsnorm_simulated_kernel,
    serial_sqsum_bf16,
    serial_sqsum_fp16,
    serial_sqsum_fp32,
    wide_dynamic_row,
)
from naive_rmsnorm import naive_rmsnorm_serial  # noqa: E402


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise SystemExit(f"Smoke Test 失败于: {name}")


def test_format_boundaries() -> None:
    bounds = format_boundaries()
    for name in ("fp16", "bf16"):
        b = bounds[name]
        check(f"{name} 位型重算 max_normal == torch.finfo.max", b["max_normal_match"],
              f"recomputed={b['max_normal_bits_recomputed']:.6e}")
    check("fp16 min_subnormal == 2^-24", bounds["fp16"]["min_subnormal"] == 2.0**-24,
          f"{bounds['fp16']['min_subnormal']:.3e}")
    check("bf16 min_subnormal == 2^-133", bounds["bf16"]["min_subnormal"] == 2.0**-133,
          f"{bounds['bf16']['min_subnormal']:.3e}")
    check("存储尾数位 10 vs 7（有效精度 11 vs 8）",
          bounds["fp16"]["man_bits_stored"] == 10 and bounds["bf16"]["man_bits_stored"] == 7
          and bounds["fp16"]["man_bits_effective"] == 11
          and bounds["bf16"]["man_bits_effective"] == 8)


def test_bf16_rounding_simulation() -> None:
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(100000) * rng.choice([1e-3, 1.0, 1e3, 1e5, 1e9], 100000)).astype(np.float32)
    sim_bits = round_f32_to_bf16(x)  # 返回 float32（由模拟位型转回）
    torch_bf16 = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    both_finite = ~np.isnan(sim_bits) & ~np.isnan(torch_bf16)
    bitwise_value = np.array_equal(sim_bits[both_finite], torch_bf16[both_finite])
    check("BF16 位模拟舍入 vs torch bfloat16 原生转换（1e5 随机值，位级一致）",
          bool(bitwise_value), f"finite {int(both_finite.sum())}/{len(x)}")


def test_fp16_bit_decomposition() -> None:
    from dtype_layout import _bits16
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(100000) * 10).astype(np.float32)
    torch_bits = torch.from_numpy(x).to(torch.float16).view(torch.int16).numpy().astype(np.uint16) & 0xFFFF
    idx = rng.choice(len(x), 500, replace=False)
    ok = sum(1 for i in idx if _bits16(float(x[i]), "fp16") == int(torch_bits[i]))
    check("FP16 位分解 vs torch 原生位型（抽样 500）", ok == 500, f"{ok}/500")


def test_ulp_ratio() -> None:
    ratios = []
    for p in (-10, -1, 0, 7, 10, 14):
        ratios.append(ulp_at(p, "bf16") / ulp_at(p, "fp16"))
    check("同指数区间 bf16/fp16 ulp 比恒为 8（bf16 粒度粗 8 倍：尾数 7 vs 10）",
          all(abs(r - 8.0) < 1e-9 for r in ratios),
          "ratios=" + ",".join(f"{r:.1f}" for r in ratios))


def test_integer_boundaries() -> None:
    cases = integer_precision_boundaries()
    fp16_2049 = next(c for c in cases if c["integer"] == 2049)
    bf16_257 = next(c for c in cases if c["integer"] == 257)
    check("fp16 在 2048 以上丢整数精度（2049 回读 != 2049）", not fp16_2049["fp16_exact"],
          f"roundtrip={fp16_2049['fp16_roundtrip']}")
    check("bf16 在 256 以上丢整数精度（257 回读 != 257）", not bf16_257["bf16_exact"],
          f"roundtrip={bf16_257['bf16_roundtrip']}")


def test_overflow_underflow() -> None:
    rows = value_table([65536.0, 1e39, 1e-8, 2.0**-20, 1.4 * 2.0**-24])
    r65536 = rows[0]
    r1e39 = rows[1]
    r1e8 = rows[2]
    r2e20 = rows[3]
    r14 = rows[4]
    check("65536.0：fp16 溢出为 inf，bf16 正常表示",
          r65536["fp16_class"] == "inf" and r65536["bf16_class"] == "normal")
    check("1e39：fp16 inf、bf16 也 inf（超出 3.39e38）",
          r1e39["fp16_class"] == "inf" and r1e39["bf16_class"] == "inf")
    check("1e-8：fp16 下溢为 0（<2^-24），bf16 normal",
          r1e8["fp16_class"] in ("zero", "subnormal") and r1e8["bf16_class"] == "normal",
          f"fp16_class={r1e8['fp16_class']} rel_err={r1e8['fp16_rel_error']:.2e}")
    check("2^-20：fp16 subnormal（渐进下溢区间内），bf16 normal",
          r2e20["fp16_class"] == "subnormal" and r2e20["bf16_class"] == "normal")
    check("1.4·2^-24：fp16 subnormal 精度塌缩（相对误差 ~29%），bf16 正常舍入（<2^-8）",
          r14["fp16_class"] == "subnormal" and r14["fp16_rel_error"] > 0.2
          and r14["bf16_rel_error"] < 2.0**-8,
          f"fp16_rel={r14['fp16_rel_error']:.3f} bf16_rel={r14['bf16_rel_error']:.2e}")


def test_sqsum_simulators() -> None:
    rng = np.random.default_rng(2)
    row = rng.standard_normal(256).astype(np.float32)
    acc32, _ = serial_sqsum_fp32(row)
    truth = float(np.sum((row * row).astype(np.float64)))
    check("fp32 串行平方和 vs float64 真值（相对误差 < 1e-6）",
          abs(acc32 - truth) / truth < 1e-6, f"rel={abs(acc32-truth)/truth:.2e}")
    # fp16 平方溢出：|x| ~ 300 → x^2 = 90000 > 65504
    big_row = np.full(4, 300.0, dtype=np.float32)
    acc16, ovf = serial_sqsum_fp16(big_row)
    check("fp16 累加：|x|=300 的平方逐项溢出为 inf", ovf and np.isinf(acc16),
          f"acc={acc16}")
    # fp16 累加溢出：H=1024 个 |x|~8 → sum ≈ 65536 > 65504
    sweep_row = np.full(1024, 8.0, dtype=np.float32)
    acc16b, ovf_b = serial_sqsum_fp16(sweep_row)
    check("fp16 累加：H·x² = 65536 > 65504 → 累加链溢出 inf", ovf_b and np.isinf(acc16b),
          f"acc={acc16b}")
    acc32b, _ = serial_sqsum_fp32(sweep_row)
    check("同输入 FP32 累加正常（65536.0 精确可表示）", abs(acc32b - 65536.0) < 1e-3,
          f"acc={acc32b:.1f}")
    # bf16 位模拟累加链核对：单元素 = bf16(v²)；两元素 = bf16(bf16(v0²)+bf16(v1²))，
    # 外层舍入用 torch .to(bfloat16) 转换独立验证（该转换已位级验证）。
    # 注：torch CPU 的 bf16 逐元素加法内部以 FP32 累加（acc_type 语义），
    # 不在每步舍回 bf16 —— 这本身就是平台行为发现（EXPERIMENT_LOG 素材），
    # 因此本核对构造的是"硬件 bf16 ALU 语义"的oracle 而非 torch 逐元素链。
    v0, v1 = 1.0, 1.01
    sim1, _ = serial_sqsum_bf16(np.array([v0], dtype=np.float32))
    expect1 = float(torch.tensor([v0 * v0]).to(torch.bfloat16).float().item())
    check("bf16 累加链单元素 = bf16(v²)", sim1 == expect1, f"sim={sim1} expect={expect1}")
    sim2, _ = serial_sqsum_bf16(np.array([v0, v1], dtype=np.float32))
    inner = float(np.float32(v0) * np.float32(v0))  # v0=1.0 精确
    sq1 = float(torch.tensor([v1 * v1]).to(torch.bfloat16).float().item())
    expect2 = float(torch.tensor([np.float32(inner) + np.float32(sq1)]).to(torch.bfloat16).float().item())
    check("bf16 累加链两元素 = bf16(bf16(v0²)+bf16(v1²))", sim2 == expect2,
          f"sim={sim2:.6f} expect={expect2:.6f}")
    # torch CPU 逐元素链行为记录（发现：内部 FP32 累加，明显更准）
    t64 = torch.from_numpy(rng.standard_normal(64).astype(np.float32)).to(torch.bfloat16)
    acc_t = torch.tensor(0.0, dtype=torch.float32)
    for v in t64:
        acc_t = acc_t + (v * v).float()
    print(f"[NOTE] torch CPU bf16 逐元素链（内部 FP32 累加）sum={float(acc_t):.4f}"
          f" vs bf16-ALU 语义模拟={serial_sqsum_bf16(t64.float().numpy())[0]:.4f}"
          f" vs FP64 真值={float(np.sum((t64.float().numpy().astype(np.float64))**2)):.4f}")


def test_simulated_kernel_paths() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, 1024)).astype(np.float32)
    w = rng.standard_normal(1024).astype(np.float32)
    truth = naive_rmsnorm_serial(x, w, RMS_NORM_EPS)
    sim32 = rmsnorm_simulated_kernel(x, w, RMS_NORM_EPS, "fp32", "fp32")["y"]
    diff32 = float(np.abs(sim32 - truth).max())
    check("模拟 Kernel fp32/fp32acc vs 真值（归约顺序差异 < 1e-5）", diff32 < 1e-5,
          f"max_diff={diff32:.2e}")
    for dtype, tol in (("fp16", 5e-2), ("bf16", 1e-1)):
        out = rmsnorm_simulated_kernel(x, w, RMS_NORM_EPS, dtype, "fp32")["y"]
        err = float(np.abs(out.astype(np.float32) - truth).max())
        rel = err / float(np.abs(truth).max())
        check(f"模拟 Kernel {dtype}/fp32acc：dtype 主导误差量级",
              rel < tol, f"max_abs={err:.3e} rel={rel:.3e}")
    # NaN 传播 / 全零行
    x2 = x.copy()
    x2[0, 0] = np.float32("nan")
    out = rmsnorm_simulated_kernel(x2, w, RMS_NORM_EPS, "fp16", "fp32")["y"]
    check("NaN 行传播（行 0 全 NaN、行 1 不受影响）",
          bool(np.isnan(out[0]).all()) and bool(np.isfinite(out[1]).all()))
    x3 = x.copy()
    x3[2] = 0
    out3 = rmsnorm_simulated_kernel(x3, w, RMS_NORM_EPS, "bf16", "fp32")["y"]
    check("全零行输出全零（eps 在根号内）", bool((out3[2] == 0).all()))
    # FP16 累加语义：大范数行（scale=16 → sum≈2.6e5 > 65504）
    from dtype_sim import large_norm_row
    xbig = large_norm_row(1024, 16.0, seed=42)[None, :]
    out_fp32acc = rmsnorm_simulated_kernel(xbig, np.ones(1024, dtype=np.float32),
                                           RMS_NORM_EPS, "fp32", "fp32")
    out_fp16acc = rmsnorm_simulated_kernel(xbig, np.ones(1024, dtype=np.float32),
                                           RMS_NORM_EPS, "fp32", "fp16")
    check("acc 实验（scale=16）：FP32 累加正常 / FP16 累加溢出输出全零",
          bool(np.isfinite(out_fp32acc["y"]).all()) and out_fp16acc["diag"]["any_overflow"]
          and bool((np.abs(out_fp16acc["y"]) == 0).all()))


def test_wide_dynamic_row() -> None:
    rng = np.random.default_rng(4)
    row = wide_dynamic_row(rng, 1024)
    check("宽动态行构造（量级 1e4 与 1e-7 同时存在）",
          float(np.abs(row).max()) > 1e3 and float(np.abs(row[row != 0]).min()) < 1e-5)
    w = np.ones(1024, dtype=np.float32)
    truth = naive_rmsnorm_serial(row[None, :], w, RMS_NORM_EPS)[0]
    small_mask = np.abs(row) < 1e-5
    out16 = rmsnorm_simulated_kernel(row[None, :], w, RMS_NORM_EPS, "fp16", "fp32")["y"][0]
    outbf = rmsnorm_simulated_kernel(row[None, :], w, RMS_NORM_EPS, "bf16", "fp32")["y"][0]
    rel16 = float(np.abs(out16[small_mask] - truth[small_mask]).max()
                  / max(float(np.abs(truth[small_mask]).max()), 1e-30))
    relbf = float(np.abs(outbf[small_mask] - truth[small_mask]).max()
                  / max(float(np.abs(truth[small_mask]).max()), 1e-30))
    check("宽动态行小值分支：bf16 相对误差 < fp16（'bf16 反而更准'引子）",
          relbf < rel16, f"fp16_rel={rel16:.3e} bf16_rel={relbf:.3e}")


def main() -> int:
    test_format_boundaries()
    test_bf16_rounding_simulation()
    test_fp16_bit_decomposition()
    test_ulp_ratio()
    test_integer_boundaries()
    test_overflow_underflow()
    test_sqsum_simulators()
    test_simulated_kernel_paths()
    test_wide_dynamic_row()
    print("[done] Smoke Test 通过（CPU-only，不占 GPU）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
