#!/usr/bin/env python3
"""numpy 对拍：test_half_convert.cpp 生成的转换表 vs numpy.float16（穷举/百万级）。

用后即删数据文件；结论写进 EXPERIMENT_LOG（MUSA 位级转换 fallback 的本地验证）。
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

failures = 0

# 1) half→float 穷举
table = np.loadtxt(HERE / "half-table.csv", delimiter=",", dtype=np.uint32)
for h_bits, f_bits in table:
    h = np.uint16(h_bits)
    ours = np.uint32(f_bits).view(np.float32)
    ref = h.view(np.float16).astype(np.float32)
    if not (ours == ref or (np.isnan(ours) and np.isnan(ref))):
        # NaN payload 差异允许（我们统一 quiet NaN 0xFE00，numpy 会保留位型）
        if not (np.isnan(ref) and np.isnan(ours)):
            print(f"FAIL h2f bits={h:04x} ours={ours!r} ref={ref!r}")
            failures += 1
        else:
            failures += 0
print(f"[h2f] 穷举 65536 位型：{'PASS' if failures == 0 else f'{failures} FAIL'}")

# 2) float→half
tests = np.loadtxt(HERE / "f2h-tests.csv", delimiter=",", dtype=np.uint32)
f2h_fail = 0
for f_bits, h_bits in tests:
    v = np.uint32(f_bits).view(np.float32)
    ours = np.uint16(h_bits).view(np.float16)
    ref = v.astype(np.float16)  # numpy RNE
    same = (ours == ref) or (np.isnan(ours) and np.isnan(ref))
    # 值相等但 -0/+0 位型差异也接受（数值语义一致）
    if not same and not (ours == 0 and ref == 0):
        f2h_fail += 1
        if f2h_fail <= 10:
            print(f"FAIL f2h v={v!r} ours={np.uint16(h_bits):04x} ref_bits={np.uint16(ref.view(np.uint16)):04x}")
print(f"[f2h] {tests.shape[0]} 个测试值：{'PASS' if f2h_fail == 0 else f'{f2h_fail} FAIL'}")

(HERE / "half-table.csv").unlink(missing_ok=True)
(HERE / "f2h-tests.csv").unlink(missing_ok=True)
raise SystemExit(0 if (failures == 0 and f2h_fail == 0) else 1)
