#!/usr/bin/env python3
"""实验 1：FP16 vs BF16 位布局对照（本地 CPU，无 GPU 依赖）。

产出 results/dtype-layout.json：
1. 格式边界表（exp/man/尾数位/偏置，位型重算 vs torch.finfo 双源核对）；
2. 构造值对照表：大数/小数/subnormal/边界值的位分解、舍入误差、溢出/下溢行为
   （直接作为 EXPERIMENT_LOG 对照表的数据源）；
3. 随机舍入误差统计：多个量级区间的 FP32->FP16/BF16 相对舍入误差
   （理论上限 2^-11 vs 2^-8，比值 8）；
4. ulp 表：同指数区间粒度对照（bf16 粗 8 倍）；
5. 整数可分辨边界（fp16 2^11 以上丢整数 / bf16 2^8 以上丢）；
6. RMSNorm 语境的引子值：65504（fp16 max）在两格式的命运、
   1.4·2^-24（subnormal 精度塌缩）等。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys_path = HERE / "implementations" / "cpu"

import sys  # noqa: E402

sys.path.insert(0, str(sys_path))

from dtype_layout import (  # noqa: E402
    FORMATS,
    format_boundaries,
    integer_precision_boundaries,
    random_roundtrip_stats,
    ulp_at,
    value_table,
)

CONSTRUCTED_VALUES = [
    1.0, -1.0, 0.5, 2.5, 0.1,                     # 常规值
    3.14159265, 2.71828183,                       # 无理数（舍入演示）
    1.0000001, 1.0009765625,                      # 1 附近的半 ulp（RNE tie 演示）
    2047.0, 2048.0, 2049.0, 4097.0,               # fp16 整数精度边界
    255.0, 256.0, 257.0,                          # bf16 整数精度边界
    65504.0, 65519.0, 65536.0, 70000.0,           # fp16 溢出区
    3.0e38, 3.5e38, 1.0e39,                       # bf16 溢出区
    1.0e-3, 1.0e-5, 6.0e-5, 1.0e-8,               # 小数区
    2.0**-14, 2.0**-15, 2.0**-20,                 # fp16 normal/subnormal 边界
    1.4 * 2.0**-24, 2.0**-24, 2.0**-25, 1.0e-45,  # fp16 subnormal/下溢
    2.0**-126, 2.0**-127, 1.0e-40, 1.0e-45,       # bf16 subnormal 区
]

RANGES = [(0.5, 1.0), (1.0, 2.0), (1.0e-3, 1.0e-2), (1.0e-8, 1.0e-7),
          (100.0, 200.0), (1.0e4, 2.0e4), (1.0e30, 2.0e30), (1.0e-38, 2.0e-38)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/dtype-layout.json")
    arguments = parser.parse_args()

    rng = np.random.default_rng(20260601)

    boundaries = format_boundaries()
    print("[layout] 格式边界（位型重算 vs torch.finfo 双源核对）:")
    for name, b in boundaries.items():
        print(f"  {name}: max_normal={b['max_normal_bits_recomputed']:.6e} (match={b['max_normal_match']})"
              f" min_normal={b['min_normal']:.3e} min_subnormal={b['min_subnormal']:.3e}"
              f" exp_bits={b['exp_bits']} man_stored={b['man_bits_stored']}"
              f" man_effective={b['man_bits_effective']} bias={b['exponent_bias']}")

    table = value_table(CONSTRUCTED_VALUES)
    print(f"[layout] 构造值表 {len(table)} 行（前 5 行）:")
    for row in table[:5]:
        print(f"  {row['value_repr']}: fp16[{row['fp16_class']}] rel={row['fp16_rel_error']:.2e} | "
              f"bf16[{row['bf16_class']}] rel={row['bf16_rel_error']:.2e}")

    roundtrip_stats = [random_roundtrip_stats(rng, lo, hi, n=200000) for lo, hi in RANGES]
    print("[layout] 随机舍入误差统计（各量级区间）:")
    for stat in roundtrip_stats:
        print(f"  {stat['range']}: fp16 max_rel={stat['fp16']['max_rel_error']:.3e} "
              f"(theory={stat['fp16']['theoretical_max_rel']:.3e}) | "
              f"bf16 max_rel={stat['bf16']['max_rel_error']:.3e} "
              f"(theory={stat['bf16']['theoretical_max_rel']:.3e}) | "
              f"ratio={stat['bf16']['max_rel_error'] / max(stat['fp16']['max_rel_error'], 1e-300):.1f}x")

    ulp_table = []
    for p in range(-25, 40):
        entry = {"exponent_power": p}
        for name in ("fp16", "bf16"):
            entry[f"ulp_{name}"] = ulp_at(p, name)
        entry["ratio_bf16_over_fp16"] = entry["ulp_bf16"] / entry["ulp_fp16"]
        ulp_table.append(entry)

    int_cases = integer_precision_boundaries()
    first_fp16_loss = next((c["integer"] for c in int_cases if not c["fp16_exact"]), None)
    first_bf16_loss = next((c["integer"] for c in int_cases if not c["bf16_exact"]), None)
    print(f"[layout] 整数精度：fp16 首个失真整数={first_fp16_loss}，bf16 首个失真整数={first_bf16_loss}")

    payload = {
        "schema_version": "kernel-chapter06.dtype-layout.v1",
        "formats": FORMATS,
        "note": "尾数口径：stored man_bits fp16=10 / bf16=7；有效精度（含隐含位）11 vs 8 ——"
                "大纲'尾数 10 vs 8 位'采用有效精度口径；ulp 比恒为 8 倍（粒度差 3 位）",
        "boundaries": boundaries,
        "constructed_value_table": table,
        "roundtrip_stats": roundtrip_stats,
        "ulp_table_power_-25_39": ulp_table,
        "integer_precision": {"cases": int_cases,
                              "first_fp16_loss": first_fp16_loss,
                              "first_bf16_loss": first_bf16_loss},
        "rmsnorm_context_values": {
            "fp16_max_65504": next(r for r in table if r["value"] == 65504.0),
            "subnormal_collapse_1p4_2en24": next(
                r for r in table if abs(r["value"] - 1.4 * 2.0**-24) < 1e-45),
        },
    }
    out = HERE / arguments.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[done] 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
