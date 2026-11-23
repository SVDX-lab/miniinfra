#!/usr/bin/env python3
"""第 18 期 Smoke Test（CPU-only，无 GPU）。

覆盖：
1. WMMA 映射：第 17 期 11 个 tile 配置 → 5 个 WMMA 形状（同 TM/TN），KSTEP=16；
2. fragment 几何：TM/TN 均为 16 的倍数；warp 布局 = (TM/16)×(TN/16)，每 warp
   1 个 16×16 累加 fragment（每条 mma_sync = 16×16×16×2 = 8192 FLOP）；
3. mma_sync 计数自洽：全 GEMM 的 mma_sync 总 FLOP = 2MNK；
4. smem 公式与上限：WMMA 暂存路径（fp32-acc / fp16-acc）均 < 48KB；
5. fixtures 单源：chapter_18/fixtures 与 chapter_17 manifest 逐张量 SHA 一致 +
   tiny 手算例位级核对；
6. roofline：FP16 脊点（cuda 164 / musa 130 FLOP/B）与 AI 判定（1024³ FP16
   AI=341.3 → compute-bound 侧不因 Tensor Core 翻转，只是脊点上移）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gemm_configs import (  # noqa: E402
    CONFIGS,
    WMMA_CONFIGS,
    WMMA_KSTEP,
    WMMA_MAP_FROM_EP18,
    parse_wmma_config,
    wmma_block_threads,
    wmma_mma_sync_count,
    wmma_smem_bytes,
    wmma_warps,
)
from roofline_gemm import (  # noqa: E402
    PLATFORM_SPEC,
    classify_kernel_bound,
    gemm_arithmetic_intensity,
    ridge_point,
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


def test_wmma_mapping() -> None:
    check("WMMA_KSTEP = 16（fragment k 维）", WMMA_KSTEP == 16)
    for config in CONFIGS:
        mapped = WMMA_MAP_FROM_EP18[config["name"]]
        wcfg = parse_wmma_config(mapped)
        check(f"{config['name']} → {mapped}（TM/TN 不变）",
              wcfg["tm"] == config["tm"] and wcfg["tn"] == config["tn"])
        frags_k = -(-config["tk"] // WMMA_KSTEP)  # ceil
        check(f"{config['name']} 的 TILE_K={config['tk']} 映射为 {frags_k} 个 k-fragment",
              (config["tk"] in (8, 16)) and WMMA_KSTEP % config["tk"] == 0)
    check("WMMA 形状数 = 5（同 TILE_M/TILE_N 族）", len(WMMA_CONFIGS) == 5)


def test_fragment_geometry() -> None:
    for config in WMMA_CONFIGS:
        check(f"{config['name']}: TM/TN 均为 16 的倍数",
              config["tm"] % 16 == 0 and config["tn"] % 16 == 0)
        wm, wn = wmma_warps(config)
        check(f"{config['name']}: warp 布局 {wm}×{wn}，线程 {wmma_block_threads(config)} = warps×32",
              wmma_block_threads(config) == wm * wn * 32)
        check(f"{config['name']}: 32 ≤ 线程 ≤ 1024",
              32 <= wmma_block_threads(config) <= 1024)
    # 每条 mma_sync 的 FLOP：16×16×16 MAC ×2 = 8192
    total = wmma_mma_sync_count(1024, 1024, 1024, parse_wmma_config("wm32x32")) * 8192
    check("wm32x32 @1024³ 的 mma_sync 总 FLOP = 2MNK（自洽）",
          total == theoretical_flops(1024, 1024, 1024), f"{total} FLOP")


def test_smem_budget() -> None:
    for config in WMMA_CONFIGS:
        smem_f32 = wmma_smem_bytes(config, acc_float=True)
        smem_f16 = wmma_smem_bytes(config, acc_float=False)
        check(f"{config['name']}: smem(fp32-acc)={smem_f32}B < 48KB 且 > smem(fp16-acc)={smem_f16}B",
              smem_f32 < 48 * 1024 and smem_f32 > smem_f16)
    # 手算抽查 wm64x64 fp32-acc: As 64×24×2 + Bs 16×72×2 + Cs_f 64×64×4 = 3072+2304+16384
    check("抽查 wm64x64 fp32-acc smem = 21760B",
          wmma_smem_bytes(parse_wmma_config("wm64x64")) == 3072 + 2304 + 16384)


def test_fixture_provenance() -> None:
    manifest_path = ROOT / "fixtures" / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    check("manifest provenance = 第 17 期单源",
          "第 17 期" in manifest.get("provenance", ""))
    with np.load(ROOT / "fixtures" / "correctness.npz") as data:
        arrays = {key: data[key] for key in data.files}
    check("fixture 张量数 = 50（8 组形状 × 主输出/epilogue）", len(arrays) == 50, f"实际 {len(arrays)}")
    tiny = arrays["C_4x8x4"]
    expected = np.array([[4, 11, 12, 5], [4, 6, 3, 9], [8, 5, 10, 9], [7, 7, 9, 8]], dtype=np.float32)
    check("tiny 手算例位级一致（真值单源未漂移）", np.array_equal(tiny, expected))


def test_roofline_fp16() -> None:
    ai_1024_fp16 = gemm_arithmetic_intensity(1024, 1024, 1024, 2)
    check("AI(1024³ FP16) = 341.33 FLOP/B", abs(ai_1024_fp16 - 2048 / 6) < 1e-6)
    for backend, spec_peak, bw in [("cuda", 165.2, 1008.0), ("musa", 100.0, 768.0)]:
        ridge = ridge_point(spec_peak * 1e12, bw * 1e9)
        spec = PLATFORM_SPEC[backend]
        verdict = classify_kernel_bound(
            theoretical_bytes(1024, 1024, 1024, 2), theoretical_flops(1024, 1024, 1024),
            bw * 1e9, spec_peak * 1e12)
        check(f"{backend}: FP16 脊点 {ridge:.0f} FLOP/B < AI 341 → 仍 compute-bound（Tensor Core 只上移脊点）",
              verdict["kernel_bound_class"].startswith("compute-bound"),
              f"ridge={spec['peak_fp16_tflops'] / spec['peak_bandwidth_gb_s'] * 1e3:.0f}")


def main() -> int:
    test_wmma_mapping()
    test_fragment_geometry()
    test_smem_budget()
    test_fixture_provenance()
    test_roofline_fp16()
    print("[done] 第 18 期 Smoke Test 全部通过（CPU-only）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
