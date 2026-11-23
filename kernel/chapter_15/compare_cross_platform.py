#!/usr/bin/env python3
"""第 15 期 MUSA ↔ CUDA 结果合并对照（回填分析用，本地只读两个 results JSON）。

产出：
- 误差分层对照表（Lb/Lc 每组两平台 + CPU La 层）；
- 性能对照表（最优配置 P50 两平台 + 延迟比 int8/fp16 各自平台内）；
- dequant v1/v2/v3 跨平台对照（M-47 闭合判定：CUDA v1≈v3 → 升格跨平台语义差异）；
- decode INT8/FP16 兑现率对照（MUSA 掩埋 vs 4090 兑现度）。
输出 stdout（markdown 表），供 EXPERIMENT_LOG 回填引用。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load(name):
    path = ROOT / "results" / name
    if not path.exists():
        return {"records": [], "entries": []}
    return json.load(open(path))


def best_by(entries, filt):
    best = None
    for e in entries:
        if filt(e) and (best is None or e["p50_ms"] < best["p50_ms"]):
            best = e
    return best


def main() -> int:
    cpu = load("correctness-results-cpu.json")
    musa_c = load("correctness-results-musa.json")
    cuda_c = load("correctness-results-cuda.json")
    musa_b = load("benchmark-results-musa.json")
    cuda_b = load("benchmark-results-cuda.json")

    la = {f"{r['m']}x{r['k']}x{r['n']}": r for r in cpu["records"]}
    mc = {f"{r['m']}x{r['k']}x{r['n']}": r for r in musa_c["records"]}
    cc = {f"{r['m']}x{r['k']}x{r['n']}": r for r in cuda_c["records"]}

    print("## 误差分层对照（每格 4 配置最差值）\n")
    print("| 组 | La1(CPU) | Lb-MUSA | Lb-CUDA | Lb−La1(CUDA) | Lc-MUSA | Lc-CUDA |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(la, key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1]))):
        r = la[key]
        la1 = r["layers"]["La1_quant_only_fp32"]["max_abs_error"]

        def worst(rec, field):
            if key not in rec:
                return None
            vals = [v[field]["max_abs_error"] for v in rec[key]["kernels"].values()]
            return max(vals)

        lbm, lbc = worst(mc, "Lb_int8_fused_vs_truth"), worst(cc, "Lb_int8_fused_vs_truth")
        lcm, lcc = worst(mc, "Lc_fp16_kernel_vs_truth"), worst(cc, "Lc_fp16_kernel_vs_truth")
        impl = None
        if key in cc:
            impl = max(v["Lb_vs_ya1_impl_delta"]["max_abs_error"] for v in cc[key]["kernels"].values())
        fmt = lambda v: f"{v:.3e}" if v is not None else "—"
        print(f"| {key} | {fmt(la1)} | {fmt(lbm)} | {fmt(lbc)} | {fmt(impl)} | {fmt(lcm)} | {fmt(lcc)} |")

    print("\n## 性能对照（最优配置 P50）\n")
    print("| 形状 | kernel | MUSA 最优 | CUDA 最优 | MUSA int8/fp16 | CUDA int8/fp16 |")
    print("|---|---|---|---|---|---|")

    def platform_pairs(entries):
        out = {}
        best_cache = {}
        for e in entries:
            if e["section"] != "tile_scan":
                continue
            key = (e["m"], e["k"], e["n"], e["kernel"])
            if key not in best_cache or e["p50_ms"] < best_cache[key]["p50_ms"]:
                best_cache[key] = e
        shapes = sorted({(k[0], k[1], k[2]) for k in best_cache})
        for s in shapes:
            f = best_cache.get((s[0], s[1], s[2], "fp16"))
            q = best_cache.get((s[0], s[1], s[2], "int8_fused"))
            if f and q:
                out[s] = (f, q)
        return out

    mp = platform_pairs(musa_b["entries"])
    cp = platform_pairs(cuda_b["entries"])
    for s in sorted(set(mp) | set(cp)):
        fm, qm = mp.get(s, (None, None))
        fc, qc = cp.get(s, (None, None))
        g = lambda e: f"{e['p50_ms']:.4f}ms ({e['tile']}_reg{e['reg']})" if e else "—"
        rm = f"{qm['p50_ms'] / fm['p50_ms']:.3f}" if fm and qm else "—"
        rc = f"{qc['p50_ms'] / fc['p50_ms']:.3f}" if fc and qc else "—"
        print(f"| {s[0]}x{s[1]}x{s[2]} | GEMM | {g(fm)} | {g(fc)} | {rm} | {rc} |")

    # decodeblock
    print("\n## decodeblock 对照（最优 P50）\n")
    for name, b in (("MUSA", musa_b), ("CUDA", cuda_b)):
        best = {}
        for e in b["entries"]:
            if e["section"] == "decodeblock":
                key = (e["m"], e["kernel"])
                if key not in best or e["p50_ms"] < best[key]["p50_ms"]:
                    best[key] = e
        for key in sorted(best):
            e = best[key]
            print(f"- {name} m={key[0]} {key[1]}: {e['p50_ms']:.4f}ms "
                  f"({e['tile']}_reg{e['reg']}, {e['effective_gb_s_p50']:.1f} GB/s)")

    # dequant 三变体跨平台（M-47 闭合判定）
    print("\n## dequant v1/v2/v3 跨平台（M-47 闭合点）\n")
    print("| 变体 | MUSA 4096x4096 | CUDA 4096x4096 | MUSA GB/s | CUDA GB/s |")
    print("|---|---|---|---|---|")
    for variant in ("dequant", "dequant2", "dequant3"):
        row = {}
        for name, b in (("MUSA", musa_b), ("CUDA", cuda_b)):
            hits = [e for e in b["entries"]
                    if e["section"] == "dequant_materialize" and e["kernel"] == variant
                    and e["k"] == 4096 and e["n"] == 4096]
            if hits:
                row[name] = hits[0]
        m, c = row.get("MUSA"), row.get("CUDA")
        print(f"| {variant} | {m['p50_ms']:.4f}ms" if m else "| —", end="")
        print(f" | {c['p50_ms']:.4f}ms" if c else " | —", end="")
        print(f" | {m['effective_gb_s_p50']:.1f}" if m else " | —", end="")
        print(f" | {c['effective_gb_s_p50']:.1f} |" if c else " | — |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
