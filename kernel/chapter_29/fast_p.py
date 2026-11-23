#!/usr/bin/env python3
"""第 29 期 fast_p 小型复刻（KernelBench 口径的本地化）。

KernelBench fast_p 定义：问题集上"输出正确且 p50 快于基线 p%"的问题比例。
本期小型复刻：
  - 问题集 = 3 算子（rmsnorm / gemm / attention-decode），p=1.0（快于基线）；
  - 基线 = 本篇人工手写版（同 fixtures、同口径、同形状对照点）；
  - 正确 = 该算子全部 correctness cases 过容差 且 确定性 3×位级一致；
  - 对照点：rmsnorm rows=2048×H1024、gemm 1024³、attention decode N=2048 B=1
    （主 dtype：fp16 / fp32 / fp16）。
另附 dtype 展开 6 问题口径（每算子 fp32/fp16 各一问题）作为讨论列。

用法：python fast_p.py   （读 results/correctness-results-cuda.json +
                          results/benchmark-results-cuda.json）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMPARISON_POINTS = {
    # (op, dtype, shape-key selector)
    "rmsnorm": {"dtype": "fp16", "shape": {"rows": 2048, "hidden": 1024}},
    "gemm": {"dtype": "fp32", "shape": {"m": 1024, "k": 1024, "n": 1024}},
    "attention": {"dtype": "fp16", "shape": {"batch": 1, "kv_len": 2048}},
}
EXPANDED_POINTS = [
    ("rmsnorm", "fp32"), ("rmsnorm", "fp16"),
    ("gemm", "fp32"), ("gemm", "fp16"),
    ("attention", "fp32"), ("attention", "fp16"),
]


def shape_matches(entry_shape: dict, target: dict) -> bool:
    return all(entry_shape.get(k) == v for k, v in target.items())


def main() -> int:
    corr_path = HERE / "results/correctness-results-cuda.json"
    bench_path = HERE / "results/benchmark-results-cuda.json"
    if not corr_path.exists() or not bench_path.exists():
        print("[fast_p] 需要 GPU 阶段产物：correctness-results-cuda.json 与 "
              "benchmark-results-cuda.json", file=sys.stderr)
        return 2
    corr = json.loads(corr_path.read_text())
    bench = json.loads(bench_path.read_text())

    def op_correct(op: str) -> dict:
        cases = [c for c in corr.get("cases", []) if c.get("operator") == op]
        det = [d for d in corr.get("determinism", []) if d.get("operator") == op]
        return {
            "cases_pass": sum(1 for c in cases if c.get("pass")),
            "cases_total": len(cases),
            "determinism_all_bitwise": bool(det) and all(d["bitwise_identical"] for d in det),
        }

    def p50_at(op: str, dtype: str, shape: dict) -> float | None:
        for e in bench.get("entries", []):
            if (e.get("operator") == op and e.get("dtype") == dtype
                    and e.get("p50_ms") and shape_matches(e.get("shape", {}), shape)):
                return e["p50_ms"]
        return None

    def manual_at(entry: dict) -> float | None:
        base = entry.get("manual_baseline")
        return base.get("p50_ms") if base and "p50_ms" in base else None

    # 对照点取 benchmark entries（带 manual_baseline）
    def ratio_at(op: str, dtype: str, shape: dict) -> dict:
        for e in bench.get("entries", []):
            if (e.get("operator") == op and e.get("dtype") == dtype
                    and e.get("p50_ms") and shape_matches(e.get("shape", {}), shape)):
                return {"ai_p50_ms": e["p50_ms"],
                        "manual_p50_ms": manual_at(e),
                        "ai_over_manual": e.get("ai_over_manual_p50")}
        return {"ai_p50_ms": p50_at(op, dtype, shape), "manual_p50_ms": None,
                "ai_over_manual": None}

    problems = []
    for op, point in COMPARISON_POINTS.items():
        c = op_correct(op)
        r = ratio_at(op, point["dtype"], point["shape"])
        correct = (c["cases_pass"] == c["cases_total"] and c["determinism_all_bitwise"])
        faster = bool(r["ai_p50_ms"] and r["manual_p50_ms"]
                      and r["ai_p50_ms"] < r["manual_p50_ms"])
        problems.append({"problem": f"{op}/{point['dtype']}", "correct": correct,
                         "faster_than_manual": faster, **c, **r})

    fast_p = sum(1 for p in problems if p["correct"] and p["faster_than_manual"]) / len(problems)

    expanded = []
    for op, dtype in EXPANDED_POINTS:
        point = COMPARISON_POINTS[op]
        c = op_correct(op)
        cases = [x for x in corr.get("cases", [])
                 if x.get("operator") == op and x.get("dtype") == dtype]
        det = [d for d in corr.get("determinism", [])
               if d.get("operator") == op and d.get("dtype") == dtype]
        correct = (bool(cases) and all(x.get("pass") for x in cases)
                   and (not det or all(d["bitwise_identical"] for d in det)))
        r = ratio_at(op, dtype, point["shape"])
        faster = bool(r["ai_p50_ms"] and r["manual_p50_ms"]
                      and r["ai_p50_ms"] < r["manual_p50_ms"])
        expanded.append({"problem": f"{op}/{dtype}", "correct": correct,
                         "faster_than_manual": faster, **r})
    fast_p_expanded = sum(1 for p in expanded
                          if p["correct"] and p["faster_than_manual"]) / len(expanded)

    report = {
        "schema_version": "kernel-chapter30.fast-p.v1",
        "definition": "fast_p（p=1.0）= 正确且 p50 快于人工基线的问题比例；"
                      "正确 = 全部 correctness cases 过容差 + 确定性位级一致",
        "baseline": "本篇人工手写版（ch03/ch18/ch21 同口径）",
        "problems_main3": problems,
        "fast_p_main3": fast_p,
        "problems_dtype_expanded6": expanded,
        "fast_p_dtype_expanded6": fast_p_expanded,
        "discussion_hook": "H-A6 预期 fast_p=0/3（正确性 H-A1、性能 H-A5 双缺陷主导）",
    }
    out = HERE / "results" / "fast-p.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[fast_p] main3 = {fast_p}（{sum(1 for p in problems if p['correct'] and p['faster_than_manual'])}/{len(problems)}）")
    print(f"[fast_p] dtype-expanded6 = {fast_p_expanded}")
    print(f"[results] 已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
