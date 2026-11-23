#!/usr/bin/env python3
"""compare_correctness.py —— 第 10 期正确性口径（工具链体检期变体）。

本期的"正确性"不是一个算子对真值，而是两组命题：
  C1 健康版命题：ch03/ch04/ch08 health 靶（fp32 rows=64 h=1024 tile=256）
     的内置自检 max_abs_error 达到第 02 期 bench --check 同级（fp32 < 1e-5）；
  C2 缺陷版命题：bad 变体的损坏证据在位（race bad unique_5rep>1）——
     "工具抓的 bug 是真 bug"的正例锚。
输入：裸跑各 binary 的 stdout（本地解析）；也可直接复用 smoke_test 的输出。
输出 JSON：correctness-results-cuda.json。
"""
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / "implementations" / "cuda" / "build"
FP32_HEALTH_MAX = 1e-5  # 第 02 期 bench --check fp32 口径


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, ""


def parse_max_err(out: str) -> float | None:
    m = re.search(r"max_abs_error=([0-9.eE+-]+)", out)
    return float(m.group(1)) if m else None


def main() -> int:
    if not BUILD.exists():
        print("先 build", file=sys.stderr)
        return 2
    results: dict = {"schema_version": "kernel-chapter11.correctness.v1", "checks": []}

    targets = [
        ("ch03-det", str(BUILD / "rmsnorm_bench"),
         ["--rows", "64", "--h", "1024", "--dtype", "fp32", "--tiles", "256",
          "--warmup", "2", "--repeats", "3", "--check", "true", "--check-rows", "4"],
         r'"max_abs_error":([0-9.eE+-]+)'),
        ("ch04-fused", str(BUILD / "health_ch04"),
         ["--target", "ch04-fused", "--rows", "64", "--h", "1024", "--tile", "256", "--iters", "1"],
         r"max_abs_error=([0-9.eE+-]+)"),
        ("ch08-det", str(BUILD / "health_ch08"),
         ["--target", "ch08-det", "--rows", "64", "--h", "1024", "--tile", "256", "--iters", "1"],
         r"max_abs_error=([0-9.eE+-]+)"),
        ("ch08-atomic-shared", str(BUILD / "health_ch08"),
         ["--target", "ch08-atomic-shared", "--rows", "64", "--h", "1024",
          "--tile", "256", "--iters", "1"],
         r"max_abs_error=([0-9.eE+-]+)"),
        ("ch08-atomic-global", str(BUILD / "health_ch08"),
         ["--target", "ch08-atomic-global", "--rows", "64", "--h", "1024",
          "--tile", "256", "--iters", "1", "--splits", "8"],
         r"max_abs_error=([0-9.eE+-]+)"),
    ]
    for name, binary, args, pat in targets:
        rc, out = run([binary, *args])
        m = re.search(pat, out)
        max_err = float(m.group(1)) if m else None
        ok = rc == 0 and max_err is not None and max_err < FP32_HEALTH_MAX
        results["checks"].append({
            "target": name, "exit": rc, "max_abs_error": max_err,
            "threshold": FP32_HEALTH_MAX, "pass": ok,
        })

    ds = str(BUILD / "defect_suite")
    rc, out = run([ds, "--case", "race", "--variant", "bad"])
    m = re.search(r"unique_outputs_5rep=(\d+)", out)
    unique = int(m.group(1)) if m else 0
    results["checks"].append({
        "target": "race-bad-corruption-anchor", "exit": rc,
        "unique_outputs_5rep": unique, "pass": unique > 1,
    })

    results["all_pass"] = all(c["pass"] for c in results["checks"])
    out_path = HERE / "results" / "correctness-results-cuda.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
