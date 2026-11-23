#!/usr/bin/env python3
"""benchmark.py —— 第 10 期统一计时口径（本期"基准" = sanitizer 开销测量）。

数据源：implementations/cuda/run_sanitizer_suite.sh 的 E3 阶段输出
（E3-bench-<mode>.jsonl 各含一行 rmsnorm_bench 的 JSON：p50/p95/mean/min，
warmup=10 repeats=100，Event 计时口径与第 01/02 期一致）。

聚合：
  - 每 mode 的 kernel p50/p95 与 p50_ratio_vs_bare（kernel 级慢化比）；
  - 每 mode 的进程 wall（从 results/sanitizer/summary.tsv 的 wall= 字段）
    与 wall_ratio_vs_bare（工具全进程开销比）。
输出：results/benchmark-results-cuda.json
"""
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
MODES = ["bare", "memcheck", "racecheck", "initcheck", "synccheck"]


def main() -> int:
    rows = []
    base_p50 = None
    base_wall = None
    for mode in MODES:
        p50 = p95 = mean = None
        jl = RES / "sanitizer" / f"E3-bench-{mode}.jsonl"
        if jl.exists():
            for line in jl.read_text().splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p50, p95, mean = d.get("p50_ms"), d.get("p95_ms"), d.get("mean_ms")
        wall = None
        summary = RES / "sanitizer" / "summary.tsv"
        if summary.exists():
            for line in summary.read_text().splitlines():
                if line.startswith("E3\t") and f"\t{mode}\t" in line:
                    m = re.search(r"wall=([0-9.]+)", line)
                    if m:
                        wall = float(m.group(1))
        if mode == "bare":
            base_p50, base_wall = p50, wall
        rows.append({
            "mode": mode, "kernel_p50_ms": p50, "kernel_p95_ms": p95, "kernel_mean_ms": mean,
            "kernel_p50_ratio_vs_bare": (p50 / base_p50) if (p50 and base_p50) else None,
            "process_wall_s": wall,
            "wall_ratio_vs_bare": (wall / base_wall) if (wall and base_wall) else None,
        })

    out = {
        "schema_version": "kernel-chapter11.benchmark.v1",
        "kernel": "rmsnorm fp32 rows=2048 h=1024 tile=256 warmup=10 repeats=100（ch03 bench 原样）",
        "timer": "cudaEvent 对 + 末尾一次 synchronize（同第 01 期 OperatorTimer 口径）",
        "device_note": "RTX 4090 24G, 驱动 580.119.02（2026-11-21 换机后新实例；旧实例 595.91.07 的旧数字保留于 results/sanitizer-old-instance-595/，两组并存对照）, CUDA 13.0, compute-sanitizer 2025.3.1",
        "rows": rows,
    }
    out_path = RES / "benchmark-results-cuda.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
