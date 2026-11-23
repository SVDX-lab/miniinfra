#!/usr/bin/env python3
"""第 08 期 nsys 时间线解析：传输与计算 kernel 并行区间证据（实验 4）。

输入：nsys stats --report cuda_gpu_trace 导出的 CSV（run_nsys.sh 产出），
每行一个 GPU op（Start/Dur/Name，ns）。

解析逻辑：
  - 行分类：Memcpy HtoD / Memcpy DtoH / rmsnorm kernel；
  - 时相切分：nsys-mode 脚本先串行后重叠，以"首个与 kernel 重叠的 memcpy"
    为界（更稳的做法：以串行相位长度近似切分）——实际上直接全段统计 +
    按连续重叠区段分组报告；
  - 证据指标：
      memcpy_busy_ms / kernel_busy_ms / intersect_busy_ms（两两区间交并）
      concurrency_ratio = intersect / span
      并行区段数（存在 memcpy∩kernel 的连续时间窗）
  - 对照预期：串行相位 intersect≈0；重叠相位 intersect>0。

输出：results/overlap-trace-{backend}.json + server-logs/gpu-trace-excerpt.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_utils import SCHEMA_VERSION, utc_now_iso, write_results_json


def load_trace(csv_path: Path) -> list[dict]:
    ops = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        def col2(candidates: list[str]) -> int | None:
            low = [h.strip().lower() for h in header]
            for c in candidates:
                c = c.lower()
                for i, h in enumerate(low):
                    if h.startswith(c):
                        return i
            return None

        i_start = col2(["start (ns)", "start(ns)", "start"])
        i_dur = col2(["dur (ns)", "dur(ns)", "dur", "duration"])
        i_name = col2(["name"])
        i_src = col2(["srcmemkd"])
        i_dst = col2(["dstmemkd"])
        if i_start is None or i_dur is None or i_name is None:
            raise SystemExit(f"CSV 列不可识别: {header}")
        for row in reader:
            if len(row) <= max(i_start, i_dur, i_name):
                continue
            try:
                start = float(row[i_start])
                dur = float(row[i_dur])
            except ValueError:
                continue
            ops.append({"start_ns": start, "dur_ns": dur, "name": row[i_name],
                        "src_kd": row[i_src] if i_src is not None and i_src < len(row) else "",
                        "dst_kd": row[i_dst] if i_dst is not None and i_dst < len(row) else ""})
    ops.sort(key=lambda o: o["start_ns"])
    return ops


def classify(name: str, src_kd: str = "", dst_kd: str = "") -> str | None:
    """方向判定优先用 nsys 的 SrcMemKd/DstMemKd 列（Pinned/Device/Pageable），
    名称匹配为后备（[CUDA memcpy Host-to-Device] / Memcpy HtoD 等多代格式）。"""
    low = name.lower()
    if "memcpy" in low:
        sk, dk = src_kd.lower(), dst_kd.lower()
        if sk and dk:
            if "host" in sk and "device" in dk:
                return "h2d"
            if "device" in sk and "host" in dk:
                return "d2h"
        if "htod" in low or "host-to-device" in low or "h to d" in low:
            return "h2d"
        if "dtoh" in low or "device-to-host" in low or "d to h" in low:
            return "d2h"
        return "memcpy_other"
    if "rmsnorm" in low:
        return "kernel"
    return None


def intersect_sum(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """两组区间两两交集总时长（ns）。两组各自内部无重叠（GPU 单资源假设近似）。"""
    total = 0.0
    j = 0
    for s0, e0 in a:
        for k in range(j, len(b)):
            s1, e1 = b[k]
            if e1 <= s0:
                j = k + 1
                continue
            if s1 >= e0:
                break
            total += min(e0, e1) - max(s0, s1)
    return total


def contiguous_windows(intervals: list[tuple[float, float]], gap_ns: float = 5e6) -> list[tuple[float, float]]:
    """把重叠发生的时间轴合并为连续窗口（gap > 5ms 切开——区分串行/重叠相位）。"""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le + gap_ns:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-csv", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ops = load_trace(Path(args.trace_csv))
    mem_all = {"h2d": [], "d2h": [], "other": []}
    kernels = []
    counts = {"h2d": 0, "d2h": 0, "memcpy_other": 0, "kernel": 0}
    for op in ops:
        kind = classify(op["name"], op.get("src_kd", ""), op.get("dst_kd", ""))
        if kind is None:
            continue
        counts[kind] += 1
        iv = (op["start_ns"], op["start_ns"] + op["dur_ns"])
        if kind == "kernel":
            kernels.append(iv)
        else:
            mem_all[kind if kind in mem_all else "other"].append(iv)

    span_s = ops[0]["start_ns"] if ops else 0
    span_e = max((o["start_ns"] + o["dur_ns"]) for o in ops) if ops else 0
    span = span_e - span_s
    mem_union = sorted(mem_all["h2d"] + mem_all["d2h"] + mem_all["other"])
    inter = intersect_sum(mem_union, kernels)
    pw = _pairwise_windows(mem_union, kernels)
    windows = contiguous_windows(pw)
    kernel_busy = sum(e - s for s, e in kernels)
    mem_busy = sum(e - s for s, e in mem_union)
    # 并行窗口内的 kernel 忙时与"被隐藏比例"：window 内 kernel 总时长里
    # 与 memcpy 重叠的部分 —— 重叠相位计算被传输完全遮蔽时 ≈ 1.0
    kernel_in_windows = 0.0
    for ks, ke in kernels:
        for ws, we in windows:
            if ks < we and ke > ws:
                kernel_in_windows += min(ke, we) - max(ks, ws)
    hidden_fraction = inter / kernel_in_windows if kernel_in_windows > 0 else None
    # 重叠相位（最后一个并行窗口）时间线摘录
    tl_lines = []
    if windows:
        ws, we = windows[-1]
        pad = 3e6  # ±3ms
        for o in ops:
            if ws - pad <= o["start_ns"] <= we + pad:
                kind = classify(o["name"], o.get("src_kd", ""), o.get("dst_kd", "")) or "other"
                tl_lines.append(f"{(o['start_ns']-span_s)/1e6:>12.4f} ms  {o['dur_ns']/1e3:>10.2f} us  {kind:<8} {o['name'][:60]}")

    result = {
        "schema_version": SCHEMA_VERSION + ".overlap-trace",
        "created_at_utc": utc_now_iso(),
        "backend": args.backend,
        "trace_csv": str(Path(args.trace_csv).resolve()),
        "op_counts": counts,
        "span_ms": span / 1e6,
        "memcpy_busy_ms": mem_busy / 1e6,
        "kernel_busy_ms": kernel_busy / 1e6,
        "intersect_busy_ms": inter / 1e6,
        "parallel_window_count": len(windows),
        "parallel_windows_ms": [(round(s / 1e6, 4), round(e / 1e6, 4)) for s, e in windows],
        "concurrency_fraction": inter / span if span else None,
        "kernel_busy_in_windows_ms": kernel_in_windows / 1e6,
        "compute_hidden_fraction": hidden_fraction,
        "evidence": ("intersect_busy_ms > 0 且 parallel_window_count >= 1 "
                     "=> memcpy 与 rmsnorm kernel 在时间线上存在并行区间（实验 4 时间线证据）；"
                     "compute_hidden_fraction = 窗口内 kernel 忙时被 memcpy 覆盖的比例"),
    }
    if tl_lines:
        tl_path = Path(args.out).parent / "server-logs" / f"overlap-phase-timeline-{args.backend}.txt"
        tl_path.parent.mkdir(parents=True, exist_ok=True)
        header = f"overlap 相位时间线（最后一个并行窗口 we={windows[-1][1]/1e6:.3f}ms，共 {len(tl_lines)} op）\n"
        tl_path.write_text(header + "\n".join(tl_lines) + "\n", encoding="utf-8")
        result["timeline_excerpt"] = str(tl_path)
    write_results_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "parallel_windows_ms"}, ensure_ascii=False, indent=2))
    # 摘录前 120 行供留档
    excerpt = Path(args.out).parent / "server-logs" / f"gpu-trace-excerpt-{args.backend}.csv"
    excerpt.parent.mkdir(parents=True, exist_ok=True)
    with open(args.trace_csv, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    excerpt.write_text("".join(lines[:121]), encoding="utf-8")
    print(f"[trace] 摘录 {excerpt}")


def _pairwise_windows(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """返回所有 (start, end) 真实交集区间，供连续窗口合并。"""
    out = []
    j = 0
    for s0, e0 in a:
        for k in range(j, len(b)):
            s1, e1 = b[k]
            if e1 <= s0:
                j = k + 1
                continue
            if s1 >= e0:
                break
            out.append((max(s0, s1), min(e0, e1)))
    return out


if __name__ == "__main__":
    main()
