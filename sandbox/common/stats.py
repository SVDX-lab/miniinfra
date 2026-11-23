"""采样与统计：P50/P95/P99 口径见 experiment-protocol.md 第 4 节。"""
from __future__ import annotations

import math


def percentile(values: list[float], p: float) -> float:
    """线性插值百分位；p ∈ (0,100]。"""
    if not values:
        raise ValueError("empty sample")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def summary(values: list[float]) -> dict:
    """延迟/耗时摘要；n<20 时不报 P99（口径：样本量足够时才报）。"""
    if not values:
        return {"n": 0}
    out = {
        "n": len(values),
        "min": round(min(values), 3),
        "mean": round(sum(values) / len(values), 3),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "max": round(max(values), 3),
    }
    if len(values) >= 20:
        out["p99"] = round(percentile(values, 99), 3)
    return out


def now_ms() -> float:
    import time
    return time.perf_counter() * 1000.0
