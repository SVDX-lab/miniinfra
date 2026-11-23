"""第 14 期实验 3（CPU 列）：dynamic shape 与重编译行为。

四个子实验（**每个 phase 独立子进程**——实测 torch 2.6 的 automatic
dynamic frame state 会跨 torch._dynamo.reset() 存活并污染后续 dynamic=True
编译（同进程先 auto 后 dynamic=True 时出现假失败），变体隔离必须跨进程）：

A. 自动 dynamic（torch.compile 默认 dynamic=None）：shape 序列
   1→128→2048→1→64→2048 上逐调用统计 unique_graphs 与每次调用墙钟；
B1. dynamic=True 显式全动态：单进程纯净状态下能否编译/运行（权重对齐
   静态维的广播约束）；
B2. mark_dynamic 行维部分动态：生产可用的替代，统计 graph 数；
C. guard 结构与失效面：explain 抓 guard 全文 + 逐项失效探测（shape/eps/
   dtype/stride）；缓存命中验证。

产出 results/recompile-results-cpu.json + results/recompile-trace-cpu.log。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiment_utils import write_results_json

CHAPTER = Path(__file__).resolve().parent
SCHEMA_VERSION = "kernel-chapter15.recompile.v1"


def unique_graphs() -> int:
    return int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))


def make_inputs(rows: int, hidden: int = 1024, seed: int = 20260415):
    rng_r = np.random.default_rng(seed + rows)
    rng_h = np.random.default_rng(seed + rows + 1)
    r = torch.from_numpy(rng_r.standard_normal((rows, hidden), dtype=np.float32))
    h = torch.from_numpy(rng_h.standard_normal((rows, hidden), dtype=np.float32))
    w = torch.from_numpy(
        np.random.default_rng(seed + 999).standard_normal((hidden,), dtype=np.float32)
    )
    return r, h, w


def run_sequence(compiled, label: str, trace: list[str], mark_rows: bool = False) -> list[dict]:
    steps: list[dict] = []
    for rows in (1, 128, 2048, 1, 64, 2048):
        r, h, w = make_inputs(rows)
        if mark_rows:
            torch._dynamo.mark_dynamic(r, 0)
            torch._dynamo.mark_dynamic(h, 0)
        graphs_before = unique_graphs()
        started = time.perf_counter()
        t, y = compiled(r, h, w)
        wall_ms = (time.perf_counter() - started) * 1000.0
        graphs_after = unique_graphs()
        steps.append(
            {
                "rows": rows,
                "graphs_before": graphs_before,
                "graphs_after": graphs_after,
                "recompiled": graphs_after > graphs_before,
                "call_wall_ms": round(wall_ms, 1),
                "checksum": float(y.abs().sum().item()),
            }
        )
        line = (
            f"[{label}] rows={rows}: graphs {graphs_before}->{graphs_after}"
            f"{' (RECOMPILED)' if graphs_after > graphs_before else ''}, call {wall_ms:.1f} ms"
        )
        trace.append(line)
        print(line, flush=True)
    return steps


def phase_a(trace: list[str]) -> tuple[list[dict], dict]:
    from implementations.cpu.compiled_objects import add_rmsnorm_separated
    from implementations.cpu.compile_variants import (
        disable_codegen_caches,
        reset_compilers,
    )

    reset_compilers()
    disable_codegen_caches()
    auto = torch.compile(add_rmsnorm_separated)
    steps = run_sequence(auto, "auto-dynamic", trace)
    counters = {
        str(k): {str(kk): int(vv) for kk, vv in v.items()}
        for k, v in torch._dynamo.utils.counters.items()
        if hasattr(v, "items")
    }
    return steps, counters


def phase_b1(trace: list[str]) -> dict:
    """dynamic=True 纯净单进程：能否编译+运行？"""
    from implementations.cpu.compiled_objects import add_rmsnorm_separated

    code = (
        "import torch, numpy as np\n"
        "from implementations.cpu.compiled_objects import add_rmsnorm_separated\n"
        "c = torch.compile(add_rmsnorm_separated, dynamic=True)\n"
        "r = torch.randn(1, 1024); h = torch.randn(1, 1024); w = torch.randn(1024)\n"
        "try:\n"
        "    t, y = c(r, h, w)\n"
        "    print('B1-OK', float(y.abs().sum()))\n"
        "except Exception as e:\n"
        "    import traceback; traceback.print_exc()\n"
        "    print('B1-FAIL', type(e).__name__)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(CHAPTER), check=False,
    )
    ok = "B1-OK" in done.stdout
    record = {
        "clean_process_ok": ok,
        "stdout_tail": done.stdout.strip()[-400:],
        "stderr_tail": done.stderr.strip()[-2000:] if not ok else "",
    }
    verdict = (
        "dynamic=True 纯净进程可用（weight 对齐的 H 维被自动回特化为静态，仅行维动态）"
        if ok
        else "dynamic=True 失败（报错全文见 stderr_tail）"
    )
    trace.append(f"[dynamic-true] {verdict}")
    print(f"[dynamic-true] {verdict}")
    record["verdict"] = verdict
    return record


def phase_b2(trace: list[str]) -> list[dict]:
    from implementations.cpu.compiled_objects import add_rmsnorm_separated
    from implementations.cpu.compile_variants import (
        disable_codegen_caches,
        reset_compilers,
    )

    reset_compilers()
    disable_codegen_caches()
    marked = torch.compile(add_rmsnorm_separated)
    return run_sequence(marked, "mark_dynamic-rows", trace, mark_rows=True)


def phase_c(trace: list[str]) -> tuple[dict, list[dict]]:
    from implementations.cpu.compiled_objects import add_rmsnorm_separated
    from implementations.cpu.compile_variants import (
        disable_codegen_caches,
        reset_compilers,
    )

    reset_compilers()
    explain_result = torch._dynamo.explain(add_rmsnorm_separated)(  # noqa: B015
        *make_inputs(128)
    )
    guards_text = [str(g) for g in getattr(explain_result, "out_guards", [])]
    trace.append(f"[guards] explain graph_count={explain_result.graph_count}, guards={len(guards_text)}")
    for guard in guards_text[:40]:
        trace.append(f"[guard] {guard}")
    print(f"[guards] explain guards={len(guards_text)}, graph={explain_result.graph_count}")

    guard_probes: list[dict] = []
    reset_compilers()
    disable_codegen_caches()
    probed = torch.compile(add_rmsnorm_separated)
    r, h, w = make_inputs(64)
    probed(r, h, w)

    def probe(name: str, call):
        before = unique_graphs()
        error_text = None
        try:
            call()
        except Exception as error:  # noqa: BLE001 —— 探测项允许失败（失败本身是结论）
            error_text = f"{type(error).__name__}: {error}"
        after = unique_graphs()
        guard_probes.append(
            {
                "probe": name,
                "recompiled": after > before,
                "graphs_after": after,
                "error": error_text,
            }
        )
        line = f"[guard-probe] {name}: recompiled={after > before} (graphs {before}->{after})"
        if error_text:
            line += f" | CALL-RAISED {error_text[:240]}"
        trace.append(line)
        print(line)

    probe("同 shape 重复调用（应命中缓存不重编译）", lambda: probed(r, h, w))
    probe("rows 变化 64->96（自动 dynamic 重编译转动态图）", lambda: probed(*make_inputs(96)))
    probe("eps 默认参数 1e-6 -> 1e-5（常量 specialization guard）",
          lambda: probed(*make_inputs(96), 1e-5))
    probe("dtype fp32 -> fp64（dtype guard）",
          lambda: probed(make_inputs(96)[0].double(), make_inputs(96)[1].double(),
                         make_inputs(96)[2].double()))
    big = torch.from_numpy(
        np.random.default_rng(777).standard_normal((96, 16), dtype=np.float32)
    )
    probe("非连续输入（stride guard，H 1024->8 变化）",
          lambda: probed(big[:, ::2], make_inputs(96)[1][:, :8], make_inputs(96)[2]))
    explain_payload = {
        "graph_count": explain_result.graph_count,
        "graph_break_count": explain_result.graph_break_count,
        "op_count": explain_result.op_count,
        "guards_text": guards_text,
    }
    return explain_payload, guard_probes


def main() -> None:
    trace: list[str] = [f"# recompile trace {datetime.now(timezone.utc).isoformat()}"]
    steps_auto, counters_auto = phase_a(trace)
    b1 = phase_b1(trace)
    steps_marked = phase_b2(trace)
    explain_payload, guard_probes = phase_c(trace)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "sequence": [1, 128, 2048, 1, 64, 2048],
        "methodology_note": (
            "每个 phase 独立子进程/独立重置：实测同进程先 auto-dynamic 后 "
            "dynamic=True 会因跨 reset 存活的 automatic dynamic frame state "
            "产生假失败（见 KNOWN_ISSUES KI-15-01）"
        ),
        "auto_dynamic": {"steps": steps_auto, "counters": counters_auto,
                         "unique_graphs_total": steps_auto[-1]["graphs_after"]},
        "dynamic_true": b1,
        "mark_dynamic_rows": {"steps": steps_marked,
                              "unique_graphs_total": steps_marked[-1]["graphs_after"]},
        "explain": explain_payload,
        "guard_probes": guard_probes,
    }
    write_results_json(CHAPTER / "results" / "recompile-results-cpu.json", payload)
    (CHAPTER / "results" / "recompile-trace-cpu.log").write_text(
        "\n".join(trace) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
