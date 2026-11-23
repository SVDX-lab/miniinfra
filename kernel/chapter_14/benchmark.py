"""第 14 期实验 1（CPU 列）：torch.compile 前后延迟对照主脚本。

矩阵：3 个被编译对象 ×（eager + 4 个编译变体）× shape {1,128,2048}×1024
（qwen3 段的 rows 语义 = 序列长度，batch=1；编译变体取 4 变量子集，
max-autotune 只用于前两档——整段上逐 kernel 调优耗时失控，如实记录取舍）。

每个 (对象, 变体, shape) 条目：
- 先测 eager 基线（OperatorTimer CPU 口径：perf_counter 逐次，P50/P95）；
- torch._dynamo.reset() + 关闭 inductor 持久缓存 → 冷编译首调墙钟
  （compile_first_call_ms = dynamo 图捕获 + AOT + inductor 代码生成 +
  kernel 编译 + 执行）；
- 首调输出 vs eager 输出快速对拍（max_abs）；
- 稳态延迟（首调后再 warmup + repeats 计时）；
- dynamo counters 快照（graph break / recompile 原因）。

产出 results/benchmark-results-cpu.json（协议 §2 风格：样本数组保留、
聚合可重算）。
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiment_utils import OperatorTimer, comparison_metrics, write_results_json

CHAPTER = Path(__file__).resolve().parent
SCHEMA_VERSION = "kernel-chapter15.cpu-latency.v1"

HIDDEN = 1024
ROW_SHAPES = (1, 128, 2048)
SEQ_SHAPES = (1, 128, 2048)

REPEAT_PLAN = {
    ("rmsnorm", 1): (5, 30), ("rmsnorm", 128): (5, 30), ("rmsnorm", 2048): (3, 10),
    ("add_rmsnorm", 1): (5, 30), ("add_rmsnorm", 128): (5, 30), ("add_rmsnorm", 2048): (3, 10),
    ("qwen3", 1): (2, 5), ("qwen3", 128): (2, 5), ("qwen3", 2048): (1, 3),
}


def seed_for(rows: int, hidden: int) -> dict[str, int]:
    base = 20260415
    r_seed = base + rows * 100000 + hidden
    return {"r": r_seed, "h": r_seed + 555555, "w": base + hidden}


def make_pair_inputs(rows: int, hidden: int):
    """与 fixtures 同 seed 规则的张量（延迟实验与正确性共用同一组数字）。"""
    seeds = seed_for(rows, hidden)
    r = torch.from_numpy(
        np.random.default_rng(seeds["r"]).standard_normal((rows, hidden), dtype=np.float32)
    )
    h = torch.from_numpy(
        np.random.default_rng(seeds["h"]).standard_normal((rows, hidden), dtype=np.float32)
    )
    w = torch.from_numpy(
        np.random.default_rng(seeds["w"]).standard_normal((hidden,), dtype=np.float32)
    )
    return r, h, w


def counters_summary() -> dict:
    """torch._dynamo.utils.counters 的可序列化快照。"""
    counters = torch._dynamo.utils.counters
    out: dict[str, dict[str, int]] = {}
    for key, sub in counters.items():
        try:
            out[str(key)] = {str(k): int(v) for k, v in sub.items()}
        except TypeError:
            continue
    return out


def build_eager_call(object_name: str, rows: int):
    """构建 (eager 调用闭包, 参数元组, eager 参考输出, 可释放引用)。"""
    holders: dict = {"module": None}

    if object_name == "qwen3":
        from implementations.cpu.qwen3_segment import build_segment

        module = build_segment()
        input_ids = torch.from_numpy(
            np.random.default_rng(20260415 + rows).integers(
                0, module.config.vocab_size, size=(1, rows), dtype=np.int64
            )
        )
        args = (input_ids,)
        holders["module"] = module

        def eager_call():
            return module(*args)

        return eager_call, args, module(*args), holders

    r, h, w = make_pair_inputs(rows, HIDDEN)
    args = (r, h, w)

    def eager_call():
        import implementations.cpu.compiled_objects as objects

        if object_name == "rmsnorm":
            return objects.rmsnorm_expr(*args)
        return objects.add_rmsnorm_separated(*args)

    return eager_call, args, eager_call(), holders


def run_object(object_name: str, variants: tuple[str, ...]) -> list[dict]:
    from implementations.cpu.compile_variants import (
        disable_codegen_caches,
        make_callable,
        reset_compilers,
    )

    entries: list[dict] = []
    shapes = list(SEQ_SHAPES) if object_name == "qwen3" else list(ROW_SHAPES)

    for rows in shapes:
        eager_call, args, reference, holders = build_eager_call(object_name, rows)
        warmup, repeats = REPEAT_PLAN[(object_name, rows)]
        eager_latency = OperatorTimer(torch.device("cpu"), warmup, repeats).measure(eager_call)

        for variant in variants:
            reset_compilers()
            disable_codegen_caches()
            module_c = None
            if object_name == "qwen3":
                from implementations.cpu.qwen3_segment import build_segment

                module_c = build_segment()
                callable_fn = make_callable("qwen3", variant, module=module_c)
            else:
                callable_fn = make_callable(object_name, variant)

            started = time.perf_counter()
            output = callable_fn(*args)
            compile_ms = (time.perf_counter() - started) * 1000.0

            entry: dict = {
                "object": object_name,
                "variant": variant,
                "rows": rows,
                "hidden": HIDDEN,
                "dtype": "float32",
                "torch": torch.__version__,
                "warmup": warmup,
                "repeats": repeats,
                "compile_first_call_ms": round(compile_ms, 1),
                "eager": eager_latency,
            }

            if isinstance(reference, tuple):
                entry["first_call_max_abs_vs_eager"] = {
                    name: comparison_metrics(out, ref)["max_abs_error"]
                    for name, out, ref in zip(("t", "y"), output, reference)
                }
            else:
                entry["first_call_max_abs_vs_eager"] = comparison_metrics(
                    output, reference
                )["max_abs_error"]

            entry["dynamo_counters_after_first_call"] = counters_summary()
            entry["latency"] = OperatorTimer(
                torch.device("cpu"), warmup, repeats
            ).measure(lambda fn=callable_fn, a=args: fn(*a))
            entry["speedup_p50_vs_eager"] = round(
                eager_latency["p50_ms"] / entry["latency"]["p50_ms"], 4
            )
            entries.append(entry)
            print(
                f"[bench] {object_name}/{variant} rows={rows}: "
                f"compile {entry['compile_first_call_ms']}ms, "
                f"eager p50 {eager_latency['p50_ms']}ms, "
                f"compiled p50 {entry['latency']['p50_ms']}ms, "
                f"speedup {entry['speedup_p50_vs_eager']}",
                flush=True,
            )
            del callable_fn, output, module_c  # qwen3 module ~0.7GB/个，及时释放
        del eager_call, reference, holders
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", default="rmsnorm,add_rmsnorm,qwen3")
    args = parser.parse_args()

    torch.manual_seed(20260415)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    import torch._dynamo as dynamo

    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "dynamo": getattr(dynamo, "__version__", "n/a（torch 2.6 无 dynamo.__version__）"),
        "inductor_config": {
            "fx_graph_cache_disabled_for_cold_compile": True,
            "num_threads": torch.get_num_threads(),
        },
        "shape_contract": {"rows": list(ROW_SHAPES), "hidden": HIDDEN},
        "entries": [],
    }
    for object_name in [s.strip() for s in args.objects.split(",") if s.strip()]:
        variants = (
            ("eager", "dynamo-eager", "aot-eager", "inductor")
            if object_name == "qwen3"
            else ("eager", "dynamo-eager", "aot-eager", "inductor", "inductor-mat")
        )
        payload["entries"].extend(run_object(object_name, variants))

    write_results_json(CHAPTER / "results" / "benchmark-results-cpu.json", payload)


if __name__ == "__main__":
    main()
