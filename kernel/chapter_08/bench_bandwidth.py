#!/usr/bin/env python3
"""第 08 期带宽/分配基准（实验 1 + 实验 3 + D2D 对照 + zero-copy 附注）。

实验 1：pageable vs pinned 的 H2D/D2H 带宽对照（尺寸扫描 1MB~1GB）。
  - pinned 三种来源分开测：cudaMallocHost（legacy）/ cudaHostAlloc(Default) /
    cudaHostRegister（pageable 原地注册，若平台支持）；
  - 同步 cudaMemcpy 与 cudaMemcpyAsync 各测一遍（async+pageable 的退化路径
    本身就是口径数据）；
  - 计时双口径：同步 API 用"NULL 流事件对"（同步拷贝按默认流次序进入），
    异步 API 用"自建流事件对"；两口径都另记 perf_counter Host 墙钟对账 ——
    若 NULL 流事件测不到拷贝（平台语义差异），以墙钟为准并标注。

实验 3：pinned 的代价。
  - 分配耗时：malloc vs cudaMallocHost vs cudaHostAlloc(Default/Portable/Mapped)
    vs cudaMalloc（设备）vs cudaHostRegister（pageable 原地 pin）
    vs torch.empty（caching allocator 工程对照）；
  - 首次传输慢：新分配缓冲的第一次传输 vs 稳态（页面 pin/GART 映射代价）；
  - 本进程冷启动单独记录；真·冷进程证据在 smoke_test。

对照分母（前序单源，直接引用不重测）：4090 D2D 拷贝峰 918.8 GB/s、
S4000 711.3 GB/s（chapter_01/performance/measure_peak.py）。本期另测 D2D 两档作为
同会话锚点（口径一致性校验）。

输出：results/bandwidth-{backend}.json（schema kernel-chapter09.bandwidth.v1）
"""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_utils import SCHEMA_VERSION, backend_environment_record, utc_now_iso, write_results_json
from transfer_bench_lib import KIND_D2D, KIND_D2H, KIND_H2D, TransferOps

ROOT = Path(__file__).resolve().parent
SEED = 20260909
MIB = 1024 * 1024
NULL_STREAM = 0  # legacy 默认流（同步拷贝的次序归属流；语义探测见 stream_probe）


def percentile(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def stats(vals: list[float]) -> dict:
    s = sorted(vals)
    return {"n": len(s), "p50": s[len(s) // 2], "min": s[0], "max": s[-1],
            "p95": percentile(s, 0.95), "mean": sum(s) / len(s)}


def gbps(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e6) if ms > 0 else 0.0


def fill_pattern(addr: int, nbytes: int) -> None:
    """Host 缓冲填定值模式（0xABABABAB 归一化 float 范围内；带宽与内容无关）。"""
    ctypes.memset(ctypes.c_void_p(addr), 0xAB, nbytes)


def event_time_block(ops: TransferOps, event_stream: int, launch, warmup: int, repeats: int) -> dict:
    """事件对口径：warmup + repeats 次，每次独立事件对（链级）。
    event_stream=0（NULL/legacy 默认流）用于同步 API；自建流用于异步 API。"""
    ev_a = ops.event_create()
    ev_b = ops.event_create()
    ops.device_sync()
    for _ in range(warmup):
        launch()
        ops.device_sync()
    wall, dev = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ops.event_record(ev_a, event_stream)
        launch()
        ops.event_record(ev_b, event_stream)
        ops.device_sync()
        dev.append(ops.event_elapsed_ms(ev_a, ev_b))
        wall.append((time.perf_counter() - t0) * 1e3)
    ops.event_destroy(ev_a)
    ops.event_destroy(ev_b)
    return {"device_ms": stats(dev), "wall_ms": stats(wall)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["cuda", "musa"])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--sizes-mb", default="1,4,16,64,256,512,1024")
    ap.add_argument("--alloc-sizes-mb", default="16,256,1024")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--fresh-trials", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-zerocopy", action="store_true",
                    help="跳过 zero-copy 段（S4000 实测：mapped 读 kernel 在驱动层长时间不完成，见 KI-09-04）")
    args = ap.parse_args()

    notes: list[str] = []
    ops = TransferOps(args.lib)
    rc_init = ops.init()
    if rc_init != 0:
        notes.append(f"t_init rc={rc_init}（flags 未能设置——上下文可能已激活，mapped pinned 受影响）")
    info = ops.dev_info()
    pci = ops.pci_bus_id()
    stream = ops.stream_create()
    ev_a, ev_b = ops.event_create(), ops.event_create()

    result: dict = {
        "schema_version": SCHEMA_VERSION + ".bandwidth",
        "created_at_utc": utc_now_iso(),
        "backend": args.backend,
        "lib": str(Path(args.lib).resolve()),
        "api_names_compiled_in": ops.api_names(),
        "features": ops.features,
        "t_init_rc": rc_init,
        "device": info,
        "pci_bus_id": pci,
        "params": {"repeats": args.repeats, "warmup": args.warmup,
                   "sizes_mb": args.sizes_mb, "seed": SEED,
                   "timing": "同步 API=NULL 流事件对；异步 API=自建流事件对；双口径均记墙钟对账"},
    }

    # ---------------------------------------------------------------- 冷启动
    t0 = time.perf_counter()
    d_probe = ops.malloc_device(1 << 20)
    cold_first_malloc_s = time.perf_counter() - t0
    ops.free_device(d_probe)
    result["cold_process_note"] = {
        "first_cuda_malloc_s_after_init": cold_first_malloc_s,
        "note": "本进程冷启动仅覆盖 init 之后的首次 cudaMalloc；"
                "真·冷进程（含上下文创建/模块加载/首次内核）证据在 smoke_test 的 first_call 节",
    }

    # ------------------------------------------------------- 分配耗时（实验 3）
    alloc_rows = []
    for size_mb in [int(x) for x in args.alloc_sizes_mb.split(",")]:
        nbytes = size_mb * MIB
        trials = max(5, args.repeats // 2)

        def time_alloc(name: str, alloc, free, touch: bool = False) -> None:
            alloc_ms, free_ms, touch_ms = [], [], []
            for _ in range(trials):
                t0 = time.perf_counter()
                p = alloc()
                t1 = time.perf_counter()
                if touch and p:
                    ctypes.memset(ctypes.c_void_p(p), 1, nbytes)  # 首触（页分配真发生处）
                    t2 = time.perf_counter()
                    touch_ms.append((t2 - t1) * 1e3)
                else:
                    t2 = t1
                    touch_ms.append(0.0)
                if p:
                    free(p)
                t3 = time.perf_counter()
                alloc_ms.append((t1 - t0) * 1e3)
                free_ms.append((t3 - t2) * 1e3)
            alloc_rows.append({"size_mb": size_mb, "allocator": name,
                               "alloc_ms": stats(alloc_ms), "free_ms": stats(free_ms),
                               "first_touch_ms": stats(touch_ms) if touch else None})

        time_alloc("malloc(pageable)", lambda: ops.malloc_pageable(nbytes), ops.free_pageable, touch=True)
        time_alloc("cudaMallocHost(pinned legacy)", lambda: ops.malloc_pinned_mallochost(nbytes), ops.free_pinned)
        time_alloc("cudaHostAlloc(Default)", lambda: ops.malloc_pinned_default(nbytes), ops.free_pinned)
        if ops.features["t_malloc_host_pinned_portable"]:
            try:
                time_alloc("cudaHostAlloc(Portable)", lambda: ops.malloc_pinned_portable(nbytes), ops.free_pinned)
            except Exception as exc:  # noqa: BLE001 —— 平台边界如实记录
                notes.append(f"Portable alloc @ {size_mb}MB: {exc}")
        if ops.features["t_malloc_host_pinned_mapped"]:
            try:
                time_alloc("cudaHostAlloc(Mapped)", lambda: ops.malloc_pinned_mapped(nbytes), ops.free_pinned)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Mapped alloc @ {size_mb}MB: {exc}")
        time_alloc("cudaMalloc(device)", lambda: ops.malloc_device(nbytes), ops.free_device)

        # cudaHostRegister：malloc 后原地 pin（pin 代价的显式形式）
        if ops.features["t_host_register"]:
            reg_ms, unreg_ms = [], []
            ok = True
            for _ in range(trials):
                p = ops.malloc_pageable(nbytes)
                try:
                    t0 = time.perf_counter()
                    ops.host_register(p, nbytes)
                    t1 = time.perf_counter()
                    ops.host_unregister(p)
                    t2 = time.perf_counter()
                    reg_ms.append((t1 - t0) * 1e3)
                    unreg_ms.append((t2 - t1) * 1e3)
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"host_register @ {size_mb}MB: {exc}")
                    ok = False
                finally:
                    ops.free_pageable(p)
            if ok and reg_ms:
                alloc_rows.append({"size_mb": size_mb, "allocator": "cudaHostRegister(pageable 原地 pin)",
                                   "alloc_ms": stats(reg_ms), "free_ms": stats(unreg_ms),
                                   "first_touch_ms": None})

        # torch caching allocator 对照（工程参照，不进性能路径）
        try:
            import torch
            dev = torch.device(args.backend)
            if args.backend == "cuda":
                torch.cuda.empty_cache()
            else:
                torch.musa.empty_cache()
            rows = []
            for _ in range(trials):
                t0 = time.perf_counter()
                x = torch.empty(nbytes // 4, dtype=torch.float32, device=dev)
                t1 = time.perf_counter()
                del x
                rows.append((t1 - t0) * 1e3)
            alloc_rows.append({"size_mb": size_mb, "allocator": f"torch.empty({args.backend}, caching)",
                               "alloc_ms": stats(rows), "free_ms": None, "first_touch_ms": None})
        except Exception as exc:  # noqa: BLE001
            notes.append(f"torch.empty 对照不可用: {exc}")
    result["allocation_cost"] = alloc_rows

    # ------------------------------------------------- 首次传输 vs 稳态（实验 3）
    SIZE_FIRST = 1024 * MIB
    first_rows = []
    for buf_name, alloc_fn, free_fn in [
        ("cudaMallocHost", ops.malloc_pinned_mallochost, ops.free_pinned),
        ("cudaHostAlloc(Default)", ops.malloc_pinned_default, ops.free_pinned),
        ("malloc(pageable)", ops.malloc_pageable, ops.free_pageable),
    ]:
        for direction, kind in [("h2d", KIND_H2D), ("d2h", KIND_D2H)]:
            firsts, steadies = [], []
            print(f"  [progress] first_vs_steady {buf_name} {direction}", flush=True)
            d_buf = ops.malloc_device(SIZE_FIRST)
            for trial in range(args.fresh_trials):
                h_buf = alloc_fn(SIZE_FIRST)
                if direction == "h2d":
                    fill_pattern(h_buf, SIZE_FIRST)
                    src, dst = h_buf, d_buf
                else:
                    src, dst = d_buf, h_buf
                ops.device_sync()
                ops.event_record(ev_a, NULL_STREAM)
                ops.memcpy_sync(dst, src, SIZE_FIRST, kind)
                ops.event_record(ev_b, NULL_STREAM)
                ops.device_sync()
                first_ms = ops.event_elapsed_ms(ev_a, ev_b)
                # 稳态：同缓冲重复 5 次取 P50
                steady = []
                for _ in range(5):
                    ops.event_record(ev_a, NULL_STREAM)
                    ops.memcpy_sync(dst, src, SIZE_FIRST, kind)
                    ops.event_record(ev_b, NULL_STREAM)
                    ops.device_sync()
                    steady.append(ops.event_elapsed_ms(ev_a, ev_b))
                firsts.append(first_ms)
                steadies.append(sorted(steady)[len(steady) // 2])
                free_fn(h_buf)
            ops.free_device(d_buf)
            steady_mean = sum(steadies) / len(steadies)
            first_rows.append({"buffer": buf_name, "direction": direction,
                               "size_mb": SIZE_FIRST // MIB,
                               "first_ms": stats(firsts), "steady_p50_ms": stats(steadies),
                               "first_over_steady_mean": stats(firsts)["p50"] / steady_mean if steady_mean else None})
    result["first_vs_steady"] = first_rows

    # ------------------------------------------------------------- 带宽扫描（实验 1）
    sweep = []
    for size_mb in [float(x) for x in args.sizes_mb.split(",")]:
        nbytes = int(size_mb * MIB)
        h_page = ops.malloc_pageable(nbytes)
        h_pin1 = ops.malloc_pinned_mallochost(nbytes)
        h_pin2 = ops.malloc_pinned_default(nbytes)
        h_reg = ops.malloc_pageable(nbytes)
        registered = False
        if ops.features["t_host_register"]:
            try:
                ops.host_register(h_reg, nbytes)
                registered = True
            except Exception as exc:  # noqa: BLE001
                notes.append(f"host_register @ {size_mb}MB sweep: {exc}")
        d_buf = ops.malloc_device(nbytes)
        fill_pattern(h_page, nbytes)
        fill_pattern(h_pin1, nbytes)
        fill_pattern(h_pin2, nbytes)
        fill_pattern(h_reg, nbytes)

        print(f"  [progress] sweep size={size_mb}MB", flush=True)
        row = {"size_mb": size_mb, "bytes": nbytes, "modes": {}}
        buffers = [("pageable", h_page), ("pinned_mallochost", h_pin1),
                   ("pinned_hostalloc_default", h_pin2)]
        if registered:
            buffers.append(("pageable_hostregistered", h_reg))
        for buf_name, h_buf in buffers:
            for direction, kind in [("h2d", KIND_H2D), ("d2h", KIND_D2H)]:
                for api in ["sync", "async"]:
                    if direction == "h2d":
                        src, dst = h_buf, d_buf
                    else:
                        src, dst = d_buf, h_buf

                    def launch(src=src, dst=dst, kind=kind, api=api, nbytes=nbytes):
                        if api == "sync":
                            ops.memcpy_sync(dst, src, nbytes, kind)
                        else:
                            ops.memcpy_async(dst, src, nbytes, kind, stream)

                    # 同步 API → NULL 流事件对；异步 API → 自建流事件对
                    ev_stream = NULL_STREAM if api == "sync" else stream
                    m = event_time_block(ops, ev_stream, launch, args.warmup, args.repeats)
                    dev_p50 = m["device_ms"]["p50"]
                    row["modes"][f"{buf_name}-{direction}-{api}"] = {
                        "event_stream": ev_stream,
                        "device": {"p50_ms": dev_p50, "min_ms": m["device_ms"]["min"],
                                   "gbps_p50": gbps(nbytes, dev_p50),
                                   "gbps_best": gbps(nbytes, m["device_ms"]["min"])},
                        "wall": {"p50_ms": m["wall_ms"]["p50"],
                                 "gbps_p50": gbps(nbytes, m["wall_ms"]["p50"])},
                        "event_sane": dev_p50 > 0.3 * m["wall_ms"]["p50"],
                    }
        # async+pageable 行为证据：发射后立即 query（驱动 staging 退化的直接观察）
        ops.memcpy_async(d_buf, h_page, nbytes, KIND_H2D, stream)
        q_right_after = ops.stream_query(stream)
        ops.stream_sync(stream)
        row["async_pageable_behavior"] = {
            "h2d_pageable_query_right_after": q_right_after,
            "note": "1=仍在流中未完成（真异步）；0=发射返回时已物理完成（退化为同步/staging 完成）",
        }
        ops.free_device(d_buf)
        if registered:
            ops.host_unregister(h_reg)
        ops.free_pageable(h_reg)
        ops.free_pinned(h_pin1)
        ops.free_pinned(h_pin2)
        ops.free_pageable(h_page)
        sweep.append(row)
    result["sweep"] = sweep

    # ------------------------------------------------------------- D2D 锚点
    d2d = []
    for size_mb in [256, 1024]:
        nbytes = size_mb * MIB
        d1 = ops.malloc_device(nbytes)
        d2 = ops.malloc_device(nbytes)
        for api in ["sync", "async"]:
            def launch(api=api, d2=d2, d1=d1, nbytes=nbytes):
                if api == "sync":
                    ops.memcpy_sync(d2, d1, nbytes, KIND_D2D)
                else:
                    ops.memcpy_async(d2, d1, nbytes, KIND_D2D, stream)
            ev_stream = NULL_STREAM if api == "sync" else stream
            m = event_time_block(ops, ev_stream, launch, args.warmup, args.repeats)
            # 拷贝口径 = 读+写两倍字节（与 chapter_01/performance/measure_peak.py 一致）
            d2d.append({"size_mb": size_mb, "api": api,
                        "p50_ms": m["device_ms"]["p50"],
                        "gbps_rw_p50": gbps(2 * nbytes, m["device_ms"]["p50"])})
        ops.free_device(d1)
        ops.free_device(d2)
    result["d2d_reference"] = {
        "note": "同会话锚点；跨期分母引用 chapter_01/performance/measure_peak.py 单源"
                "（4090: 918.8 GB/s，S4000: 711.3 GB/s），口径=读+写",
        "rows": d2d,
    }

    # ------------------------------------------------------- zero-copy 附注
    if args.skip_zerocopy:
        notes.append("zero-copy 段按 --skip-zerocopy 跳过（S4000 实测 mapped 读 kernel 驱动层长时间不完成，KI-09-04）")
    elif ops.features["t_malloc_host_pinned_mapped"]:
        try:
            import numpy as np
            H = 1024
            zc = []
            for size_mb in [64, 256]:
                nbytes = size_mb * MIB
                rows = nbytes // 4 // H
                h_map = ops.malloc_pinned_mapped(nbytes)
                if not h_map:
                    notes.append(f"mapped alloc 返回 NULL @ {size_mb}MB（flags 未生效/平台边界）")
                    break
                rng = np.random.default_rng(SEED)
                data = rng.standard_normal(nbytes // 4, dtype=np.float32).tobytes()
                ctypes.memmove(ctypes.c_void_p(h_map), data, nbytes)
                x_alias = ops.host_get_device_pointer(h_map)
                w_dev = ops.malloc_device(H * 4)
                w_host = ops.malloc_pinned_default(H * 4)
                wdata = np.random.default_rng(SEED + 7).uniform(0.5, 2.0, H).astype(np.float32).tobytes()
                ctypes.memmove(ctypes.c_void_p(w_host), wdata, H * 4)
                ops.memcpy_sync(w_dev, w_host, H * 4, KIND_H2D)
                y_dev = ops.malloc_device(nbytes)

                def launch(x_alias=x_alias, w_dev=w_dev, y_dev=y_dev, rows=rows):
                    ops.rmsnorm(x_alias, w_dev, y_dev, rows, H, 1e-6, 256, stream)

                m = event_time_block(ops, stream, launch, args.warmup, args.repeats)
                zc.append({"size_mb": size_mb, "rows": rows,
                           "kernel_read_mapped_p50_ms": m["device_ms"]["p50"],
                           "gbps_read_p50": gbps(nbytes, m["device_ms"]["p50"])})
                ops.free_device(w_dev)
                ops.free_device(y_dev)
                ops.free_pinned(h_map)
                ops.free_pinned(w_host)
            if zc:
                result["zerocopy_compute"] = {
                    "note": "kernel 直接读 mapped host 内存（零显式传输）——"
                            "'把数据留在设备'的对照路径；对照 H2D+计算",
                    "rows": zc,
                }
        except Exception as exc:  # noqa: BLE001
            notes.append(f"zero-copy 探测失败: {exc}")
    else:
        notes.append("平台无 Mapped pinned 导出（T9_HAVE_HOSTALLOC_MAPPED 未注入）——zero-copy 路径平台边界")

    result["notes"] = notes
    ops.event_destroy(ev_a)
    ops.event_destroy(ev_b)
    ops.stream_destroy(stream)
    result["environment"] = backend_environment_record(args.backend)
    write_results_json(args.out, result)
    print(f"[bandwidth] 写出 {args.out}")
    for row in result["sweep"]:
        pin = row["modes"].get("pinned_mallochost-h2d-async", {}).get("device", {})
        pag = row["modes"].get("pageable-h2d-sync", {}).get("device", {})
        print(f"  {row['size_mb']:>8} MB: pinned-h2d-async {pin.get('gbps_p50', 0):8.2f} GB/s | "
              f"pageable-h2d-sync {pag.get('gbps_p50', 0):8.2f} GB/s")


if __name__ == "__main__":
    main()
