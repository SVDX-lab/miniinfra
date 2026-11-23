#!/usr/bin/env python3
"""第 08 期冒烟测试：库加载 → API 面 → 数据完整性 → 最小重叠 → 流/事件。

每个用例独立 PASS/FAIL，任何 FAIL 都退出码非零。本测试在 FRESH 进程运行，
顺带记录真·冷进程证据（进程起点 → t_init → 首次 malloc/memcpy/kernel 的耗时
——"为什么第一次跑特别慢"验收问题的冷进程一侧证据）。

用法：python3 smoke_test.py --backend cuda --lib implementations/cuda/build/libtransfer_ops_cuda.so
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transfer_bench_lib import KIND_D2H, KIND_H2D, memcmp_equal, TransferOps

T0 = time.perf_counter()  # 进程起点（冷启动证据）
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["cuda", "musa"])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--h", type=int, default=1024)
    args = ap.parse_args()

    ops = TransferOps(args.lib)
    t_load = time.perf_counter() - T0
    check("lib_load", True, f"{t_load*1e3:.1f}ms（含 CDLL dlopen）")

    # 1. 初始化与设备信息
    t0 = time.perf_counter()
    rc = ops.init()
    t_init = time.perf_counter() - t0
    check("t_init", rc == 0, f"rc={rc} {t_init*1e3:.1f}ms（首次=上下文创建，冷启动证据）")
    info = ops.dev_info()
    print(f"    device: {info}")
    check("dev_info_json", "name" in info and "warp_size_capi" in info)
    print(f"    api_names_compiled_in: {ops.api_names()}")
    print(f"    features(T9_HAVE_*): {ops.features}")

    # 2. 冷启动首笔操作
    t0 = time.perf_counter()
    d = ops.malloc_device(1 << 20)
    t_first_malloc = time.perf_counter() - t0
    check("first_cudaMalloc", True, f"{t_first_malloc*1e3:.1f}ms")
    ops.free_device(d)

    # 3. memcpy 数据完整性（pageable + pinned 各自 H2D→D2H 往返）
    nbytes = 4 << 20
    h_page = ops.malloc_pageable(nbytes)
    h_pin = ops.malloc_pinned_mallochost(nbytes)
    h_gold = ops.malloc_pageable(nbytes)
    d_buf = ops.malloc_device(nbytes)
    ctypes.memset(ctypes.c_void_p(h_page), 0x5A, nbytes)
    ctypes.memset(ctypes.c_void_p(h_pin), 0x5A, nbytes)
    ctypes.memset(ctypes.c_void_p(h_gold), 0x5A, nbytes)
    for buf_name, h_buf in [("pageable", h_page), ("pinned", h_pin)]:
        ops.memcpy_sync(d_buf, h_buf, nbytes, KIND_H2D)
        ctypes.memset(ctypes.c_void_p(h_buf), 0, nbytes)
        ops.memcpy_sync(h_buf, d_buf, nbytes, KIND_D2H)
        check(f"memcpy_integrity_{buf_name}", memcmp_equal(h_buf, h_gold, nbytes))
    check("memcpy_integrity_roundtrip", memcmp_equal(h_page, h_pin, nbytes))

    # 4. RMSNorm 小例 vs numpy 参考
    h = args.h
    rows = 8
    rn = rows * h * 4
    x_host = ops.malloc_pinned_default(rn)
    y_host = ops.malloc_pinned_default(rn)
    x_dev, y_dev = ops.malloc_device(rn), ops.malloc_device(rn)
    x_np = np.random.default_rng(20260909).standard_normal((rows, h), dtype=np.float32)
    ctypes.memmove(ctypes.c_void_p(x_host), x_np.tobytes(), rn)
    ops.memcpy_sync(x_dev, x_host, rn, KIND_H2D)
    w_host = ops.malloc_pinned_default(h * 4)
    w_np = np.random.default_rng(20260910).uniform(0.5, 2.0, h).astype(np.float32)
    ctypes.memmove(ctypes.c_void_p(w_host), w_np.tobytes(), h * 4)
    w_dev = ops.malloc_device(h * 4)
    ops.memcpy_sync(w_dev, w_host, h * 4, KIND_H2D)
    ops.rmsnorm(x_dev, w_dev, y_dev, rows, h, 1e-6, 256, 0)
    ops.device_sync()
    ops.memcpy_sync(y_host, y_dev, rn, KIND_D2H)
    y_np = np.frombuffer((ctypes.c_char * rn).from_address(y_host), dtype=np.float32).reshape(rows, h)
    ref = x_np.astype(np.float64) / np.sqrt((x_np.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6) * w_np
    rel = np.abs(y_np.astype(np.float64) - ref).max() / np.abs(ref).max()
    check("rmsnorm_vs_numpy_fp32", rel < 1e-4, f"max_rel={rel:.3e}（rsqrtf 近似级）")

    # 5. async + 事件次序（async 发射 → 事件 → 同步 → 数值正确）
    ops.memcpy_sync(x_dev, x_host, rn, KIND_H2D)
    s = ops.stream_create()
    e = ops.event_create()
    ops.memcpy_async(x_dev, x_host, rn, KIND_H2D, s)
    ops.event_record(e, s)
    ops.rmsnorm(x_dev, w_dev, y_dev, rows, h, 1e-6, 256, s)
    ops.event_record(e, s)  # 复用同一事件再记录
    ops.stream_sync(s)
    ops.memcpy_sync(y_host, y_dev, rn, KIND_D2H)
    y2 = np.frombuffer((ctypes.c_char * rn).from_address(y_host), dtype=np.float32).reshape(rows, h)
    rel2 = np.abs(y2.astype(np.float64) - ref).max() / np.abs(ref).max()
    check("async_event_ordering", rel2 < 1e-4, f"max_rel={rel2:.3e}")
    ops.event_destroy(e)
    ops.stream_destroy(s)

    # 6. 最小重叠（2 chunk × 1MB, K=2）：overlap vs serial 位级一致
    chunk = 1 << 20
    payload = 2 * chunk
    src = ops.malloc_pinned_mallochost(payload)
    dst_s = ops.malloc_pinned_mallochost(payload)
    dst_o = ops.malloc_pinned_mallochost(payload)
    data = np.random.default_rng(20260911).standard_normal(payload // 4, dtype=np.float32).tobytes()
    ctypes.memmove(ctypes.c_void_p(src), data, payload)
    in_a, in_b = ops.malloc_device(chunk), ops.malloc_device(chunk)
    out_a, out_b = ops.malloc_device(chunk), ops.malloc_device(chunk)
    sa, sb = ops.stream_create(), ops.stream_create()
    n = 2
    # 串行
    for i in range(n):
        ops.memcpy_async([in_a, in_b][i], src + i * chunk, chunk, KIND_H2D, sa)
        ops.rmsnorm([in_a, in_b][i], w_dev, [out_a, out_b][i], chunk // (h * 4), h, 1e-6, 256, sa)
        ops.rmsnorm([out_a, out_b][i], w_dev, [out_a, out_b][i], chunk // (h * 4), h, 1e-6, 256, sa)
        ops.memcpy_async(dst_s + i * chunk, [out_a, out_b][i], chunk, KIND_D2H, sa)
    ops.device_sync()
    # 重叠（深缓冲三段式）
    ev_h = [ops.event_create() for _ in range(n)]
    ev_c = [ops.event_create() for _ in range(n)]
    for i in range(n):
        ops.memcpy_async([in_a, in_b][i], src + i * chunk, chunk, KIND_H2D, sa)
        ops.event_record(ev_h[i], sa)
    for i in range(n):
        ops.stream_wait_event(sb, ev_h[i])
        ops.rmsnorm([in_a, in_b][i], w_dev, [out_a, out_b][i], chunk // (h * 4), h, 1e-6, 256, sb)
        ops.rmsnorm([out_a, out_b][i], w_dev, [out_a, out_b][i], chunk // (h * 4), h, 1e-6, 256, sb)
        ops.event_record(ev_c[i], sb)
    for i in range(n):
        ops.stream_wait_event(sa, ev_c[i])
        ops.memcpy_async(dst_o + i * chunk, [out_a, out_b][i], chunk, KIND_D2H, sa)
    ops.device_sync()
    ok = all(memcmp_equal(dst_s + i * chunk, dst_o + i * chunk, chunk) for i in range(n))
    check("tiny_overlap_bitwise_equal", ok)
    for e_ in ev_h + ev_c:
        ops.event_destroy(e_)
    ops.stream_destroy(sa)
    ops.stream_destroy(sb)
    for p in (in_a, in_b, out_a, out_b):
        ops.free_device(p)
    ops.free_pinned(src)
    ops.free_pinned(dst_s)
    ops.free_pinned(dst_o)

    # 7. 流创建/查询 + 非阻塞流（平台能力面）
    s2 = ops.stream_create()
    check("stream_query_idle", ops.stream_query(s2) in (0, 1, -1000), f"rc={ops.stream_query(s2)}")
    ops.stream_destroy(s2)
    if ops.features["t_stream_create_nonblocking"]:
        s3 = ops.stream_create_nonblocking()
        check("stream_nonblocking", s3 != 0)
        try:
            ops.stream_destroy(s3)
            print("[PASS] stream_nonblocking_destroy")
        except Exception as exc:  # noqa: BLE001
            # S4000 实测（SDK 5.2.0）：musaStreamCreateWithFlags 的流创建成功但
            # musaStreamDestroy 报 rc=201 "invalid device context" —— 平台缺陷，
            # 非致命（记录进 KNOWN_ISSUES/台账），不阻断实验。
            print(f"[WARN] stream_nonblocking_destroy（平台边界，留档）: {exc}")
    else:
        print("[SKIP] stream_nonblocking（平台无此符号，T9_HAVE_STREAM_NONBLOCKING 未注入）")

    # 清理
    ops.free_pageable(h_page)
    ops.free_pageable(h_gold)
    ops.free_pinned(h_pin)
    ops.free_device(d_buf)
    ops.free_pinned(x_host)
    ops.free_pinned(y_host)
    ops.free_pinned(w_host)
    ops.free_device(x_dev)
    ops.free_device(y_dev)
    ops.free_device(w_dev)

    t_total = time.perf_counter() - T0
    print(f"[smoke] 总耗时 {t_total:.2f}s（含冷启动 {t_load*1e3:.0f}ms load + {t_init*1e3:.0f}ms init）")
    if FAILURES:
        print(f"[smoke] FAILED: {FAILURES}")
        sys.exit(1)
    print("[smoke] 全部通过")


if __name__ == "__main__":
    main()
