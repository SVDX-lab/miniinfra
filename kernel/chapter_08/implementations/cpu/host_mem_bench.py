#!/usr/bin/env python3
"""Host 侧计时基准（第 08 期"无 CPU 列"声明的落点 —— 本期主体是 Host↔设备
路径，Host 侧只做计时基准，不参与设备端对照）。

角色与口径（README 同步声明）：
  1. malloc/calloc/free 分配与释放耗时（16MB/256MB/1GB）——pinned 分配代价
     对照的 Host 基线（同一物理机上 malloc vs cudaMallocHost 才可比）；
  2. 首触（first-touch）页错误代价：malloc 后逐 4KB 页触碰 vs 全量 memset；
  3. Host DRAM 单线程拷贝带宽（ctypes.memmove，读+写口径）—— 传输实验的
     "如果数据根本不出 Host"参照；
  4. 大页（THP）状态记录 —— 分配耗时差异的解释变量。

输出：results/host-mem-{label}.json（本地跑 label=cpu；服务器跑 label=cuda-host/musa-host）
"""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_utils import SCHEMA_VERSION, host_record, utc_now_iso, write_results_json

MIB = 1024 * 1024


def stats(vals: list[float]) -> dict:
    s = sorted(vals)
    return {"n": len(s), "p50": s[len(s) // 2], "min": s[0], "max": s[-1],
            "mean": sum(s) / len(s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="cpu")
    ap.add_argument("--sizes-mb", default="16,256,1024")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    libc = ctypes.CDLL(None)
    libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    libc.memcpy.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.malloc.restype = ctypes.c_void_p  # 关键：默认 restype=c_int 会截断 64 位指针
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free.restype = None

    result: dict = {
        "schema_version": SCHEMA_VERSION + ".host-mem",
        "created_at_utc": utc_now_iso(),
        "role": "Host 侧计时基准（第 08 期无 CPU 列声明：本期主体是 Host↔设备路径）",
        "label": args.label,
        "host": host_record(),
        "params": vars(args),
    }

    alloc_rows = []
    for size_mb in [int(x) for x in args.sizes_mb.split(",")]:
        nbytes = size_mb * MIB
        a_alloc, a_free, a_touch, a_page = [], [], [], []

        for _ in range(args.trials):
            t0 = time.perf_counter()
            buf = (ctypes.c_char * nbytes)()
            t1 = time.perf_counter()
            ctypes.memset(buf, 1, nbytes)  # 首触（页分配真发生处）
            t2 = time.perf_counter()
            del buf
            t3 = time.perf_counter()
            a_alloc.append((t1 - t0) * 1e3)
            a_touch.append((t2 - t1) * 1e3)
            a_free.append((t3 - t2) * 1e3)

        # 逐 4KB 页首触（缺页中断密度实验）：新缓冲按 stride 触碰
        for _ in range(args.trials):
            buf = (ctypes.c_char * nbytes)()
            t0 = time.perf_counter()
            addr = ctypes.addressof(buf)
            for page in range(0, nbytes, 4096):
                ctypes.c_char.from_address(addr + page).value = b"\x01"
            t1 = time.perf_counter()
            del buf
            a_page.append((t1 - t0) * 1e3)

        # libc malloc 口径（同一物理机的第二分配器对照）
        m_alloc, m_free = [], []
        for _ in range(args.trials):
            t0 = time.perf_counter()
            p = libc.malloc(nbytes) if hasattr(libc, "malloc") else None
            t1 = time.perf_counter()
            libc.free(p)
            t2 = time.perf_counter()
            m_alloc.append((t1 - t0) * 1e3)
            m_free.append((t2 - t1) * 1e3)

        # Host DRAM 拷贝带宽（同尺寸：读+写 = 2×bytes 口径）
        src = (ctypes.c_char * nbytes)()
        dst = (ctypes.c_char * nbytes)()
        ctypes.memset(src, 2, nbytes)
        copy_ms = []
        for _ in range(max(5, args.trials // 2)):
            t0 = time.perf_counter()
            libc.memcpy(dst, src, nbytes)
            copy_ms.append((time.perf_counter() - t0) * 1e3)
        del src, dst

        alloc_rows.append({
            "size_mb": size_mb,
            "python_c_char_array": {"alloc_ms": stats(a_alloc), "first_touch_ms": stats(a_touch),
                                    "page_stride_touch_ms": stats(a_page), "free_ms": stats(a_free)},
            "libc_malloc_free": {"alloc_ms": stats(m_alloc), "free_ms": stats(m_free)},
            "host_dram_copy": {"p50_ms": stats(copy_ms)["p50"],
                               "gbps_rw": 2 * nbytes / (stats(copy_ms)["p50"] * 1e6)},
        })
    result["rows"] = alloc_rows
    write_results_json(args.out, result)
    print(f"[host-mem] 写出 {args.out}")
    for r in result["rows"]:
        print(f"  {r['size_mb']:>5} MB: malloc {r['libc_malloc_free']['alloc_ms']['p50']:8.3f} ms | "
              f"首触 {r['python_c_char_array']['first_touch_ms']['p50']:8.3f} ms | "
              f"DRAM 拷贝 {r['host_dram_copy']['gbps_rw']:6.2f} GB/s(r+w)")


if __name__ == "__main__":
    main()
