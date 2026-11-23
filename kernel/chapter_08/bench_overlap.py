#!/usr/bin/env python3
"""第 08 期传输-计算重叠基准（实验 2 + 实验 4）。

流水线设计（chunked + 双 stream，全部经 ctypes 直传，M-17 模式）：
  - stream A（传输）：全部 H2D 与 D2H；stream B（计算）：全部 RMSNorm 轮次；
  - 深缓冲：每 chunk 一对 in_buf[i]/out_buf[i]（设备侧 = 2×payload）——不用
    双缓冲复用。原因：三段式发射下，若设备跑得比 Host 发射快，跨流 wait 引用
    "尚未 record 的事件"时按 CUDA 语义是 no-op（wait 立即通过），双缓冲复用
    依赖该 wait 做保护 → 出现"H2D 覆盖正在被计算的输入缓冲"的发射滞后竞态
    （首版实现本地推演抓出，见 EXPERIMENT_LOG/KNOWN_ISSUES）。深缓冲下所有
    wait 引用的事件都在更早的发射段必然已 record，正确性与发射速率解耦；
  - 事件依赖（无环）：
      phase1: A: H2D(i) → record ev_h[i]
      phase2: B: wait ev_h[i] → K 轮 RMSNorm(i) → record ev_c[i]
      phase3: A: wait ev_c[i] → D2H(i)（ev_d[i] 仅记录不使用）
  - 理论上限 = max(T_transfer, T_compute)：A 忙 = H2D 链 + D2H 链（=T_transfer），
    B 忙 = K×RMSNorm 链（=T_compute），两流互不串行阻塞 → 结构上逼近上限，
    剩余稀释 = 流水线填充/排空尾部 + per-op launch 地板 + Host 发射速率。

传输/计算比场景（{0.25, 1, 4} + natural）：
  - RMSNorm 是 ~1 FLOP/byte 的 memory-bound 算子：payload 与行数 1:1 耦合时，
    T_transfer/T_compute = BW_device/BW_pcie ≈ 硬件常数（4090 实测 ~36）——
    单靠"改 payload 大小 vs 行数"无法构造三档比例，这是本期要显式记录的
    结构性结论（验收问题 4 的证据）；
  - 因此比例旋钮 = 每块 K 轮 RMSNorm（K = round(r_ref/ratio)，r_ref 为 K=1
    时同分块的 T_transfer/T_compute 实测值）——计算负载按比例放大，payload
    不变。natural 档（K=1）保留原始比例作为对照。

正确性（实验 4）：overlap 流水线结果 vs 单流串行结果 位级一致（memcmp）；
  同 kernel 同 K 轮同序，仅跨 chunk 的发射次序不同 → 位级相等是确定性前提
  （第 07 期结论）下的硬校验。

nsys 证据：--nsys-mode 跑单次小配置（供 run_nsys.sh 包裹），时间线上可见
memcpy 与 rmsnorm kernel 的并行区间（check_overlap_trace.py 解析）。

输出：results/overlap-{backend}.json（schema kernel-chapter09.overlap.v1）
"""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_utils import SCHEMA_VERSION, backend_environment_record, utc_now_iso, write_results_json
from transfer_bench_lib import KIND_D2H, KIND_H2D, memcmp_equal, TransferOps

ROOT = Path(__file__).resolve().parent
SEED = 20260909
MIB = 1024 * 1024
K_MAX = 2000


def percentile(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def stats(vals: list[float]) -> dict:
    s = sorted(vals)
    return {"n": len(s), "p50": s[len(s) // 2], "min": s[0], "max": s[-1],
            "p95": percentile(s, 0.95), "mean": sum(s) / len(s)}


class Pipeline:
    """一套 (payload, chunk) 缓冲/事件资源；K 与运行模式按需组合。

    设备侧缓冲 = 每 chunk 一对（深缓冲，2×payload）：消除缓冲复用依赖后，
    流水线正确性不依赖 Host 发射速率（发射滞后竞态的教训见模块 docstring）。
    """

    def __init__(self, ops: TransferOps, payload_bytes: int, chunk_bytes: int, h: int, w_dev: int):
        assert payload_bytes % chunk_bytes == 0, "payload 必须整除 chunk（本期合同）"
        self.ops = ops
        self.payload = payload_bytes
        self.chunk = chunk_bytes
        self.n = payload_bytes // chunk_bytes
        self.h = h
        self.w_dev = w_dev
        self.host_src = ops.malloc_pinned_mallochost(payload_bytes)
        self.host_dst_serial = ops.malloc_pinned_mallochost(payload_bytes)
        self.host_dst_overlap = ops.malloc_pinned_mallochost(payload_bytes)
        self.in_buf = [ops.malloc_device(chunk_bytes) for _ in range(self.n)]
        self.out_buf = [ops.malloc_device(chunk_bytes) for _ in range(self.n)]
        self.ev_h = [ops.event_create() for _ in range(self.n)]
        self.ev_c = [ops.event_create() for _ in range(self.n)]
        rng = np.random.default_rng(SEED)
        data = rng.standard_normal(payload_bytes // 4, dtype=np.float32).tobytes()
        ctypes.memmove(ctypes.c_void_p(self.host_src), data, payload_bytes)
        del data

    def destroy(self) -> None:
        for e in self.ev_h + self.ev_c:
            self.ops.event_destroy(e)
        for p in self.in_buf + self.out_buf:
            self.ops.free_device(p)
        self.ops.free_pinned(self.host_src)
        self.ops.free_pinned(self.host_dst_serial)
        self.ops.free_pinned(self.host_dst_overlap)

    # -- 基本动作 -----------------------------------------------------------

    def h2d(self, i: int, s: int) -> None:
        self.ops.memcpy_async(self.in_buf[i], self.host_src + i * self.chunk, self.chunk, KIND_H2D, s)

    def d2h_serial(self, i: int, s: int) -> None:
        self.ops.memcpy_async(self.host_dst_serial + i * self.chunk, self.out_buf[i], self.chunk, KIND_D2H, s)

    def d2h_overlap(self, i: int, s: int) -> None:
        self.ops.memcpy_async(self.host_dst_overlap + i * self.chunk, self.out_buf[i], self.chunk, KIND_D2H, s)

    def compute_chunk(self, i: int, k_rounds: int, s: int, rows: int) -> None:
        """K 轮 RMSNorm：round 0 in→out，round>=1 原地（H<=2048 时 kernel 经
        shared 读 x，in-place 安全；位级确定性与串行一致）。"""
        self.ops.rmsnorm(self.in_buf[i], self.w_dev, self.out_buf[i], rows, self.h, 1e-6, 256, s)
        for _ in range(k_rounds - 1):
            self.ops.rmsnorm(self.out_buf[i], self.w_dev, self.out_buf[i], rows, self.h, 1e-6, 256, s)

    # -- 独立段测量（链级事件对，单次；数字稳定性由外层 repeat 承担） ----------

    def time_h2d(self, stream: int) -> float:
        ev_a, ev_b = self.ops.event_create(), self.ops.event_create()
        self.ops.event_record(ev_a, stream)
        for i in range(self.n):
            self.h2d(i, stream)
        self.ops.event_record(ev_b, stream)
        self.ops.device_sync()
        ms = self.ops.event_elapsed_ms(ev_a, ev_b)
        self.ops.event_destroy(ev_a)
        self.ops.event_destroy(ev_b)
        return ms

    def time_d2h(self, stream: int) -> float:
        ev_a, ev_b = self.ops.event_create(), self.ops.event_create()
        self.ops.event_record(ev_a, stream)
        for i in range(self.n):
            self.d2h_overlap(i, stream)
        self.ops.event_record(ev_b, stream)
        self.ops.device_sync()
        ms = self.ops.event_elapsed_ms(ev_a, ev_b)
        self.ops.event_destroy(ev_a)
        self.ops.event_destroy(ev_b)
        return ms

    def time_comp(self, stream: int, k_rounds: int, rows: int) -> float:
        ev_a, ev_b = self.ops.event_create(), self.ops.event_create()
        self.ops.event_record(ev_a, stream)
        for i in range(self.n):
            self.compute_chunk(i, k_rounds, stream, rows)
        self.ops.event_record(ev_b, stream)
        self.ops.device_sync()
        ms = self.ops.event_elapsed_ms(ev_a, ev_b)
        self.ops.event_destroy(ev_a)
        self.ops.event_destroy(ev_b)
        return ms

    # -- 两种流水线 ----------------------------------------------------------

    def run_serial(self, stream_a: int, k_rounds: int, rows: int) -> None:
        s = stream_a
        for i in range(self.n):
            self.h2d(i, s)
            self.compute_chunk(i, k_rounds, s, rows)
            self.d2h_serial(i, s)

    def run_overlap(self, stream_a: int, stream_b: int, k_rounds: int, rows: int) -> float:
        """三段式发射；返回发射耗时 ms（Host 侧，用于发射瓶颈归因）。
        所有 wait 引用的事件都在更早发射段必然已 record（深缓冲设计）。"""
        A, B = stream_a, stream_b
        t0 = time.perf_counter()
        # phase1：全部 H2D（A）
        for i in range(self.n):
            self.h2d(i, A)
            self.ops.event_record(self.ev_h[i], A)
        # phase2：全部计算（B）
        for i in range(self.n):
            self.ops.stream_wait_event(B, self.ev_h[i])
            self.compute_chunk(i, k_rounds, B, rows)
            self.ops.event_record(self.ev_c[i], B)
        # phase3：全部 D2H（A）
        for i in range(self.n):
            self.ops.stream_wait_event(A, self.ev_c[i])
            self.d2h_overlap(i, A)
        emit_ms = (time.perf_counter() - t0) * 1e3
        return emit_ms


def measure_pair(pipe: Pipeline, ops: TransferOps, stream_a: int, stream_b: int,
                 k_rounds: int, rows: int, warmup: int, repeats: int) -> dict:
    ev_a = ops.event_create()
    ev_b = ops.event_create()
    serial_ms, overlap_ms, overlap_emit_ms = [], [], []
    ops.device_sync()
    for _ in range(warmup):
        pipe.run_serial(stream_a, k_rounds, rows)
        ops.device_sync()
        ops.event_record(ev_a, stream_a)
        pipe.run_overlap(stream_a, stream_b, k_rounds, rows)
        ops.event_record(ev_b, stream_a)
        ops.device_sync()
    for _ in range(repeats):
        # 串行
        ops.device_sync()
        ops.event_record(ev_a, stream_a)
        pipe.run_serial(stream_a, k_rounds, rows)
        ops.event_record(ev_b, stream_a)
        ops.device_sync()
        serial_ms.append(ops.event_elapsed_ms(ev_a, ev_b))
        # 重叠（含发射耗时记录）
        ops.device_sync()
        ops.event_record(ev_a, stream_a)
        emit = pipe.run_overlap(stream_a, stream_b, k_rounds, rows)
        ops.event_record(ev_b, stream_a)
        ops.device_sync()
        overlap_ms.append(ops.event_elapsed_ms(ev_a, ev_b))
        overlap_emit_ms.append(emit)
    ops.event_destroy(ev_a)
    ops.event_destroy(ev_b)
    return {"serial_ms": stats(serial_ms), "overlap_ms": stats(overlap_ms),
            "overlap_emit_ms": stats(overlap_emit_ms)}


def run_config(ops: TransferOps, stream_a: int, stream_b: int, payload_bytes: int,
               chunk_bytes: int, ratio: float | None, h: int,
               warmup: int, repeats: int, w_dev: int, launch_floor_us: float,
               do_correctness: bool = True) -> dict:
    rows = chunk_bytes // (h * 4)
    pipe = Pipeline(ops, payload_bytes, chunk_bytes, h, w_dev)
    out: dict = {
        "payload_mb": payload_bytes / MIB, "chunk_mb": chunk_bytes / MIB,
        "n_chunks": pipe.n, "rows_per_chunk": rows, "h": h,
    }

    # ---- 独立段（K=1）与比例校准 ----
    t_h2d = pipe.time_h2d(stream_a)
    t_d2h = pipe.time_d2h(stream_a)
    t_comp1 = pipe.time_comp(stream_b, 1, rows)
    t_transfer = t_h2d + t_d2h
    r_ref = t_transfer / t_comp1 if t_comp1 > 0 else float("inf")
    k_rounds = 1 if ratio is None else max(1, min(K_MAX, round(r_ref / ratio)))
    t_comp_k = pipe.time_comp(stream_b, k_rounds, rows) if k_rounds > 1 else t_comp1
    achieved_ratio = t_transfer / t_comp_k if t_comp_k > 0 else None

    out["standalone"] = {
        "h2d_ms": t_h2d, "d2h_ms": t_d2h, "transfer_ms": t_transfer,
        "comp_k1_ms": t_comp1, "comp_k_ms": t_comp_k, "k_rounds": k_rounds,
        "ratio_target": ratio, "ratio_ref_k1": r_ref, "ratio_achieved": achieved_ratio,
        "per_chunk": {"h2d_ms": t_h2d / pipe.n, "d2h_ms": t_d2h / pipe.n,
                      "comp_k_ms": t_comp_k / pipe.n},
    }

    # ---- 流水线 ----
    pair = measure_pair(pipe, ops, stream_a, stream_b, k_rounds, rows, warmup, repeats)
    t_serial = pair["serial_ms"]["p50"]
    t_overlap = pair["overlap_ms"]["p50"]
    upper = max(t_transfer, t_comp_k)
    n_ops = pipe.n * (k_rounds + 2)
    out["pipeline"] = {
        **pair,
        "upper_bound_ms": upper,
        "speedup_measured": t_serial / t_overlap if t_overlap else None,
        "speedup_ideal": t_serial / upper if upper else None,
        "overlap_efficiency": upper / t_overlap if t_overlap else None,
        "dilution_fraction": 1 - upper / t_overlap if t_overlap else None,
        "dilution_attribution": {
            "n_device_ops": n_ops,
            "launch_floor_us": launch_floor_us,
            "launch_overhead_est_ms": n_ops * launch_floor_us / 1e3,
            "fill_drain_tail_est_ms": (t_h2d + t_d2h + t_comp_k) / pipe.n,
            "host_emit_p50_ms": pair["overlap_emit_ms"]["p50"],
            "emit_bound": pair["overlap_emit_ms"]["p50"] > t_overlap,
        },
    }

    # ---- 正确性（位级） ----
    if do_correctness:
        ops.device_sync()
        pipe.run_serial(stream_a, k_rounds, rows)
        ops.device_sync()
        pipe.run_overlap(stream_a, stream_b, k_rounds, rows)
        ops.device_sync()
        mismatches = []
        for i in range(pipe.n):
            if not memcmp_equal(pipe.host_dst_serial + i * pipe.chunk,
                                pipe.host_dst_overlap + i * pipe.chunk, pipe.chunk):
                mismatches.append(i)
        out["correctness"] = {
            "bitwise_equal_all_chunks": not mismatches,
            "mismatch_chunks": mismatches[:10],
            "compared_bytes": payload_bytes,
            "method": "libc memcmp（零拷贝）",
        }
    pipe.destroy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["cuda", "musa"])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--payload-mb", type=int, default=512)
    ap.add_argument("--chunk-mb", type=int, default=16)
    ap.add_argument("--ratios", default="0.25,1,4")
    ap.add_argument("--include-natural", action="store_true", default=True)
    ap.add_argument("--sweep-chunks-mb", default="1,4,16,64,256")
    ap.add_argument("--sweep-payload-mb", type=int, default=256)
    ap.add_argument("--sweep-ratio", type=float, default=1.0)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--floor-samples", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsys-mode", action="store_true",
                    help="单次小配置（供 nsys 包裹采集时间线）")
    args = ap.parse_args()

    ops = TransferOps(args.lib)
    rc_init = ops.init()
    if rc_init != 0:
        print(f"[overlap] t_init rc={rc_init}（记录）")
    info = ops.dev_info()
    stream_a = ops.stream_create()
    stream_b = ops.stream_create()

    h = args.h
    w_host = ops.malloc_pinned_default(h * 4)
    w_arr = np.random.default_rng(SEED + 1).uniform(0.5, 2.0, h).astype(np.float32)
    ctypes.memmove(ctypes.c_void_p(w_host), w_arr.tobytes(), h * 4)
    w_dev = ops.malloc_device(h * 4)
    ops.memcpy_sync(w_dev, w_host, h * 4, KIND_H2D)
    ops.free_pinned(w_host)

    # ---- launch/event 地板（M-12 口径）：GPU 空闲时单 op 事件对 P50 ----
    ev_a, ev_b = ops.event_create(), ops.event_create()
    tiny_in = ops.malloc_device(4 * h * 4)
    tiny_out = ops.malloc_device(4 * h * 4)
    ops.device_sync()
    floors, enqueues = [], []
    for _ in range(args.floor_samples):
        ops.event_record(ev_a, stream_a)
        ops.rmsnorm(tiny_in, w_dev, tiny_out, 4, h, 1e-6, args.tile, stream_a)
        ops.event_record(ev_b, stream_a)
        ops.device_sync()
        floors.append(ops.event_elapsed_ms(ev_a, ev_b) * 1e3)  # µs
        t0 = time.perf_counter()
        for _ in range(10):
            ops.rmsnorm(tiny_in, w_dev, tiny_out, 4, h, 1e-6, args.tile, stream_a)
        enqueues.append((time.perf_counter() - t0) / 10 * 1e6)  # µs/op
    ops.device_sync()
    launch_floor_us = sorted(floors)[len(floors) // 2]
    enqueue_floor_us = sorted(enqueues)[len(enqueues) // 2]
    ops.event_destroy(ev_a)
    ops.event_destroy(ev_b)
    ops.free_device(tiny_in)
    ops.free_device(tiny_out)

    result: dict = {
        "schema_version": SCHEMA_VERSION + ".overlap",
        "created_at_utc": utc_now_iso(),
        "backend": args.backend,
        "lib": str(Path(args.lib).resolve()),
        "device": info,
        "params": {k: v for k, v in vars(args).items()} | {"seed": SEED},
        "launch_floor": {"event_pair_p50_us": launch_floor_us,
                         "enqueue_us_per_op": enqueue_floor_us,
                         "samples": args.floor_samples,
                         "note": "M-12 口径：GPU 空闲单 op 事件对地板（MUSA ~28µs 量级预期）"},
    }

    if args.nsys_mode:
        # nsys 目标：串行相位 + 重叠相位各一次（时间线上对比并行区间）
        chunk = args.chunk_mb * MIB
        payload = args.payload_mb * MIB
        rows = chunk // (h * 4)
        pipe = Pipeline(ops, payload, chunk, h, w_dev)
        k_nsys = 8
        pipe.run_serial(stream_a, k_nsys, rows)
        ops.device_sync()
        pipe.run_overlap(stream_a, stream_b, k_nsys, rows)
        ops.device_sync()
        pipe.destroy()
        result["nsys_mode"] = {"k_rounds": k_nsys, "rows_per_chunk": rows,
                               "payload_mb": args.payload_mb, "chunk_mb": args.chunk_mb}
        write_results_json(args.out, result)
        print("[overlap] nsys-mode 完成")
        return

    # ---- 场景表：ratio ∈ {0.25, 1, 4} + natural（K=1） ----
    scenarios = []
    ratio_list: list[float | None] = [float(tok) for tok in args.ratios.split(",")]
    if args.include_natural:
        ratio_list.append(None)
    for ratio in ratio_list:
        cfg = run_config(ops, stream_a, stream_b, args.payload_mb * MIB,
                         args.chunk_mb * MIB, ratio, h,
                         args.warmup, args.repeats, w_dev, launch_floor_us)
        scenarios.append(cfg)
        st = cfg["standalone"]
        pl = cfg["pipeline"]
        ach = st["ratio_achieved"]
        print(f"[overlap] ratio={st['ratio_target']} K={st['k_rounds']:>4} "
              f"achieved={ach and round(ach, 2)} "
              f"serial={pl['serial_ms']['p50']:.2f}ms overlap={pl['overlap_ms']['p50']:.2f}ms "
              f"upper={pl['upper_bound_ms']:.2f}ms "
              f"speedup={pl['speedup_measured'] and round(pl['speedup_measured'], 3)} "
              f"eff={pl['overlap_efficiency'] and round(pl['overlap_efficiency'], 3)} "
              f"bitwise={cfg['correctness']['bitwise_equal_all_chunks']}")
    result["scenario_table"] = {
        "payload_mb": args.payload_mb, "chunk_mb": args.chunk_mb,
        "note": "ratio=传输时间/计算时间（K 轮校准）；natural=K=1 原始比例",
        "rows": scenarios,
    }

    # ---- 分块大小扫描（ratio=sweep-ratio） ----
    sweep = []
    for chunk_mb in [float(x) for x in args.sweep_chunks_mb.split(",")]:
        payload_mb = max(args.sweep_payload_mb, 4 * chunk_mb)
        if int(payload_mb) % int(chunk_mb) != 0:
            payload_mb = chunk_mb * 4
        cfg = run_config(ops, stream_a, stream_b, int(payload_mb) * MIB,
                         int(chunk_mb) * MIB, args.sweep_ratio, h,
                         args.warmup, args.repeats, w_dev, launch_floor_us)
        sweep.append(cfg)
        pl = cfg["pipeline"]
        print(f"[sweep] chunk={chunk_mb:>6}MB n={cfg['n_chunks']:>4} "
              f"overlap={pl['overlap_ms']['p50']:.2f}ms "
              f"eff={pl['overlap_efficiency'] and round(pl['overlap_efficiency'], 3)} "
              f"emit_bound={pl['dilution_attribution']['emit_bound']} "
              f"bitwise={cfg['correctness']['bitwise_equal_all_chunks']}")
    result["chunk_sweep"] = {
        "ratio_target": args.sweep_ratio,
        "note": "payload=max(256MB, 4×chunk) 保证流水线深度 ≥4",
        "rows": sweep,
    }

    ops.stream_destroy(stream_a)
    ops.stream_destroy(stream_b)
    ops.free_device(w_dev)
    result["environment"] = backend_environment_record(args.backend)
    write_results_json(args.out, result)
    print(f"[overlap] 写出 {args.out}")


if __name__ == "__main__":
    main()
