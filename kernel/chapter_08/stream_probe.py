#!/usr/bin/env python3
"""第 08 期 stream 语义探测（ep05 遗留核对项 + MUSA 多流结论载体）。

核对目标（chapter_04/KNOWN_ISSUES 未决 3 —— "MUSA 多流语义：默认流与框架
当前流的等价性边界"）：
  T1  torch.<backend>.Stream API 可用性（构造/stream 上下文/current/default/Event）
  T2  torch 流的裸句柄是否可直接作为 musaStream_t/cudaStream_t 传给手写 kernel
      （ctypes 直传，M-17 路径）—— "框架流 == 运行时流句柄" 的行为证据
  T3  默认流等价性：torch default/current 流句柄数值 + 行为判别
      （torch op 写 x 于默认流；kernel 于 NULL 流无同步读 x 写 y ——
        同流 → 顺序化 → y=f(x1)；异流 → 并发 → y=f(x0) 或混合）
  T4  事件互操作：ctypes 事件 record/wait 与 torch 流的组合（跨体系次序）
  T5  torch_musa/torch 源码与二进制中流 API 的存在证据（grep/strings）
  T6  双流并发行为证据：两流各自 kernel 的总耗时 vs 单流串行（并发 → 更短）

输出：results/stream-probe-{backend}.json + server-logs/stream-probe-{backend}.log
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_utils import SCHEMA_VERSION, utc_now_iso, write_results_json
from transfer_bench_lib import KIND_H2D, TransferOps

ROOT = Path(__file__).resolve().parent
SEED = 20260909


def backend_mod(backend: str):
    import torch
    return torch.cuda if backend == "cuda" else torch.musa, torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["cuda", "musa"])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    mod, torch = backend_mod(args.backend)
    dev = torch.device(args.backend)
    ops = TransferOps(args.lib)
    ops.init()
    h = args.h
    rows = args.rows
    nbytes = rows * h * 4

    result: dict = {
        "schema_version": SCHEMA_VERSION + ".stream-probe",
        "created_at_utc": utc_now_iso(),
        "backend": args.backend,
        "torch": torch.__version__,
        "tests": {},
    }

    # ---- T1: torch 流 API 面 -------------------------------------------------
    t1: dict = {
        "has_Stream": hasattr(mod, "Stream"),
        "has_stream_ctx": hasattr(mod, "stream"),
        "has_current_stream": hasattr(mod, "current_stream"),
        "has_default_stream": hasattr(mod, "default_stream"),
        "has_Event": hasattr(mod, "Event"),
        "has_synchronize": hasattr(mod, "synchronize"),
        "stream_methods": sorted(a for a in dir(mod.Stream) if "stream" in a.lower() or a in ("record_event", "wait_event", "wait_stream", "synchronize")) if hasattr(mod, "Stream") else None,
    }
    stream = None
    if t1["has_Stream"]:
        try:
            stream = mod.Stream()
            t1["construct_ok"] = True
        except Exception as exc:  # noqa: BLE001
            t1["construct_ok"] = False
            t1["construct_error"] = repr(exc)
    result["tests"]["T1_torch_api_surface"] = t1
    log(f"[T1] torch API surface: {t1}")

    # ---- 准备数据与 kernel 句柄 ----------------------------------------------
    x_host = ops.malloc_pinned_default(nbytes)
    y_ref_host = ops.malloc_pinned_default(nbytes)
    w_host = ops.malloc_pinned_default(h * 4)
    x_dev = ops.malloc_device(nbytes)
    y_dev = ops.malloc_device(nbytes)
    w_dev = ops.malloc_device(h * 4)
    rng = np.random.default_rng(SEED)
    x_np = rng.standard_normal((rows, h), dtype=np.float32)
    w_np = rng.uniform(0.5, 2.0, h).astype(np.float32)
    ctypes.memmove(ctypes.c_void_p(x_host), x_np.tobytes(), nbytes)
    ctypes.memmove(ctypes.c_void_p(w_host), w_np.tobytes(), h * 4)
    ops.memcpy_sync(w_dev, w_host, h * 4, KIND_H2D)

    def rmsnorm_on(stream_arg: int) -> None:
        ops.rmsnorm(x_dev, w_dev, y_dev, rows, h, 1e-6, 256, stream_arg)

    def check_y(tag: str) -> bool:
        ops.memcpy_sync(y_ref_host, y_dev, nbytes, 1)  # D2H
        y = np.frombuffer((ctypes.c_char * nbytes).from_address(y_ref_host), dtype=np.float32)
        # 参考：单流 rmsnorm（eps 1e-6, tile 256）—— 与 kernel 同语义 numpy 复算
        mean_sq = (x_np.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
        scale = 1.0 / np.sqrt(mean_sq + 1e-6)
        ref = x_np * scale * w_np
        diff = np.abs(y.reshape(rows, h).astype(np.float64) - ref)
        rel = diff.max() / np.abs(ref).max()
        log(f"  [{tag}] max_rel_diff vs numpy fp32 ref = {rel:.3e}")
        return rel < 1e-4

    # ---- T2: torch 流裸句柄 → ctypes kernel ---------------------------------
    t2: dict = {"torch_stream_handle_attr": None, "handle_value": None, "launch_ok": None}
    try:
        ops.memcpy_sync(x_dev, x_host, nbytes, KIND_H2D)
        ops.device_sync()
        if stream is not None:
            attrs = [a for a in dir(stream) if "stream" in a.lower()]
            t2["torch_stream_handle_attr"] = attrs
            handle = None
            for a in attrs:
                v = getattr(stream, a, None)
                if isinstance(v, int) and a != "priority":
                    handle = v
                    t2["handle_attr_used"] = a
                    break
            t2["handle_value"] = handle
            if handle is not None:
                rmsnorm_on(handle)
                mod.synchronize()
                t2["launch_ok"] = check_y("T2 torch-stream-handle launch")
            else:
                t2["launch_ok"] = None
                log("  [T2] 未找到裸句柄属性（记录平台边界）")
    except Exception as exc:  # noqa: BLE001
        t2["launch_ok"] = False
        t2["error"] = repr(exc)
    result["tests"]["T2_torch_handle_interop"] = t2
    log(f"[T2] torch 裸句柄 → ctypes kernel: {t2.get('launch_ok')} (attr={t2.get('handle_attr_used')}, value={t2.get('handle_value')})")

    # ---- T3: 默认流等价性 ----------------------------------------------------
    # 行为判别设计：x_t（torch 张量）持 x0；torch eager op（x_t.add_）写 x；
    # 紧接着（无任何同步）kernel 于候选流读 x_t 写 y。
    #   - 候选流与 eager op 同队 → kernel 严格排在 add_ 之后 → y = f(x0+3000)；
    #   - 不同队 → 并发 → y = f(x0) 或混合值。
    # 判别须用 add_ 而非 mul_：RMSNorm 对行尺度不变（f(c·x)=f(x)），乘法不可见。
    # S4000 首跑发现：kernel 于 NULL(0) 流与 torch eager op 并发（20/20 raced）
    # → 本版补 current_stream 句柄矩阵，定位 eager op 实际所在队列。
    def _get_handle(stream_obj) -> int | None:
        for a in [x for x in dir(stream_obj) if "stream" in x.lower()]:
            v = getattr(stream_obj, a, None)
            if isinstance(v, int) and a != "priority":
                return v
        return None

    t3: dict = {}
    try:
        d_stream = mod.default_stream()
        t3["default_stream_handle"] = _get_handle(d_stream)
        t3["current_stream_handle_outside_ctx"] = _get_handle(mod.current_stream())
        t3["current_is_default"] = bool(mod.current_stream() == mod.default_stream())
        t3["handle_is_null_zero"] = (t3["default_stream_handle"] == 0) if t3["default_stream_handle"] is not None else None
        with mod.stream(mod.default_stream()):
            t3["current_stream_handle_inside_default_ctx"] = _get_handle(mod.current_stream())

        x_t = torch.empty(rows * h, dtype=torch.float32, device=dev)
        x_t.copy_(torch.from_numpy(x_np.reshape(-1)).to(dev))
        mod.synchronize()
        x_ptr = x_t.data_ptr()  # 关键：kernel 与 torch op 操作同一块设备缓冲
        def rmsnorm_on_x_t(stream_arg: int) -> None:
            ops.rmsnorm(x_ptr, w_dev, y_dev, rows, h, 1e-6, 256, stream_arg)

        def read_y_np() -> np.ndarray:
            ops.memcpy_sync(y_ref_host, y_dev, nbytes, 1)
            return np.frombuffer((ctypes.c_char * nbytes).from_address(y_ref_host),
                                 dtype=np.float32).reshape(rows, h).astype(np.float64)

        mean_sq = (x_np.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
        scale0 = 1.0 / np.sqrt(mean_sq + 1e-6)
        y_f_x0 = x_np * scale0 * w_np          # f(x0)
        ADD_TIMES = 30  # 30 次 add_ 拉长 eager 流忙窗口（单次 add_ ~10µs 与 kernel 发射同量级，判别力不足）
        x1_np = x_np + 100.0 * ADD_TIMES
        y_f_x1 = x1_np / np.sqrt((x1_np.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6) * w_np  # f(x0+3000)

        def behavior(trials: int, eager_ctx: str, kernel_handle: int | None, tag: str) -> dict:
            ordered = raced = mixed = 0
            for _ in range(trials):
                ops.device_sync()
                x_t.copy_(torch.from_numpy(x_np.reshape(-1)).to(dev))  # 复位 x（防跨 trial 累加）
                if eager_ctx == "default":
                    with mod.stream(mod.default_stream()):
                        for _ in range(ADD_TIMES):
                            x_t.add_(100.0)
                else:  # "plain"：无显式流上下文（torch 内部当前流，最贴近真实 eager）
                    for _ in range(ADD_TIMES):
                        x_t.add_(100.0)
                rmsnorm_on_x_t(0 if kernel_handle is None else kernel_handle)
                mod.synchronize()
                y = read_y_np()
                d1 = np.abs(y - y_f_x1).max()
                d0 = np.abs(y - y_f_x0).max()
                tol = 1e-3 * np.abs(y_f_x1).max()
                if d1 <= tol and d0 > tol:
                    ordered += 1
                elif d0 <= tol and d1 > tol:
                    raced += 1
                else:
                    mixed += 1
            return {"tag": tag, "trials": trials, "ordered_f_of_x1": ordered,
                    "raced_f_of_x0": raced, "mixed": mixed}

        cur_handle = t3["current_stream_handle_outside_ctx"]
        t3["behavior"] = [
            behavior(20, "plain", 0, "kernel@NULL(0) vs eager(无上下文)"),
            behavior(20, "default", 0, "kernel@NULL(0) vs eager(default_stream 上下文)"),
        ]
        if cur_handle is not None and cur_handle != 0:
            t3["behavior"].append(
                behavior(20, "plain", cur_handle, f"kernel@current({cur_handle}) vs eager(无上下文)"))
        t3["interpretation"] = ("ordered>0 且 raced=0 ⇒ kernel 与 eager op 同队次序；"
                                "raced>0 ⇒ 两队并发（异流实锤）。current 句柄行定位 eager 实际队列")
        ops.clear_last_error()
    except Exception as exc:  # noqa: BLE001
        t3["error"] = repr(exc)
    result["tests"]["T3_default_stream_equivalence"] = t3
    log(f"[T3] default handle={t3.get('default_stream_handle')} ==0? {t3.get('handle_is_null_zero')} "
        f"current_outside={t3.get('current_stream_handle_outside_ctx')} "
        f"current_inside_default_ctx={t3.get('current_stream_handle_inside_default_ctx')}")
    for b in t3.get("behavior", []) or []:
        log(f"[T3] {b}")

    # ---- T4: 事件互操作 ------------------------------------------------------
    t4: dict = {}
    try:
        ops.clear_last_error()
        s1 = mod.Stream() if stream is not None else None
        ops.memcpy_sync(x_dev, x_host, nbytes, KIND_H2D)
        ops.device_sync()
        # 方向 1：torch 流上记录事件 → ctypes 流 wait 该事件 → kernel → torch 同步可见
        # （S4000 首跑发现：torch Event 裸句柄属性只有 event_id=0（计数器非句柄），
        #   拿不到 musaEvent_t → 该方向按平台边界记录，不再用 0 当句柄调用）
        if s1 is not None:
            ev = mod.Event()
            ev.record(s1)
            ev_handle = None
            for a in dir(ev):
                if "event" in a.lower():
                    v = getattr(ev, a, None)
                    if isinstance(v, int):
                        ev_handle = v
                        t4["event_handle_attr"] = a
                        break
            t4["event_handle_value"] = ev_handle
            t4["event_handle_is_valid"] = bool(ev_handle)  # 0/None = 拿不到真句柄
            if ev_handle:
                raw_a = ops.stream_create()
                try:
                    ops.stream_wait_event(raw_a, ev_handle)
                    rmsnorm_on(raw_a)
                    ops.stream_sync(raw_a)
                    t4["torch_event_waited_by_ctypes_stream"] = check_y("T4 event interop")
                finally:
                    ops.stream_destroy(raw_a)
            else:
                t4["torch_event_waited_by_ctypes_stream"] = None
                t4["boundary_note"] = ("torch Event 无裸 musaEvent 句柄导出（属性仅 "
                                       f"{t4.get('event_handle_attr')}={ev_handle}）→ "
                                       "方向 1 平台边界；torch 事件与手写流无法跨体系等待")
        # 方向 2：ctypes 事件 → torch 流 wait_event（torch 流等待手写事件）
        ev2 = ops.event_create()
        ops.event_record(ev2, 0)
        try:
            if s1 is not None:
                s1.wait_event(ev2)  # torch_musa 2.11: 只接受 torch Event，raw int 会 AttributeError
            with mod.stream(s1):
                torch.mul(torch.ones(4, device=dev), 2.0)
            mod.synchronize()
            t4["ctypes_event_waited_by_torch_stream"] = True
        except Exception as exc2:  # noqa: BLE001
            t4["ctypes_event_waited_by_torch_stream"] = False
            t4["ctypes_event_wait_error"] = repr(exc2)
            t4["boundary_note_2"] = ("torch 流不能等待裸 musaEvent 句柄（wait_event 只接受 torch Event）"
                                     "→ 跨体系事件等待双向都不通；混用场景以 torch 层 synchronize 兜底")
        ops.event_destroy(ev2)
    except Exception as exc:  # noqa: BLE001
        t4["error"] = repr(exc)
    result["tests"]["T4_event_interop"] = t4
    log(f"[T4] event interop: {t4}")

    # ---- T6: 双流并发行为证据 ------------------------------------------------
    t6: dict = {}
    try:
        ops.clear_last_error()  # T4 的无效句柄错误若未被清除会污染本段 kernel 的 GetLastError 检查
        reps = 20
        # 单流串行基线：两份 rmsnorm 背靠背
        ops.memcpy_sync(x_dev, x_host, nbytes, KIND_H2D)
        ops.device_sync()
        ev_a, ev_b = ops.event_create(), ops.event_create()
        ops.event_record(ev_a, 0)
        for _ in range(reps):
            rmsnorm_on(0)
            rmsnorm_on(0)
        ops.event_record(ev_b, 0)
        ops.device_sync()
        serial_ms = ops.event_elapsed_ms(ev_a, ev_b)
        # 双流：A 跑 reps 次，B 跑 reps 次（共享只读 x + 各自独立 y：并行安全）
        y2_dev = ops.malloc_device(nbytes)
        raw_a = ops.stream_create()
        raw_b = ops.stream_create()
        ev_c = ops.event_create()
        ops.event_record(ev_a, raw_a)
        for _ in range(reps):
            ops.rmsnorm(x_dev, w_dev, y_dev, rows, h, 1e-6, 256, raw_a)
            ops.rmsnorm(x_dev, w_dev, y2_dev, rows, h, 1e-6, 256, raw_b)
        ops.event_record(ev_b, raw_a)
        ops.event_record(ev_c, raw_b)
        ops.device_sync()
        span_a = ops.event_elapsed_ms(ev_a, ev_b)
        span_b = ops.event_elapsed_ms(ev_a, ev_c)
        dual_ms = max(span_a, span_b)  # 两流中较晚结束者（真实并发跨度）
        ops.event_destroy(ev_a)
        ops.event_destroy(ev_b)
        ops.event_destroy(ev_c)
        ops.stream_destroy(raw_a)
        ops.stream_destroy(raw_b)
        ops.free_device(y2_dev)
        t6 = {"serial_two_chains_ms": serial_ms, "dual_stream_two_chains_ms": dual_ms,
              "concurrency_ratio": serial_ms / dual_ms if dual_ms else None,
              "note": "ratio>1.1 ⇒ 两流真并发（SM 资源允许时）"}
    except Exception as exc:  # noqa: BLE001
        t6 = {"error": repr(exc)}
    result["tests"]["T6_dual_stream_concurrency"] = t6
    log(f"[T6] dual-stream concurrency: {t6}")

    # ---- T5: 源码/二进制证据 -------------------------------------------------
    t5: dict = {}
    try:
        try:
            import torch_musa  # noqa: F401  （CUDA 平台无此包 → 走 torch 本体分支）
            pkg = Path(torch_musa.__file__).resolve().parent
            symbol = "musaStreamCreate"
        except ImportError:
            pkg = Path(torch.__file__).resolve().parent
            symbol = "cudaStreamCreate"
        # py 源码证据
        py_grep = subprocess.run(
            ["bash", "-c", f"grep -rl '{symbol}' {pkg} --include='*.py' 2>/dev/null | head -5"],
            capture_output=True, text=True, timeout=120)
        # 二进制证据：包下全部 .so 中符号出现次数（grep -a 按文本扫二进制；strings 未必安装）
        so_grep = subprocess.run(
            ["bash", "-c",
             f"for so in $(find {pkg} -name '*.so*' -size +1M 2>/dev/null | head -8); do "
             f"c=$(grep -ac '{symbol}' $so 2>/dev/null); echo \"$so : $c\"; done"],
            capture_output=True, text=True, timeout=300)
        t5 = {"package": str(pkg), "symbol": symbol,
              "py_files_with_symbol": py_grep.stdout.strip(),
              "so_counts": so_grep.stdout.strip()}
    except Exception as exc:  # noqa: BLE001
        t5 = {"error": repr(exc)}
    result["tests"]["T5_impl_evidence"] = t5
    log(f"[T5] impl evidence: {t5}")

    # 清理
    ops.free_pinned(x_host)
    ops.free_pinned(y_ref_host)
    ops.free_pinned(w_host)
    ops.free_device(x_dev)
    ops.free_device(y_dev)
    ops.free_device(w_dev)

    result["log"] = log_lines
    write_results_json(args.out, result)
    log_path = ROOT / "results" / "server-logs" / f"stream-probe-{args.backend}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[stream-probe] 写出 {args.out}")


if __name__ == "__main__":
    main()
