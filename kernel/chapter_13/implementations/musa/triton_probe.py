#!/usr/bin/env python3
"""第 13 期 Triton 后端可用性探测（本期首要实验——大纲 §14 准入第 7 条）。

探测内容（全部实测取证，输出 JSON）：
  A. 包层：pip 元数据（triton / triton_musa / torch_musa 版本与路径）、
     triton/backends 目录清单（是否存在非 CUDA/NVIDIA 后端）、
     torch 依赖树里对 triton 的声明（pip show torch / torch_musa Requires）；
  B. import 层：import triton 版本、triton.language 可用性、
     import triton_musa 尝试（MUSA 侧）、torch_musa 里的 triton 集成探测；
  C. driver 层：triton.runtime.driver.active 类型、get_current_target()
     （GPUTarget: backend/arch/warp_size）、get_device_capability()；
  D. 编译层：首个 kernel 从 import 到跑通的分段墙钟
     （t_import / t_define / t_first_launch_compile / t_first_sync / t_second_launch）
     —— "开发耗时"数据点的机器可测部分；
  E. 正确性：首个 kernel 输出 vs torch 组合表达式同设备对照；
  F. （MUSA 侧）vllm-musa 目录只读参照：/data/feihu/vllm-musa-029 内
     triton 生态痕迹（ls + grep 级别，只读）。

用法：python triton_probe.py --backend cuda|musa [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import triton.language as tl  # noqa: PLC0415  # 模块级：jit 前端只读模块全局（3.2 实测）

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

BACKEND = os.environ.get("CH13_BACKEND") or os.environ.get("CH14_BACKEND", "cuda")


def step(name: str):
    print(f"\n===== [probe/{BACKEND}] {name} =====")


def record(store: dict, name: str, fn):
    try:
        store[name] = fn()
    except Exception as error:  # noqa: BLE001
        store[name] = {"error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()[-2000:]}
        print(f"[probe] {name} 失败: {type(error).__name__}: {error}")
    return store[name]


def pip_metadata(names: list[str]) -> dict:
    out = {}
    for name in names:
        try:
            dist = importlib.metadata.distribution(name)
            out[name] = {
                "version": dist.version,
                "location": str(dist.locate_file("")),
                "requires": dist.requires,
            }
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def ls_dir(path: str) -> dict:
    p = Path(path)
    if not p.is_dir():
        return {"exists": False}
    try:
        entries = sorted(e.name for e in p.iterdir())
        return {"exists": True, "n_entries": len(entries), "entries": entries[:200]}
    except Exception as error:  # noqa: BLE001
        return {"exists": True, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=BACKEND, choices=["cuda", "musa"])
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    backend = arguments.backend
    result: dict = {"backend": backend, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- A. 包层 ----
    step("A. 包层（pip 元数据 / backends 目录）")
    result["pip"] = record(result, "pip", lambda: pip_metadata(
        ["triton", "triton_musa", "torch", "torch_musa", "pytorch-triton"]))
    try:
        import triton  # noqa: PLC0415
        triton_pkg_dir = Path(triton.__file__).resolve().parent
        result["triton_package_dir"] = str(triton_pkg_dir)
        result["triton_backends_dir"] = record(
            result, "triton_backends_dir", lambda: ls_dir(triton_pkg_dir / "backends"))
        result["triton_version"] = triton.__version__
    except Exception as error:  # noqa: BLE001
        result["triton_import_error"] = f"{type(error).__name__}: {error}"

    # ---- B. import 层（MUSA 特有：triton_musa / torch_musa 集成） ----
    step("B. import 层")
    if backend == "musa":
        def probe_triton_musa():
            spec = importlib.util.find_spec("triton_musa")
            info = {"findable": spec is not None}
            if spec is not None:
                info["origin"] = str(spec.origin)
                mod = importlib.import_module("triton_musa")
                info["module_file"] = getattr(mod, "__file__", None)
                info["attrs_head"] = [a for a in dir(mod) if not a.startswith("_")][:60]
            return info
        record(result, "triton_musa_import", probe_triton_musa)
        def probe_torch_musa_triton():
            import torch_musa  # noqa: PLC0415
            tm = Path(torch_musa.__file__).resolve().parent
            hits = []
            for pattern in ("*triton*", "*inductor*"):
                hits += [str(p.relative_to(tm)) for p in tm.rglob(pattern)][:20]
            return {"torch_musa_dir": str(tm), "triton_related_files": hits}
        record(result, "torch_musa_triton_tree", probe_torch_musa_triton)
        def probe_vllm_musa():
            return ls_dir("/data/feihu/vllm-musa-029")
        record(result, "vllm_musa_dir", probe_vllm_musa)
        def grep_vllm_triton():
            base = Path("/data/feihu/vllm-musa-029")
            if not base.is_dir():
                return {"exists": False}
            hits = []
            for pattern in ("triton*",):
                for p in list(base.rglob(pattern))[:40]:
                    hits.append(str(p))
            return {"exists": True, "triton_path_hits": hits[:40]}
        record(result, "vllm_musa_triton_hits", grep_vllm_triton)

    # ---- C/D/E. driver 层 + 首个 kernel 全过程计时 + 正确性 ----
    step("C/D/E. driver.active / target / 首个 kernel 全过程")
    import torch  # noqa: PLC0415
    result["torch_version"] = torch.__version__

    def driver_probe():
        from triton.runtime import driver  # noqa: PLC0415
        active = driver.active
        info = {"active_type": type(active).__module__ + "." + type(active).__name__}
        for attr in ("get_current_target", "get_device_capability", "get_current_device"):
            fn = getattr(active, attr, None)
            if fn is None:
                info[attr] = "missing"
            else:
                try:
                    value = fn()
                except TypeError:
                    # fork 实测：MusaDriver._get_device_capability 需要 device 参数
                    value = fn(0) if attr == "get_device_capability" else fn()
                info[attr] = repr(value)
        return info
    record(result, "driver_active", driver_probe)

    def first_kernel_journey():
        # 分段墙钟：import -> 定义 -> 首次 launch（含编译）-> 同步 -> 第二次 launch
        import triton  # noqa: PLC0415

        timings = {}

        @triton.jit
        def probe_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.store(y_ptr + offs, x * 2.0, mask=mask)

        n = 4096
        if backend == "cuda":
            x = torch.arange(n, device="cuda", dtype=torch.float32)
            y = torch.empty_like(x)
        else:
            x = torch.arange(n, device="musa", dtype=torch.float32)
            y = torch.empty_like(x)

        t0 = time.perf_counter()
        probe_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
        t1 = time.perf_counter()
        if backend == "cuda":
            torch.cuda.synchronize()
        else:
            torch.musa.synchronize()
        t2 = time.perf_counter()
        timings["first_launch_wall_s"] = round(t1 - t0, 4)
        timings["first_sync_wall_s"] = round(t2 - t0, 4)
        t3 = time.perf_counter()
        probe_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
        if backend == "cuda":
            torch.cuda.synchronize()
        else:
            torch.musa.synchronize()
        t4 = time.perf_counter()
        timings["second_launch_wall_s"] = round(t4 - t3, 4)
        timings["speedup_cold_to_warm"] = round((t2 - t0) / max(t4 - t3, 1e-9), 1)
        ok = bool(torch.equal(y, x * 2.0))
        timings["correct_vs_torch"] = ok
        return timings
    record(result, "first_kernel_journey", first_kernel_journey)

    def rmsnorm_journey():
        # 本期主算子的"import 到跑通"全过程 + 与 torch 组合表达式对照
        from rmsnorm_triton import rmsnorm  # noqa: PLC0415

        torch.manual_seed(0)
        rows, hidden = 128, 1024
        eps = 1e-6
        if backend == "cuda":
            device = torch.device("cuda")
        else:
            device = torch.device("musa")
        x = torch.randn(rows, hidden, device=device, dtype=torch.float32)
        w = torch.randn(hidden, device=device, dtype=torch.float32)
        t0 = time.perf_counter()
        y = rmsnorm(x, w, eps)
        if backend == "cuda":
            torch.cuda.synchronize()
        else:
            torch.musa.synchronize()
        t1 = time.perf_counter()
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        ref = w * (x.float() * torch.rsqrt(var + eps)).to(x.dtype)
        max_abs = float((y.float() - ref.float()).abs().max().item())
        return {"first_call_incl_compile_s": round(t1 - t0, 4),
                "max_abs_vs_torch_composite": max_abs,
                "shape": f"{rows}x{hidden}"}
    record(result, "rmsnorm_first_journey", rmsnorm_journey)

    # ---- F. 工具链查询（编译产物/ISA 线索） ----
    step("F. 工具链与缓存目录")
    def tool_probe():
        info = {}
        for cmd in (["nvcc", "--version"], ["mcc", "--version"], ["llvm-as", "--version"],
                    ["ptxas", "--version"]):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                info[cmd[0]] = (proc.stdout + proc.stderr).strip()[:300] if proc.returncode == 0 else None
            except FileNotFoundError:
                info[cmd[0]] = "not found"
        info["triton_cache_dir_env"] = os.environ.get("TRITON_CACHE_DIR")
        info["home_cache"] = str(Path.home() / ".triton" / "cache")
        return info
    record(result, "toolchain", tool_probe)

    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path = Path(arguments.out) if arguments.out else HERE.parent.parent / "results" / f"triton-probe-{backend}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[probe] 已写入 {out_path}")
    print(json.dumps({k: v for k, v in result.items() if k in (
        "triton_version", "pip", "driver_active", "first_kernel_journey",
        "rmsnorm_first_journey")}, ensure_ascii=False, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
