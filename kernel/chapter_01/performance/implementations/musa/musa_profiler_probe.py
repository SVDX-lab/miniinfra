#!/usr/bin/env python3
"""MUSA 侧 Profiler / 性能分析工具生态探索（第 02 期，平台边界调查）。

问题：CUDA 有 Nsight Compute/Systems；MUSA（摩尔线程）侧有什么等价物？
torch.profiler（kineto）能否带上 torch_musa 设备事件？MUPTI 是否可用？
有厂商级 profiler 工具吗？

方法：逐项尝试，能跑通的保存实际输出，跑不通的保存完整报错文本——
"取不到"本身是生态结论，全部写入 results/musa-profiler-probe.json。

用法：
    python3 musa_profiler_probe.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path

import torch

SCHEMA_VERSION = "kernel-chapter02.musa-profiler-probe.v1"
RMS_NORM_EPS = 1e-6


def composite_rmsnorm(x, weight, eps=RMS_NORM_EPS):
    input_dtype = x.dtype
    float_states = x.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def attempt(name: str, function) -> dict:
    try:
        value = function()
        return {"item": name, "status": "ok", "result": value}
    except Exception as error:  # noqa: BLE001 —— 探索性脚本，报错文本就是数据
        return {
            "item": name,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_text": str(error)[:2000],
            "traceback_tail": traceback.format_exc()[-1500:],
        }


def probe_environment() -> dict:
    results = []

    results.append(attempt("torch_version", lambda: torch.__version__))
    results.append(attempt("torch_musa_import+version", lambda: __import__("torch_musa").__version__
                           if hasattr(__import__("torch_musa"), "__version__") else "imported(no __version__)"))
    results.append(attempt("musa_available", lambda: torch.musa.is_available()))
    results.append(attempt("musa_device_name", lambda: str(torch.musa.get_device_properties(0))))

    def list_musa_sdk():
        root = Path("/usr/local/musa")
        if not root.exists():
            return "(/usr/local/musa 不存在)"
        out = {"root_exists": True}
        for sub in ["lib", "include", "bin"]:
            directory = root / sub
            if directory.exists():
                entries = sorted(p.name for p in directory.iterdir())
                out[sub] = [e for e in entries if any(k in e.lower() for k in
                            ["prof", "trace", "mupti", "musight", "tool"])][:60]
                out[f"{sub}_count_total"] = len(entries)
        return out

    results.append(attempt("musa_sdk_profiler_related_files", list_musa_sdk))

    def which_tools():
        candidates = ["muprof", "musight", "mthreads-profiler", "muTrace", "musatrace",
                      "mthreads-gmi", "mcc", "musa-gdb", "mthreads-diagnostic"]
        found = {}
        for name in candidates:
            path = shutil.which(name)
            found[name] = path
        return found

    results.append(attempt("profiler_tools_in_path", which_tools))

    def find_mupti():
        hits = subprocess.run(
            ["find", "/usr/local/musa", "-maxdepth", "3", "-iname", "*mupti*"],
            capture_output=True, text=True, timeout=30,
        )
        return hits.stdout.strip().splitlines() or "(none)"

    results.append(attempt("find_mupti_under_sdk", find_mupti))

    def kineto_backends():
        import torch.autograd.profiler as profiler
        info = {
            "kineto_available": getattr(profiler, "kineto_available", lambda: None)(),
        }
        try:
            from torch.autograd import profiler as p2
            info["supported_activities"] = [str(a) for a in torch.profiler.ProfilerActivity.__members__.values()]
        except Exception as error:  # noqa: BLE001
            info["supported_activities_error"] = str(error)
        return info

    results.append(attempt("kineto_info", kineto_backends))
    return results


def probe_torch_profiler() -> dict:
    results = []

    def run_profile(activities, label):
        device = torch.device("musa")
        weight = torch.randn(1024, device=device, dtype=torch.float16)
        x = torch.randn(2048, 1024, device=device, dtype=torch.float16)
        for _ in range(5):
            composite_rmsnorm(x, weight)
        torch.musa.synchronize()
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(10):
                composite_rmsnorm(x, weight)
            torch.musa.synchronize()
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
        events = []
        for event in prof.key_averages():
            events.append({
                "name": event.key[:80],
                "count": event.count,
                "self_cpu_us": round(event.self_cpu_time_total, 1),
                "device_time_us": round(getattr(event, "self_device_time_total", 0.0), 1)
                if hasattr(event, "self_device_time_total") else
                round(getattr(event, "self_cuda_time_total", 0.0), 1),
            })
        return {"label": label, "table": table[:4000], "events_top": events[:15]}

    results.append(attempt(
        "torch.profiler activities=[CPU,PrivateUse1]",
        lambda: run_profile([torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.PrivateUse1],
                            "CPU+PrivateUse1"),
    ))
    results.append(attempt(
        "torch.profiler activities=[CPU]",
        lambda: run_profile([torch.profiler.ProfilerActivity.CPU], "CPU-only"),
    ))

    def legacy_profiler():
        device = torch.device("musa")
        weight = torch.randn(1024, device=device, dtype=torch.float16)
        x = torch.randn(2048, 1024, device=device, dtype=torch.float16)
        with torch.autograd.profiler.profile(use_musa=True) as prof:
            for _ in range(10):
                composite_rmsnorm(x, weight)
            torch.musa.synchronize()
        return prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)[:3000]

    results.append(attempt("legacy autograd.profiler use_musa=True", legacy_profiler))

    def chrome_trace_export():
        device = torch.device("musa")
        weight = torch.randn(1024, device=device, dtype=torch.float16)
        x = torch.randn(2048, 1024, device=device, dtype=torch.float16)
        out_path = Path("results/musa-trace.json")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            for _ in range(5):
                composite_rmsnorm(x, weight)
            torch.musa.synchronize()
        prof.export_chrome_trace(str(out_path))
        size = out_path.stat().st_size
        # 检查 trace 中是否出现 musa 设备事件
        text = out_path.read_text(errors="ignore")[:400000]
        markers = {m: text.count(f'"cat": "{m}"') for m in ["kernel", "gpu_memcpy", "cuda_runtime", "cpu_op", "musa"]}
        return {"file": str(out_path), "bytes": size, "category_markers": markers}

    results.append(attempt("chrome_trace_export(CPU activity, device= musa)", chrome_trace_export))
    return results


def probe_runtime_level() -> dict:
    results = []

    def try_mupti_python():
        # MUPTI 是否暴露 Python 绑定？（预期没有，报告验证过程）
        import importlib
        for name in ["mupti", "torch_musa.profiler", "torch_musa.mupti"]:
            importlib.import_module(name)
        return "imported"

    results.append(attempt("python_import_mupti/torch_musa.profiler", try_mupti_python))

    def mupti_header_symbols():
        header = Path("/usr/local/musa/include/mupti.h")
        if not header.exists():
            candidates = list(Path("/usr/local/musa/include").glob("mupti*")) if Path("/usr/local/musa/include").exists() else []
            return {"header": None, "candidates": [str(c) for c in candidates]}
        text = header.read_text(errors="ignore")
        activities = [line.strip()[:120] for line in text.splitlines() if "ActivityKind" in line or "MUPTI_ACTIVITY" in line][:20]
        return {"header": str(header), "size_bytes": header.stat().st_size, "sample_lines": activities}

    results.append(attempt("mupti_header_symbols", mupti_header_symbols))
    return results


def main() -> int:
    Path("results").mkdir(exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "MUSA 侧 profiler 生态探索：能取到的取，取不到的记录报错全文",
        "environment_probe": probe_environment(),
        "torch_profiler_probe": probe_torch_profiler(),
        "runtime_level_probe": probe_runtime_level(),
        "note_env": f"PATH={os.environ.get('PATH', '')[:400]}",
    }
    output = Path("results") / "musa-profiler-probe.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[probe] 已写入 {output}")
    for section in ["environment_probe", "torch_profiler_probe", "runtime_level_probe"]:
        for item in payload[section]:
            status = item["status"]
            print(f"  [{status}] {item['item']}"
                  + (f" -> {str(item.get('error_text', ''))[:120]}" if status == "failed" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
