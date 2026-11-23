#!/usr/bin/env python3
"""第 09 期 MUSA Graph 深挖（一）：API 可用性探测——探测过程与结论本身就是实验内容。

M-02 背景（台账）：ch01 实测 MUSA Graph Replay 0.30ms 慢于 eager 0.17ms
@2048×1024（CUDA 相反快 3.5×）；ch02 实测 MUSA 1-op 图回放 25.5µs/op 高于
eager 发射 16.7µs/op。本期先回答"torch_musa 的 Graph 到底是什么实现路径"：

  探测面 1：torch 级 API——torch.musa 有无 CUDAGraph/graph/Stream/graph_pool_handle，
            与 torch.cuda 同名 API 的行为差异；
  探测面 2：torch_musa Python 源码——graph 相关模块与实现入口（grep 证据）；
  探测面 3：符号级证据——libtorch_musa*.so 是否引用 musaGraph* 系 API
            （nm -D），证明捕获/回放走 musaGraph 驱动 API 而非逐 kernel 发射；
  探测面 4：SDK 头文件——/usr/local/musa/include 有无 musaGraph*.h、
            musaStreamBeginCapture/musaGraphInstantiate/musaGraphLaunch 原型；
  探测面 5：功能冒烟——小 shape 与 2048×1024（M-02 原始 shape）捕获/回放/
            eager 位级对账；
  探测面 6：vllm(-musa) 是否在环境内（生产引擎内置路径的参考证据，不安装）。

输出：results/graph-probe-musa.json（探测即实验，全部证据进 JSON）
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]
for _path in (str(CHAPTER), str(CHAPTER / "implementations"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import (  # noqa: E402
    backend_module,
    backend_sync,
    device_environment_record,
    graph_support_level,
    resolve_device,
    write_results_json,
)

SCHEMA_VERSION = "kernel-chapter10.graph-probe-musa.v1"
DEFAULT_BACKEND = "musa"


def run_command(command: list[str], timeout: int = 60) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True,
                                text=True, timeout=timeout)
        return (result.stdout or result.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {type(error).__name__}"


def torch_musa_source_evidence() -> dict:
    """面 2：torch_musa Python 源码里 graph 相关实现入口。"""
    evidence: dict = {}
    try:
        import torch_musa  # noqa: F401
        package_root = Path(torch_musa.__file__).resolve().parent
        evidence["torch_musa_package"] = str(package_root)
        py_files = list(package_root.rglob("*.py"))
        graph_files = [str(p.relative_to(package_root)) for p in py_files
                       if "graph" in p.name.lower()]
        evidence["graph_named_modules"] = graph_files
        # grep 关键 API 名
        for keyword in ("CUDAGraph", "MUSAGraph", "graph_pool_handle", "make_graphed_callables"):
            hits = []
            for path in py_files:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if keyword in text:
                    lines = [f"{path.relative_to(package_root)}:{i + 1}: {line.strip()[:120]}"
                             for i, line in enumerate(text.splitlines()) if keyword in line][:5]
                    hits.extend(lines)
            evidence[f"python_grep_{keyword}"] = hits[:10] or ["no-hit"]
    except ImportError as error:
        evidence["torch_musa_import"] = f"failed: {error}"
    return evidence


def symbol_evidence() -> dict:
    """面 3：torch_musa 二进制与 libmusart 的 musaGraph 符号。"""
    evidence: dict = {}
    libs = []
    try:
        import torch_musa  # noqa: F401
        package_root = Path(torch_musa.__file__).resolve().parent
        libs = [str(p) for p in package_root.rglob("*.so")]
    except ImportError:
        pass
    evidence["torch_musa_so_count"] = len(libs)
    for lib in libs[:8]:
        out = run_command(["bash", "-c", f"nm -D '{lib}' 2>/dev/null | grep -ci musagraph"])
        count = None if out is None or not out.isdigit() else int(out)
        out_sample = run_command(["bash", "-c",
                                  f"nm -D '{lib}' 2>/dev/null | grep -i musagraph | head -20"])
        if count:
            evidence[f"musaGraph_symbols:{Path(lib).name}"] = {"count": count, "sample": out_sample}
    out = run_command(["bash", "-c",
                       "nm -D /usr/local/musa/lib/libmusart.so 2>/dev/null | grep -i graph | head -40"])
    evidence["libmusart_graph_symbols"] = (out or "no-hit")[:2000]
    return evidence


def sdk_header_evidence() -> dict:
    """面 4：MUSA SDK 头文件的 musaGraph API 原型。"""
    evidence: dict = {}
    evidence["include_graph_headers"] = run_command(
        ["bash", "-c", "ls /usr/local/musa/include 2>/dev/null | grep -i graph"])
    evidence["begin_capture_prototype"] = run_command(
        ["bash", "-c",
         "grep -rn 'musaStreamBeginCapture' /usr/local/musa/include 2>/dev/null | head -5"])
    evidence["instantiate_prototype"] = run_command(
        ["bash", "-c",
         "grep -rn 'musaGraphInstantiate' /usr/local/musa/include 2>/dev/null | head -5"])
    evidence["graph_launch_prototype"] = run_command(
        ["bash", "-c",
         "grep -rn 'musaGraphLaunch' /usr/local/musa/include 2>/dev/null | head -5"])
    evidence["driver_api_graph"] = run_command(
        ["bash", "-c",
         "grep -rln 'musaGraphLaunch' /usr/local/musa/include 2>/dev/null | head -10"])
    return evidence


def vllm_probe() -> dict:
    """面 6：vllm(-musa) 是否已安装（只查不装）。"""
    out = run_command(["bash", "-c", "python3 -c 'import importlib.util as u; print(bool(u.find_spec(\"vllm\")))'"])
    return {"vllm_installed": out, "note": "vllm-musa 内置 Graph 路径仅作参考证据，不在本环境安装"}


def functional_smoke(device, elements: int, chain_k: int = 8) -> dict:
    """面 5：最小捕获回放链 + 位级对账（两档 shape：1024 与 2048×1024=M-02 原始 shape）。"""
    module = backend_module(device)
    x = torch.arange(elements, device=device, dtype=torch.float32)
    x = x.to(torch.float16).clamp(-100.0, 100.0)
    t_eager = x.clone()
    for _ in range(chain_k):
        t_eager.add_(1.0)
    t_ref = t_eager.clone()

    side_stream = module.Stream()
    current = module.current_stream()
    side_stream.wait_stream(current)
    t_warm = x.clone()
    with module.stream(side_stream):
        for _ in range(3):
            t_warm.add_(1.0)
    current.wait_stream(side_stream)
    backend_sync(device)

    graph = module.CUDAGraph()
    t_static = x.clone()
    with module.graph(graph):
        for _ in range(chain_k):
            t_static.add_(1.0)
    backend_sync(device)
    graph.replay()
    backend_sync(device)
    bitwise = bool(torch.equal(t_static, t_ref))
    max_abs = float((t_static.float() - t_ref.float()).abs().max().item())
    return {
        "elements": elements,
        "chain_k": chain_k,
        "capture_ok": True,
        "replay_bitwise_equal": bitwise,
        "max_abs_error": max_abs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["cuda", "musa"])
    arguments = parser.parse_args()
    device = resolve_device(arguments.backend)

    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "torch_level_api": graph_support_level(device),
        "torch_musa_python_evidence": torch_musa_source_evidence(),
        "symbol_evidence": symbol_evidence(),
        "sdk_header_evidence": sdk_header_evidence(),
        "vllm_probe": vllm_probe(),
        "functional_smoke": {},
        "environment": device_environment_record(device),
    }
    try:
        results["functional_smoke"]["small_1024"] = functional_smoke(device, 1024)
    except Exception as error:  # noqa: BLE001
        results["functional_smoke"]["small_1024"] = {"error": f"{type(error).__name__}: {error}"}
    try:
        results["functional_smoke"]["m02_shape_2048x1024"] = functional_smoke(device, 2048 * 1024)
    except Exception as error:  # noqa: BLE001
        results["functional_smoke"]["m02_shape_2048x1024"] = {"error": f"{type(error).__name__}: {error}"}

    output = Path("results") / f"graph-probe-{arguments.backend}.json"
    write_results_json(output, results)
    print(f"[probe/{arguments.backend}] smoke={json.dumps(results['functional_smoke'], ensure_ascii=False)[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
