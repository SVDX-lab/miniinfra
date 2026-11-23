#!/usr/bin/env python3
"""第 09 期 Graph API 可用性探测（CUDA 列，轻量；MUSA 列的同名脚本是重点实验）。

记录本期实现路径选择：
  torch.cuda.CUDAGraph 流捕获（torch 级 API）——选择理由：
  1) 与模型/融合 op 同层，无需手写 kernel 节点；2) ch01/ch02 已验证该路径；
  3) 原生 cudaGraph API（cudaGraphAddKernelNode 等）需自行管理节点依赖，
     本期主题是"调用组织"而非"图构造 API"，列入 out_of_scope（合同）。

探测项：
  1. torch 级 API 清单（CUDAGraph/graph/Stream/graph_pool_handle/make_graphed_callables）；
  2. 功能冒烟：捕获→回放→与 eager 位级对账；
  3. 运行时 API 存在性（ctypes 查 libcudart 符号，仅记录不调用）；
  4. nsys 可用性与版本（graph trace 证据链工具）。

输出：results/graph-probe-{backend}.json
"""

from __future__ import annotations

import argparse
import ctypes
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

SCHEMA_VERSION = "kernel-chapter10.graph-probe.v1"
DEFAULT_BACKEND = "cuda"


def runtime_api_probe() -> dict:
    """ctypes 查 libcudart 的 Graph API 符号（存在性证据，不实际调用）。"""
    found = {}
    candidates = ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    api_names = [
        "cudaStreamBeginCapture", "cudaStreamEndCapture",
        "cudaGraphInstantiate", "cudaGraphLaunch", "cudaGraphDestroy",
        "cudaGraphAddKernelNode", "cudaGraphExecUpdate", "cudaGraphGetNodes",
    ]
    library = None
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            found["library_loaded"] = candidate
            break
        except OSError:
            continue
    if library is None:
        return {"library_loaded": None, "note": "libcudart 不可加载（仅记录）"}
    for name in api_names:
        found[name] = hasattr(library, name)
    return found


def nsys_probe() -> dict:
    for tool in ("nsys", "ncu"):
        try:
            result = subprocess.run([tool, "--version"], check=False,
                                    capture_output=True, text=True, timeout=30)
            first_line = (result.stdout or result.stderr).strip().splitlines()
            return {"available": True, "version": first_line[0] if first_line else None}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return {"available": False}


def functional_smoke(device) -> dict:
    """最小捕获回放链：8 次 add_ 捕获成图，回放结果与 eager 链位级对账。"""
    module = backend_module(device)
    x = torch.arange(1024, device=device, dtype=torch.float16).clamp(-100, 100)
    t_eager = x.clone()
    for _ in range(8):
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
        for _ in range(8):
            t_static.add_(1.0)
    backend_sync(device)
    graph.replay()
    backend_sync(device)
    bitwise = bool(torch.equal(t_static, t_ref))
    max_abs = float((t_static.float() - t_ref.float()).abs().max().item())
    return {
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
        "path_choice": {
            "selected": "torch.CUDAGraph stream capture（torch 级）",
            "reasons": [
                "与模型/注册融合 op 同层，无需手写图节点",
                "ch01/ch02 已验证该捕获/回放路径可运行",
                "原生 cudaGraphAddKernelNode 建图列为 out_of_scope（合同）",
            ],
        },
        "runtime_api_presence": runtime_api_probe(),
        "profiler_tools": nsys_probe(),
        "functional_smoke": functional_smoke(device),
        "environment": device_environment_record(device),
    }
    output = Path("results") / f"graph-probe-{arguments.backend}.json"
    write_results_json(output, results)
    print(f"[probe/{arguments.backend}] smoke={results['functional_smoke']} "
          f"nsys={results['profiler_tools']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
