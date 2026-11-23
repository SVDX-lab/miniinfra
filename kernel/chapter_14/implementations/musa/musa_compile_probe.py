"""第 14 期 MUSA 列：torch.compile 行为矩阵探测 v2（S4000 / torch_musa 2.11）。

v2 要点（v1 会话证据驱动）：
- **逐 case 子进程隔离**：v1 实测失败的 compile 会毒化 dynamo 状态，连
  torch._dynamo.reset() 都崩溃（cudagraph_trees weakref 错误），后续 case 全废；
- 根因已实机定位：torch_musa 把 **torch.cuda 完全别名到自身**
  （`torch.cuda is torch_musa == True`），而 dynamo guard 构造期要访问
  `torch.cuda.streams.Stream`（torch/_dynamo/guards.py EQUALS_MATCH 常量类型表）
  → `torch_musa.streams` 不存在 → InternalTorchDynamoError；
- Inductor MUSA 代码生成本身**已经跑通**（生成的 output_code 含
  `@triton.jit` + `triton_heuristics.persistent_reduction` kernel，wrapper 用
  `torch_musa._MUSAC._musa_getCurrentRawStream`，缓存落盘）——败在 guard 一道关；
- 新增实验性 case：monkeypatch `torch_musa.streams = SimpleNamespace(Stream=
  torch.musa.Stream)` 后重试（如实标注"实验性补丁，非生产建议"）；
- 新增 case：musagraphs（torch_musa 自有 backend，已注册进 dynamo registry）。

输出 results/musa-compile-probe.json；可用 backend 的 kernel 缓存源码另存
results/inductor-musa/。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"

CASE_TEMPLATE = '''
import json, sys, types, traceback
import torch
import torch_musa

case = sys.argv[1]

def add_rmsnorm(r, h, w, eps=1e-6):
    t = r + h
    ft = t.float()
    var = ft.pow(2).mean(-1, keepdim=True)
    y = w * (ft * torch.rsqrt(var + eps)).to(t.dtype)
    return t, y

record = {{"case": case, "compile": None, "run": None, "error": None,
           "max_abs_y_vs_eager": None, "t_bitwise_vs_eager": None}}

try:
    if case == "inductor-default":
        compiled = torch.compile(add_rmsnorm)
    elif case == "inductor-mode-reduce-overhead":
        compiled = torch.compile(add_rmsnorm, mode="reduce-overhead")
    elif case == "musagraphs":
        compiled = torch.compile(add_rmsnorm, backend="musagraphs")
    elif case == "streams-monkeypatch+inductor":
        torch_musa.streams = types.SimpleNamespace(Stream=torch.musa.Stream)
        compiled = torch.compile(add_rmsnorm)
    else:
        compiled = torch.compile(add_rmsnorm, backend=case)
    record["compile"] = "ok"
except Exception as error:
    record["compile"] = f"FAIL at torch.compile(): {{type(error).__name__}}"
    record["error"] = traceback.format_exc()
    print("CASE-RESULT" + json.dumps(record))
    sys.exit(0)

gen = torch.Generator(device="cpu").manual_seed(20260415)
r = torch.randn(128, 1024, generator=gen).to("musa")
h = torch.randn(128, 1024, generator=gen).to("musa")
w = torch.randn(1024, generator=gen).to("musa")
try:
    t_c, y_c = compiled(r, h, w)
    torch.musa.synchronize()
    record["run"] = "ok"
    t_e, y_e = add_rmsnorm(r, h, w)
    torch.musa.synchronize()
    record["max_abs_y_vs_eager"] = float((y_c - y_e).abs().max().item())
    record["t_bitwise_vs_eager"] = bool(torch.equal(t_c, t_e))
except Exception as error:
    record["run"] = f"FAIL at first call: {{type(error).__name__}}: {{error}}"
    record["error"] = traceback.format_exc()
print("CASE-RESULT" + json.dumps(record))
'''


def run_case(case: str, python: str = sys.executable) -> dict:
    done = subprocess.run(
        [python, "-c", CASE_TEMPLATE.format(), case],
        capture_output=True, text=True, timeout=600, check=False,
    )
    for line in done.stdout.splitlines():
        if line.startswith("CASE-RESULT"):
            return json.loads(line[len("CASE-RESULT"):])
    return {
        "case": case,
        "compile": "FAIL before CASE-RESULT（进程级崩溃）",
        "run": None,
        "error": (done.stderr or done.stdout)[-3000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    args = parser.parse_args()
    del args  # v2 固定 128×1024（矩阵探测不需要多 shape）

    import torch_musa  # noqa: F401

    payload: dict = {
        "schema_version": "kernel-chapter15.musa-compile-probe.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "torch_musa": getattr(torch_musa, "__version__", "unknown"),
        "device": None,
        "cuda_alias_root_cause": None,
        "registered_backends": None,
        "matrix": [],
        "usable": None,
        "workaround": None,
    }

    properties = torch.musa.get_device_properties(0)
    payload["device"] = {
        "name": properties.name,
        "cc": f"{properties.major}.{properties.minor}",
        "multi_processor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
    }
    payload["cuda_alias_root_cause"] = {
        "torch_cuda_is_torch_musa": bool(torch.cuda is torch_musa),
        "torch_musa_has_streams": hasattr(torch_musa, "streams"),
        "torch_musa_has_Stream": hasattr(torch_musa, "Stream"),
        "mechanism": (
            "torch_musa 把 torch.cuda 完全别名到自身；dynamo guard 构造期访问 "
            "torch.cuda.streams.Stream（guards.py EQUALS_MATCH 常量类型表）解析到 "
            "torch_musa.streams → AttributeError → InternalTorchDynamoError。"
            "Inductor MUSA Triton 代码生成本身已跑通（生成 @triton.jit kernel 落盘）。"
        ),
    }

    from torch._dynamo.backends.registry import list_backends

    payload["registered_backends"] = sorted(list_backends())

    for case in (
        "inductor-default",
        "eager",
        "aot_eager",
        "inductor-mode-reduce-overhead",
        "cudagraphs",
        "musagraphs",
        "streams-monkeypatch+inductor",
    ):
        record = run_case(case)
        payload["matrix"].append(record)
        print(
            f"[probe] {case}: compile={record['compile']} run={record.get('run')} "
            f"max_abs={record.get('max_abs_y_vs_eager')}",
            flush=True,
        )

    usable_cases = [m for m in payload["matrix"] if m.get("run") == "ok"]
    if usable_cases:
        winner = usable_cases[0]
        payload["usable"] = winner["case"]
        payload["workaround"] = (
            "torch_musa.streams = types.SimpleNamespace(Stream=torch.musa.Stream)"
            if winner["case"].startswith("streams-monkeypatch")
            else "无需 workaround（backend 原生可用）"
        )
    else:
        payload["usable"] = None
        payload["workaround"] = None

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "musa-compile-probe.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[probe] written {out}")


if __name__ == "__main__":
    main()
