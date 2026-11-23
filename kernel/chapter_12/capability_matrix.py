#!/usr/bin/env python3
"""第 12 期实验 2：能力查询矩阵 + "能力缺失显式报错"行为验证。

矩阵：三后端 × {dtype 支持（fp32/fp16/bf16 四项探测）、融合 op 可用、
注册路径、设备探测、同步原语、遥测签名}。不可用后端也入表（available=false
+ 原因）——能力矩阵是数据，缺位本身就是平台边界记录。

显式报错验证（不静默回退的行为学证据，逐项采集错误原文）：
  E1 不可用后端的选择（无 GPU 机器选 cuda → BackendNotAvailableError）；
  E2 合同 dtype 缺失（bf16 进融合 op → 显式报错）；
  E3 设备不符（meta 张量进已 setup 后端 → strict 报错）；
  E4 未注册设备进路由器（列出已注册集合）；
  E5（GPU 列）框架级对照：CPU 张量直调 torch.ops 注册 op → 框架路由到库内
     CPU composite impl（框架允许、runtime 拒绝——两层语义的差异证据）；
     no_cpu_fallback 族 → NotImplementedError（框架级显式报错原文）。

输出：results/capability-matrix.json + stdout markdown 表。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import write_results_json  # noqa: E402
from implementations.runtime import (  # noqa: E402
    BackendNotAvailableError,
    CapabilityError,
    create_backend,
    list_backends,
)
from implementations.runtime.qwen3_runtime import bootstrap_compat  # noqa: E402


def truncate(text: str, limit: int = 260) -> str:
    return text[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="cpu,cuda,musa")
    parser.add_argument("--out", default=str(HERE / "results" / "capability-matrix.json"))
    arguments = parser.parse_args()

    bootstrap_compat()
    requested = [name.strip() for name in arguments.backends.split(",") if name.strip()]
    unknown = [name for name in requested if name not in list_backends()]
    if unknown:
        raise ValueError(f"未知后端 {unknown}（已注册：{list_backends()}）")

    payload: dict = {
        "schema_version": "kernel-chapter13.capability.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "requested_backends": requested,
        "backends": {},
        "explicit_error_cases": {},
    }

    # ---- 能力矩阵（可用后端 setup + deep 查询；不可用后端只静态探测）----
    for name in requested:
        backend = create_backend(name)
        available, reason = backend.probe_availability()
        if available:
            backend.setup()
            report = backend.query_capabilities(deep=True)
        else:
            report = backend.query_capabilities(deep=False)
        payload["backends"][name] = report.to_dict()
        print(f"\n=== {name}: available={available}"
              + (f" reason={reason}" if reason else f" device={report.device_name}"))

    # ---- 显式报错行为验证（机器能力允许的子集逐项采集）----
    cases: dict[str, dict] = {}

    # E1: 不可用后端的选择报错（对本机不可用的每个后端采集）。
    for name in requested:
        backend = create_backend(name)
        available, _ = backend.probe_availability()
        if not available:
            try:
                backend.ensure_available()
                cases[f"E1_select_unavailable_{name}"] = {
                    "raised": False, "note": "未抛错（本机该后端意外可用）"}
            except BackendNotAvailableError as error:
                cases[f"E1_select_unavailable_{name}"] = {
                    "raised": True, "error_type": type(error).__name__,
                    "message": truncate(str(error))}
    if not any(key.startswith("E1_") for key in cases):
        cases["E1_select_unavailable"] = {
            "raised": None, "note": "三后端在本机全部可用——无不可用选择场景（GPU 机器上的 CPU 列亦然）"}

    # E2–E5：对每个**可用**后端逐个采集（GPU 服务器上 CPU 可用不代表 GPU 可用，
    # 框架级用例 E5 必须绑在对应 GPU 后端上，不随列表顺序滑到 CPU）。
    for backend_name in requested:
        backend = create_backend(backend_name)
        available, _ = backend.probe_availability()
        if not available:
            continue
        backend.setup()
        device = backend.resolve_device()

        # E2: 合同 dtype 缺失（bf16 → 融合 op 显式报错）。
        bf16_h = torch.randn(2, 8, device=device, dtype=torch.bfloat16)
        bf16_r = torch.randn(2, 8, device=device, dtype=torch.bfloat16)
        bf16_w = torch.ones(8, device=device, dtype=torch.bfloat16)
        try:
            backend.fused_add_rmsnorm(bf16_h, bf16_r, bf16_w, 1e-6)
            cases[f"E2_bf16_into_fused_op_{backend_name}"] = {
                "raised": False, "note": "融合 op 接受了 bf16（合同外 dtype 未拒绝！）"}
        except (CapabilityError, ValueError, RuntimeError) as error:
            cases[f"E2_bf16_into_fused_op_{backend_name}"] = {
                "raised": True, "backend": backend_name,
                "error_type": type(error).__name__, "message": truncate(str(error))}

        # E3: 设备不符（meta 张量 → strict 报错，不静默改道）。
        meta_h = torch.zeros(2, 8, device="meta")
        meta_r = torch.zeros(2, 8, device="meta")
        w = torch.ones(8)
        try:
            backend.fused_add_rmsnorm(meta_h, meta_r, w, 1e-6)
            cases[f"E3_device_mismatch_{backend_name}"] = {
                "raised": False, "note": "meta 张量被接受（静默改道风险！）"}
        except CapabilityError as error:
            cases[f"E3_device_mismatch_{backend_name}"] = {
                "raised": True, "error_type": type(error).__name__,
                "message": truncate(str(error))}

        # E4: 未注册设备进路由器（列出已注册集合）。
        try:
            backend.fused_ops_protocol().resolve(meta_h)
            cases[f"E4_unregistered_device_route_{backend_name}"] = {"raised": False}
        except RuntimeError as error:
            cases[f"E4_unregistered_device_route_{backend_name}"] = {
                "raised": True, "error_type": type(error).__name__,
                "message": truncate(str(error))}

        # E5: 框架级对照（仅 torch.ops 扩展后端）：CPU 张量直调 torch.ops →
        # 库内 CPU composite impl（框架允许、runtime 拒绝——两层语义差异证据）；
        # no_cpu_fallback 族 → NotImplementedError（框架级显式报错原文）。
        namespace = None
        module_path = {"cuda": "implementations.cuda.build",
                       "musa": "implementations.musa.build"}.get(backend_name)
        if module_path:
            try:
                import importlib

                module = importlib.import_module(module_path)
                _, namespace, *_ = module.get_ops()
            except Exception:  # noqa: BLE001
                namespace = None
        if namespace is None:
            cases[f"E5_framework_level_{backend_name}"] = {
                "applicable": False,
                "note": f"{backend_name} 后端无 torch.ops 扩展（CPU composite 无框架注册层）"}
            continue
        h_in = torch.randn(2, 8, dtype=torch.float32)
        r_in = torch.randn(2, 8, dtype=torch.float32)
        w_in = torch.ones(8, dtype=torch.float32)
        h_buf, r_buf = h_in.clone(), r_in.clone()
        try:
            namespace.fused_add_rmsnorm(h_buf, r_buf, w_in, 1e-6)
            t_ref = r_in + h_in
            fs = t_ref.float()
            y_ref = w_in * (fs * torch.rsqrt(fs.pow(2).mean(-1, keepdim=True) + 1e-6)).to(t_ref.dtype)
            cases[f"E5a_framework_routes_cpu_tensor_{backend_name}"] = {
                "raised": False,
                "note": "框架层 CPU 张量被路由到库内 CPU composite impl"
                        "（框架允许；runtime strict 层拒绝——两层语义差异证据）",
                "residual_updated": torch.equal(r_buf, t_ref),
                "output_vs_composite_max_abs": round(
                    float((h_buf - y_ref).abs().max()), 8)}
        except Exception as error:  # noqa: BLE001
            cases[f"E5a_framework_routes_cpu_tensor_{backend_name}"] = {
                "raised": True, "error_type": type(error).__name__,
                "message": truncate(str(error))}
        try:
            namespace.fused_add_rmsnorm_no_cpu_fallback(
                h_in.clone(), r_in.clone(), w_in.clone(), 1e-6)
            cases[f"E5b_no_cpu_fallback_family_{backend_name}"] = {
                "raised": False, "note": "no_cpu_fallback 族接受了 CPU 张量"}
        except Exception as error:  # noqa: BLE001
            cases[f"E5b_no_cpu_fallback_family_{backend_name}"] = {
                "raised": True, "error_type": type(error).__name__,
                "message": truncate(str(error))}

    payload["explicit_error_cases"] = cases

    # ---- markdown 表（stdout）----
    print("\n## 能力矩阵\n")
    header = "| 后端 | available | 设备 | fp32 | fp16 | bf16 | fused op | 注册路径 | 同步 |"
    print(header)
    print("|---|---|---|---|---|---|---|---|---|")
    for name in requested:
        report = payload["backends"][name]
        dtypes = report.get("dtype_capabilities", {})
        def cell(dtype_key):
            cap = dtypes.get(dtype_key)
            if not report.get("available"):
                return "N/A"
            if cap is None:
                return "?"
            if cap["fused_op_ok"] is None:
                return "create+linear" if cap["tensor_create"] else "x"
            return "ok" if (cap["tensor_create"] and cap["linear_ok"]
                            and cap["composite_rmsnorm_ok"]) else "部分"
        fused = "ok" if report.get("fused_op_available") else \
            ("N/A" if not report.get("available") else "待setup")
        path = (report.get("fused_op_path") or "-")
        path = path[:40] + ("…" if len(path) > 40 else "")
        print(f"| {name} | {report.get('available')} | {report.get('device_name') or '-'} "
              f"| {cell('float32')} | {cell('float16')} | {cell('bfloat16')} "
              f"| {fused} | {path} | {report.get('sync_api')} |")

    print("\n## 显式报错用例\n")
    for key, case in cases.items():
        print(f"- {key}: raised={case.get('raised')} "
              f"{case.get('error_type', '')} {case.get('note', '')}")

    write_results_json(Path(arguments.out), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
