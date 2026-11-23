"""第 14 期 CUDA 列：数值正确性（compile vs eager vs fixtures FP32 真值）。

fixtures = chapter_14/fixtures/correctness.npz（本地生成，CPU FP32 串行真值，
SHA256 manifest，协议"两台服务器共用同一份"）。5 组 (rows,H)：
(1,1024),(128,1024),(2048,1024),(128,127),(128,1025)。

对照链（设备 cuda）：
1. eager 组合表达式 fp32 vs 真值：t 位级（加法无累加）、y 容差 1e-5（合同）；
2. compile(inductor) fp32 vs 真值 + vs eager；
3. compile fp16 vs eager fp16 + y vs fp32 真值（容差 2e-2，ch04 fp16 口径，
   ch04 实测 fp16 y worst 1.169e-02）；t vs 真值（fp16 舍入，ch04 t worst
   4.404e-03 量级）；
4. 重编译计数顺带记录（H 1024→127→1025 触发 dtype/shape guard）。

产出 results/correctness-results-cuda.json。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiment_utils import comparison_metrics, write_results_json

CHAPTER = Path(__file__).resolve().parent.parent.parent
SCHEMA_VERSION = "kernel-chapter15.correctness-cuda.v1"


def add_rmsnorm(r, h, w, eps: float = 1e-6):
    t = r + h
    ft = t.float()
    var = ft.pow(2).mean(-1, keepdim=True)
    y = w * (ft * torch.rsqrt(var + eps)).to(t.dtype)
    return t, y


def rmsnorm_expr(hidden, weight, eps: float = 1e-6):
    input_dtype = hidden.dtype
    float_states = hidden.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def metrics_vs_truth(actual: torch.Tensor, truth_np: np.ndarray) -> dict:
    m = comparison_metrics(actual, torch.from_numpy(truth_np))
    m["bitwise_equal"] = bool(
        actual.detach().float().cpu().contiguous().numpy().tobytes()
        == truth_np.tobytes()
    )
    return m


def main() -> None:
    npz = np.load(CHAPTER / "fixtures" / "correctness.npz")
    manifest = json.loads((CHAPTER / "fixtures" / "manifest.json").read_text())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")
    if torch.__version__ != "2.12.1+cu130":
        raise SystemExit(f"torch 版本门禁失败：{torch.__version__}（等待降级）")

    torch._dynamo.reset()
    compiled_add = torch.compile(add_rmsnorm)
    compiled_rms = torch.compile(rmsnorm_expr)

    entries: list[dict] = []
    for group in manifest["groups"]:
        rows, hidden = group["rows"], group["hidden"]
        prefix = f"{rows}x{hidden}"
        r_np = npz[f"{prefix}/r"]
        h_np = npz[f"{prefix}/h"]
        w_np = npz[f"{prefix}/w"]
        t_true = npz[f"{prefix}/t"]
        y_true = npz[f"{prefix}/y"]
        entry: dict = {"group": prefix, "rows": rows, "hidden": hidden}

        # ---- fp32：eager 与 compiled 共享同一 GPU 输入 ----
        r32 = torch.from_numpy(r_np).to("cuda")
        h32 = torch.from_numpy(h_np).to("cuda")
        w32 = torch.from_numpy(w_np).to("cuda")

        t_e, y_e = add_rmsnorm(r32, h32, w32)
        torch.cuda.synchronize()
        entry["eager_t_vs_truth"] = metrics_vs_truth(t_e, t_true)
        entry["eager_y_vs_truth"] = metrics_vs_truth(y_e, y_true)

        t_c, y_c = compiled_add(r32, h32, w32)
        torch.cuda.synchronize()
        entry["compiled_t_vs_truth"] = metrics_vs_truth(t_c, t_true)
        entry["compiled_y_vs_truth"] = metrics_vs_truth(y_c, y_true)
        entry["compiled_vs_eager_t"] = comparison_metrics(t_c, t_e)
        entry["compiled_vs_eager_y"] = comparison_metrics(y_c, y_e)
        del r32, h32, w32, t_e, y_e, t_c, y_c

        # ---- fp16：compile vs eager + vs fp32 真值（dtype 舍入口径） ----
        r16 = torch.from_numpy(r_np).to("cuda", dtype=torch.float16)
        h16 = torch.from_numpy(h_np).to("cuda", dtype=torch.float16)
        w16 = torch.from_numpy(w_np).to("cuda", dtype=torch.float16)
        t16_e, y16_e = add_rmsnorm(r16, h16, w16)
        torch.cuda.synchronize()
        t16_c, y16_c = compiled_add(r16, h16, w16)
        torch.cuda.synchronize()
        entry["fp16_compiled_vs_eager_y"] = comparison_metrics(y16_c, y16_e)
        entry["fp16_compiled_y_vs_fp32_truth"] = metrics_vs_truth(y16_c, y_true)
        entry["fp16_compiled_t_vs_fp32_truth"] = metrics_vs_truth(t16_c, t_true)
        entry["fp16_eager_y_vs_fp32_truth"] = metrics_vs_truth(y16_e, y_true)
        del r16, h16, w16, t16_e, y16_e, t16_c, y16_c

        # ---- rmsnorm 单对象（fp32） ----
        y_er = rmsnorm_expr(torch.from_numpy(t_true).to("cuda"),
                            torch.from_numpy(w_np).to("cuda"))
        torch.cuda.synchronize()
        entry["eager_rmsnorm_vs_truth"] = metrics_vs_truth(y_er, y_true)
        y_cr = compiled_rms(torch.from_numpy(t_true).to("cuda"),
                            torch.from_numpy(w_np).to("cuda"))
        torch.cuda.synchronize()
        entry["compiled_rmsnorm_vs_truth"] = metrics_vs_truth(y_cr, y_true)
        del y_er, y_cr

        graphs = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
        entry["unique_graphs_cumulative"] = int(graphs)
        entries.append(entry)
        print(
            f"[correctness] {prefix}: eager y {entry['eager_y_vs_truth']['max_abs_error']:.3e} | "
            f"compiled y {entry['compiled_y_vs_truth']['max_abs_error']:.3e} | "
            f"t bitwise {entry['compiled_t_vs_truth']['bitwise_equal']} | "
            f"fp16 compiled y vs truth {entry['fp16_compiled_y_vs_fp32_truth']['max_abs_error']:.3e} | "
            f"graphs {graphs}",
            flush=True,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "truth": "ch04 separated_add_rmsnorm_serial（CPU FP32，fixtures manifest SHA256）",
        "tolerances": {"fp32_y": "1e-5（合同）", "fp16_y": "2e-2（ch04 口径）", "t_fp32": "位级"},
        "entries": entries,
    }
    write_results_json(CHAPTER / "results" / "correctness-results-cuda.json", payload)


if __name__ == "__main__":
    main()
