#!/usr/bin/env python3
"""第 06 期三平台 stride 行为探针（torch CPU/CUDA/MUSA 同脚本逐项 diff）。

采内容：
  1. stride 语义：连续/转置视图/行切片/expand 的 .stride()/is_contiguous()
     （ATen 层语义，预期与后端无关——实测核对）；
  2. transpose-back 陷阱：tv.t().contiguous() 是否零拷贝返回别名（data_ptr）；
  3. .contiguous() 裸成本：转置视图 / 行切片 / 连续 clone 的延迟与有效带宽
     （rows=2048/32768，fp32+fp16）——torch_musa vs torch CUDA 的 layout
     copy 路径差异（迁移台账素材）。

用法：python3 probe_layout.py --device cuda|musa|cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import OperatorTimer, backend_sync, resolve_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "musa"])
    arguments = parser.parse_args()
    device = resolve_device(arguments.device) if arguments.device != "cpu" else torch.device("cpu")

    rows, hidden = 8, 1024
    x = torch.randn(rows, hidden, device=device)
    tv = x.t().contiguous().t()
    rs = torch.randn(rows, hidden + 8, device=device)[:, :hidden]
    expanded = torch.randn(1, hidden, device=device).expand(rows, hidden)

    semantics = {
        "torch_version": torch.__version__,
        "device": arguments.device,
        "contiguous": {"stride": list(x.stride()), "is_contiguous": x.is_contiguous()},
        "transposed_view": {"stride": list(tv.stride()), "is_contiguous": tv.is_contiguous(),
                            "values_equal_x": bool(torch.equal(tv, x))},
        "row_sliced_view": {"stride": list(rs.stride()), "is_contiguous": rs.is_contiguous(),
                            "is_view": rs._base is not None},
        "expanded": {"stride": list(expanded.stride()), "is_contiguous": expanded.is_contiguous()},
        "transpose_back_trap": {
            "tv_t_is_contiguous": tv.t().is_contiguous(),
            "contiguous_same_ptr": tv.t().contiguous().data_ptr() == tv.t().data_ptr(),
            "back_stride": list(tv.t().contiguous().t().stride()),
        },
        "contiguous_output": {"stride": list(tv.contiguous().stride()),
                              "equal_x": bool(torch.equal(tv.contiguous(), x))},
    }

    costs = {"device": arguments.device, "entries": []}
    for rows_b in [2048, 32768]:
        xb = torch.randn(rows_b, hidden, device=device)
        for dtype in ([torch.float32] if arguments.device == "cpu" else [torch.float32, torch.float16]):
            # 注意：视图必须先转 dtype 再派生——.to(dtype) 会物化连续输出，
            # 把非连续视图变连续（探针自身踩坑一次，修正记录）。
            xb_d = xb.to(dtype)
            tvb_d = xb_d.t().contiguous().t()      # 转置视图（fp16 也保持 stride=(1,rows)）
            rsb_d = torch.randn(rows_b, hidden + 8, device=device,
                                dtype=torch.float32).to(dtype)[:, :hidden]
            copy_bytes = rows_b * hidden * xb_d.element_size()
            timer = OperatorTimer(device, warmup=5, repeats=20 if arguments.device != "cpu" else 5)
            entry = {"rows": rows_b, "dtype": str(dtype).replace("torch.", ""),
                     "copy_bytes": copy_bytes}
            p50s = {}
            for name, op in {
                "transpose_copy": lambda: tvb_d.contiguous(),
                "row_gather_copy": lambda: rsb_d.contiguous(),
                "contig_clone": lambda: xb_d.clone(),
            }.items():
                t = timer.measure(op)
                p50s[name] = t["p50_ms"]
                entry[name + "_p50_ms"] = t["p50_ms"]
                entry[name + "_bw_gb_s"] = round(copy_bytes / (t["p50_ms"] / 1000.0) / 1e9, 2)
            entry["transpose_vs_clone_ratio"] = round(p50s["transpose_copy"] / p50s["contig_clone"], 3)
            costs["entries"].append(entry)
            print(f"[probe/{arguments.device}/{dtype}] rows={rows_b}: "
                  f"transpose={p50s['transpose_copy']:.3f}ms "
                  f"({entry['transpose_copy_bw_gb_s']}GB/s) "
                  f"gather={p50s['row_gather_copy']:.3f}ms "
                  f"clone={p50s['contig_clone']:.3f}ms "
                  f"ratio={entry['transpose_vs_clone_ratio']}")

    backend_sync(device)
    out = HERE / "results" / f"layout-probe-{arguments.device}.json"
    out.write_text(json.dumps({"schema_version": "kernel-chapter07.layout-probe.v1",
                               "semantics": semantics, "costs": costs},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[probe] 已写入 {out}")
    print(json.dumps(semantics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
