#!/usr/bin/env python3
"""第 01 期性能诊断 fixtures：固定 seed 生成 benchmark / ncu / launch-bound 共用输入。

第 01 期性能诊断没有新的数值合同（被测对象与第 01 期是同一个 PyTorch 组合表达式，
正确性锚点沿用第 01 期四件套），因此 fixtures 的作用是：把两台服务器上
benchmark 与 ncu 用的输入张量位级固定（逐张量 SHA256），保证证据链可复现。

seed 规则与第 01 期 benchmark 一致：
    seed = 20260101 + rows * 100000 + hidden_size
CPU FP32 生成后以 .npy 落盘；GPU 侧 benchmark 优先读取本目录（缺失时按同一
seed 规则内联生成，两路位级一致，由 smoke_test.py 校验）。

用法：
    python generate_fixtures.py            # 在任一有 torch CPU 的机器执行
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from roofline import HIDDEN_SIZE, ROW_COUNTS, RMS_NORM_EPS

FIXTURES_SEED_BASE = 20260101
SCHEMA_VERSION = "kernel-chapter02.fixtures.v1"


def seed_for(rows: int, hidden_size: int) -> int:
    return FIXTURES_SEED_BASE + rows * 100000 + hidden_size


def sha256_of_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir.mkdir(exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "fixtures_seed_base": FIXTURES_SEED_BASE,
        "seed_rule": "20260101 + rows*100000 + hidden_size（与第 01 期一致）",
        "purpose": "benchmark/ncu 输入位级固定；正确性锚点沿用第 01 期 fixtures",
        "tensors": {},
    }

    weight = torch.randn(HIDDEN_SIZE, generator=torch.Generator().manual_seed(seed_for(0, HIDDEN_SIZE)))
    weight_path = out_dir / "weight-fp32.npy"
    np.save(weight_path, weight.numpy())
    manifest["tensors"]["weight_fp32"] = {
        "file": weight_path.name,
        "shape": list(weight.shape),
        "sha256": sha256_of_array(weight.numpy()),
    }

    for rows in ROW_COUNTS:
        generator = torch.Generator().manual_seed(seed_for(rows, HIDDEN_SIZE))
        x = torch.randn(rows, HIDDEN_SIZE, generator=generator)
        path = out_dir / f"x-{rows}x{HIDDEN_SIZE}-fp32.npy"
        np.save(path, x.numpy())
        manifest["tensors"][f"x_{rows}x{HIDDEN_SIZE}_fp32"] = {
            "file": path.name,
            "shape": list(x.shape),
            "sha256": sha256_of_array(x.numpy()),
        }
        print(f"[fixtures] {path.name} sha256={manifest['tensors'][f'x_{rows}x{HIDDEN_SIZE}_fp32']['sha256'][:16]}…")

    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[fixtures] 清单已写入 {manifest_path}")
    print(f"[fixtures] rms_norm_eps={RMS_NORM_EPS}（仅记录，不参与生成）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
