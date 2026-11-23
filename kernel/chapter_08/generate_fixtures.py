#!/usr/bin/env python3
"""第 08 期 fixtures 生成（本地无 GPU 机器执行，两台服务器共用同一份）。

内容：RMSNorm 小例的输入/权重/CPU FP32 参考真值（与第 02 期朴素参考同语义：
FP32 串行、FP64 中间计算后写回 FP32 —— 容差按 rsqrtf 近似 ~1e-6 设定）。
固定 seed=20260909；逐张量 SHA256 清单写入 manifest.json。

注意：本期主体是 Host↔设备路径（无 CPU 列，期首声明），fixtures 仅作
管线级回归锚（kernel 正确性已在第 02 期全量验证）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 20260909
H = 1024
ROWS = 128

ROOT = Path(__file__).resolve().parent


def sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> None:
    out = ROOT / "fixtures"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((ROWS, H), dtype=np.float32)
    w = rng.uniform(0.5, 2.0, H).astype(np.float32)
    # CPU FP32 参考真值（FP64 中间计算，与 ch03 口径一致）
    x64 = x.astype(np.float64)
    scale = 1.0 / np.sqrt((x64 ** 2).mean(axis=1, keepdims=True) + 1e-6)
    y_ref = (x64 * scale * w.astype(np.float64)).astype(np.float32)

    np.savez(out / "rmsnorm-fixtures.npz", x=x, w=w, y_ref=y_ref)

    manifest = {
        "schema": "kernel-chapter09.fixtures.v1",
        "seed": SEED,
        "created_at_utc": None,  # 由调用环境记录；sha256 是身份的主键
        "tensors": {
            "x": {"shape": list(x.shape), "dtype": "float32", "sha256": sha256(x)},
            "w": {"shape": list(w.shape), "dtype": "float32", "sha256": sha256(w)},
            "y_ref": {"shape": list(y_ref.shape), "dtype": "float32", "sha256": sha256(y_ref)},
        },
        "note": "y_ref = CPU FP64 中间计算写回 FP32（第 02 期朴素参考语义，eps=1e-6）",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
