"""第 14 期 fixtures 生成（固定 seed，参照第 01/03 期口径）。

组集合：合同 shape {1,128,2048}×1024 三组 + 非 2 幂边界 (128,127)/(128,1025)
两组，共 5 组 × 5 张量（r/h/w/t/y）。t/y 由第 03 期真值
separated_add_rmsnorm_serial 生成（CPU FP32，随代码上服务器共用）。

seed 规则（ch04 同构，基点换为本期 20260415）：
    r_seed = 20260415 + rows*100000 + H
    h_seed = r_seed + 555555（独立流）
    w_seed = 20260415 + H
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parent
GROUPS = [
    (1, 1024),
    (128, 1024),
    (2048, 1024),
    (128, 127),
    (128, 1025),
]
SEED_BASE = 20260415


def seed_for(rows: int, hidden: int) -> dict[str, int]:
    r_seed = SEED_BASE + rows * 100000 + hidden
    return {"r": r_seed, "h": r_seed + 555555, "w": SEED_BASE + hidden}


def sha256_of_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    from implementations.cpu.reference_serial import separated_add_rmsnorm_serial

    out_dir = CHAPTER / "fixtures"
    out_dir.mkdir(exist_ok=True)
    manifest: dict = {
        "episode": 15,
        "seed_base": SEED_BASE,
        "groups": [],
        "tensors": {},
        "truth": "separated_add_rmsnorm_serial (ch04 原样复用，CPU FP32)",
    }
    arrays: dict[str, np.ndarray] = {}
    for rows, hidden in GROUPS:
        started = time.perf_counter()
        seeds = seed_for(rows, hidden)
        rng_r = np.random.default_rng(seeds["r"])
        rng_h = np.random.default_rng(seeds["h"])
        rng_w = np.random.default_rng(seeds["w"])
        r = rng_r.standard_normal((rows, hidden), dtype=np.float32)
        h = rng_h.standard_normal((rows, hidden), dtype=np.float32)
        w = rng_w.standard_normal((hidden,), dtype=np.float32)
        t, y = separated_add_rmsnorm_serial(r, h, w)
        prefix = f"{rows}x{hidden}"
        for name, array in (("r", r), ("h", h), ("w", w), ("t", t), ("y", y)):
            key = f"{prefix}/{name}"
            arrays[key] = array
            manifest["tensors"][key] = {
                "seed": seeds[name] if name in seeds else None,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": sha256_of_array(array),
            }
        manifest["groups"].append(
            {"rows": rows, "hidden": hidden, "seeds": seeds,
             "generation_seconds": round(time.perf_counter() - started, 2)}
        )
        print(f"[fixtures] {prefix} 完成（{manifest['groups'][-1]['generation_seconds']}s）")

    npz_path = out_dir / "correctness.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest["npz_bytes"] = npz_path.stat().st_size
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"[fixtures] 写出 {npz_path}（{manifest['npz_bytes']/1e6:.1f} MB）")


if __name__ == "__main__":
    main()
