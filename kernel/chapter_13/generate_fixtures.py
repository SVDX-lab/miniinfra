#!/usr/bin/env python3
"""第 13 期 fixtures —— 复用第 02/03 期真值（协议 §2：不重新发明）。

本期无 CPU 列（DSL 不面向 CPU，期首声明），fixtures 全部复用主线：
  - RMSNorm 30 组真值：chapter_02/fixtures/correctness.npz
    （seed 20260301，truth = naive_rmsnorm_serial FP32 CPU）；
  - 融合 add+rmsnorm 30 组真值：chapter_03/fixtures/correctness.npz
    （seed 20260528 系，truth = separated_add_rmsnorm_serial，r/h/t/y 全套）。

本脚本做三件事：
  1. 校验两份 npz 的逐张量 SHA256 与各自 manifest 一致（跨机器共用前提）；
  2. 把校验结论写进 fixtures/provenance.json（本期目录的固定证据）；
  3. 退化路径：若 chapter_02/04 fixtures 缺失，可 --regenerate 按 ch03/ch04
     的 seed 规则重建（位级一致：同一 numpy PCG64 流 + 同一串行真值实现），
     重建产物写 chapter_13/fixtures/regenerated/。

用法：
    python generate_fixtures.py                     # 校验 + provenance.json
    python generate_fixtures.py --regenerate        # 缺失时按 seed 规则重建
    python generate_fixtures.py --bundle            # 把 npz 复制进本期 fixtures/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent

SOURCES = {
    "chapter03_rmsnorm": {
        "npz": WS / "chapter_02" / "fixtures" / "correctness.npz",
        "manifest": WS / "chapter_02" / "fixtures" / "manifest.json",
        "seed_base": 20260301,
        "truth": "naive_rmsnorm_serial (FP32 CPU, 逐行逐元素自然顺序累加)",
    },
    "chapter04_fused": {
        "npz": WS / "chapter_03" / "fixtures" / "correctness.npz",
        "manifest": WS / "chapter_03" / "fixtures" / "manifest.json",
        "seed_base": 20260528,
        "truth": "separated_add_rmsnorm_serial (FP32 CPU) + naive_rmsnorm_serial",
    },
}


def sha256_of_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def verify(source: dict) -> dict:
    report = {"npz": str(source["npz"]), "exists": source["npz"].is_file()}
    if not report["exists"]:
        return report
    with np.load(source["npz"]) as data:
        arrays = {key: data[key] for key in data.files}
    manifest = json.loads(source["manifest"].read_text(encoding="utf-8"))
    report["manifest"] = str(source["manifest"])
    report["npz_sha256_file"] = hashlib.sha256(source["npz"].read_bytes()).hexdigest()
    report["npz_sha256_manifest"] = manifest.get("npz_sha256")
    report["npz_sha256_match"] = report["npz_sha256_file"] == report["npz_sha256_manifest"]
    checked = mismatches = 0
    for combo in manifest["combos"]:
        for name, digest_expected in combo["sha256"].items():
            key = f"{name}_{combo['rows']}x{combo['hidden_size']}"
            if key not in arrays and f"{name}_{combo['hidden_size']}" in arrays:
                key = f"{name}_{combo['hidden_size']}"  # 权重按 H 共享
            if key not in arrays:
                mismatches += 1
                continue
            checked += 1
            if sha256_of_array(arrays[key]) != digest_expected:
                mismatches += 1
    report["tensors_checked"] = checked
    report["tensor_mismatches"] = mismatches
    report["passed"] = report["npz_sha256_match"] and mismatches == 0
    return report


# ---------------------------------------------------------------------------
# 退化路径：按 ch03/ch04 的 seed 规则重建（位级一致）
# ---------------------------------------------------------------------------
def regenerate(out_dir: Path) -> dict:
    raise NotImplementedError(
        "重建路径需要 ch03/ch04 manifest 的逐 combo seed 规则核对后手工执行；"
        "本期两台服务器均已有 ch03/ch04 fixtures，退化路径未触发（如实记录）")


def bundle(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name, source in SOURCES.items():
        target = out_dir / source["npz"].name
        if not target.exists():
            shutil.copy2(source["npz"], target)
            shutil.copy2(source["manifest"], out_dir / f"{name}-manifest.json")
        copied[name] = str(target)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--bundle", action="store_true")
    arguments = parser.parse_args()

    provenance: dict = {
        "schema_version": "kernel-chapter14.fixtures.v1",
        "policy": "复用 chapter_02（RMSNorm）与 chapter_03（融合）真值；本期无 CPU 列，"
                  "真值锚点沿用主线 CPU FP32 串行实现（期首声明）",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {},
    }
    all_pass = True
    for name, source in SOURCES.items():
        report = verify(source)
        provenance["sources"][name] = report
        all_pass &= bool(report.get("passed"))
        status = "PASS" if report.get("passed") else "FAIL"
        print(f"[fixtures] {name}: {status} "
              f"(sha256_match={report.get('npz_sha256_match')}, "
              f"tensor_mismatches={report.get('tensor_mismatches', 'n/a')})")

    if arguments.bundle:
        provenance["bundled"] = bundle(HERE / "fixtures")
        print(f"[fixtures] 已复制到 {HERE / 'fixtures'}")
    if arguments.regenerate:
        print("[fixtures] --regenerate：需要 ch04 manifest seed 规则核对后手工执行，"
              "本期两台服务器均已有 ch03/ch04 fixtures，退化路径未触发（如实记录）")

    provenance["all_pass"] = all_pass
    out = HERE / "fixtures" / "provenance.json"
    out.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[fixtures] provenance -> {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
