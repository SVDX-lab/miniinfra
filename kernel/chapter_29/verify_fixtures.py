#!/usr/bin/env python3
"""第 29 期 fixtures 复用校验：SHA256 对拍 ch03/18/21（不重新生成）。

对拍三层：
  1. npz 文件级 SHA256 vs 原章 manifest 的 npz_sha256 字段（ch21 无该字段时跳过并记录）；
  2. 逐张量 SHA256 vs 原章 manifest（ch03 combo.sha256{x,w,y}、ch18 combo.sha256
     按数组名、ch21 tensors{name:{sha256,...}}）；
  3. 本期使用的数组集合存在性（decode 6 组 / rmsnorm 30 组 / gemm 8 组）。

用法：
  python verify_fixtures.py                 # 全量校验，写 results/fixtures-verification.json
  python verify_fixtures.py --quiet         # 只打印摘要
退出码：0=全部通过；1=校验失败；2=fixtures 找不到。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """原章逐张量哈希配方（ch03/ch18/ch21 三家一致）：shape + dtype + 原始字节。"""
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(contiguous.shape).encode("utf-8"))
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def resolve_npz(rel_path: str) -> Path:
    env_root = os.environ.get("CH29_FIXTURES_ROOT") or os.environ.get("CH30_FIXTURES_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root) / rel_path)          # 仓库镜像结构
        candidates.append(Path(env_root) / Path(rel_path).name)  # 平铺
    candidates.append(REPO_ROOT / rel_path)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"fixtures 未找到: {rel_path}（尝试: " + "; ".join(str(c) for c in candidates) + "）"
    )


def verify_ch03(manifest: dict, npz_path: Path) -> dict:
    out: dict = {"source": "chapter_02", "npz": str(npz_path)}
    expected_npz = manifest.get("npz_sha256")
    got_npz = sha256_file(npz_path)
    out["npz_sha256"] = got_npz
    out["npz_sha256_match"] = (expected_npz == got_npz) if expected_npz else None
    data = np.load(npz_path)
    combos = manifest["combos"]
    tensor_checks = []
    for combo in combos:
        r, h = combo["rows"], combo["hidden_size"]
        names = {"x": f"x_{r}x{h}", "w": f"w_{h}", "y": f"y_{r}x{h}"}
        for key, arr_name in names.items():
            expected = combo["sha256"][key]
            got = sha256_array(data[arr_name])
            tensor_checks.append(
                {"array": arr_name, "expected": expected, "got": got,
                 "match": expected == got}
            )
    out["tensors_checked"] = len(tensor_checks)
    out["tensors_matched"] = sum(c["match"] for c in tensor_checks)
    out["tensor_checks"] = tensor_checks if len(tensor_checks) <= 5 else tensor_checks[:3] + [
        {"note": f"... 其余 {len(tensor_checks) - 4} 条省略（match 状态见 tensors_matched）"},
        tensor_checks[-1],
    ]
    out["pass"] = bool(out["npz_sha256_match"] is not False
                       and out["tensors_matched"] == out["tensors_checked"])
    return out


def verify_ch18(manifest: dict, npz_path: Path) -> dict:
    out: dict = {"source": "chapter_17", "npz": str(npz_path)}
    expected_npz = manifest.get("npz_sha256")
    got_npz = sha256_file(npz_path)
    out["npz_sha256"] = got_npz
    out["npz_sha256_match"] = (expected_npz == got_npz) if expected_npz else None
    data = np.load(npz_path)
    tensor_checks = []
    for combo in manifest["combos"]:
        m, k, n = combo["shape"]
        for key, expected in combo["sha256"].items():
            # ch18 manifest 两种键风格并存：combo0 用完整数组名（A_4x8x4），
            # 其余 combo 用 split 前缀（A/B/bias/C/...）——按 npz 实际文件名解析。
            arr_name = key if key in data.files else f"{key}_{m}x{k}x{n}"
            if arr_name not in data.files:
                tensor_checks.append({"array": arr_name, "match": False,
                                      "note": "npz 中不存在"})
                continue
            got = sha256_array(data[arr_name])
            tensor_checks.append(
                {"array": arr_name, "expected": expected, "got": got,
                 "match": expected == got}
            )
    out["tensors_checked"] = len(tensor_checks)
    out["tensors_matched"] = sum(
        1 for c in tensor_checks if c["match"] is True
    )
    out["tensor_checks_sample"] = tensor_checks[:3]
    out["pass"] = bool(out["npz_sha256_match"] is not False
                       and out["tensors_matched"] == out["tensors_checked"]
                       and out["tensors_checked"] > 0)
    return out


def verify_ch21(manifest: dict, npz_path: Path) -> dict:
    out: dict = {"source": "chapter_20", "npz": str(npz_path)}
    expected_npz = manifest.get("npz_sha256")
    got_npz = sha256_file(npz_path)
    out["npz_sha256"] = got_npz
    out["npz_sha256_match"] = (expected_npz == got_npz) if expected_npz else None
    data = np.load(npz_path)
    tensors = manifest["tensors"]
    needed = [f"d_n{n}_b{b}_{k}" for n in (64, 512, 2048) for b in (1, 4)
              for k in ("q", "k", "v", "truth32")]
    tensor_checks = []
    for arr_name in needed:
        if arr_name not in tensors or arr_name not in data.files:
            tensor_checks.append({"array": arr_name, "match": False,
                                  "note": "manifest 或 npz 中不存在"})
            continue
        expected = tensors[arr_name]["sha256"]
        got = sha256_array(data[arr_name])
        tensor_checks.append(
            {"array": arr_name, "expected": expected, "got": got,
             "match": expected == got}
        )
    out["decode_tensors_checked"] = len(tensor_checks)
    out["decode_tensors_matched"] = sum(c["match"] is True for c in tensor_checks)
    out["tensor_checks_sample"] = tensor_checks[:3]
    out["pass"] = bool(out["decode_tensors_matched"] == out["decode_tensors_checked"]
                       and out["decode_tensors_checked"] > 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    report: dict = {"schema_version": "kernel-chapter30.fixtures-verification.v1",
                    "policy": "复用不重生成；SHA256 对拍原章 manifest", "sources": []}

    try:
        m3 = json.loads((REPO_ROOT / "chapter_02/fixtures/manifest.json").read_text())
        npz3 = resolve_npz("chapter_02/fixtures/correctness.npz")
        report["sources"].append(verify_ch03(m3, npz3))
    except FileNotFoundError as e:
        print(f"[fixtures] FAIL: {e}", file=sys.stderr)
        return 2

    try:
        m18 = json.loads((REPO_ROOT / "chapter_17/fixtures/manifest.json").read_text())
        npz18 = resolve_npz("chapter_17/fixtures/correctness.npz")
        report["sources"].append(verify_ch18(m18, npz18))
    except FileNotFoundError as e:
        print(f"[fixtures] FAIL: {e}", file=sys.stderr)
        return 2

    try:
        m21 = json.loads((REPO_ROOT / "chapter_20/fixtures/manifest.json").read_text())
        npz21 = resolve_npz("chapter_20/fixtures/correctness.npz")
        report["sources"].append(verify_ch21(m21, npz21))
    except FileNotFoundError as e:
        print(f"[fixtures] FAIL: {e}", file=sys.stderr)
        return 2

    report["pass"] = all(s["pass"] for s in report["sources"])
    out_path = HERE / "results" / "fixtures-verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    if not args.quiet:
        for s in report["sources"]:
            print(f"[fixtures] {s['source']}: npz_match={s['npz_sha256_match']} "
                  f"pass={s['pass']}")
        print(f"[fixtures] 总体: pass={report['pass']}（详情 {out_path}）")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
