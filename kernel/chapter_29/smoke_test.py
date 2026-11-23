#!/usr/bin/env python3
"""第 29 期 smoke test（边界行为快速验证）。

--backend cpu : 本地（无 GPU）：fixtures 对拍 + kernel_sim 小规模 + Python 脚手架自检
--backend cuda: AI 三二进制：--selftest + 最小边界集（零形状/非法形状/文件不符）
                ——正式全量边界在 compare_correctness.py --backend cuda

用法：python smoke_test.py --backend cpu|cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

BINARIES = {
    "rmsnorm": HERE / "implementations/ai/rmsnorm/ai_rmsnorm",
    "gemm": HERE / "implementations/ai/gemm/ai_gemm",
    "attention": HERE / "implementations/ai/attention/ai_decode_attn",
}

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[smoke:{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def smoke_cpu() -> None:
    import kernel_sim

    # fixtures 快速对拍（npz 级）
    import verify_fixtures as vf
    try:
        vf.resolve_npz("chapter_02/fixtures/correctness.npz")
        vf.resolve_npz("chapter_17/fixtures/correctness.npz")
        vf.resolve_npz("chapter_20/fixtures/correctness.npz")
        check("fixtures_resolution", True)
    except FileNotFoundError as e:
        check("fixtures_resolution", False, str(e))
        return

    # kernel_sim 小规模
    sim = kernel_sim.run_all(max_rows=4, max_n=64)
    check("kernel_sim_runs", sim["case_count"] >= 10,
          f"cases={sim['case_count']}")
    # 结构性缺陷预判（本期实验对象：AI 起草缺陷应在预审可见）
    rms_bad = [c for c in sim["cases"]
               if c["operator"] == "rmsnorm" and c.get("structural_fail")]
    att_bad = [c for c in sim["cases"]
               if c["operator"] == "attention" and c.get("structural_fail")]
    gem_bad = [c for c in sim["cases"]
               if c["operator"] == "gemm" and c.get("structural_fail")]
    check("sim_rmsnorm_structural_fail_seen", len(rms_bad) > 0,
          f"{len(rms_bad)} 组结构失败（归约广播缺陷，H>32 触发）")
    check("sim_attention_structural_fail_seen", len(att_bad) > 0,
          f"{len(att_bad)} 组结构失败")
    check("sim_gemm_clean", len(gem_bad) == 0, "gemm 无结构失败（预期）")

    # 源码始终校验。原始提示词由书稿工程保存，可按需一起校验。
    import json
    import hashlib

    rec = json.loads((HERE / "prompts/generation-record.json").read_text())
    prompts_dir = os.environ.get("CH29_PROMPTS_DIR") or os.environ.get("CH30_PROMPTS_DIR")
    for key, meta in rec["artifacts"].items():
        if "sha256" not in meta:
            continue
        relative = Path(meta["path"])
        if relative.parent == Path("prompts"):
            if prompts_dir is None:
                continue
            path = Path(prompts_dir) / relative.name
        else:
            path = HERE / relative
        if not path.is_file():
            check(f"frozen_{key}", False, f"文件缺失：{path}")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        check(f"frozen_{key}", got == meta["sha256"],
              "与 generation-record.json 一致" if got == meta["sha256"] else "被改动!")


def smoke_cuda() -> None:
    missing = [op for op, p in BINARIES.items() if not p.exists()]
    check("binaries_built", not missing, f"missing={missing}" if missing else "")
    if missing:
        return

    # selftest 冒烟（结果本身是实验素材：AI 自检的检出力）
    for op, p in BINARIES.items():
        proc = subprocess.run([str(p), "--selftest"], capture_output=True,
                              text=True, timeout=120)
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        check(f"{op}_selftest_runs", proc.returncode in (0, 1),
              f"exit={proc.returncode} last={tail}")

    with tempfile.TemporaryDirectory(prefix="ch30smoke_") as td:
        tmp = Path(td)
        rng = np.random.default_rng(7)
        h = 128
        w = rng.uniform(0.5, 1.5, h).astype(np.float32)
        x = rng.standard_normal((2, h)).astype(np.float32)
        xp, wp, op_ = tmp / "x.bin", tmp / "w.bin", tmp / "y.bin"
        x.tofile(xp)
        w.tofile(wp)
        proc = subprocess.run([str(BINARIES["rmsnorm"]), "--run", "--rows", "0",
                               "--hidden", str(h), "--dtype", "fp32",
                               "--x", str(xp), "--w", str(wp), "--out", str(op_)],
                              capture_output=True, text=True)
        check("rmsnorm_rows0_empty_exit0",
              proc.returncode == 0 and op_.exists() and op_.stat().st_size == 0,
              f"exit={proc.returncode}")
        proc = subprocess.run([str(BINARIES["rmsnorm"]), "--run", "--rows", "2",
                               "--hidden", "0", "--dtype", "fp32",
                               "--x", str(xp), "--w", str(wp), "--out", str(op_)],
                              capture_output=True, text=True)
        check("rmsnorm_hidden0_nonzero_exit", proc.returncode != 0,
              f"exit={proc.returncode}")
        proc = subprocess.run([str(BINARIES["attention"]), "--bench", "--batch", "1",
                               "--kv-len", "32768", "--dtype", "fp16",
                               "--warmup", "0", "--repeats", "1"],
                              capture_output=True, text=True, timeout=120)
        check("attention_n32768_probe_recorded", True,
              f"exit={proc.returncode}（非 0 = AI-SCALING 证据；实验素材不是失败）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    if args.backend == "cpu":
        smoke_cpu()
    else:
        smoke_cuda()
    print(f"[smoke] {'PASS' if not failures else 'FAIL'} "
          f"（{len(failures)} failures）")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
