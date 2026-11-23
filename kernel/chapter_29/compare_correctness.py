#!/usr/bin/env python3
"""第 29 期正确性验收（AI 版走与人工版完全相同的比较器口径）。

--backend cpu  : fixtures SHA256 对拍 + kernel_sim 行为仿真（静态预审，本地可跑）
--backend cuda : AI 三二进制 --selftest 记录 + 全合同对拍（30/8/6 组 × dtype）
                 + 确定性 3×位级 + 边界全集（非整 Tile/NaN/错误 Shape/零形状）
                 → results/correctness-results-cuda.json
容差与判级来自 experiment-contract.yaml（rmsnorm 1e-5/2e-2、gemm 5e-3/0.15、
attention 5e-4/2e-2，与人工版合同逐字对应）。

用法：python compare_correctness.py --backend cpu|cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import comparison_metrics, write_results_json  # noqa: E402

BINARIES = {
    "rmsnorm": HERE / "implementations/ai/rmsnorm/ai_rmsnorm",
    "gemm": HERE / "implementations/ai/gemm/ai_gemm",
    "attention": HERE / "implementations/ai/attention/ai_decode_attn",
}

TOL = {
    "rmsnorm": {"fp32": 1e-5, "fp16": 2e-2},
    "gemm": {"fp32": 5e-3, "fp16": 1.5e-1},
    "attention": {"fp32": 5e-4, "fp16": 2e-2},
}


# ---------------------------------------------------------------------------
# fixtures 加载（复用校验逻辑）
# ---------------------------------------------------------------------------

def load_fixtures() -> dict:
    import verify_fixtures as vf

    root = HERE.parent
    out: dict = {}
    m3 = json.loads((root / "chapter_02/fixtures/manifest.json").read_text())
    out["rmsnorm"] = {"manifest": m3, "npz": np.load(vf.resolve_npz("chapter_02/fixtures/correctness.npz"))}
    m18 = json.loads((root / "chapter_17/fixtures/manifest.json").read_text())
    out["gemm"] = {"manifest": m18, "npz": np.load(vf.resolve_npz("chapter_17/fixtures/correctness.npz"))}
    m21 = json.loads((root / "chapter_20/fixtures/manifest.json").read_text())
    out["attention"] = {"manifest": m21, "npz": np.load(vf.resolve_npz("chapter_20/fixtures/correctness.npz"))}
    return out


def run_binary(binary: Path, args: list[str], timeout: int = 300) -> dict:
    proc = subprocess.run([str(binary)] + args, capture_output=True, text=True,
                          timeout=timeout)
    return {"exit_code": proc.returncode, "stdout_tail": proc.stdout.strip().splitlines()[-8:],
            "stderr_tail": proc.stderr.strip().splitlines()[-8:]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    return arr.astype(np.float16) if dtype == "fp16" else arr.astype(np.float32)


def eval_case(op: str, name: str, out_arr: np.ndarray, truth: np.ndarray, dtype: str) -> dict:
    # 统一展平为 1-D 再比较（二进制读回的是裸数组，真值带形状）
    metrics = comparison_metrics(torch.from_numpy(out_arr.astype(np.float32).reshape(-1)),
                                 torch.from_numpy(truth.astype(np.float32).reshape(-1)))
    metrics.update({"operator": op, "case": name, "dtype": dtype,
                    "tolerance": TOL[op][dtype]})
    metrics["pass"] = bool(
        metrics["max_abs_error"] <= TOL[op][dtype]
        and metrics["nan_count"] == 0 and metrics["inf_count"] == 0)
    return metrics


# ---------------------------------------------------------------------------
# CPU 列（静态预审；与 static_precheck.py 互补的轻量入口）
# ---------------------------------------------------------------------------

def correctness_cpu(results: dict) -> None:
    import kernel_sim
    import verify_fixtures as vf

    fx = load_fixtures()
    results["fixtures_verification"] = {
        "note": "全量对拍见 verify_fixtures.py 输出 results/fixtures-verification.json",
        "sim": kernel_sim.run_all(max_rows=2048, max_n=2048),
    }
    del fx


# ---------------------------------------------------------------------------
# CUDA 列：AI 二进制全合同对拍
# ---------------------------------------------------------------------------

def bench_arrays(tmp: Path, op: str, combo: dict, dtype: str, fx: dict) -> dict:
    """把 fixture 数组按 dtype 落成裸二进制；extra 必须保持 flag/value 顺序。"""
    files = {}
    if op == "rmsnorm":
        r, h = combo["rows"], combo["hidden_size"]
        x = to_dtype(fx["rmsnorm"]["npz"][f"x_{r}x{h}"], dtype)
        w = to_dtype(fx["rmsnorm"]["npz"][f"w_{h}"], dtype)
        truth = fx["rmsnorm"]["npz"][f"y_{r}x{h}"].astype(np.float32)
        files = {"x": (x, f"x_{r}x{h}.bin"), "w": (w, f"w_{h}.bin")}
        extra = ["--rows", str(r), "--hidden", str(h)]
        label = f"R{r}xH{h}"
    elif op == "gemm":
        m, k, n = combo["shape"]
        a = to_dtype(_ch18_arr(fx, m, k, n, "A"), dtype)
        b = to_dtype(_ch18_arr(fx, m, k, n, "B"), dtype)
        truth = _ch18_arr(fx, m, k, n, "C").astype(np.float32)
        files = {"x": (a, f"a_{m}_{k}_{n}.bin"), "w": (b, f"b_{m}_{k}_{n}.bin")}
        extra = ["--m", str(m), "--k", str(k), "--n", str(n)]
        label = f"{m}x{k}x{n}"
    else:
        n, bdim = combo["kv_len"], combo["batch"]
        q = to_dtype(fx["attention"]["npz"][f"d_n{n}_b{bdim}_q"], dtype)
        kk = to_dtype(fx["attention"]["npz"][f"d_n{n}_b{bdim}_k"], dtype)
        vv = to_dtype(fx["attention"]["npz"][f"d_n{n}_b{bdim}_v"], dtype)
        truth = fx["attention"]["npz"][f"d_n{n}_b{bdim}_truth32"].astype(np.float32)
        files = {"x": (q, f"q_{n}_{bdim}.bin"), "k": (kk, f"k_{n}_{bdim}.bin"),
                 "v": (vv, f"v_{n}_{bdim}.bin")}
        extra = ["--batch", str(bdim), "--kv-len", str(n)]
        label = f"N{n}B{bdim}"
    paths = {}
    for key, (arr, name) in files.items():
        p = tmp / name
        arr.tofile(p)
        paths[key] = p
    return {"paths": paths, "extra": extra, "truth": truth, "label": label}


def _ch18_arr(fx: dict, m: int, k: int, n: int, pref: str) -> np.ndarray:
    npz = fx["gemm"]["npz"]
    name = f"{pref}_{m}x{k}x{n}"
    return npz[name] if name in npz.files else npz[pref]


FLAG_FOR_KEY = {
    "rmsnorm": {"x": "--x", "w": "--w"},
    "gemm": {"x": "--a", "w": "--b"},
    "attention": {"x": "--q", "k": "--k", "v": "--v"},
}


def run_op_case(op: str, dtype: str, prepared: dict, out_path: Path) -> np.ndarray | None:
    binp = BINARIES[op]
    args = ["--run", "--dtype", dtype]
    args += prepared["extra"]
    for key, path in prepared["paths"].items():
        args += [FLAG_FOR_KEY[op][key], str(path)]
    args += ["--out", str(out_path)]
    proc = subprocess.run([str(binp)] + args, capture_output=True, text=True, timeout=600)
    prepared["last_run"] = {"exit_code": proc.returncode,
                            "stderr_tail": proc.stderr.strip().splitlines()[-5:]}
    if proc.returncode != 0 or not out_path.exists():
        return None
    dtype_np = np.float16 if dtype == "fp16" else np.float32
    return np.fromfile(out_path, dtype=dtype_np)


def correctness_cuda(results: dict) -> None:
    fx = load_fixtures()
    missing = [op for op, p in BINARIES.items() if not p.exists()]
    if missing:
        raise RuntimeError(f"AI 二进制未构建: {missing}（先 bash implementations/ai/*/build.sh）")

    cases: list[dict] = []
    determinism: list[dict] = []
    boundary: list[dict] = []
    selftests: dict = {}

    combos = {
        "rmsnorm": fx["rmsnorm"]["manifest"]["combos"],
        "gemm": fx["gemm"]["manifest"]["combos"],
        "attention": [{"kv_len": n, "batch": b} for n in (64, 512, 2048) for b in (1, 4)],
    }

    with tempfile.TemporaryDirectory(prefix="ch30cmp_") as td:
        tmp = Path(td)
        for op, op_combos in combos.items():
            # selftest 记录（AI 自己的检出力评估素材）
            selftests[op] = run_binary(BINARIES[op], ["--selftest"], timeout=120)
            for dtype in ("fp32", "fp16"):
                for idx, combo in enumerate(op_combos):
                    prepared = bench_arrays(tmp, op, combo, dtype, fx)
                    out_path = tmp / f"out_{op}_{idx}_{dtype}.bin"
                    out_arr = run_op_case(op, dtype, prepared, out_path)
                    name = prepared["label"]
                    if out_arr is None:
                        cases.append({"operator": op, "case": name, "dtype": dtype,
                                      "pass": False, "run_error": prepared.get("last_run")})
                        continue
                    if out_arr.size == 0 and prepared["truth"].size == 0:
                        cases.append({"operator": op, "case": name, "dtype": dtype,
                                      "pass": True, "note": "empty shape, empty output"})
                        continue
                    cases.append(eval_case(op, name, out_arr.astype(np.float32),
                                           prepared["truth"], dtype))

                # 确定性：每 op×dtype 取第 2 组，3 次 --run 对拍 SHA256
                combo = op_combos[1]
                prepared = bench_arrays(tmp, op, combo, dtype, fx)
                hashes = []
                for i in range(3):
                    out_path = tmp / f"det_{op}_{dtype}_{i}.bin"
                    run_op_case(op, dtype, prepared, out_path)
                    hashes.append(sha256_file(out_path) if out_path.exists() else "MISSING")
                determinism.append({
                    "operator": op, "dtype": dtype, "combo": combo,
                    "sha256_runs": hashes,
                    "bitwise_identical": len(set(hashes)) == 1 and hashes[0] != "MISSING",
                })

        # ---- 边界全集（合同 boundary_cases）----
        boundary += rmsnorm_boundary(tmp, fx)
        boundary += gemm_boundary(tmp, fx)
        boundary += attention_boundary(tmp, fx)

    results["selftest_capture"] = selftests
    results["cases"] = cases
    results["determinism"] = determinism
    results["boundary"] = boundary
    results["summary"] = {
        "cases_total": len(cases),
        "cases_pass": sum(1 for c in cases if c.get("pass")),
        "determinism_pass": sum(1 for d in determinism if d["bitwise_identical"]),
        "determinism_total": len(determinism),
        "boundary_entries": len(boundary),
    }


def rmsnorm_boundary(tmp: Path, fx: dict) -> list[dict]:
    out = []
    binp = BINARIES["rmsnorm"]
    h = 128
    rng = np.random.default_rng(20260301)
    w = rng.uniform(0.5, 1.5, h).astype(np.float32)
    x = rng.standard_normal((4, h)).astype(np.float32)

    # 全零行：输出该行全零
    xz = x.copy()
    xz[2, :] = 0.0
    xp, wp, op_ = tmp / "bz_x.bin", tmp / "bz_w.bin", tmp / "bz_out.bin"
    xz.tofile(xp)
    w.tofile(wp)
    proc = subprocess.run([str(binp), "--run", "--rows", "4", "--hidden", str(h),
                           "--dtype", "fp32", "--x", str(xp), "--w", str(wp),
                           "--out", str(op_)], capture_output=True, text=True)
    yz = np.fromfile(op_, dtype=np.float32).reshape(4, h) if op_.exists() else None
    out.append({"case": "rmsnorm/all_zero_row", "exit_code": proc.returncode,
                "row2_all_zero": bool(yz is not None and np.all(yz[2] == 0)),
                "other_rows_finite": bool(yz is not None and np.isfinite(yz[[0, 1, 3]]).all())})

    # NaN 行：该行全 NaN、其余行不受影响（合同语义）
    xn = x.copy()
    xn[1, 5] = np.nan
    xn.tofile(xp)
    op2 = tmp / "bz_out_nan.bin"
    proc = subprocess.run([str(binp), "--run", "--rows", "4", "--hidden", str(h),
                           "--dtype", "fp32", "--x", str(xp), "--w", str(wp),
                           "--out", str(op2)], capture_output=True, text=True)
    yn = np.fromfile(op2, dtype=np.float32).reshape(4, h) if op2.exists() else None
    out.append({"case": "rmsnorm/nan_row", "exit_code": proc.returncode,
                "row1_all_nan": bool(yn is not None and np.isnan(yn[1]).all()),
                "other_rows_finite": bool(yn is not None and np.isfinite(yn[[0, 2, 3]]).all()),
                "note": "合同：NaN 行全 NaN 其余行不受影响；缺陷组织下预期 row1 只有"
                        "warp0 元素是 NaN（kernel_sim 已仿真预判）"})

    # rows=0：空输出 + 退出码 0
    op3 = tmp / "bz_out_empty.bin"
    proc = subprocess.run([str(binp), "--run", "--rows", "0", "--hidden", str(h),
                           "--dtype", "fp32", "--x", str(xp), "--w", str(wp),
                           "--out", str(op3)], capture_output=True, text=True)
    out.append({"case": "rmsnorm/rows_zero", "exit_code": proc.returncode,
                "empty_file": bool(op3.exists() and op3.stat().st_size == 0)})

    # H=0：非零退出
    proc = subprocess.run([str(binp), "--run", "--rows", "2", "--hidden", "0",
                           "--dtype", "fp32", "--x", str(xp), "--w", str(wp),
                           "--out", str(op3)], capture_output=True, text=True)
    out.append({"case": "rmsnorm/hidden_zero", "exit_code": proc.returncode,
                "expect_nonzero_exit": proc.returncode != 0,
                "stderr": proc.stderr.strip()[:200]})

    # 文件大小不符：非零退出
    bad = tmp / "bz_bad.bin"
    (x[:, :64]).tofile(bad)
    proc = subprocess.run([str(binp), "--run", "--rows", "4", "--hidden", str(h),
                           "--dtype", "fp32", "--x", str(bad), "--w", str(wp),
                           "--out", str(op3)], capture_output=True, text=True)
    out.append({"case": "rmsnorm/file_size_mismatch", "exit_code": proc.returncode,
                "expect_nonzero_exit": proc.returncode != 0,
                "stderr": proc.stderr.strip()[:200]})
    return out


def gemm_boundary(tmp: Path, fx: dict) -> list[dict]:
    out = []
    binp = BINARIES["gemm"]
    rng = np.random.default_rng(20261118)
    a = rng.standard_normal((8, 8)).astype(np.float32)
    b = rng.standard_normal((8, 8)).astype(np.float32)
    ap, bp = tmp / "bg_a.bin", tmp / "bg_b.bin"
    a.tofile(ap)
    b.tofile(bp)
    op_ = tmp / "bg_out.bin"
    for label, args in (
        ("gemm/m_zero", ["--m", "0", "--k", "8", "--n", "8"]),
        ("gemm/n_zero", ["--m", "8", "--k", "8", "--n", "0"]),
        ("gemm/k_zero_invalid", ["--m", "8", "--k", "0", "--n", "8"]),
    ):
        proc = subprocess.run([str(binp), "--run"] + args + ["--dtype", "fp32",
                              "--a", str(ap), "--b", str(bp), "--out", str(op_)],
                              capture_output=True, text=True)
        entry = {"case": label, "exit_code": proc.returncode,
                 "stderr": proc.stderr.strip()[:200]}
        if "invalid" not in label:
            entry["empty_file_if_zero"] = bool(
                op_.exists() and op_.stat().st_size == 0) if proc.returncode == 0 else None
        out.append(entry)
    # 13x17x11 已在 correctness cases（非整 Tile）——此处记录指针
    out.append({"case": "gemm/non_tile_13x17x11",
                "note": "在 correctness cases 内（ch18 fixtures 组合 2）"})
    return out


def attention_boundary(tmp: Path, fx: dict) -> list[dict]:
    out = []
    binp = BINARIES["attention"]
    rng = np.random.default_rng(20261121)
    b_, n_, hq, hkv, d = 1, 64, 16, 8, 128
    q = rng.standard_normal((b_, hq, d)).astype(np.float32)
    kk = rng.standard_normal((b_, hkv, n_, d)).astype(np.float32)
    vv = rng.standard_normal((b_, hkv, n_, d)).astype(np.float32)
    qp, kp, vp = tmp / "ba_q.bin", tmp / "ba_k.bin", tmp / "ba_v.bin"
    q.tofile(qp)
    kk.tofile(kp)
    vv.tofile(vp)
    op_ = tmp / "ba_out.bin"

    # kv_len=0：非零退出
    proc = subprocess.run([str(binp), "--run", "--batch", "1", "--kv-len", "0",
                           "--dtype", "fp32", "--q", str(qp), "--k", str(kp),
                           "--v", str(vp), "--out", str(op_)],
                          capture_output=True, text=True)
    out.append({"case": "attention/kv_len_zero", "exit_code": proc.returncode,
                "expect_nonzero_exit": proc.returncode != 0,
                "stderr": proc.stderr.strip()[:200]})

    # 全零 q：softmax 均匀 → 输出 = V 均值
    q0 = np.zeros_like(q)
    q0.tofile(qp)
    proc = subprocess.run([str(binp), "--run", "--batch", "1", "--kv-len", str(n_),
                           "--dtype", "fp32", "--q", str(qp), "--k", str(kp),
                           "--v", str(vp), "--out", str(op_)],
                          capture_output=True, text=True)
    yo = np.fromfile(op_, dtype=np.float32).reshape(b_, hq, d) if op_.exists() else None
    vmean = vv.mean(axis=2)                     # (B, HKV, D)
    vmean_hq = np.repeat(vmean, hq // hkv, axis=1)  # GQA：kv 头 g → q 头 {2g,2g+1}
    out.append({"case": "attention/zero_q_uniform_softmax",
                "exit_code": proc.returncode,
                "close_to_v_mean": bool(yo is not None and
                                        np.abs(yo - vmean_hq).max() <= 1e-4)})

    # 缩放上限探针：N=32768（内部合成 bench 数据，--bench 不读文件）
    proc = subprocess.run([str(binp), "--bench", "--batch", "1", "--kv-len", "32768",
                           "--dtype", "fp16", "--warmup", "0", "--repeats", "1"],
                          capture_output=True, text=True, timeout=120)
    out.append({"case": "attention/N32768_smem_probe", "exit_code": proc.returncode,
                "expect_nonzero_exit": proc.returncode != 0,
                "stderr": proc.stderr.strip()[:300],
                "classification_hint": "退出码非 0 = AI-SCALING 缺陷类证据；"
                                       "stderr 应含 launch 失败信息"})

    # N=63（非整块，run 路径；真值由本地 kernel_sim/参考另算——此处只验退出）
    proc = subprocess.run([str(binp), "--bench", "--batch", "1", "--kv-len", "63",
                           "--dtype", "fp16", "--warmup", "1", "--repeats", "2"],
                          capture_output=True, text=True, timeout=120)
    out.append({"case": "attention/N63_bench_exit", "exit_code": proc.returncode})
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()

    results: dict = {
        "schema_version": "kernel-chapter30.correctness.v1",
        "backend": args.backend,
        "comparison_policy": "ch01 比较器口径（max/mean abs、rel、cos、NaN/Inf、"
                             "误差位置）；容差=人工版合同",
    }
    if args.backend == "cpu":
        correctness_cpu(results)
        out = HERE / "results" / "correctness-results-cpu.json"
    else:
        correctness_cuda(results)
        out = HERE / "results" / "correctness-results-cuda.json"
    write_results_json(out, results)

    if args.backend == "cuda":
        s = results["summary"]
        print(f"[compare] cases {s['cases_pass']}/{s['cases_total']} pass；"
              f"determinism {s['determinism_pass']}/{s['determinism_total']}；"
              f"boundary entries {s['boundary_entries']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
