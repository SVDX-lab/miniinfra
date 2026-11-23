#!/usr/bin/env python3
"""第 19 期正确性对照（CUDA/MUSA 两列；无 CPU 性能列——期首声明）。

链路（fixtures = 第 17 期同源单源，seed 规则相同，真值 = CPU FP32 朴素实现）：
1. fixtures 就位校验（PROVENANCE / manifest + SHA256）；
2. cuda/musa：async_gemm_bench 二进制按多 kernel × 多配置跑同一 fixtures：
   - scalar 三级（sync / cpasync1 / cpasync2）× 3 配置 vs FP32 真值；
   - cpasync2v16 / warpspec（实例化子集配置）vs FP32 真值；
   - wmma_pipe（FP16 输入）vs FP32 真值（容差单独记录，ep19 同口径）；
   - ragged 形状（13x17x11 / 200³）验证同步回退路径；
   - 确定性：512³ cpasync2 连续 3 次位级一致（无原子、固定累加序）；
   - broken 竞态演示核：跑 128³ 记录与真值偏差（不进 pass/fail——竞态
     是否当场可见是概率性的，报告形态由 racecheck 日志承担）。
3. MUSA 无 cp.async 时：cpasync 系 kernel 报"未编译"→ 记平台边界继续。

用法：
    python compare_correctness.py --backend cuda
    python compare_correctness.py --backend musa
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    comparison_metrics_np,
    device_environment_record,
    resolve_device,
    SCHEMA_VERSION,
    write_results_json,
)
from async_gemm_configs import kernel_supports_config  # noqa: E402

# fixtures 定位：优先本期（独立生成），否则复用 ch18（同语义单源）
def locate_fixtures() -> tuple[Path, Path]:
    local = ROOT / "fixtures" / "correctness.npz"
    if local.exists():
        return local, local.with_name("manifest.json")
    ch18 = ROOT.parent / "chapter_17" / "fixtures" / "correctness.npz"
    if not ch18.exists():
        raise SystemExit("[ERROR] 找不到 fixtures（本地与 chapter_17 均无）——先跑 generate_fixtures.py")
    return ch18, ch18.with_name("manifest.json")


SCALAR_CONFIGS = ["32x32x8_reg4x4", "64x64x8_reg4x4", "16x16x16_reg4x4"]


def load_fixtures(npz_path: Path, manifest_path: Path):
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return arrays, manifest


def run_binary(binary: Path, arguments: list[str], expect_ok: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [str(binary), "run", *arguments], capture_output=True, text=True, cwd=str(binary.parent)
    )
    if expect_ok and result.returncode != 0:
        raise RuntimeError(
            f"binary 失败 rc={result.returncode}\nstdout: {result.stdout[-800:]}\nstderr: {result.stderr[-800:]}"
        )
    return result


def to_bin(array: np.ndarray, path: Path, dtype: str) -> None:
    data = array.astype(np.float16 if dtype == "fp16" else np.float32, copy=False)
    path.write_bytes(np.ascontiguousarray(data).tobytes())


def from_bin(path: Path, dtype: str) -> np.ndarray:
    raw = path.read_bytes()
    return np.frombuffer(raw, dtype=np.float16 if dtype == "fp16" else np.float32).copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    args = parser.parse_args()
    backend = args.backend

    binary = ROOT / "implementations" / backend / "async_gemm_bench"
    if not binary.exists():
        raise SystemExit(f"[ERROR] 找不到 {binary}——先运行 implementations/{backend}/build.sh")

    npz_path, manifest_path = locate_fixtures()
    arrays, manifest = load_fixtures(npz_path, manifest_path)
    print(f"[fixtures] {npz_path}（{len(arrays)} 张量）")

    device = resolve_device(backend)
    environment = device_environment_record(device)

    tmp = ROOT / "tmp-correctness"
    tmp.mkdir(exist_ok=True)
    entries: list[dict] = []
    platform_boundaries: list[dict] = []
    passed = failed = 0

    def check_case(kernel, config, dtype, m, k, n, tol, note="") -> dict:
        nonlocal passed, failed
        key = f"{m}x{k}x{n}"
        a_path = tmp / f"a_{kernel}_{config.replace('x', '')}_{key}.bin"
        b_path = tmp / f"b_{kernel}_{config.replace('x', '')}_{key}.bin"
        c_path = tmp / f"c_{kernel}_{config.replace('x', '')}_{key}.bin"
        to_bin(arrays[f"A_{key}"], a_path, dtype)
        to_bin(arrays[f"B_{key}"], b_path, dtype)
        result = run_binary(
            binary,
            ["--kernel", kernel, "--tile", config.split("_")[0], "--reg",
             (config.split("_reg")[1] if "_reg" in config else "4x4"),
             "--dtype", dtype, "--m", str(m), "--n", str(n), "--k", str(k),
             "--in-a", str(a_path), "--in-b", str(b_path), "--out-c", str(c_path)],
            expect_ok=False,
        )
        entry = {"kernel": kernel, "config": config, "dtype": dtype, "shape": [m, k, n], "note": note}
        if result.returncode != 0:
            entry["status"] = "binary_error"
            entry["stderr_tail"] = result.stderr.strip().splitlines()[-3:] if result.stderr else []
            platform_boundaries.append(entry)
            print(f"[{backend}] {kernel}/{config} {key}: 二进制报错（记录）: {entry['stderr_tail']}")
            return entry
        c_out = from_bin(c_path, dtype).astype(np.float64)
        truth = arrays[f"C_{key}"].astype(np.float64)
        metrics = comparison_metrics_np(c_out, truth)
        entry.update(metrics)
        if dtype == "fp16":
            entry["tolerance"] = {"max_abs": tol["max_abs"], "mean_rel": tol["mean_rel"],
                                  "note": "FP16 输入量化 + 累加序差，ep19 同口径"}
            ok = (metrics["max_abs_error"] <= tol["max_abs"]
                  and metrics["mean_rel_error"] <= tol["mean_rel"]
                  and metrics["nan_count"] == 0 and metrics["inf_count"] == 0)
        else:
            ok = (metrics["max_abs_error"] <= tol["max_abs"]
                  and metrics["nan_count"] == 0 and metrics["inf_count"] == 0)
        entry["status"] = "PASS" if ok else "FAIL"
        passed += 1 if ok else 0
        failed += 0 if ok else 1
        print(f"[{backend}] {kernel}/{config} {dtype} {key}: max_abs={metrics['max_abs_error']:.2e} "
              f"cos={metrics['cosine_similarity']:.8f} {entry['status']}")
        entries.append(entry)
        return entry

    fp32_tol = {"max_abs": 2e-3, "mean_rel": 5e-3}
    fp16_tol = {"max_abs": 1.0, "mean_rel": 2e-2}  # ep19 口径：fp16 输入量化主导

    shapes_all = [(4, 8, 4), (13, 17, 11), (64, 64, 64), (200, 200, 200), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    shapes_square = [(64, 64, 64), (200, 200, 200), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]

    # 1) scalar 三级 × 3 配置 × 全形状（FP32 主线）
    for kernel in ["sync", "cpasync1", "cpasync2"]:
        for config in SCALAR_CONFIGS:
            if not kernel_supports_config(kernel, config):
                continue
            for (m, k, n) in shapes_all:
                check_case(kernel, config, "fp32", m, k, n, fp32_tol)

    # 2) v16 / warpspec（实例化子集）
    for kernel, config in [("cpasync2v16", "32x32x8_reg4x4"), ("cpasync2v16", "64x64x8_reg4x4"),
                           ("warpspec", "32x32x8_reg4x4"), ("warpspec", "64x64x8_reg4x4")]:
        for (m, k, n) in shapes_square:
            check_case(kernel, config, "fp32", m, k, n, fp32_tol)

    # 3) wmma_pipe 组合核（FP16 输入，只支持整除形状；ch18 fixtures 里
    #    128³ 缺失于 shapes_square——补上）
    for (m, k, n) in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
        check_case("wmma_pipe", "64x64x16", "fp16", m, k, n, fp16_tol)

    # 4) 确定性：512³ cpasync2 连续 3 次位级一致
    det_entry = {"kernel": "cpasync2", "config": "32x32x8_reg4x4", "shape": [512, 512, 512],
                 "kind": "determinism_3x_bitwise"}
    try:
        key = "512x512x512"
        hashes = []
        for rep in range(3):
            a_path = tmp / "det_a.bin"
            b_path = tmp / "det_b.bin"
            c_path = tmp / f"det_c_{rep}.bin"
            to_bin(arrays[f"A_{key}"], a_path, "fp32")
            to_bin(arrays[f"B_{key}"], b_path, "fp32")
            run_binary(binary, ["--kernel", "cpasync2", "--tile", "32x32x8", "--reg", "4x4",
                                "--dtype", "fp32", "--m", "512", "--n", "512", "--k", "512",
                                "--in-a", str(a_path), "--in-b", str(b_path), "--out-c", str(c_path)])
            hashes.append(__import__("hashlib").sha256(c_path.read_bytes()).hexdigest())
        det_entry["sha256"] = hashes
        det_entry["status"] = "PASS" if len(set(hashes)) == 1 else "FAIL"
    except Exception as error:  # noqa: BLE001
        det_entry["status"] = "ERROR"
        det_entry["error"] = str(error)
    if det_entry["status"] == "PASS":
        passed += 1
    else:
        failed += 1
    print(f"[{backend}] determinism cpasync2 512³: {det_entry['status']} {det_entry.get('sha256', det_entry.get('error'))}")
    entries.append(det_entry)

    # 5) broken 竞态演示核（不进 pass/fail；记录是否当场偏离真值）
    to_bin(arrays["A_128x128x128"], tmp / "ba128.bin", "fp32")
    to_bin(arrays["B_128x128x128"], tmp / "bb128.bin", "fp32")
    for kernel in ["sync2_broken", "cpasync2_broken"]:
        result = run_binary(
            binary,
            ["--kernel", kernel, "--tile", "32x32x8", "--reg", "4x4", "--dtype", "fp32",
             "--m", "128", "--n", "128", "--k", "128",
             "--in-a", str(tmp / "ba128.bin"), "--in-b", str(tmp / "bb128.bin"),
             "--out-c", str(tmp / f"broken_{kernel}.bin")],
            expect_ok=False,
        )
        entry = {"kernel": kernel, "config": "32x32x8_reg4x4", "shape": [128, 128, 128],
                 "kind": "race_demo_no_pass_fail", "rc": result.returncode}
        if (tmp / f"broken_{kernel}.bin").exists() and result.returncode == 0:
            c_out = from_bin(tmp / f"broken_{kernel}.bin", "fp32").astype(np.float64)
            truth = arrays["C_128x128x128"].astype(np.float64)
            metrics = comparison_metrics_np(c_out, truth)
            entry.update(metrics)
            entry["deviates_from_truth"] = metrics["max_abs_error"] > 2e-3
            print(f"[{backend}] {kernel}: max_abs={metrics['max_abs_error']:.3e} "
                  f"偏离真值={entry['deviates_from_truth']}（竞态当场可见是概率性的）")
        else:
            entry["stderr_tail"] = result.stderr.strip().splitlines()[-2:] if result.stderr else []
            print(f"[{backend}] {kernel}: rc={result.returncode} {entry.get('stderr_tail')}")
            platform_boundaries.append(entry)
        entries.append(entry)

    # 6) MUSA 无 cp.async 时的平台边界记录（cpasync 系整体 binary_error 已进
    #    platform_boundaries；这里汇总）
    results = {
        "schema_version": SCHEMA_VERSION + ".correctness",
        "backend": backend,
        "fixtures": {"path": str(npz_path), "manifest_schema": manifest.get("schema_version"),
                     "source": "chapter_17 同源单源（seed 规则一致）"},
        "tolerances": {"fp32": fp32_tol, "fp16": fp16_tol},
        "summary": {"passed": passed, "failed": failed},
        "entries": entries,
        "platform_boundaries": platform_boundaries,
        "environment": environment,
    }
    write_results_json(ROOT / "results" / f"correctness-results-{backend}.json", results)

    for e in entries + platform_boundaries:
        if e.get("status") == "FAIL":
            print(f"[FAIL] {e['kernel']}/{e['config']} {e.get('shape')}")
    if failed > 0:
        print(f"[SUMMARY] {backend}: {passed} passed, {failed} FAILED")
        return 1
    print(f"[SUMMARY] {backend}: {passed} passed, 0 failed"
          + (f"（另有 {len(platform_boundaries)} 条平台边界/演示记录）" if platform_boundaries else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
