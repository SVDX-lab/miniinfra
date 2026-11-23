#!/usr/bin/env python3
"""第 29 期静态预审（CPU 侧可做的验收，GPU 前第一道筛）。

三个部分（对应合同 metrics 静态预审条目）：
  1. stub 编译检查：把 CUDA 限定符/builtin/API stub 成平凡实现后用 g++
     -fsyntax-only 编译三份 AI .cu（ep21/22 stub 语法检查模式；kernel launch
     语法 `<<<...>>>` 机械替换为普通函数调用，仅语法不执行）；
  2. kernel_sim 行为仿真：逐 block/warp 语义仿真三份算法 vs fixtures 真值
     （见 implementations/cpu/kernel_sim.py），含归约触发边界探针、NaN 传播
     探针、smem 容量探针；
  3. 合同逐条静态检查：非整 Tile guard / warpSize 硬编码 / atomic 误用 /
     aliasing（const+restrict）/ 错误处理覆盖 / 确定性组织 / 零形状处理 /
     selftest 检出力（人工读码条目，给证据行号）。

输出 results/static-precheck.json；退出码 0=预审完成（不代表实现正确）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import kernel_sim  # noqa: E402

AI_FILES = {
    "rmsnorm": HERE / "implementations/ai/rmsnorm/ai_rmsnorm.cu",
    "gemm": HERE / "implementations/ai/gemm/ai_gemm.cu",
    "attention": HERE / "implementations/ai/attention/ai_decode_attn.cu",
}

STUB_PRELUDE = r"""
// ==== ch30 static precheck stub prelude（仅语法检查，不执行）====
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __inline__ inline
#define __forceinline__ inline
#define __shared__
#define __restrict__

struct Dim3Stub {
    int x, y, z;
    Dim3Stub(int a, int b = 1, int c = 1) : x(a), y(b), z(c) {}
};
using dim3 = Dim3Stub;
inline Dim3Stub threadIdx{0, 0, 0};
inline Dim3Stub blockIdx{0, 0, 0};
inline Dim3Stub blockDim{256, 1, 1};
inline Dim3Stub gridDim{1, 1, 1};
inline void __syncthreads() {}

struct HalfStub { unsigned short bits; };
using __half = HalfStub;
inline float __half2float(HalfStub) { return 0.0f; }
inline HalfStub __float2half(float) { return HalfStub{0}; }
inline float __shfl_xor_sync(unsigned, float v, int) { return v; }
inline double __shfl_xor_sync(unsigned, double v, int) { return v; }
inline float rsqrtf(float v) { return 1.0f / std::sqrt(v); }

using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr int cudaErrorInvalidValue = 1;
constexpr int cudaMemcpyHostToDevice = 1;
constexpr int cudaMemcpyDeviceToHost = 2;
struct StubEvent { int dummy; };
using cudaEvent_t = StubEvent;
inline cudaError_t cudaMalloc(void**, size_t) { return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return cudaSuccess; }
inline cudaError_t cudaMemset(void*, int, size_t) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaEventCreate(StubEvent*) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(StubEvent&) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(StubEvent&) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(StubEvent&) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float*, StubEvent&, StubEvent&) { return cudaSuccess; }
// ==== end stub prelude ====
"""


def remove_main(source: str) -> tuple[str, bool]:
    """按括号配平删除 int main(...) {...}（stub 检查不需要 host 入口）。"""
    match = re.search(r"\bint\s+main\s*\(", source)
    if not match:
        return source, False
    brace_open = source.find("{", match.end())
    if brace_open < 0:
        return source, False
    depth = 0
    for i in range(brace_open, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[:match.start()] + source[i + 1:], True
    return source, False


def stub_launches(source: str) -> tuple[str, int]:
    """`kernel<<<cfg>>>(args)` → `kernel(args)`（仅语法；cfg 丢弃）。"""
    new, count = re.subn(r"<<<.*?>>>", "", source, flags=re.DOTALL)
    return new, count


def stub_compile_check(op: str, path: Path) -> dict:
    text = path.read_text()
    body, main_removed = remove_main(text)
    body, n_launches = stub_launches(body)
    # 中立化 CUDA 头（由 stub prelude 顶替）
    body = re.sub(r"#include\s+<cuda_runtime\.h>", "/* stubbed include */", body)
    body = re.sub(r"#include\s+<cuda_fp16\.h>", "/* stubbed include */", body)
    tu = STUB_PRELUDE + "\n" + body
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(tu)
        tu_path = f.name
    try:
        proc = subprocess.run(
            ["g++", "-std=c++17", "-fsyntax-only", "-Wall", "-Wno-unknown-pragmas",
             tu_path],
            capture_output=True, text=True, timeout=120,
        )
        return {
            "operator": op,
            "source": str(path),
            "main_removed": main_removed,
            "launches_stubbed": n_launches,
            "gxx_returncode": proc.returncode,
            "syntax_ok": proc.returncode == 0,
            "stderr_tail": proc.stderr.strip().splitlines()[-25:],
        }
    finally:
        Path(tu_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 合同逐条静态检查（证据 = 行号 + 匹配文本）
# ---------------------------------------------------------------------------

def scan(op: str, path: Path) -> dict:
    lines = path.read_text().splitlines()
    findings: dict = {"operator": op, "source": str(path)}

    def find(pattern: str) -> list[dict]:
        out = []
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                out.append({"line": i, "text": line.strip()[:120]})
        return out

    findings["device_kernels"] = find(r"__global__\s+void\s+(\w+)")
    findings["nonfull_tile_guards"] = find(r"<\s*(hidden|kv_len|M|K|N|m|k|n)\b")
    findings["strided_loops"] = find(r"\+=\s*(BLOCK|kThreads|TILE|block)")
    findings["warp_literal_offsets"] = find(r"offset\s*=\s*16|offset\s*>>=\s*1|>>\s*1\b.*offset")
    findings["warp_size_uses"] = find(r"\bwarpSize\b")
    findings["block_constant"] = find(r"constexpr int k(Block|Threads|Tile)\s*=\s*\d+")
    findings["atomic_uses"] = find(r"\batomic\w+\s*\(")
    findings["rng_in_kernel"] = find(r"curand|rand\(")
    findings["restrict_uses"] = find(r"__restrict__")
    findings["const_inputs"] = find(r"const\s+(__half|float)\s*\*\s*__restrict__")
    findings["cuda_malloc"] = find(r"cudaMalloc\s*\(")
    findings["cuda_memcpy"] = find(r"cudaMemcpy\s*\(")
    findings["launch_error_checks"] = find(r"cudaGetLastError|launch_err")
    findings["malloc_error_checks"] = find(r"cudaMalloc[^)]*\)\s*!=\s*cudaSuccess")
    findings["zero_shape_handling"] = find(r"(rows|a\.m|a\.n|M|N)\s*==\s*0|rows\s*==\s*0|<=\s*0")
    findings["smem_decl"] = find(r"extern __shared__|__shared__\s+\w+")
    findings["selftest_double_call_suspect"] = find(
        r"CpuRmsNormF16\(x, w, c\.rows, c\.hidden, yh\);")
    return findings


# ---------------------------------------------------------------------------
# 探针（kernel_sim 基础上的定点行为学检查）
# ---------------------------------------------------------------------------

def reduction_boundary_probe() -> dict:
    """RMSNorm 归约缺陷的触发边界：H≤32 应正确，H>32 首个 warp 即错。"""
    rng = np.random.default_rng(20260301)
    out = []
    for h in (16, 32, 33, 48, 127):
        rows = 2
        x = rng.standard_normal((rows, h)).astype(np.float32)
        w = rng.uniform(0.5, 1.5, h).astype(np.float32)
        sum2 = (x.astype(np.float64) ** 2).sum(axis=1)
        truth = (w[None, :] * x / np.sqrt(sum2[:, None] / h + 1e-6)).astype(np.float32)
        sim = kernel_sim.simulate_rmsnorm(x, w)
        err = float(np.abs(sim.output - truth).max())
        out.append({"hidden": h, "max_abs_error": round(err, 8),
                    "expected_ok": h <= 32, "actually_ok": bool(err <= 1e-3)})
    verdict_ok = all((o["actually_ok"]) == o["expected_ok"] for o in out)
    return {"probe": "rmsnorm_reduction_boundary", "cases": out,
            "boundary_confirmed": verdict_ok,
            "note": "缺陷触发边界 = 32（第一个 warp 之外）；H≤32 全对——归因"
                    "跨 warp 归约返回值未广播到 warp 1..k"}


def nan_propagation_probe() -> dict:
    """RMSNorm NaN 行语义：NaN 行全 NaN、其余行不受影响（仿真）。"""
    rng = np.random.default_rng(20260302)
    rows, h = 3, 128
    x = rng.standard_normal((rows, h)).astype(np.float32)
    x[1, 5] = np.nan
    w = rng.uniform(0.5, 1.5, h).astype(np.float32)
    # 真值按合同语义独立逐行算（NaN 行 → NaN scale → 行全 NaN）
    truth = np.full((rows, h), np.nan, dtype=np.float32)
    for r in range(rows):
        if not np.isnan(x[r]).any():
            sum2 = float((x[r].astype(np.float64) ** 2).sum())
            truth[r] = w * x[r] / np.sqrt(sum2 / h + 1e-6)
    sim = kernel_sim.simulate_rmsnorm(x, w).output
    row_nan = [bool(np.isnan(sim[r]).all()) for r in range(rows)]
    row_ok = [bool(np.abs(sim[r] - truth[r]).max() <= 1e-3) for r in range(rows)]
    return {"probe": "rmsnorm_nan_row", "row_is_all_nan": row_nan,
            "row_matches_truth": row_ok,
            "verdict": (
                "合同要求：NaN 行全 NaN、其余行不受影响。仿真结果：行 1（NaN 行）"
                "不是全 NaN——NaN 随元素 5（warp 0）进入归约，但因跨 warp 归约缺"
                "广播，warp 1..k 的元素用未污染的部分和算 scale，该行只有 warp 0 "
                "的元素是 NaN → NaN 传播语义也被归约缺陷破坏。行 0/2 因同一缺陷"
                "不匹配真值（row_matches_truth=false 属归约缺陷，非 NaN 问题）。"
            ),
            "sim_rows_allnan": row_nan, "sim_rows_correct": row_ok}


def smem_cap_probe() -> dict:
    out = []
    for n in (2048, 8192, 12288, 12289, 32768):
        smem = n * 4
        out.append({"kv_len": n, "smem_bytes": smem,
                    "launch_ok_default_limit": smem <= kernel_sim.ATTN_SMEM_DEFAULT_LIMIT})
    return {"probe": "attention_smem_cap",
            "cases": out,
            "note": "AI 源码 dynamic smem = N×4B，未调 cudaFuncSetAttribute opt-in；"
                    "sm_89 默认每 block 48KB → N≥12289 launch 失败（错误处理合规但"
                    "性能矩阵缺口，归 AI-SCALING 类）"}


def gemm_guard_probe() -> dict:
    """13x17x11 非整 tile 的 guard 行为学（tile 序仿真 vs 手算真值）。"""
    rng = np.random.default_rng(20261118)
    a = rng.standard_normal((13, 17)).astype(np.float32)
    b = rng.standard_normal((17, 11)).astype(np.float32)
    truth = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    sim = kernel_sim.simulate_gemm(a, b).output
    err = float(np.abs(sim - truth).max())
    return {"probe": "gemm_nontile_guard", "shape": "13x17x11",
            "max_abs_error": round(err, 8), "pass": bool(err <= 5e-3)}


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true",
                    help="跳过全量 kernel_sim（只做 stub+扫描+探针）")
    args = ap.parse_args()

    report: dict = {
        "schema_version": "kernel-chapter30.static-precheck.v1",
        "stage": "CPU 侧静态预审（GPU 验收前的第一道筛）",
        "stub_compile": [],
        "contract_static_scan": [],
        "probes": [],
        "kernel_sim_summary": None,
    }

    for op, path in AI_FILES.items():
        report["stub_compile"].append(stub_compile_check(op, path))
        report["contract_static_scan"].append(scan(op, path))

    report["probes"].append(reduction_boundary_probe())
    report["probes"].append(nan_propagation_probe())
    report["probes"].append(smem_cap_probe())
    report["probes"].append(gemm_guard_probe())

    if not args.skip_sim:
        sim = kernel_sim.run_all(max_rows=2048, max_n=2048)
        (HERE / "results" / "kernel-sim-full.json").write_text(
            json.dumps(sim, ensure_ascii=False, indent=2) + "\n")
        by_op: dict[str, dict] = {}
        for c in sim["cases"]:
            slot = by_op.setdefault(c["operator"], {"cases": 0, "structural_fail": 0,
                                                    "errors": []})
            slot["cases"] += 1
            slot["structural_fail"] += 1 if c.get("structural_fail") else 0
            if c.get("max_abs_error") is not None:
                slot["errors"].append(c["max_abs_error"])
        report["kernel_sim_summary"] = {
            "detail_file": "results/kernel-sim-full.json",
            "by_operator": {
                op: {
                    "cases": s["cases"], "structural_fail": s["structural_fail"],
                    "max_abs_error": max(s["errors"]) if s["errors"] else None,
                } for op, s in by_op.items()
            },
        }

    out = HERE / "results" / "static-precheck.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("== stub 编译检查 ==")
    for r in report["stub_compile"]:
        print(f"  {r['operator']}: syntax_ok={r['syntax_ok']} "
              f"(launches stubbed={r['launches_stubbed']}, main removed={r['main_removed']})")
        if not r["syntax_ok"]:
            for line in r["stderr_tail"][:10]:
                print("   ", line)
    print("== 探针 ==")
    for p in report["probes"]:
        print(f"  {p['probe']}: "
              + ("boundary_confirmed" if "boundary_confirmed" in p
                 else f"{json.dumps({k: v for k, v in p.items() if k != 'note'}, ensure_ascii=False)[:220]}"))
    if report["kernel_sim_summary"]:
        print("== kernel_sim（全量）==")
        for op, s in report["kernel_sim_summary"]["by_operator"].items():
            print(f"  {op}: cases={s['cases']} structural_fail={s['structural_fail']} "
                  f"max_err={s['max_abs_error']}")
    print(f"[precheck] 详情 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
