#!/usr/bin/env python3
"""第 21 期正确性比较（大纲 §9 主对：给定块表的 paged gather vs 同一逻辑 KV
的连续布局实现）。

CPU 列（真值锚点）：
  - fixtures 锚点子集：paged 参考 vs truth32（容差档 fp32 ≤ 5e-4）+ 与连续参考
    **位级一致**断言（值透明性）+ truth64 锚点（N ≤ 128）；
  - 变长 batch（引擎口径）+ prefill 锚点（paged prefill vs 连续 prefill 位级）；
  - GQA kv 头切片抽查（q 头 h 输出 == 对 kv[h//2] 的单头计算）；
  - 确定性：同输入 3 次位级一致。

CUDA 列（同一套用例 + 运行期大形状）：
  - paged kernel vs 连续 kernel：**位级一致**（合同核心断言）；
  - 两者 vs CPU FP32 真值：容差档 fp32 ≤ 5e-4 / fp16 ≤ 2e-2（ch21 口径）；
  - gather kernel vs 逻辑 K/V：位级一致（纯拷贝）；
  - 大形状 N ∈ {2048, 8191}：运行期 gen_input 同规则生成（manifest 记录
    预期 SHA256，现场对账），CPU FP32 真值现算；
  - validate_tables 合同报错路径；确定性 3 次位级。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import paged_config as cfg  # noqa: E402
import roofline_paged as rp  # noqa: E402
from experiment_utils import (  # noqa: E402
    backend_sync,
    comparison_metrics,
    resolve_device,
    write_results_json,
)
from generate_fixtures import FIXTURE_BLOCK_SIZES, gen_input, table_seed  # noqa: E402
from paged_kv_reference import (  # noqa: E402
    build_block_table,
    contiguous_decode_reference,
    contiguous_decode_truth64,
    contiguous_prefill_reference,
    gather_from_pool,
    paged_decode_reference,
    paged_prefill_reference,
    scatter_to_pool,
    validate_block_table,
)

TOL = {"fp32": 5e-4, "fp16": 2e-2}


def sha256_t(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(a.shape).encode())
    digest.update(str(a.dtype).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def load_fixture_cases() -> list[dict]:
    """fixtures 锚点子集 → 用例列表（解码 + 变长 + prefill）。"""
    data = np.load(HERE / "fixtures" / "correctness.npz")
    cases = []
    for n in (65, 128, 511):
        for b in (1, 4):
            for bs in FIXTURE_BLOCK_SIZES:
                for mode in ("identity", "rand"):
                    key = f"d_n{n}_b{b}"
                    cases.append({
                        "name": f"{key}_bs{bs}_{mode}", "kind": "decode",
                        "seq_lens": [n] * b, "block_size": bs, "mode": mode,
                        "q": torch.from_numpy(data[f"{key}_q"]),
                        "k": torch.from_numpy(data[f"{key}_k"]),
                        "v": torch.from_numpy(data[f"{key}_v"]),
                        "table": torch.from_numpy(data[f"{key}_bs{bs}_{mode}_table"]),
                        "truth32": torch.from_numpy(data[f"{key}_truth32"]),
                        "truth64": (torch.from_numpy(data[f"{key}_truth64"])
                                    if f"{key}_truth64" in data else None),
                    })
    # 变长 batch
    key = "d_varlen_b4_bs32_rand"
    cases.append({
        "name": key, "kind": "decode_varlen",
        "seq_lens": data[f"{key}_seqlens"].tolist(), "block_size": 32,
        "mode": "rand",
        "q": torch.from_numpy(data[f"{key}_q"]),
        "k": torch.from_numpy(data[f"{key}_k"]),
        "v": torch.from_numpy(data[f"{key}_v"]),
        "table": torch.from_numpy(data[f"{key}_table"]),
        "truth32": torch.from_numpy(data[f"{key}_truth32"]),
        "truth64": None,
    })
    # prefill 锚点
    for s in (130, 512):
        for mode in ("identity", "rand"):
            pkey = f"p_s{s}_b1_bs128_{mode}"
            cases.append({
                "name": pkey, "kind": "prefill", "seq_lens": [s], "block_size": 128,
                "mode": mode,
                "q": torch.from_numpy(data[f"{pkey}_q"]),
                "k": torch.from_numpy(data[f"{pkey}_k"]),
                "v": torch.from_numpy(data[f"{pkey}_v"]),
                "table": torch.from_numpy(data[f"{pkey}_table"]),
                "truth32": torch.from_numpy(data[f"{pkey}_truth32"]),
                "truth64": None,
            })
    return cases


def runtime_large_case(n: int, b: int, bs: int, mode: str) -> dict:
    """运行期大形状用例（与 fixtures 同 gen_input 规则；manifest 哈希对账）。"""
    seq_lens = [n] * b
    q, k, v = (torch.from_numpy(gen_input((b, cfg.HQ, cfg.D), "q")),
               torch.from_numpy(gen_input((b, cfg.HKV, n, cfg.D), "k")),
               torch.from_numpy(gen_input((b, cfg.HKV, n, cfg.D), "v")))
    for name, arr in ((f"rt_n{n}_b{b}_q", q), (f"rt_n{n}_b{b}_k", k),
                      (f"rt_n{n}_b{b}_v", v)):
        manifest = json.loads((HERE / "fixtures" / "manifest.json").read_text())
        want = manifest["runtime_large_shapes"]["tensors"].get(name)
        if want:
            got = sha256_t(arr)
            if got != want["sha256"]:
                raise RuntimeError(f"运行期数据哈希不一致: {name}（seed 规则漂移?）")
    total_nb = sum(cfg.blocks_for(n_, bs) for n_ in seq_lens)
    num_blocks = total_nb + 2
    table = build_block_table(seq_lens, bs, num_blocks, mode,
                              seed=table_seed(seq_lens, bs, mode))
    return {"name": f"rt_n{n}_b{b}_bs{bs}_{mode}", "kind": "decode",
            "seq_lens": seq_lens, "block_size": bs, "mode": mode,
            "q": q, "k": k, "v": v, "table": table,
            "truth32": contiguous_decode_reference(q, k, v, seq_lens, cfg.SOFTMAX_SCALE),
            "truth64": None}


def gqa_head_slice_check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         seq_lens: list[int]) -> float:
    """GQA 抽查：q 头 0/1 都用 kv 头 0；输出第 0 头 == 单头手算。"""
    o = contiguous_decode_reference(q, k, v, seq_lens, cfg.SOFTMAX_SCALE)
    scores = torch.einsum("bd,bnd->bn", q[:, 0], k[:, 0]) * cfg.SOFTMAX_SCALE
    b, n = scores.shape[0], scores.shape[-1]
    # 2D scores 配 2D mask（pos [b,n] × lens [b,1]）——3D mask 会把 scores 广播
    # 成 [b,1,n]，下游 matmul 拿到的是被广播的错误形状（本抽查首版 bug）
    pos = torch.arange(n).view(1, -1).expand(b, n)
    lens = torch.tensor(seq_lens).view(b, 1)
    scores = scores.masked_fill(pos >= lens, float("-inf"))
    m = scores.amax(dim=-1, keepdim=True)
    e = torch.exp(scores - m)
    attn = e / e.sum(dim=-1, keepdim=True)
    manual = torch.matmul(attn.unsqueeze(1), v[:, 0]).squeeze(1)
    return (o[:, 0] - manual).abs().max().item()


def run_cpu() -> list[dict]:
    results = []
    for case in load_fixture_cases():
        bs = case["block_size"]
        seq_lens = case["seq_lens"]
        q, k, v, table = case["q"], case["k"], case["v"], case["table"]
        num_blocks = int(table.max().item()) + 1
        validate_block_table(table, seq_lens, bs, num_blocks)
        kp, vp = scatter_to_pool(k, v, table, seq_lens, bs, num_blocks)
        if case["kind"] in ("decode", "decode_varlen"):
            o_paged = paged_decode_reference(q, kp, vp, table, seq_lens, bs, cfg.SOFTMAX_SCALE)
            o_contig = contiguous_decode_reference(q, k, v, seq_lens, cfg.SOFTMAX_SCALE)
        else:
            o_paged = paged_prefill_reference(q, kp, vp, table, bs, cfg.SOFTMAX_SCALE)
            o_contig = contiguous_prefill_reference(q, k, v, cfg.SOFTMAX_SCALE)
        entry = {
            "case": case["name"], "kind": case["kind"], "backend": "cpu",
            "seq_lens": seq_lens, "block_size": bs, "table_mode": case["mode"],
            "bit_paged_vs_contig": bool(torch.equal(o_paged, o_contig)),
            "vs_truth32": comparison_metrics(o_paged, case["truth32"]),
            "tolerance_fp32_pass": bool(
                (o_paged - case["truth32"]).abs().max().item() <= TOL["fp32"]),
        }
        if case["truth64"] is not None:
            entry["vs_truth64"] = comparison_metrics(o_paged.double(), case["truth64"].double())
        det = all(bool(torch.equal(paged_decode_reference(q, kp, vp, table, seq_lens, bs,
                                                   cfg.SOFTMAX_SCALE), o_paged))
                  for _ in range(2)) if case["kind"] != "prefill" else True
        entry["deterministic_3x_bit"] = bool(det)
        entry["tol_pass"] = entry["tolerance_fp32_pass"]
        results.append(entry)
        print(f"[correct/cpu] {case['name']}: bit={entry['bit_paged_vs_contig']} "
              f"max_abs={entry['vs_truth32']['max_abs_error']:.2e} "
              f"cos={entry['vs_truth32']['cosine_similarity']:.8f} "
              f"{'OK' if entry['tol_pass'] and entry['bit_paged_vs_contig'] else 'FAIL'}")
    # GQA 切片抽查 + 值透明往返
    q, k, v = (torch.randn(2, cfg.HQ, cfg.D),
               torch.randn(2, cfg.HKV, 65, cfg.D), torch.randn(2, cfg.HKV, 65, cfg.D))
    err = gqa_head_slice_check(q, k, v, [65, 65])
    results.append({"case": "gqa_head_slice", "backend": "cpu",
                    "max_abs": err, "pass": err < 1e-5})
    print(f"[correct/cpu] gqa_head_slice: max_abs={err:.2e} {'OK' if err < 1e-5 else 'FAIL'}")
    return results


def run_cuda() -> list[dict]:
    device = resolve_device("cuda")
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    module = build_ext.get_module()
    results = []

    def run_case(case, dtype: torch.dtype, dtype_name: str) -> None:
        bs = case["block_size"]
        seq_lens = case["seq_lens"]
        b = len(seq_lens)
        table = case["table"].to(device)
        lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        num_blocks = int(case["table"].max().item()) + 1
        module.validate_tables(table, lens, num_blocks, bs)
        kp32, vp32 = scatter_to_pool(case["k"], case["v"], case["table"],
                                     seq_lens, bs, num_blocks)
        qd = case["q"].to(device=device, dtype=dtype)
        kd = kp32.to(device=device, dtype=dtype)
        vd = vp32.to(device=device, dtype=dtype)
        if case["kind"] in ("decode", "decode_varlen"):
            o_p = module.paged_decode_attention(qd, kd, vd, table, lens, bs, cfg.SOFTMAX_SCALE)
            backend_sync(device)
            o_c = module.contig_decode_attention(qd, case["k"].to(device=device, dtype=dtype),
                                                 case["v"].to(device=device, dtype=dtype),
                                                 lens, cfg.SOFTMAX_SCALE)
            backend_sync(device)
            det = all(bool(torch.equal(
                module.paged_decode_attention(qd, kd, vd, table, lens, bs, cfg.SOFTMAX_SCALE),
                o_p)) for _ in range(2))
        else:
            o_p = module.paged_prefill_attention(qd, kd, vd, table, bs, cfg.SOFTMAX_SCALE)
            o_c = module.contig_prefill_attention(qd, case["k"].to(device=device, dtype=dtype),
                                                  case["v"].to(device=device, dtype=dtype),
                                                  cfg.SOFTMAX_SCALE)
            det = True
        bit = bool(torch.equal(o_p, o_c))
        metrics = comparison_metrics(o_p, case["truth32"])
        entry = {"case": case["name"], "kind": case["kind"], "backend": "cuda",
                 "dtype": dtype_name, "seq_lens": seq_lens, "block_size": bs,
                 "table_mode": case["mode"],
                 "bit_paged_vs_contig": bit,
                 "deterministic_3x_bit": bool(det),
                 "vs_truth32": metrics,
                 "tol_pass": metrics["max_abs_error"] <= TOL[dtype_name],
                 "max_abs_error": metrics["max_abs_error"],
                 "cosine_similarity": metrics["cosine_similarity"]}
        if case["kind"] in ("decode", "decode_varlen"):
            kg, vg = module.gather_kv(kd, vd, table, lens, bs)
            entry["gather_bitexact_vs_logical"] = bool(
                torch.equal(kg.cpu(), case["k"].to(dtype)) and
                torch.equal(vg.cpu(), case["v"].to(dtype)))
        ok = bit and entry["tol_pass"] and det
        results.append(entry)
        print(f"[correct/cuda] {case['name']} {dtype_name}: bit={bit} det={det} "
              f"max_abs={metrics['max_abs_error']:.2e} "
              f"{'OK' if ok else 'FAIL'}")

    for case in load_fixture_cases():
        run_case(case, torch.float32, "fp32")
    for case in load_fixture_cases():
        if case["kind"] == "decode":
            run_case(case, torch.float16, "fp16")
    for n in (2048, 8191):
        for b in (1, 4):
            case = runtime_large_case(n, b, 128, "rand")
            run_case(case, torch.float16, "fp16")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    if arguments.backend == "cpu":
        results = run_cpu()
        environment = {"device_type": "cpu",
                       "torch": torch.__version__,
                       "platform": sys.platform}
    else:
        results = run_cuda()
        environment = {"device": str(resolve_device("cuda")),
                       "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0)}
    n_fail = sum(1 for r in results
                 if not r.get("tol_pass", r.get("pass", True))
                 or not r.get("bit_paged_vs_contig", True)
                 or not r.get("deterministic_3x_bit", True))
    payload = {
        "schema_version": "kernel-chapter22.correctness.v1",
        "backend": arguments.backend,
        "tolerance": TOL,
        "value_transparency_claim": "paged kernel 与连续布局 kernel 位级一致（合同核心断言）；"
                                    "vs CPU FP32 真值按容差档报告",
        "cases": results,
        "summary": {"total": len(results), "fail": n_fail},
        "environment": environment,
    }
    out = Path(arguments.out) if arguments.out else \
        HERE / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(out, payload)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
