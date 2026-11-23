#!/usr/bin/env python3
"""第 21 期统一计时口径基准（实验 3：paged vs 连续布局 kernel 性能对照）。

测量对象（decode 形态为主战场——ep21 结论：decode memory-dominated，
AI≈2 恒定，带宽是唯一变量）：
  A contig kernel（同核去块表化控制组）
  B paged kernel + 顺序块表（恒等；+表查找/整除取模代码路径）
  C paged kernel + 随机块表（+局部性损失；行粒度 = bs×D×itemsize）
  D gather-only kernel（纯 gather 归因层：块表查找+取数的访存代价）
  E prefill paged vs contig（可选对照项）
扫描：N ∈ {512, 2048, 8192} × B ∈ {1, 4, 16} × fp16（fp32 抽 N=2048）。
  B=16 → B·HQ=256 blocks > 128 SM（4090），占用率层退出瓶颈——归因分层第三层。

归因分层（任务书）：
  层1 占用率：B=1（16 blocks，ep21 的 4-22% 带宽瓶颈）vs B=16；
  层2 间接寻址代码路径：A vs B（恒等表——数据布局相同，仅多表查找）；
  层3 局部性：B vs C（随机表——数据被打散到 bs×D 粒度）；
  层4 纯 gather 代价：D 的有效带宽 vs A/C。

口径：ch01 OperatorTimer（事件对逐样本、末尾一次同步，P50/P95）；
理论字节 = roofline_paged 单源（unique 连续口径 + 块表字节）；
实测有效带宽 = bytes_unique/P50；带宽达成率 = /918.8（4090 实测拷贝峰）；
nsys 互验由 run_nsys.sh 承担（双口径）。
"""

from __future__ import annotations

import argparse
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
    OperatorTimer,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from paged_kv_reference import build_block_table, scatter_to_pool  # noqa: E402

SEED_BASE = 20261122
TAG_MAP = {"q": 0, "k": 1, "v": 2}


def gen_input(shape: tuple[int, ...], tag: str) -> np.ndarray:
    """与 generate_fixtures 同规则（单源）。"""
    seed = SEED_BASE + TAG_MAP[tag] * 1000003 + sum(shape) * 7 + len(shape)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def bench_plan(n: int, b: int) -> tuple[int, int]:
    if n <= 512:
        return 3, 30
    if n <= 2048:
        return 2, 20
    return 2, 10 if b <= 4 else 5


def make_paged_data(device: torch.device, dtype: torch.dtype, b: int, n: int,
                    bs: int, mode: str):
    """构造池 + 块表（bench 前一次性构造；计时循环不含构造）。"""
    n_max = n
    q = torch.from_numpy(gen_input((b, cfg.HQ, cfg.D), "q")).to(device=device, dtype=dtype)
    k = torch.from_numpy(gen_input((b, cfg.HKV, n_max, cfg.D), "k")).to(device=device, dtype=dtype)
    v = torch.from_numpy(gen_input((b, cfg.HKV, n_max, cfg.D), "v")).to(device=device, dtype=dtype)
    total_nb = b * cfg.blocks_for(n, bs)
    num_blocks = total_nb + 2
    table = build_block_table([n] * b, bs, num_blocks, mode,
                              seed=SEED_BASE + n + b * 13 + (7 if mode == "rand" else 0))
    kp, vp = scatter_to_pool(k.cpu(), v.cpu(), table, [n] * b, bs, num_blocks)
    kp = kp.to(device=device, dtype=dtype)
    vp = vp.to(device=device, dtype=dtype)
    table_d = table.to(device)
    lens = torch.tensor([n] * b, dtype=torch.int32, device=device)
    return q, k, v, kp, vp, table_d, lens, num_blocks


def attach_theory(entry: dict, platform: str) -> dict:
    prof = rp.decode_profile(entry["batch"], entry["kv_len"], entry["dtype"],
                             platform, entry["block_size"], entry["table_mode"])
    if entry.get("p50_ms", 0) > 0:
        t_s = entry["p50_ms"] / 1000.0
        prof["achieved_gbs_unique_contig"] = round(prof["bytes_unique_contig"] / t_s / 1e9, 1)
        prof["achieved_gbs_unique_paged"] = round(prof["bytes_unique_paged"] / t_s / 1e9, 1)
        prof["achieved_tflops"] = round(prof["flops"] / t_s / 1e12, 3)
        copy_peak = cfg.PLATFORM_PEAKS[platform]["copy_gb_s_measured"]
        prof["gbs_vs_copy_peak_pct"] = round(prof["achieved_gbs_unique_contig"] / copy_peak * 100.0, 1)
    entry["theory"] = prof
    return entry


def bench_cuda(platform: str, decode_lens=None, batches=None,
               dtypes=None, variants=None) -> dict:
    device = resolve_device("cuda")
    cuda_dir = HERE / "implementations" / "cuda"
    sys.path.insert(0, str(cuda_dir))
    import build_ext  # noqa: PLC0415
    module = build_ext.get_module()
    module.validate_tables(torch.zeros(1, 1, dtype=torch.int32),
                           torch.ones(1, dtype=torch.int32), 1, 16)  # 冒烟预检
    decode_lens = decode_lens or cfg.BENCH_DECODE_KV_LENS
    batches = batches or cfg.BENCH_BATCHES
    dtypes = dtypes or [("fp16", torch.float16)]
    variants = variants or ["contig", "paged_seq", "paged_rand", "gather", "prefill"]
    results: dict = {"backend": "cuda", "entries": [],
                     "warp_device": int(module.warp_size_device()),
                     "denominators": {"copy_gb_s": cfg.PLATFORM_PEAKS[platform]["copy_gb_s_measured"],
                                      "fp16_tf": cfg.PLATFORM_PEAKS[platform]["fp16_gemm_tf_measured"]}}
    bs = cfg.DEFAULT_BLOCK_SIZE
    for dtype_name, dtype in dtypes:
        for n in decode_lens:
            for b in batches:
                warm, rep = bench_plan(n, b)
                # 预建数据（恒等表 + 随机表各一套）
                q, k, v, kp_seq, vp_seq, tab_seq, lens, nb = make_paged_data(
                    device, dtype, b, n, bs, "identity")
                _, _, _, kp_rand, vp_rand, tab_rand, _, _ = make_paged_data(
                    device, dtype, b, n, bs, "rand")
                if "contig" in variants:
                    e = {"variant": "contig_kernel", "batch": b, "kv_len": n,
                         "dtype": dtype_name, "block_size": bs, "table_mode": "none",
                         "blocks_launched": b * cfg.HQ}
                    e.update(OperatorTimer(device, warm, rep).measure(
                        lambda: module.contig_decode_attention(q, k, v, lens, cfg.SOFTMAX_SCALE)))
                    attach_theory(e, platform)
                    results["entries"].append(e)
                    print(f"[bench/cuda] contig {dtype_name} N={n} B={b}: "
                          f"p50={e['p50_ms']}ms bw%={e['theory']['gbs_vs_copy_peak_pct']}")
                if "paged_seq" in variants:
                    e = {"variant": "paged_kernel", "batch": b, "kv_len": n,
                         "dtype": dtype_name, "block_size": bs, "table_mode": "seq",
                         "blocks_launched": b * cfg.HQ}
                    e.update(OperatorTimer(device, warm, rep).measure(
                        lambda: module.paged_decode_attention(q, kp_seq, vp_seq, tab_seq,
                                                              lens, bs, cfg.SOFTMAX_SCALE)))
                    attach_theory(e, platform)
                    results["entries"].append(e)
                    print(f"[bench/cuda] paged-seq {dtype_name} N={n} B={b}: "
                          f"p50={e['p50_ms']}ms bw%={e['theory']['gbs_vs_copy_peak_pct']}")
                if "paged_rand" in variants:
                    e = {"variant": "paged_kernel", "batch": b, "kv_len": n,
                         "dtype": dtype_name, "block_size": bs, "table_mode": "rand",
                         "blocks_launched": b * cfg.HQ}
                    e.update(OperatorTimer(device, warm, rep).measure(
                        lambda: module.paged_decode_attention(q, kp_rand, vp_rand, tab_rand,
                                                              lens, bs, cfg.SOFTMAX_SCALE)))
                    attach_theory(e, platform)
                    results["entries"].append(e)
                    print(f"[bench/cuda] paged-rand {dtype_name} N={n} B={b}: "
                          f"p50={e['p50_ms']}ms bw%={e['theory']['gbs_vs_copy_peak_pct']}")
                if "gather" in variants and dtype_name == "fp16" and b == 1:
                    e = {"variant": "gather_only", "batch": b, "kv_len": n,
                         "dtype": dtype_name, "block_size": bs, "table_mode": "seq"}
                    e.update(OperatorTimer(device, warm, rep).measure(
                        lambda: module.gather_kv(kp_seq, vp_seq, tab_seq, lens, bs, n)))
                    gather_bytes = 2 * 2 * b * cfg.HKV * n * cfg.D * (2 if dtype_name == "fp16" else 4)
                    e["theory"] = {"bytes_moved": gather_bytes,
                                   "achieved_gbs": round(gather_bytes / (e["p50_ms"] / 1000.0) / 1e9, 1)}
                    results["entries"].append(e)
                    print(f"[bench/cuda] gather {dtype_name} N={n} B={b}: "
                          f"p50={e['p50_ms']}ms bw={e['theory']['achieved_gbs']}GB/s")
                del q, k, v, kp_seq, vp_seq, tab_seq, kp_rand, vp_rand, tab_rand
                torch.cuda.empty_cache()
        # fp32 只扫 N=2048（dtype 抽查档）
        if dtype_name == "fp32":
            continue
    # prefill 对照（fp16 B=1）
    if "prefill" in variants:
        for s in cfg.BENCH_PREFILL_SEQ_LENS:
            b = 1
            q = torch.from_numpy(gen_input((b, cfg.HQ, s, cfg.D), "q")).to(
                device=device, dtype=torch.float16)
            k = torch.from_numpy(gen_input((b, cfg.HKV, s, cfg.D), "k"))
            v = torch.from_numpy(gen_input((b, cfg.HKV, s, cfg.D), "v"))
            bs2 = 128
            total_nb = b * cfg.blocks_for(s, bs2)
            table = build_block_table([s] * b, bs2, total_nb, "rand",
                                      seed=SEED_BASE + s + 5)
            kp, vp = scatter_to_pool(k, v, table, [s] * b, bs2, total_nb)
            kpd, vpd = kp.to(device=device, dtype=torch.float16), vp.to(device=device, dtype=torch.float16)
            kd, vd = k.to(device=device, dtype=torch.float16), v.to(device=device, dtype=torch.float16)
            qd = q
            warm, rep = bench_plan(s, 1)
            e = {"variant": "contig_prefill", "batch": b, "seq": s, "dtype": "fp16",
                 "block_size": bs2}
            e.update(OperatorTimer(device, warm, rep).measure(
                lambda: module.contig_prefill_attention(qd, kd, vd, cfg.SOFTMAX_SCALE)))
            prof = rp.prefill_bytes_unique(b, s, 2)
            e["theory"] = {"bytes_unique": prof,
                           "achieved_gbs": round(prof / (e["p50_ms"] / 1000.0) / 1e9, 1)}
            results["entries"].append(e)
            e2 = {"variant": "paged_prefill", "batch": b, "seq": s, "dtype": "fp16",
                  "block_size": bs2, "table_mode": "rand"}
            e2.update(OperatorTimer(device, warm, rep).measure(
                lambda: module.paged_prefill_attention(qd, kpd, vpd, table.to(device), bs2,
                                                       cfg.SOFTMAX_SCALE)))
            e2["theory"] = {"bytes_unique": prof + rp.block_table_bytes(b, s, bs2),
                            "achieved_gbs": round((prof + rp.block_table_bytes(b, s, bs2))
                                                  / (e2["p50_ms"] / 1000.0) / 1e9, 1)}
            results["entries"].append(e2)
            print(f"[bench/cuda] prefill fp16 S={s}: contig={e['p50_ms']}ms "
                  f"paged-rand={e2['p50_ms']}ms")
    return results


def bench_cpu() -> dict:
    device = resolve_device("cpu")
    from paged_kv_reference import (  # noqa: PLC0415
        contiguous_decode_reference,
        paged_decode_reference,
    )
    results: dict = {"backend": "cpu", "entries": []}
    bs = cfg.DEFAULT_BLOCK_SIZE
    for n in (64, 512, 2048):
        for b in (1, 4):
            q, k, v, kp, vp, table, lens, _ = make_paged_data(
                torch.device("cpu"), torch.float32, b, n, bs, "rand")
            e = {"variant": "paged_reference", "batch": b, "kv_len": n, "dtype": "fp32",
                 "block_size": bs, "table_mode": "rand"}
            e.update(OperatorTimer(device, 1, 5).measure(
                lambda: paged_decode_reference(q, kp, vp, table, [n] * b, bs, cfg.SOFTMAX_SCALE)))
            e2 = {"variant": "contig_reference", "batch": b, "kv_len": n, "dtype": "fp32",
                  "block_size": bs, "table_mode": "none"}
            e2.update(OperatorTimer(device, 1, 5).measure(
                lambda: contiguous_decode_reference(q, k, v, [n] * b, cfg.SOFTMAX_SCALE)))
            results["entries"].extend([e, e2])
            print(f"[bench/cpu] N={n} B={b}: paged_ref={e['p50_ms']}ms "
                  f"contig_ref={e2['p50_ms']}ms")
    return results


def bench_musa(platform: str) -> dict:
    bench_bin = HERE / "implementations" / "musa" / "paged_bench"
    if not bench_bin.exists():
        raise RuntimeError("paged_bench 未构建（先跑 implementations/musa/build.sh）")
    proc = subprocess_run([str(bench_bin), "bench"], cwd=str(bench_bin.parent))
    mu = json_loads(bench_bin.parent / "musa-benchmark.json")
    results: dict = {"backend": "musa", "entries": [],
                     "warp_device_builtin": mu.get("warp_device_builtin")}
    for raw in mu["entries"]:
        entry = dict(raw)
        attach_theory(entry, platform)
        results["entries"].append(entry)
    return results


def subprocess_run(cmd, cwd):
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=cwd)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        raise RuntimeError(f"musa bench rc={proc.returncode}: {proc.stderr[-2000:]}")
    return proc


def json_loads(path):
    import json
    return json.loads(Path(path).read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--decode-lens", default=None)
    parser.add_argument("--batches", default=None)
    parser.add_argument("--dtypes", default=None)
    parser.add_argument("--variants", default=None,
                        help="逗号分隔：contig,paged_seq,paged_rand,gather,prefill")
    arguments = parser.parse_args()
    if arguments.backend == "cpu":
        results = bench_cpu()
        results["environment"] = device_environment_record(resolve_device("cpu"))
        suffix = "cpu"
    elif arguments.backend == "cuda":
        decode_lens = [int(x) for x in arguments.decode_lens.split(",")] if arguments.decode_lens else None
        batches = [int(x) for x in arguments.batches.split(",")] if arguments.batches else None
        dtypes = [(x, torch.float16 if x == "fp16" else torch.float32) for x in arguments.dtypes.split(",")] \
            if arguments.dtypes else None
        variants = arguments.variants.split(",") if arguments.variants else None
        results = bench_cuda("cuda", decode_lens, batches, dtypes, variants)
        results["environment"] = device_environment_record(resolve_device("cuda"))
        suffix = "cuda"
    else:
        results = bench_musa("musa")
        try:
            results["environment"] = device_environment_record(resolve_device("musa"))
        except RuntimeError as error:
            results["environment_error"] = str(error)
        suffix = "musa"
    results["schema_version"] = "kernel-chapter22.benchmark.v1"
    results["scan"] = {"decode_lens": cfg.BENCH_DECODE_KV_LENS,
                       "batches": cfg.BENCH_BATCHES,
                       "block_size_default": cfg.DEFAULT_BLOCK_SIZE,
                       "prefill_lens": cfg.BENCH_PREFILL_SEQ_LENS}
    out = Path(arguments.out) if arguments.out else \
        HERE / "results" / f"benchmark-results-{suffix}.json"
    write_results_json(out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
