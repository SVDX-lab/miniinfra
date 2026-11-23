#!/usr/bin/env python3
"""第 27 期实验 5：三条"压缩 KV"路线并置总结。

路线（跨期引用已测数字 + 本期实测）：
  A. ch22 分页 KV（PagedAttention）：不改数学，改"组织"——KV 全历史保留，
     消除连续预留的容量碎片；代价 = gather/表查找的访存路径变长。
  B. ch26 稀疏化（DSA）：不改数学，改"读取集合"——只读 top-k 历史；
     代价 = 索引器开销 + 检索质量依赖（k—误差—性能三元关系）。
  C. ch28 线性注意力（Gated DeltaNet）：改数学本身——KV 被固定大小递推状态
     替换，O(N) 增长变 O(1)；代价 = 信息有损（衰减遗忘 + 单次写覆盖），
     换取结构级的访存解耦。

产出 results/three-routes.json：跨期引用数字（读前序 results JSON，注明出处）
+ 本期 memory-curve / gate-probe 的数字，并置为对照表数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from experiment_utils import write_results_json  # noqa: E402

CH22 = Path(__file__).resolve().parent.parent / "chapter_21" / "results"
CH26 = Path(__file__).resolve().parent.parent / "chapter_25" / "results"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()

    # ---- ch22 引用：paged vs contig decode 延迟比（MUSA 实测）----
    ch22 = json.loads((CH22 / "benchmark-results-musa.json").read_text())
    ratios = []
    by_key = {}
    for e in ch22["entries"]:
        key = (e["batch"], e["kv_len"], e["dtype"])
        by_key.setdefault(key, []).append((e["variant"], e["p50_ms"]))
    for key, variants in sorted(by_key.items()):
        contig = [ms for v, ms in variants if v == "contig_kernel"]
        paged = [ms for v, ms in variants if v == "paged_kernel"]
        if contig and paged:
            ratios.append({"batch": key[0], "kv_len": key[1], "dtype": key[2],
                           "paged_over_contig_p50": round(paged[-1] / contig[-1], 3),
                           "note": "各 table_mode/block_size 组合逐对取末值"})
    kv_tok = json.loads((CH22 / "capacity-accounting.json").read_text())["kv_bytes_per_token"]

    # ---- ch26 引用：每 token KV 字节与索引开销 ----
    ch26 = json.loads((CH26 / "calculation-comparison.json").read_text())
    kv26 = ch26["kv_bytes_per_token"]

    # ---- ch28 本期数字 ----
    ours = json.loads((HERE / "results" / "memory-curve.json").read_text())
    gate = json.loads((HERE / "results" / "gate-probe.json").read_text())

    table = {
        "schema_version": "kernel-chapter28.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "routes": [
            {
                "route": "A_paged_kv (ch22)",
                "what_changes": "组织：KV 全历史保留，块表间接寻址（数学不变）",
                "bytes_growth": "O(N)——每 token 仍写 K/V",
                "bytes_per_token_per_layer_fp16": kv_tok["fp16"]["per_layer_bytes"],
                "info_loss": "无损（位级值透明，37 组 gather 全对）",
                "measured_cost": {
                    "source": "chapter_21/results/benchmark-results-musa.json",
                    "paged_over_contig_decode_p50_ratio": ratios,
                    "headline": "decode 恒 ~1.8-2.2×（间接寻址+表查找路径代价；"
                                "paged-seq≈paged-rand 差<0.5%——局部性效应在 MUSA 上不可见）",
                },
                "serial_depth": "decode 每 token 读全部 N 历史（无并行结构改变）",
            },
            {
                "route": "B_sparse_dsa (ch26)",
                "what_changes": "读取集合：每 query 只读 top-k 历史（数学不变，索引器新增）",
                "bytes_growth": "O(N) 缓存不变，但每 token 读取 O(k)",
                "bytes_per_token_per_layer_fp16": kv26["MLA_latent512_rope64"],
                "info_loss": "检索可控：k 大误差小；支撑集外信息不可恢复（ep26 三元曲线）",
                "measured_cost": {
                    "source": "chapter_25/results/{calculation-comparison,benchmark-results-cuda}.json",
                    "kv_bytes_token": {k: v for k, v in kv26.items() if isinstance(v, (int, float))},
                    "indexer_overhead": "N=65536、k=2048 时索引段占主口径 ~50%+（naive 核）",
                    "mac_ratio_vs_dense": "Design B（索引 K 独立缓存）= 0.49× @N=128K,k=2048",
                },
                "serial_depth": "索引器 + 稀疏 gather 均可并行（无递推）",
            },
            {
                "route": "C_linear_state (ch28)",
                "what_changes": "数学本身：KV → 固定递推状态 S∈R^{16×128×128}，衰减+delta 写入",
                "bytes_growth": "O(1)——与 N 解耦（512KB/层 fp16 常数）",
                "bytes_per_token_per_layer_fp16": 512 * 1024,
                "info_loss": "有损且不可控恢复：指数衰减（探针 retention α^D，"
                             "α=0.819 时 D=64 已 2.5%）+ 秩-1 写覆盖",
                "measured_cost": {
                    "source": "chapter_27/results/{memory-curve,gate-probe,benchmark-results-cpu}.json",
                    "storage_crossover_tokens": ours["crossover_tokens_per_layer"],
                    "decode_read_ratio_by_n": {str(c["seq_len"]): c["decode_read_ratio"]
                                               for c in ours["curve"]},
                    "prefill_read_ratio_by_n": {str(c["seq_len"]): c["prefill_read_ratio"]
                                                for c in ours["curve"]},
                    "cpu_latency_chunked_over_recurrent": "0.161–0.316（N=512–2048，B=16–64；净环境复跑口径）",
                },
                "serial_depth": "递推 T 步（chunked 后 (C−1)+2NC 步批量矩阵算子）",
            },
        ],
        "cross_route_notes": {
            "共同点": "三条路线都在'带宽墙'上做文章——decode 全注意力读全历史是"
                      "memory-bound（ch22 AI≈2 FLOP/Byte，深度访存受限）",
            "互斥性": "A/B 与 C 不互斥：混合架构（Qwen3-Next 型）对部分层用 C 换 O(1) "
                      "状态、对关键层保留 A/B 的无损全注意力",
            "误差性质分层": "A 无损 / B 有损但可验（ep26 给定索引集 vs 子集 dense 位级对照）"
                            "/ C 有损且无参考可恢复（本期以两形式等价保证'算得对'，"
                            "信息取舍由训练学到的门控决定）",
        },
    }
    write_results_json(HERE / "results" / "three-routes.json", table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
