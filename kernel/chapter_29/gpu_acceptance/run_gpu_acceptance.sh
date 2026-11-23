#!/usr/bin/env bash
# 第 29 期 GPU 验收总编排（4090 修复后执行；必须已持有 cuda4090.lock）。
#
# 大纲实验清单 4 项 + fast_p，顺序即执行序：
#   ① 合同验收：compare_correctness.py --backend cuda
#      （非整 Tile/NaN/错误 Shape 边界全集 + 30/8/6 组 × dtype 对拍 + 确定性 3×）
#   ② 确定性检查：含在 ①（determinism 段：每 op×dtype 3 次 --run 位级对拍）
#   ③ sanitizer 体检：gpu_acceptance/sanitizer_suite.sh（四件套）
#   ④ 性能达成率对照：benchmark.py --backend cuda（AI vs 人工版同口径比值）
#   ⑤ fast_p 小型复刻：fast_p.py
#
# 用法（服务器上，chapter_29 目录内）：
#   BACKENDS="cuda" ./run_all.sh          # 等价于本文件全部步骤
# 或分步执行（推荐，编译失败时止损快）：
#   bash implementations/ai/rmsnorm/build.sh && bash implementations/ai/gemm/build.sh \
#     && bash implementations/ai/attention/build.sh
#   python3 smoke_test.py --backend cuda
#   python3 compare_correctness.py --backend cuda
#   python3 benchmark.py --backend cuda
#   bash gpu_acceptance/sanitizer_suite.sh
#   python3 fast_p.py
#
# 完成后：结果 JSON scp 回本地 chapter_29/results/（协议 §4），锁释放前
# 把 results/ 与 EXPERIMENT_LOG.md 回填状态写全。

set -uo pipefail
cd "$(dirname "$0")/.."

echo "[gpu-acceptance] BACKENDS=cuda ./run_all.sh 即完整执行序；本文件为步骤说明与防呆入口"
BACKENDS="cuda" ./run_all.sh
