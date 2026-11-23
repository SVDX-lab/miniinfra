#!/usr/bin/env bash
# 第 26 期一键验收（BACKENDS 风格与 ch01 一致；本期 CPU 列为主，cuda/musa 待实机）
#
# 用法：
#   BACKENDS="cpu" bash run_all.sh            # 本地全列（默认）
#   BACKENDS="cpu musa" bash run_all.sh       # S4000 锁内追加 musa 小验证
#   BACKENDS="cpu cuda" bash run_all.sh       # 4090 修复后追加 cuda 回填
set -euo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"
PYTHON="${PYTHON:-python3}"

echo "[run_all] 第 26 期 MTP/投机解码——backends: ${BACKENDS}"

# 1) fixtures（幂等，SHA256 入 manifest）
"${PYTHON}" generate_fixtures.py

# 2) smoke（结构级 + 真模型）
"${PYTHON}" smoke_test.py --with-real-model

# 3) 实验 1：正确性 vs ch01 真值（6 draft × 6 k）
"${PYTHON}" compare_correctness.py

# 4) 实验 3 归因：draft 排序级诊断
"${PYTHON}" draft_quality_probe.py

# 5) 实验 2-5：接受率 / 加速比 / verify 批量化（CPU 口径）
"${PYTHON}" benchmark.py --exps 2,3,4,5

if [[ "${BACKENDS}" == *"musa"* ]]; then
  echo "[run_all] musa 小验证（须已在 S4000 锁内；模型目录用绝对路径）"
  (cd implementations/musa && python3 speculative_smoke_musa.py \
    --model-dir ../../../chapter_01/models/qwen3-0.6b --max-new-tokens 16 --k 4)
fi

if [[ "${BACKENDS}" == *"cuda"* ]]; then
  echo "[run_all] cuda verify 批量回填（须已在 4090；见 implementations/cuda/CUDA_PENDING.md）"
  (cd implementations/cuda && bash build.sh && ./verify_batch_bench --selftest \
    && ./verify_batch_bench > ../../results/verify-batch-results-cuda.json)
fi

echo "[run_all] 完成——结果见 results/，记录见 EXPERIMENT_LOG.md"
