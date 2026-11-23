#!/usr/bin/env bash
# 第 22 期一键验收。
# 用法：
#   ./run_all.sh                          # 本地 CPU 列全链（无 GPU）
#   ./run_all.sh --musa                   # S4000：MUSA 列单会话（内部转 run_session.sh）
#   ./run_all.sh --cuda                   # 4090：CUDA 列单会话【待修复】
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"

if [ "${1:-}" = "--musa" ]; then
  exec bash implementations/musa/run_session.sh
fi
if [ "${1:-}" = "--cuda" ]; then
  echo "CUDA 列待 4090 修复（见 implementations/cuda/CUDA_PENDING.md）" >&2
  exec bash implementations/cuda/run_session.sh
fi

echo "=== [1/4] Smoke Test（S1-S12：合同校验/数学性质/kernel_sim 复跑）==="
$PY smoke_test.py

echo "=== [2/4] 生成 fixtures（固定 seed + mu_hash 校验和锚点）==="
$PY generate_fixtures.py

echo "=== [3/4] 正确性：HF 对照 + 正确性矩阵 + cache dtype 误差放大 ==="
$PY compare_correctness.py

echo "=== [4/4] CPU 基准：两形态延迟/带宽 + cache 吞吐 + 形态切换 ==="
$PY benchmark.py

echo "=== 完成。结果见 results/ ==="
