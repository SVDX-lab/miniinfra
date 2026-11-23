#!/usr/bin/env bash
# 第 28 期一键验收（CPU 列 + 可选 MUSA 列；CUDA 列待回填见 implementations/cuda/CUDA_PENDING.md）
# 用法：
#   PYTHON=/home/aiuser/venvs/ml/bin/python bash run_all.sh              # 本地全链
#   BACKENDS="cpu musa" bash run_all.sh                                   # S4000 上（torch_musa 已配）
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
BACKENDS=${BACKENDS:-cpu}
echo "=== ch29 run_all: backends=$BACKENDS ==="

# 1) fixtures 完整性（跨机共享真值；权重按 seed 重建 + SHA256 对拍）
$PY generate_fixtures.py --verify-only

# 2) 边界行为
$PY smoke_test.py

# 3) 实验 1：正确性主对（分组 vs 全量专家朴素 + 结合律 + tie-break + 确定性）
$PY compare_correctness.py --backend cpu

# 4) 实验 2：路由开销占比矩阵（tokens×experts×topk 30 格；约 10 分钟）
$PY benchmark.py --backend cpu

# 5) 实验 3：负载不均（均匀 vs 80/20-Zipf；含线程敏感度探针）
$PY load_imbalance.py

# 6) 实验 4：激活参数估算（附录 B 题 11：5 公开 config 校核 + V4-Pro 反推）
$PY param_estimate.py

# 7) 实验 5：省计算不省权重读取（字节核算 + 触碰曲线 + LRU cache 模拟）
$PY weight_traffic.py

# 8) kernel_sim：CUDA 核分解语义仿真（本地无 GPU）
$PY implementations/cpu/kernel_sim.py

for B in $BACKENDS; do
  if [ "$B" = "musa" ]; then
    echo "=== MUSA 列（torch_musa eager 小验证）==="
    $PY implementations/musa/musa_column.py
  fi
done

echo "=== 完成：results/ 下 JSON 为全部实测产物 ==="
