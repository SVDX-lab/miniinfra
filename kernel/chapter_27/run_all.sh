#!/usr/bin/env bash
# 第 27 期一键验收（CPU 列 + 可选 MUSA 列；CUDA 列待回填见 CUDA_PENDING.md）
# 用法：
#   BACKENDS="cpu" bash run_all.sh           # 本地全链
#   BACKENDS="cpu musa" bash run_all.sh      # S4000 上（mus 环境已配 torch_musa）
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
BACKENDS=${BACKENDS:-cpu}
echo "=== ch28 run_all: backends=$BACKENDS ==="

# 1) fixtures 完整性（跨机共享真值；权重按 seed 重建 + SHA256 对拍）
$PY generate_fixtures.py --verify-only

# 2) 边界行为
$PY smoke_test.py

# 3) 实验 1：两形式数学等价（§9 主对 + fixtures 锚点 + 确定性）
$PY compare_correctness.py --backend cpu

# 4) 实验 2：状态访存对比曲线（确定性核算）
$PY memory_curve.py

# 5) 实验 3：递推 vs chunked 延迟（CPU 列；N=8192/32768 会话数分钟）
$PY benchmark.py --backend cpu

# 6) 实验 4：门控影响注记
$PY gate_probe.py

# 7) 实验 5：三条压缩 KV 路线并置（引用 ch22/ch26 已测数字）
$PY three_routes.py

# 8) kernel_sim：CUDA 核分解算法仿真（本地，无 GPU）
$PY implementations/cpu/kernel_sim.py

for B in $BACKENDS; do
  if [ "$B" = "musa" ]; then
    echo "=== MUSA 列（torch_musa eager 复验）==="
    $PY implementations/musa/musa_column.py
  fi
done

echo "=== 完成：results/ 下 JSON 为全部实测产物 ==="
