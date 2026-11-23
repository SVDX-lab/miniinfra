#!/usr/bin/env bash
# 第 09 期一键验收（在 GPU 服务器上、chapter_09/ 根目录执行）。
#   BACKENDS="cuda" ./run_all.sh        # 4090
#   BACKENDS="musa" ./run_all.sh        # S4000
# 无 CPU 列（期首声明：Graph 是设备侧调用组织概念）。
set -uo pipefail
cd "$(dirname "$0")"

BACKENDS=${BACKENDS:-cuda}
PYTHON=${PYTHON:-python3}

if [[ ! -f fixtures/spot-rmsnorm-2048x1024.pt || ! -f fixtures/decode-step-inputs.json ]]; then
  "$PYTHON" generate_fixtures.py || exit 1
fi

for B in $BACKENDS; do
  echo "=== [ep10] backend=$B probe ==="
  $PYTHON "implementations/$B/graph_probe.py" --backend "$B" || exit 1
  echo "=== [ep10] backend=$B smoke（含模型冒烟） ==="
  $PYTHON smoke_test.py --backend "$B" --with-model || exit 1
  echo "=== [ep10] backend=$B correctness ==="
  $PYTHON compare_correctness.py --backend "$B" || exit 1
  echo "=== [ep10] backend=$B launch-bound 图化（实验1+2 RMSNorm档） ==="
  $PYTHON "implementations/$B/launch_bound_graph.py" --backend "$B" || exit 1
  echo "=== [ep10] backend=$B graph 成本矩阵（实验5 M-02 深挖） ==="
  $PYTHON "implementations/$B/graph_matrix.py" --backend "$B" || exit 1
  echo "=== [ep10] backend=$B decode 三级归因阶梯（实验3+4） ==="
  $PYTHON "implementations/$B/decode_step_graph.py" --backend "$B" || exit 1
  echo "=== [ep10] backend=$B recapture 成本与显存驻留（实验3） ==="
  $PYTHON "implementations/$B/recapture_memory.py" --backend "$B" || exit 1
done

echo "=== [ep10] 全部完成，results/ 清点 ==="
ls -la results/
