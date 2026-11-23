#!/usr/bin/env bash
# ============================================================================
# 第 25 期一键验收（BACKENDS="cpu cuda" 环境变量风格，参照 ch01/ch03）。
# 用法：
#   BACKENDS="cpu" bash run_all.sh        # 只跑 CPU 列（本地可跑）
#   BACKENDS="cpu cuda" bash run_all.sh   # CUDA 列需在 4090 服务器上执行
# 前置：fixtures 已随仓库分发（generate_fixtures.py --verify-only 可复核）。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
BACKENDS="${BACKENDS:-cpu}"
PY="${PY:-python3}"

echo "=== [0] fixtures 校验（--verify-only，免重新生成）==="
"$PY" generate_fixtures.py --verify-only || "$PY" generate_fixtures.py

echo "=== [1] smoke test ==="
if [[ "$BACKENDS" == *"cuda"* ]]; then "$PY" smoke_test.py --backend cuda; else "$PY" smoke_test.py; fi

for BACKEND in $BACKENDS; do
  echo "=== [2] correctness ($BACKEND) ==="
  "$PY" compare_correctness.py --backend "$BACKEND" || exit 1
  echo "=== [3] benchmark ($BACKEND) ==="
  "$PY" benchmark.py --backend "$BACKEND" || exit 1
done

if [[ "$BACKENDS" == *"cuda"* ]]; then
  echo "=== [4] nsys 证据采集 ==="
  bash implementations/cuda/run_nsys.sh
fi

echo "=== 完成：results/ 下 correctness-results-*.json / benchmark-results-*.json / nsys-*.txt ==="
