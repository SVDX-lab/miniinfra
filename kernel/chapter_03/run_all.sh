#!/usr/bin/env bash
# ============================================================================
# 第 03 期一键验收。BACKENDS 环境变量风格（参照 ch01/ch03）：
#   BACKENDS="cpu" ./run_all.sh                  # 本地（无 GPU）
#   BACKENDS="cpu cuda" ./run_all.sh             # 4090 锁内
#   BACKENDS="cpu musa" ./run_all.sh             # S4000 锁内
# 步骤：smoke → fixtures 完整性（compare 内置）→ correctness → benchmark。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"
PY="${PYTHON:-python3}"
mkdir -p results

echo "=== [1/3] 冒烟测试 ==="
"$PY" -I smoke_test.py || echo "[warn] 冒烟有失败项（见上方输出，如实记录）"

echo "=== [2/3] 正确性（30 组 × 双产物核对）==="
for backend in $BACKENDS; do
  echo "--- backend=$backend ---"
  "$PY" -I compare_correctness.py --backend "$backend" || echo "[warn] correctness 失败（backend=$backend）"
done

echo "=== [3/3] 性能（合同形状 + DRAM 探针 + launch 计数）==="
for backend in $BACKENDS; do
  echo "--- backend=$backend ---"
  "$PY" -I benchmark.py --backend "$backend" || echo "[warn] benchmark 失败（backend=$backend）"
done

echo "=== 完成：results/ 下 JSON 见 ls ==="
ls -la results/*.json 2>/dev/null
