#!/usr/bin/env bash
# ============================================================================
# run_all.sh —— 第 10 期一键验收。
#   BACKENDS="cpu"     本地：TSAN 四口径
#   BACKENDS="cuda"    4090：build + smoke + correctness + suite(E1-E5) + benchmark
#   BACKENDS="musa"    S4000：tool_probe + build_and_run（需持锁）
# 用法：BACKENDS="cuda" ./run_all.sh [suite 阶段...]（缺省全阶段）
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKENDS="${BACKENDS:-cpu cuda}"
STAGES="${*:-}"

for be in $BACKENDS; do
  case "$be" in
    cpu)
      echo "===== CPU 列（本地，TSAN）====="
      (cd "$HERE/implementations/cpu" && ./run_tsan.sh)
      ;;
    cuda)
      echo "===== CUDA 列（4090）====="
      (cd "$HERE/implementations/cuda" && ./build.sh)
      (cd "$HERE" && /usr/bin/python3 smoke_test.py | tail -15)
      (cd "$HERE/implementations/cuda" && ./run_sanitizer_suite.sh $STAGES)
      (cd "$HERE" && /usr/bin/python3 compare_correctness.py | tail -12)
      (cd "$HERE" && /usr/bin/python3 benchmark.py | tail -40)
      ;;
    musa)
      echo "===== MUSA 列（S4000，需持锁）====="
      (cd "$HERE/implementations/musa" && bash tool_probe.sh | tail -20)
      (cd "$HERE/implementations/musa" && bash build_and_run.sh)
      ;;
    *) echo "未知后端 $be"; exit 2;;
  esac
done
echo "[run_all] 完成"
