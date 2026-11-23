#!/usr/bin/env bash
# ============================================================================
# 第 13 期一键验收（锁内执行；BACKENDS 环境变量风格，参照 ch01/ch03）。
#
# 用法：
#   BACKENDS="cuda" ./run_all.sh                  # CUDA 4090 全套
#   BACKENDS="musa" ./run_all.sh                  # S4000 全套
#   WITH_HANDWRITTEN=1 BACKENDS="cuda" ./run_all.sh   # 加手写同会话对照
#   SKIP_NSYS=1 ...                               # 跳过 nsys（MUSA 无 nsys 自动跳）
#
# 顺序：fixtures 校验 → probe → smoke → correctness → benchmark → autotune
#       → compile_artifacts →（CUDA）nsys 证据
# 每步日志落 results/server-logs/，失败继续下一步（各步独立取证），
# 最终打印产物清单。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cuda}"
WITH_HANDWRITTEN="${WITH_HANDWRITTEN:-0}"
mkdir -p results/server-logs

for BACKEND in $BACKENDS; do
  LOG="results/server-logs"
  echo "===== [ch14] backend=$BACKEND fixtures 校验 ====="
  python3 generate_fixtures.py 2>&1 | tee "$LOG/fixtures-$BACKEND.log"

  echo "===== [ch14] backend=$BACKEND triton probe ====="
  python3 "implementations/$BACKEND/triton_probe.py" --backend "$BACKEND" 2>&1 | tee "$LOG/probe-$BACKEND.log"

  echo "===== [ch14] backend=$BACKEND smoke ====="
  if [ "$WITH_HANDWRITTEN" = "1" ]; then
    python3 smoke_test.py --backend "$BACKEND" --with-handwritten 2>&1 | tee "$LOG/smoke-$BACKEND.log"
  else
    python3 smoke_test.py --backend "$BACKEND" 2>&1 | tee "$LOG/smoke-$BACKEND.log"
  fi

  echo "===== [ch14] backend=$BACKEND correctness ====="
  if [ "$WITH_HANDWRITTEN" = "1" ]; then
    python3 compare_correctness.py --backend "$BACKEND" --with-handwritten 2>&1 | tee "$LOG/compare-$BACKEND.log"
  else
    python3 compare_correctness.py --backend "$BACKEND" 2>&1 | tee "$LOG/compare-$BACKEND.log"
  fi

  echo "===== [ch14] backend=$BACKEND benchmark ====="
  if [ "$WITH_HANDWRITTEN" = "1" ]; then
    python3 benchmark.py --backend "$BACKEND" --with-handwritten 2>&1 | tee "$LOG/benchmark-$BACKEND.log"
  else
    python3 benchmark.py --backend "$BACKEND" 2>&1 | tee "$LOG/benchmark-$BACKEND.log"
  fi

  echo "===== [ch14] backend=$BACKEND autotune ====="
  python3 autotune_experiment.py --backend "$BACKEND" 2>&1 | tee "$LOG/autotune-$BACKEND.log"

  echo "===== [ch14] backend=$BACKEND compile artifacts ====="
  python3 "implementations/$BACKEND/compile_artifacts.py" --backend "$BACKEND" 2>&1 | tee "$LOG/artifacts-$BACKEND.log"

  if [ "$BACKEND" = "cuda" ] && [ "${SKIP_NSYS:-0}" != "1" ]; then
    echo "===== [ch14] backend=$BACKEND nsys 证据 ====="
    bash implementations/$BACKEND/run_nsys.sh 2>&1 | tee "$LOG/nsys-$BACKEND.log" || echo "[ch14] nsys 失败（继续）"
  fi
done

echo "===== [ch14] 产物清单 ====="
ls -la results/ | grep -v server-logs || true
ls results/ir/*/ 2>/dev/null | head -30 || true
