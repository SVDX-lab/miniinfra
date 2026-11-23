#!/usr/bin/env bash
# ============================================================================
# 第 20 期一键验收（BACKENDS="cpu cuda" / "cpu musa" 环境变量风格，参照 ch01/ch03）。
# 用法：
#   BACKENDS="cpu" bash run_all.sh          # 只跑 CPU 列（本地可跑）
#   BACKENDS="cpu cuda" bash run_all.sh     # CUDA 列（4090 服务器上执行）
#   BACKENDS="cpu musa" bash run_all.sh     # MUSA 列（S4000 服务器上执行；
#                                           # 先跑 implementations/musa/build.sh）
# 前置：fixtures 已随仓库分发（generate_fixtures.py --verify-only 可复核）。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
BACKENDS="${BACKENDS:-cpu}"
PY="${PY:-python3}"

echo "=== [0] fixtures 校验（--verify-only，免重新生成）==="
"$PY" generate_fixtures.py --verify-only || "$PY" generate_fixtures.py

echo "=== [1] smoke test ==="
if [[ "$BACKENDS" == *"cuda"* ]]; then "$PY" smoke_test.py --backend cuda || exit 1;
elif [[ "$BACKENDS" == *"musa"* ]]; then "$PY" smoke_test.py || true;  # musa 核 smoke 在 .mu selftest
else "$PY" smoke_test.py || exit 1; fi

for BACKEND in $BACKENDS; do
  if [[ "$BACKEND" == "musa" ]]; then
    echo "=== [2] musa binary selftest + correctness ==="
    (cd implementations/musa && ./attention_bench attrs && ./attention_bench selftest) || exit 1
    echo "=== [3] correctness ($BACKEND) ==="
    "$PY" compare_correctness.py --backend musa || exit 1
    echo "=== [4] benchmark ($BACKEND) ==="
    "$PY" benchmark.py --backend musa || exit 1
  else
    echo "=== [2] correctness ($BACKEND) ==="
    "$PY" compare_correctness.py --backend "$BACKEND" || exit 1
    echo "=== [3] benchmark ($BACKEND) ==="
    "$PY" benchmark.py --backend "$BACKEND" || exit 1
  fi
done

echo "=== 完成：results/ 下 correctness-results-*.json / benchmark-results-*.json ==="
