#!/usr/bin/env bash
# 第 12 期一键验收（BACKENDS 环境变量风格，参照 ch01）。
# 用法：
#   本地 CPU 列:        BACKENDS="cpu" bash run_all.sh
#   S4000 MUSA 列:      BACKENDS="musa" bash run_all.sh
#   4090 CUDA 列:       BACKENDS="cpu,cuda" bash run_all.sh
# 模型目录：优先本期 models/qwen3-0.6b（本地），缺省回退 chapter_01（服务器）。
set -uo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"
PYTHON="${PYTHON:-python3}"
FAILURES=0
MODEL_ARGS=()
if [ -d "models/qwen3-0.6b" ]; then
  MODEL_ARGS=(--model-dir "models/qwen3-0.6b")
fi

echo "== 第 12 期 run_all：backends=${BACKENDS} python=${PYTHON} =="

echo "-- [0] fixtures（不存在时生成） --"
if [ ! -f fixtures/probes.npz ]; then
  "$PYTHON" generate_fixtures.py || FAILURES=$((FAILURES+1))
else
  echo "fixtures/probes.npz 已在位，跳过"
fi

echo "-- [1] smoke test（本地行为学验证，无需 GPU） --"
"$PYTHON" smoke_test.py || FAILURES=$((FAILURES+1))

for BACKEND in ${BACKENDS//,/ }; do
  echo "-- [2] capability matrix（含 ${BACKEND}） --"
  "$PYTHON" capability_matrix.py --backends "cpu,${BACKEND}" \
    --out "results/capability-matrix-${BACKEND}.json" || FAILURES=$((FAILURES+1))

  echo "-- [3] exp1 correctness（${BACKEND} vs ch01 truth） --"
  "$PYTHON" compare_correctness.py --backend "${BACKEND}" "${MODEL_ARGS[@]}" \
    --out "results/correctness-results-${BACKEND}.json" || FAILURES=$((FAILURES+1))

  echo "-- [4] exp4 benchmark/telemetry（${BACKEND}） --"
  "$PYTHON" benchmark.py --backend "${BACKEND}" "${MODEL_ARGS[@]}" \
    --out "results/benchmark-results-${BACKEND}.json" || FAILURES=$((FAILURES+1))
done

echo "== run_all 完成：FAILURES=${FAILURES} =="
exit "$FAILURES"
