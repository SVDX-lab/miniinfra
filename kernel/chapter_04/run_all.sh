#!/usr/bin/env bash
# ============================================================================
# 第 04 期一键验收。BACKENDS 环境变量风格（参照 ch01–ch04）：
#   BACKENDS="cpu" ./run_all.sh                  # 本地（无 GPU；模型级自动跳过）
#   BACKENDS="cpu cuda" ./run_all.sh             # 4090 锁内（模型权重+truth 就位）
#   BACKENDS="cpu musa" ./run_all.sh             # S4000 锁内
# 步骤：smoke → fixtures（复用 ep04，compare 内置核对）→ correctness → benchmark。
# 模型级实验（正确性链 6–8 / 层 2/3）需要 chapter_01 的权重与 truth：
#   默认路径 ../chapter_01/models/qwen3-0.6b 与 ../chapter_01/truth，
#   本地无权重时 run_all 的 CPU 列自动降级为算子级（--skip-model）。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"
PY="${PYTHON:-python3}"
mkdir -p results

MODEL_DIR="../chapter_01/models/qwen3-0.6b"
TRUTH_DIR="../chapter_01/truth"
MODEL_ARGS="--model-dir ${MODEL_DIR} --truth-dir ${TRUTH_DIR}"
if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
  echo "[run_all] 未找到模型权重（${MODEL_DIR}），模型级实验降级为 --skip-model"
  MODEL_ARGS="--skip-model"
fi

echo "=== [1/3] 冒烟测试 ==="
"$PY" -I smoke_test.py || echo "[warn] 冒烟有失败项（见上方输出，如实记录）"

echo "=== [2/3] 正确性（fixtures 注册回归 + 边界/fallback + 正确性链 6–8）==="
for backend in $BACKENDS; do
  echo "--- backend=$backend ---"
  "$PY" -I compare_correctness.py --backend "$backend" $MODEL_ARGS \
    || echo "[warn] correctness 失败（backend=$backend）"
done

echo "=== [3/3] 三层测量（微基准 / 单层 / 完整推理）==="
for backend in $BACKENDS; do
  echo "--- backend=$backend ---"
  "$PY" -I benchmark.py --backend "$backend" $MODEL_ARGS \
    || echo "[warn] benchmark 失败（backend=$backend）"
done

echo "=== 完成：results/ 下 JSON ==="
ls -la results/*.json 2>/dev/null
