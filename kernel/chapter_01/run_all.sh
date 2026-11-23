#!/usr/bin/env bash
# 第 01 期 RMSNorm 基线与性能分析。
# 用法：
#   BACKENDS="cpu cuda" ./run_all.sh          # 4090 服务器（CPU 真值 + CUDA 列）
#   BACKENDS="cpu musa" ./run_all.sh          # S4000 服务器（MUSA 列 + CPU 自洽校验）
#   BACKENDS=cuda bash run_all.sh performance # 性能诊断（CPU、CUDA 或 MUSA）
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  performance) exec bash performance/run_all.sh ;;
  "") ;;
  *)
    echo "Usage: bash run_all.sh [performance]" >&2
    exit 2
    ;;
esac

BACKENDS="${BACKENDS:-cpu}"
MODEL_DIR="${MODEL_DIR:-models/qwen3-0.6b}"

echo "=== [1/4] Smoke Test（CPU-only）==="
python smoke_test.py

echo "=== [2/4] 生成 fixtures（CPU，固定 seed）==="
python generate_fixtures.py

if [ -f "$MODEL_DIR/model.safetensors" ]; then
  echo "=== [3/4] 模型真值（CPU FP32；仅真值机器执行一次）==="
  if [ ! -f truth/truth-manifest.json ] || [ ! -f truth/probes.npz ]; then
    python model_truth.py --model-dir "$MODEL_DIR"
  else
    echo "truth/ 已存在，跳过（跨平台对照使用同一份真值）"
  fi
else
  echo "=== [3/4] 跳过模型真值（未找到 $MODEL_DIR）==="
fi

echo "=== [4/4] 各后端正确性对照 + Baseline 性能 ==="
for backend in $BACKENDS; do
  if [ -f "$MODEL_DIR/model.safetensors" ]; then
    python compare_correctness.py --backend "$backend" --model-dir "$MODEL_DIR"
  else
    python compare_correctness.py --backend "$backend" --skip-model
  fi
  if [ "$backend" != "cpu" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
    python benchmark.py --backend "$backend" --with-model --model-dir "$MODEL_DIR"
  else
    python benchmark.py --backend "$backend"
  fi
done

echo "=== 完成。结果见 results/ ==="
