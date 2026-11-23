#!/usr/bin/env bash
# 第 14 期实验 2 运行器：TORCH_LOGS=output_code 捕获 Inductor 生成代码。
# 用法: ./codegen_capture.sh <python 解释器路径，默认本地 venv>
set -uo pipefail
PY="${1:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results/inductor-cpu"
mkdir -p "$OUT"
cd "$HERE"
export TORCH_LOGS=output_code
"$PY" codegen_capture.py --objects rmsnorm,add_rmsnorm > "$OUT/rmsnorm-add_rmsnorm-output_code.txt" 2>&1
"$PY" codegen_capture.py --objects add_rmsnorm_dynamic,add_rmsnorm_inplace > "$OUT/dynamic-inplace-output_code.txt" 2>&1
"$PY" codegen_capture.py --objects qwen3 --rows 128 > "$OUT/qwen3-128-output_code.txt" 2>&1
echo "captured:"; wc -l "$OUT"/*.txt
