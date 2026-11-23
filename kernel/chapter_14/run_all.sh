#!/usr/bin/env bash
# 第 14 期一键验收（协议 §1 run_all.sh 约定）。
# BACKENDS="cpu"（本地默认）；BACKENDS="cpu musa" 在 S4000 上跑 MUSA 探测段；
# CUDA 列待补（4090 修复后 BACKENDS="cpu cuda" 于服务器执行）。
set -uo pipefail
BACKENDS="${BACKENDS:-cpu}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p results

echo "== [1/6] fixtures 校验/生成"
if [ ! -f fixtures/correctness.npz ]; then "$PY" generate_fixtures.py; fi
"$PY" -c "import json; m=json.load(open('fixtures/manifest.json')); print('groups:', len(m['groups']), 'npz sha256:', m['npz_sha256'][:16], '...')"

echo "== [2/6] smoke"
"$PY" smoke_test.py || exit 1

echo "== [3/6] correctness"
"$PY" compare_correctness.py || exit 1

echo "== [4/6] codegen 捕获"
bash codegen_capture.sh "$PY"

echo "== [5/6] 延迟主表（含冷编译；约 7 分钟）"
"$PY" benchmark.py || exit 1

echo "== [6/6] dynamic/重编译行为"
"$PY" recompile_experiment.py || exit 1

if [[ "$BACKENDS" == *"musa"* ]]; then
  echo "== [MUSA] torch.compile 行为矩阵探测"
  bash implementations/musa/run_probe_session.sh
fi
if [[ "$BACKENDS" == *"cuda"* ]]; then
  echo "== [CUDA] 待补：implementations/cuda/CUDA_PENDING.md"
  "$PY" implementations/cuda/compile_bench_cuda.py || echo "CUDA 列不可用（待补）"
fi

echo "== 完成。结果清单："
ls -la results/
