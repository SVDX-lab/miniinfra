#!/usr/bin/env bash
# 第 22 期 CUDA 列单会话脚本（RTX 4090）——【待 4090 修复后执行】
# 链与 MUSA 列逐项对应（对照口径一致）。
set -uo pipefail
cd "$(dirname "$0")"
LOG=session-run.log
: > "$LOG"

echo "== [1] build ==" | tee -a "$LOG"
bash build.sh >> "$LOG" 2>&1 || { echo "BUILD FAILED"; tail -30 build-log-cuda.txt; exit 1; }
echo "== [2] attrs ==" | tee -a "$LOG"
./rope_bench attrs cuda 2>&1 | tee -a "$LOG"
echo "== [3] selftest ==" | tee -a "$LOG"
./rope_bench selftest cuda 2>&1 | tee -a "$LOG" || { echo "SELFTEST FAILED"; exit 1; }
echo "== [4] correctness ==" | tee -a "$LOG"
./rope_bench correctness cuda 2>&1 | tee -a "$LOG" || echo "correctness rc=$?（继续）"
echo "== [5] cachedtype ==" | tee -a "$LOG"
./rope_bench cachedtype cuda 2>&1 | tee -a "$LOG" || echo "cachedtype rc=$?（继续）"
echo "== [6] bench ==" | tee -a "$LOG"
./rope_bench bench cuda 2>&1 | tee -a "$LOG" || echo "bench rc=$?（继续）"
echo "== 全部完成 ==" | tee -a "$LOG"
ls -la cuda-*.json 2>/dev/null | tee -a "$LOG"
