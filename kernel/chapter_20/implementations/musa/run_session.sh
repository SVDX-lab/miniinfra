#!/usr/bin/env bash
# 第 20 期 MUSA 列单会话脚本（S4000，主机恢复后一键跑完）
set -uo pipefail
cd "$(dirname "$0")"
LOG=session-run.log
: > "$LOG"

echo "== [1] build ==" | tee -a "$LOG"
bash build.sh >> "$LOG" 2>&1 || { echo "BUILD FAILED"; tail -30 build-log-musa.txt; exit 1; }
echo "== [2] attrs ==" | tee -a "$LOG"
./attention_bench attrs 2>&1 | tee -a "$LOG"
echo "== [3] selftest ==" | tee -a "$LOG"
./attention_bench selftest 2>&1 | tee -a "$LOG" || { echo "SELFTEST FAILED"; exit 1; }
echo "== [4] correctness ==" | tee -a "$LOG"
./attention_bench correctness 2>&1 | tee -a "$LOG" || echo "correctness rc=$?（继续 bench）"
echo "== [5] bench ==" | tee -a "$LOG"
./attention_bench bench 2>&1 | tee -a "$LOG"
echo "== 全部完成：musa-env.json / musa-correctness.json / musa-benchmark.json ==" | tee -a "$LOG"
ls -la musa-*.json
