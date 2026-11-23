#!/usr/bin/env bash
# ============================================================================
# 第 21 期 MUSA 列单会话脚本（S4000，主机恢复后一键跑完；ep21 模式：
# 自含数据 + hash-RNG，不依赖 fixtures 上传）
# 用法：cd chapter_21/implementations/musa && bash run_session.sh
# 产出（本目录）：musa-env.json / musa-correctness.json / musa-benchmark.json
#                 + session-run.log / build-log-musa.txt
# 恢复后流程：重新 mkdir 抢锁 → tar 上传 chapter_21 → 跑本脚本 → 拉回 JSON。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
LOG=session-run.log
: > "$LOG"

echo "== [1] build ==" | tee -a "$LOG"
bash build.sh >> "$LOG" 2>&1 || { echo "BUILD FAILED"; tail -40 build-log-musa.txt; exit 1; }
echo "== [2] attrs（设备属性 + warpSize 双口径 + smem 预算）==" | tee -a "$LOG"
./paged_bench attrs 2>&1 | tee -a "$LOG"
echo "== [3] selftest ==" | tee -a "$LOG"
./paged_bench selftest 2>&1 | tee -a "$LOG" || { echo "SELFTEST FAILED"; exit 1; }
echo "== [4] correctness ==" | tee -a "$LOG"
./paged_bench correctness 2>&1 | tee -a "$LOG" || echo "correctness rc=$?（继续 bench）"
echo "== [5] bench ==" | tee -a "$LOG"
./paged_bench bench 2>&1 | tee -a "$LOG"
echo "== 全部完成：musa-env.json / musa-correctness.json / musa-benchmark.json ==" | tee -a "$LOG"
ls -la musa-*.json
