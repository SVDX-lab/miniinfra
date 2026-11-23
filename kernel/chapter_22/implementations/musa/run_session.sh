#!/usr/bin/env bash
# 第 22 期 MUSA 列单会话脚本（S4000；ch21 run_session 模式）
# 链：build → attrs → selftest → correctness → cachedtype → bench → apply 桥自检
# 产物：musa-env.json / musa-correctness.json / musa-cache-dtype.json /
#       musa-benchmark.json / apply-summary.json / 各 run log
set -uo pipefail
cd "$(dirname "$0")"
LOG=session-run.log
: > "$LOG"

echo "== [1] build ==" | tee -a "$LOG"
bash build.sh >> "$LOG" 2>&1 || { echo "BUILD FAILED"; tail -30 build-log-musa.txt; exit 1; }
echo "== [2] attrs ==" | tee -a "$LOG"
./rope_bench attrs musa 2>&1 | tee -a "$LOG"
echo "== [3] selftest ==" | tee -a "$LOG"
./rope_bench selftest musa 2>&1 | tee -a "$LOG" || { echo "SELFTEST FAILED"; exit 1; }
echo "== [4] correctness（seq {1,64,512,2048,32768} × fp32/fp16 × 两形态 + 确定性）==" | tee -a "$LOG"
./rope_bench correctness musa 2>&1 | tee -a "$LOG" || echo "correctness rc=$?（继续）"
echo "== [5] cachedtype（theta × seq × cache dtype 误差放大）==" | tee -a "$LOG"
./rope_bench cachedtype musa 2>&1 | tee -a "$LOG" || echo "cachedtype rc=$?（继续）"
echo "== [6] bench（prefill/decode/build-cache/switch/launch-rate）==" | tee -a "$LOG"
./rope_bench bench musa 2>&1 | tee -a "$LOG" || echo "bench rc=$?（继续）"
echo "== [7] apply 桥自检（随机数据端到端，验证文件桥供真实权重验证用）==" | tee -a "$LOG"
if [ -f ../../verify_qwen3_layer.py ]; then
  python3 ../../verify_qwen3_layer.py --apply-selfcheck --binary ./rope_bench 2>&1 | tee -a "$LOG"
fi
echo "== 全部完成 ==" | tee -a "$LOG"
ls -la musa-*.json apply-summary.json 2>/dev/null | tee -a "$LOG"
