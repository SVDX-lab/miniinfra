#!/usr/bin/env bash
# 第 22 期 MUSA 构建（MTT S4000，MUSA SDK 5.2.0 / mcc 5.2.0，ch18/ch21 回退链模式）
#
# 源码：rope_bench.mu（入口 shim）+ ../cuda/rope_kernels.cuh（核）
#       + ../cuda/rope_bench_main.cuh（入口体，CUDA/MUSA 共用）。
# 回退链（以实机为准，逐次记 build-log-musa.txt）：
#   1. 默认参数（sincosf 可用则一次过；ch21 默认目标 mp_21 实测可编译）
#   2. -DRPE_NO_SINCOS（若 sincosf 缺失 → 手工加开关改 sinf/cosf，见 KNOWN_ISSUES）
#   3. -DDISABLE_FUNC_ATTRS（防御性保留）
# 本核零 shared memory——M-43（静态 shared 28,672B 上限）不适用。
set -uo pipefail
cd "$(dirname "$0")"

LOG="build-log-musa.txt"
: > "$LOG"

echo "== mcc 版本 ==" | tee -a "$LOG"
mcc --version 2>&1 | tee -a "$LOG"

FLAGS_LIST=(
  ""
  "-DRPE_NO_SINCOS"
  "-DDISABLE_FUNC_ATTRS"
)

RC=1
for FLAGS in "${FLAGS_LIST[@]}"; do
  echo "" | tee -a "$LOG"
  echo "== 尝试: mcc -O3 -std=c++17 $FLAGS -I../cuda -o rope_bench rope_bench.mu ==" | tee -a "$LOG"
  START=$(date +%s.%N)
  # shellcheck disable=SC2086
  mcc -O3 -std=c++17 $FLAGS -I../cuda -o rope_bench rope_bench.mu >> "$LOG" 2>&1
  RC=$?
  END=$(date +%s.%N)
  echo "rc=$RC 耗时 $(echo "$END $START" | awk '{printf "%.1f", $1-$2}')s" | tee -a "$LOG"
  if [ "$RC" -eq 0 ]; then
    echo "== 成功，标志集: \"$FLAGS\" ==" | tee -a "$LOG"
    echo "$FLAGS" > build-flags.txt
    break
  fi
  tail -30 "$LOG" | sed 's/^/    | /'
done

if [ "$RC" -ne 0 ]; then
  echo "== 全部尝试失败，见 $LOG ==" | tee -a "$LOG"
  exit 1
fi

COMPILE_S=$(echo "$END $START" | awk '{printf "%.1f", $1-$2}')
echo "{\"compile_wall_seconds\": ${COMPILE_S}, \"flags\": \"${FLAGS}\"}" > compile-time.json
echo "== 构建完成（${COMPILE_S}s），跑 attrs + selftest ==" | tee -a "$LOG"
./rope_bench attrs musa | tee -a "$LOG"
./rope_bench selftest musa | tee -a "$LOG"
