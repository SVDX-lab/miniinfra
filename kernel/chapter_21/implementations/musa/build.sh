#!/usr/bin/env bash
# ============================================================================
# 第 21 期 MUSA 构建（MTT S4000，MUSA SDK 5.2.0 / mcc 5.2.0；ch21 回退链模式）
#
# 源码：paged_bench.mu（standalone 二进制，与 CUDA 列共用
# ../cuda/paged_kernels.cuh——"机械替换单点化"迁移设计）。
# 预期路径（ch21 实测沉淀，以实机为准逐项记录进 build-log-musa.txt）：
#   - .mu 必须自带 #define USE_MUSA 1（ch18 模式必要条件）；
#   - musa_fp16.h 存在（M-42 实测可用）；mcc 默认目标 mp_21，需要时
#     --offload-arch=mp_22（ch19 M-43 系）；核 smem 预算已按 28,672B 设计。
# 回退链：
#   1. mcc -O3 -std=c++17 -I../cuda -o paged_bench paged_bench.mu
#   2. + -DMUSA_IGNORE_FP16_H（防御）
#   3. + --offload-arch=mp_22（显式目标）
#   4. + -DDISABLE_FUNC_ATTRS
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

LOG="build-log-musa.txt"
: > "$LOG"

echo "== mcc 版本 ==" | tee -a "$LOG"
mcc --version 2>&1 | tee -a "$LOG"

FLAGS_LIST=(
  ""
  "-DMUSA_IGNORE_FP16_H"
  "--offload-arch=mp_22"
  "--offload-arch=mp_22 -DMUSA_IGNORE_FP16_H"
  "-DDISABLE_FUNC_ATTRS"
)

RC=1
FLAGS=""
for FLAGS_TRY in "${FLAGS_LIST[@]}"; do
  echo "" | tee -a "$LOG"
  echo "== 尝试: mcc -O3 -std=c++17 $FLAGS_TRY -I../cuda -o paged_bench paged_bench.mu ==" | tee -a "$LOG"
  START=$(date +%s.%N)
  # shellcheck disable=SC2086
  mcc -O3 -std=c++17 $FLAGS_TRY -I../cuda -o paged_bench paged_bench.mu >> "$LOG" 2>&1
  RC=$?
  END=$(date +%s.%N)
  echo "rc=$RC 耗时 $(echo "$END $START" | awk '{printf "%.1f", $1-$2}')s" | tee -a "$LOG"
  if [ "$RC" -eq 0 ]; then
    FLAGS="$FLAGS_TRY"
    echo "== 成功，标志集: \"$FLAGS\" ==" | tee -a "$LOG"
    break
  fi
  tail -40 "$LOG" | sed 's/^/    | /'
done

if [ "$RC" -ne 0 ]; then
  echo "== 全部尝试失败，见 $LOG ==" | tee -a "$LOG"
  exit 1
fi

COMPILE_S=$(echo "$END $START" | awk '{printf "%.1f", $1-$2}')
echo "{\"compile_wall_seconds\": ${COMPILE_S}, \"flags\": \"${FLAGS}\"}" > compile-time.json
echo "== 构建完成（${COMPILE_S}s）==" | tee -a "$LOG"
