#!/usr/bin/env bash
# 第 15 期 MUSA 构建（MTT S4000，MUSA SDK 5.2.0 / mcc 5.2.0）
#
# 构建回退链（ch18 同款风格；ch18 实证默认参数即可过，回退链只留最小集合）：
#   1. 默认参数（ch18 实证：mcc 默认目标 mp_21 自动选择）
#   2. -march=mp_21（显式声明；ch18 实测 mp_22 报 unknown target CPU）
#   3. -DDISABLE_FUNC_ATTRS（musaFuncGetAttributes 缺失时）
# 每次尝试结果写 build-log-musa.txt（台账素材）。
set -uo pipefail
cd "$(dirname "$0")"

LOG="build-log-musa.txt"
: > "$LOG"

echo "== mcc 版本 ==" | tee -a "$LOG"
mcc --version 2>&1 | tee -a "$LOG"

FLAGS_LIST=(
  ""
  "-march=mp_21"
  "-DDISABLE_FUNC_ATTRS"
)

RC=1
for FLAGS in "${FLAGS_LIST[@]}"; do
  echo "" | tee -a "$LOG"
  echo "== 尝试: mcc -O3 -std=c++17 $FLAGS -o int8_gemm_bench int8_gemm.mu ==" | tee -a "$LOG"
  START=$(date +%s.%N)
  # shellcheck disable=SC2086
  mcc -O3 -std=c++17 $FLAGS -o int8_gemm_bench int8_gemm.mu >> "$LOG" 2>&1
  RC=$?
  END=$(date +%s.%N)
  echo "rc=$RC 耗时 $(echo "$END $START" | awk '{printf "%.1f", $1-$2}')s" | tee -a "$LOG"
  if [ "$RC" -eq 0 ]; then
    echo "== 成功，标志集: \"$FLAGS\" ==" | tee -a "$LOG"
    echo "$FLAGS" > build-flags.txt
    break
  fi
  tail -25 "$LOG" | sed 's/^/    | /'
done

if [ "$RC" -ne 0 ]; then
  echo "== 全部尝试失败，见 $LOG ==" | tee -a "$LOG"
  exit 1
fi

# 编译时间记录（§10.1：编译时间与首次运行时间）
COMPILE_S=$(echo "$END $START" | awk '{printf "%.1f", $1-$2}')
echo "{\"compile_wall_seconds\": ${COMPILE_S}, \"flags\": \"${FLAGS}\"}" > compile-time.json

echo "== 首次运行时间（attrs 模式，即冷启动+查询）=="
ATTR_START=$(date +%s.%N)
./int8_gemm_bench attrs > first-run-attrs.json 2> first-run-attrs.err
ATTR_RC=$?
ATTR_END=$(date +%s.%N)
echo "{\"first_run_seconds\": $(echo "$ATTR_END $ATTR_START" | awk '{printf "%.2f", $1-$2}'), \"rc\": $ATTR_RC}" > first-run.json
head -c 400 first-run-attrs.json; echo
exit 0
