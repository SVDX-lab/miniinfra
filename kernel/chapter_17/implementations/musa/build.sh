#!/usr/bin/env bash
# 第 17 期 MUSA 构建（MTT S4000，MUSA SDK 5.2.0 / mcc 5.2.0）
#
# mcc 实机能力未知项（大纲标注"以实机为准"），构建走自动回退链并全程记日志：
#   1. 默认参数（依赖 mcc 默认目标架构）
#   2. 架构标志变体（-march=mp_22 / --march=mp_22，CC 2.2 猜测，实机核对）
#   3. FP16 头文件能力回退（-DMUSA_IGNORE_FP16_H：跳过 musa_fp16.h 走位级转换）
#   4. half 类型名变体（-DMUSA_HALF_TYPE_NAME=half）
#   5. 关闭 func attrs 查询（musaFuncGetAttributes 不存在时）
#   6. 最终回退：-DDISABLE_FP16（仅 FP32 主线，FP16 记平台边界）
# 每次尝试的结果写 build-log-musa.txt——迁移台账的直接素材。
set -uo pipefail
cd "$(dirname "$0")"

LOG="build-log-musa.txt"
: > "$LOG"

echo "== mcc 版本 ==" | tee -a "$LOG"
mcc --version 2>&1 | tee -a "$LOG"

FLAGS_LIST=(
  ""
  "-march=mp_22"
  "--march=mp_22"
  "-DMUSA_IGNORE_FP16_H"
  "-DMUSA_IGNORE_FP16_H -DMUSA_HALF_TYPE_NAME=half"
  "-DDISABLE_FUNC_ATTRS"
  "-DDISABLE_FUNC_ATTRS -DMUSA_IGNORE_FP16_H"
  "-DDISABLE_FUNC_ATTRS -DDISABLE_FP16"
)

RC=1
for FLAGS in "${FLAGS_LIST[@]}"; do
  echo "" | tee -a "$LOG"
  echo "== 尝试: mcc -O3 -std=c++17 $FLAGS -o naive_gemm_bench naive_gemm.mu ==" | tee -a "$LOG"
  START=$(date +%s.%N)
  # shellcheck disable=SC2086
  mcc -O3 -std=c++17 $FLAGS -o naive_gemm_bench naive_gemm.mu >> "$LOG" 2>&1
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

# 编译时间记录（§10.1 指标：编译时间与首次运行时间）
COMPILE_S=$(echo "$END $START" | awk '{printf "%.1f", $1-$2}')
echo "{\"compile_wall_seconds\": ${COMPILE_S}, \"flags\": \"${FLAGS}\"}" > compile-time.json

echo "== 首次运行时间（attrs 模式，即冷启动+查询）=="
ATTR_START=$(date +%s.%N)
./naive_gemm_bench attrs > first-run-attrs.json 2> first-run-attrs.err
ATTR_RC=$?
ATTR_END=$(date +%s.%N)
echo "{\"first_run_seconds\": $(echo "$ATTR_END $ATTR_START" | awk '{printf "%.2f", $1-$2}'), \"rc\": $ATTR_RC}" > first-run.json
head -c 400 first-run-attrs.json; echo
exit 0
