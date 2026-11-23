#!/usr/bin/env bash
# 第 20 期 MUSA 构建（MTT S4000，MUSA SDK 5.2.0 / mcc 5.2.0，ch18 回退链模式）
#
# 源码：attention_bench.mu（standalone 二进制，与 CUDA 列共用
# ../cuda/attention_kernels.cuh——"机械替换单点化"迁移设计）。
# 回退链（以实机为准，逐次记 build-log-musa.txt）：
#   1. 默认参数（mcc 默认目标 mp_21，ch18 实测）
#   2. -DMUSA_IGNORE_FP16_H（musa_fp16.h 不可用时——ch18 实测存在，防御性保留）
# 注意：核代码 warpSize-generic（WARP=32/128 双实例 + 设备探针运行期选择，
# M-11/M-20 双口径），无 TILE%warpSize 类 host guard 需求。
set -uo pipefail
cd "$(dirname "$0")"

LOG="build-log-musa.txt"
: > "$LOG"

echo "== mcc 版本 ==" | tee -a "$LOG"
mcc --version 2>&1 | tee -a "$LOG"

FLAGS_LIST=(
  ""
  "-DMUSA_IGNORE_FP16_H"
  "-DDISABLE_FUNC_ATTRS"
)

RC=1
for FLAGS in "${FLAGS_LIST[@]}"; do
  echo "" | tee -a "$LOG"
  echo "== 尝试: mcc -O3 -std=c++17 $FLAGS -I../cuda -o attention_bench attention_bench.mu ==" | tee -a "$LOG"
  START=$(date +%s.%N)
  # shellcheck disable=SC2086
  mcc -O3 -std=c++17 $FLAGS -I../cuda -o attention_bench attention_bench.mu >> "$LOG" 2>&1
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
./attention_bench attrs | tee -a "$LOG"
./attention_bench selftest | tee -a "$LOG"
