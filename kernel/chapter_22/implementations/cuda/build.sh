#!/usr/bin/env bash
# 第 22 期 CUDA 构建（RTX 4090，CUDA 13.0 / nvcc）——【待 4090 修复后执行】
#
# 源码：rope_bench.cu（入口 shim）+ rope_kernels.cuh（核）+ rope_bench_main.cuh
# （入口体）——与 MUSA 列共用，仅平台 shim 块机械替换差异。
set -uo pipefail
cd "$(dirname "$0")"

LOG="build-log-cuda.txt"
: > "$LOG"

echo "== nvcc 版本 ==" | tee -a "$LOG"
nvcc --version 2>&1 | tee -a "$LOG"

FLAGS_LIST=(
  "-gencode arch=compute_89,code=sm_89"
  ""
)

RC=1
for FLAGS in "${FLAGS_LIST[@]}"; do
  echo "" | tee -a "$LOG"
  echo "== 尝试: nvcc -O3 -std=c++17 $FLAGS -I. -o rope_bench rope_bench.cu ==" | tee -a "$LOG"
  START=$(date +%s.%N)
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 $FLAGS -I. -o rope_bench rope_bench.cu >> "$LOG" 2>&1
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
./rope_bench attrs cuda | tee -a "$LOG"
./rope_bench selftest cuda | tee -a "$LOG"
