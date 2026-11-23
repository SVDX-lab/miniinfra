#!/usr/bin/env bash
# 第 18 期 MUSA 构建：WMMA 探测（头文件/命名空间/编译）+ 同源编译回退链
set -uo pipefail
cd "$(dirname "$0")"
LOG="build-log-musa.txt"
: > "$LOG"

echo "== mcc 版本 ==" | tee -a "$LOG"
mcc --version 2>&1 | tee -a "$LOG"

echo "== [probe/0] 头文件探测：__has_include 视角 ==" | tee -a "$LOG"
for hdr in musa_wmma.h mma.h musa_mma.h musa_mma.hpp wmma.h musa_tensor.h; do
  found=$(find /usr/local/musa/include -name "$hdr" 2>/dev/null | head -2)
  echo "$hdr -> ${found:-MISSING}" | tee -a "$LOG"
done

echo "== [attempt 1] 全量 WMMA（--offload-arch=mp_22 -DWMMA_NS=mtmusa；mp_22 目标激活 __MUSA_QY2_MMA__）==" | tee -a "$LOG"
START=$(date +%s.%N)
mcc --offload-arch=mp_22 -DWMMA_NS=mtmusa -O3 -std=c++17 -o naive_gemm_bench naive_gemm_wmma.mu >> "$LOG" 2>&1
RC=$?
END=$(date +%s.%N)
echo "rc=$RC 耗时 $(echo "$END $START" | awk '{printf "%.1f", $1-$2}')s" | tee -a "$LOG"
if [ "$RC" -eq 0 ]; then
  echo "== 成功：WMMA 在 MUSA 上可用（详见 grep wmma build-log）==" | tee -a "$LOG"
  echo "FULL_WITH_WMMA" > build-flags.txt
  echo "{\"compile_wall_seconds\": $(echo "$END $START" | awk '{printf "%.1f", $1-$2}'), \"mode\": \"with_wmma\"}" > compile-time.json
  grep -iE "wmma|mma_sync|fragment" "$LOG" | head -10 | tee -a "$LOG"
  exit 0
fi
tail -20 "$LOG" | sed 's/^/    | /'

echo "== [attempt 2] -DDISABLE_WMMA mp_21（朴素版回退）==" | tee -a "$LOG"
START=$(date +%s.%N)
mcc -O3 -std=c++17 -DDISABLE_WMMA -o naive_gemm_bench naive_gemm_wmma.mu >> "$LOG" 2>&1
RC=$?
END=$(date +%s.%N)
echo "rc=$RC 耗时 $(echo "$END $START" | awk '{printf "%.1f", $1-$2}')s" | tee -a "$LOG"
if [ "$RC" -eq 0 ]; then
  echo "DISABLE_WMMA" > build-flags.txt
  echo "{\"compile_wall_seconds\": $(echo "$END $START" | awk '{printf "%.1f", $1-$2}'), \"mode\": \"no_wmma\"}" > compile-time.json
  exit 0
fi
tail -20 "$LOG" | sed 's/^/    | /'

echo "== [attempt 3] -DDISABLE_WMMA -DDISABLE_FUNC_ATTRS ==" | tee -a "$LOG"
mcc -O3 -std=c++17 -DDISABLE_WMMA -DDISABLE_FUNC_ATTRS -o naive_gemm_bench naive_gemm_wmma.mu >> "$LOG" 2>&1
RC=$?
echo "rc=$RC" | tee -a "$LOG"
if [ "$RC" -eq 0 ]; then
  echo "DISABLE_WMMA,DISABLE_FUNC_ATTRS" > build-flags.txt
  exit 0
fi
echo "== 全部尝试失败，见 $LOG ==" | tee -a "$LOG"
exit 1
