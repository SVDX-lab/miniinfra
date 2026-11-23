#!/usr/bin/env bash
# 第 19 期 MUSA 构建（MTT S4000，MUSA SDK 5.2.0 / mcc）
#
# 构建决策由探测驱动（async_probe/probe.sh 的 probe-result.json）：
#   - P2（cp.async.ca 4B 全链）通过 → 先试 -DHAVE_MUSA_CP_ASYNC（全部异步级）；
#   - 失败 → 直接编 sync-only 二进制（cp.async 级记平台边界，报错全文在探测日志）。
# 回退链（每步结果记 build-log-musa.txt，台账素材）：
#   1. -DHAVE_MUSA_CP_ASYNC（默认目标）
#   2. -DHAVE_MUSA_CP_ASYNC --offload-arch=mp_22（M-18/M-21/M-22 的有效旗标）
#   3. （无 cp.async）默认目标
#   4. （无 cp.async）--offload-arch=mp_22
set -uo pipefail
cd "$(dirname "$0")"

LOG="build-log-musa.txt"
: > "$LOG"

echo "== mcc 版本 ==" | tee -a "$LOG"
mcc --version 2>&1 | tee -a "$LOG"

# ---- 探测先行（探测结果决定主链是否带 cp.async）----
echo "== 探测驱动：运行 async_probe/probe.sh ==" | tee -a "$LOG"
if [ ! -f async_probe/probe-result.json ]; then
  (cd async_probe && bash probe.sh) 2>&1 | tail -20 | tee -a "$LOG"
fi
HAVE_CP_ASYNC=0
if [ -f async_probe/probe-result.json ]; then
  grep -q '"build_decision": "-DHAVE_MUSA_CP_ASYNC"' async_probe/probe-result.json && HAVE_CP_ASYNC=1
fi
echo "HAVE_CP_ASYNC=$HAVE_CP_ASYNC（来自 probe-result.json）" | tee -a "$LOG"

if [ "$HAVE_CP_ASYNC" = "1" ]; then
  FLAGS_LIST=(
    "-DHAVE_MUSA_CP_ASYNC"
    "-DHAVE_MUSA_CP_ASYNC --offload-arch=mp_22"
    ""
    "--offload-arch=mp_22"
  )
else
  FLAGS_LIST=(
    ""
    "--offload-arch=mp_22"
  )
fi

RC=1
for FLAGS in "${FLAGS_LIST[@]}"; do
  echo "" | tee -a "$LOG"
  echo "== 尝试: mcc -O3 -std=c++17 $FLAGS -o async_gemm_bench naive_gemm_async.mu ==" | tee -a "$LOG"
  START=$(date +%s.%N)
  # shellcheck disable=SC2086
  mcc -O3 -std=c++17 $FLAGS -o async_gemm_bench naive_gemm_async.mu >> "$LOG" 2>&1
  RC=$?
  END=$(date +%s.%N)
  echo "rc=$RC 耗时 $(echo "$END $START" | awk '{printf "%.1f", $1-$2}')s" | tee -a "$LOG"
  if [ "$RC" -eq 0 ]; then
    echo "== 成功，标志集: \"$FLAGS\" ==" | tee -a "$LOG"
    echo "$FLAGS" > build-flags.txt
    break
  fi
  tail -30 "$LOG" | sed 's/^/    | /' | tee -a "$LOG"
done

if [ "$RC" -ne 0 ]; then
  echo "== 全部尝试失败，见 $LOG ==" | tee -a "$LOG"
  exit 1
fi

COMPILE_S=$(echo "$END $START" | awk '{printf "%.1f", $1-$2}')
echo "{\"compile_wall_seconds\": ${COMPILE_S}, \"flags\": \"${FLAGS}\", \"have_cp_async\": ${HAVE_CP_ASYNC}}" > compile-time.json

echo "== 首次运行（attrs 模式，含 cp_async_enabled 字段）=="
ATTR_START=$(date +%s.%N)
./async_gemm_bench attrs > first-run-attrs.json 2> first-run-attrs.err
ATTR_RC=$?
ATTR_END=$(date +%s.%N)
echo "{\"first_run_seconds\": $(echo "$ATTR_END $ATTR_START" | awk '{printf "%.2f", $1-$2}'), \"rc\": $ATTR_RC}" > first-run.json
head -c 400 first-run-attrs.json; echo
exit 0
