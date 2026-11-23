#!/usr/bin/env bash
# ============================================================================
# build_and_run.sh —— 第 10 期 MUSA 列（持锁执行）。
#
# 步骤：
#   M1 编译期探测：mcc 对 shim 复用源（defect_suite_mu.cc / printf_trap_mu.cc）
#      直接编译——失败行即工具链边界证据；-g 与否两版（调试信息格式 readelf 看）；
#   M2 FIFO limit 枚举三候选探测（musa_fifo_limit_probe.cc，PROBE_CANDIDATE=1/2/3）；
#   M3 运行：能编译出的二进制全跑（order/trunc/frag/cost + 缺陷集 good/bad）；
#      sync bad 预期死锁 → timeout 30 截断；
#   M4 sanitizer 类工具存在则对缺陷集跑同口径（tool_probe.sh 的发现决定）。
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../results"
mkdir -p "$RES" "$HERE/build"
OUT="$RES/musa-build-run.txt"

{
echo "##### MUSA 列 build+run $(date -u +%FT%TZ) #####"

echo; echo "== M1a printf_trap_mu 编译（无 FIFO limit 映射，stub 路径）=="
mcc -O3 --offload-arch=mp_22 -DMUSA_HAS_PRINTF_FIFO_LIMIT=0 \
    -I"$HERE" printf_trap_mu.cc -o build/printf_trap 2>&1 \
  && echo "printf_trap BUILD OK" || echo "printf_trap BUILD FAILED（证据在上方报错）"

echo; echo "== M1b defect_suite_mu 编译 =="
mcc -O3 --offload-arch=mp_22 -DMUSA_HAS_PRINTF_FIFO_LIMIT=0 \
    -I"$HERE" defect_suite_mu.cc -o build/defect_suite 2>&1 \
  && echo "defect_suite BUILD OK" || echo "defect_suite BUILD FAILED（证据在上方报错）"

echo; echo "== M1c mcc -g 调试信息格式 =="
mcc -g -O1 --offload-arch=mp_22 -DMUSA_HAS_PRINTF_FIFO_LIMIT=0 \
    -I"$HERE" defect_suite_mu.cc -o build/defect_suite_g 2>&1 \
  && echo "defect_suite -g BUILD OK" || echo "-g BUILD FAILED"
for b in build/defect_suite build/defect_suite_g; do
  [ -f "$b" ] && { echo "-- $b debug sections:"; readelf -S "$b" 2>/dev/null | grep -ic debug; }
done

echo; echo "== M2 FIFO limit 枚举探测 =="
for cand in 1 2 3; do
  echo "-- candidate $cand:"
  mcc -DPROBE_CANDIDATE=$cand musa_fifo_limit_probe.cc -o build/fifo_probe_$cand 2>&1 \
    && echo "   compile OK" || echo "   compile FAILED"
done

echo; echo "== M3 运行 =="
if [ -x build/printf_trap ]; then
  for mode in order frag cost; do
    echo "-- printf $mode:"
    timeout 120 build/printf_trap --mode $mode 2>&1 | head -25
    echo "   (rc=$?)"
  done
  echo "-- printf trunc (blocks=1024 block=256 per_thread=1, 行数统计):"
  timeout 300 build/printf_trap --mode trunc --blocks 1024 --block 256 --per-thread 1 2>&1 | tee /tmp/ep11_musa_trunc.log | head -5
  echo "   received_lines=$(grep -c '^row ' /tmp/ep11_musa_trunc.log)"
fi
if [ -x build/defect_suite ]; then
  for c in oob race uninit; do
    for v in good bad; do
      echo "-- defect $c/$v:"
      timeout 120 build/defect_suite --case $c --variant $v 2>&1 | head -5
      echo "   (rc=$?)"
    done
  done
  echo "-- defect sync/good:"
  timeout 60 build/defect_suite --case sync --variant good 2>&1 | head -3
  echo "   (rc=$?)"
  echo "-- defect sync/bad（预期死锁）:"
  timeout 30 build/defect_suite --case sync --variant bad 2>&1 | head -3
  echo "   (rc=$?; 124=死锁被截断)"
fi

echo; echo "== M4 sanitizer 类工具复跑（若 tool_probe 发现）=="
SAN_BIN=$(find /usr/local/musa -maxdepth 3 -iname '*sanit*' -type f 2>/dev/null | head -1)
if [ -n "$SAN_BIN" ] && [ -x build/defect_suite ]; then
  echo "发现 $SAN_BIN，对 oob/bad 试跑："
  timeout 300 "$SAN_BIN" --tool memcheck build/defect_suite --case oob --variant bad 2>&1 | tail -8
else
  echo "未发现 sanitizer 类工具（tool_probe P2 计数=0 即平台边界证据）——缺陷集无法在 MUSA 上体检"
fi
} | tee "$OUT"
echo "[musa] 结果已写入 $OUT"
