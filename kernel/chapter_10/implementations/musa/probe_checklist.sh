#!/usr/bin/env bash
# ============================================================================
# probe_checklist.sh —— MUSA 调试工具链一键探测清单（S4000 恢复后执行）。
#
# 背景：2026-11-21 S4000 实例被 AutoDL 停止（平台通报，恢复时间未知），
# 本期 MUSA 侧实测暂缓；本脚本把全部探测/实测命令固化为可一键重放的清单。
# 分级：
#   [只读] 不需要锁：ls/find/grep/nm/readelf——恢复后可立即跑；
#   [持锁] 编译+GPU 运行：mcc 编译、缺陷集/printf 运行——按锁协议持锁后跑。
# 产出：results/musa-tool-probe.txt（只读）+ results/musa-build-run.txt（持锁）
# 对照回填位：EXPERIMENT_LOG §7 / cross-backend-comparison §2-4 的 MUSA 列。
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-readonly}"   # readonly | locked
RES="$HERE/../../results"
mkdir -p "$RES"

if [ "$MODE" = "readonly" ]; then
  echo "##### [只读] MUSA 工具链盘点 $(date -u +%FT%TZ) #####"
  {
    echo; echo "== P0 基本盘 =="
    uname -a; mcc --version | head -3
    echo; echo "== P1 /usr/local/musa/bin 全清单（sanitizer/gdb 类有无的主证据）=="
    ls -la /usr/local/musa/bin 2>/dev/null
    echo; echo "== P2 全盘名字扫描（sanit|gdb|racecheck|memcheck|initcheck|synccheck|debug|lldb）=="
    find /usr/local/musa -maxdepth 3 \( -iname '*sanit*' -o -iname '*gdb*' -o -iname '*racecheck*' \
        -o -iname '*memcheck*' -o -iname '*initcheck*' -o -iname '*synccheck*' \
        -o -iname '*debug*' -o -iname '*lldb*' \) 2>/dev/null
    echo "(count: $(find /usr/local/musa -maxdepth 3 \( -iname '*sanit*' -o -iname '*gdb*' -o -iname '*racecheck*' -o -iname '*memcheck*' -o -iname '*initcheck*' -o -iname '*synccheck*' -o -iname '*debug*' -o -iname '*lldb*' \) 2>/dev/null | wc -l))"
    echo "-- 系统 PATH 上 gdb/lldb --"; command -v gdb lldb || true
    echo; echo "== P4 MUPTI debug 面（ep02 已探 1.3.0 无 CLI/Python；看 debug API）=="
    ls /usr/local/musa/include 2>/dev/null | grep -i mupti
    grep -rn -ilE 'debug|instrument' /usr/local/musa/include/mupti* 2>/dev/null | head
    echo; echo "== P5 printf FIFO limit 原型面 =="
    grep -rn -iE 'printf.*fifo|LimitPrintf|PRINTF_FIFO' /usr/local/musa/include/ 2>/dev/null | head -10
    grep -n 'musaDeviceSetLimit\|musaDeviceGetLimit' /usr/local/musa/include/musa_runtime_api.h 2>/dev/null | head -4
    echo; echo "== P6 libmusart 导出符号 sanit/debug 面 =="
    SO=$(find /usr/local/musa -name 'libmusart.so*' 2>/dev/null | head -1)
    [ -n "$SO" ] && { nm -D "$SO" 2>/dev/null | grep -icE 'sanit|debug'; nm -D "$SO" 2>/dev/null | grep -iE 'sanit|debug' | head -8; }
  } | tee "$RES/musa-tool-probe.txt"
  echo "[readonly 探测完成] -> $RES/musa-tool-probe.txt（无需锁，恢复后立即执行）"
  exit 0
fi

# ---------------- 持锁段（锁协议：mkdir 锁 + owner.json + GRANT 后执行）----------------
echo "##### [持锁] MUSA 编译与实测 $(date -u +%FT%TZ) #####"
{
  echo; echo "== P3 mcc -g 调试信息格式（.mu 设备编译模式；对照无 -g 版）=="
  MFLAGS="-O3 -std=c++17 --offload-arch=mp_22 -I$HERE"
  mcc -g $MFLAGS -DMUSA_HAS_PRINTF_FIFO_LIMIT=1 -DMUSA_PRINTF_FIFO_LIMIT_ENUM=musaLimitPrintfFifoSize \
      "$HERE/defect_suite_mu.mu" -o build/defect_suite_g 2>&1 \
    && { echo "mcc -g OK; debug sections:"; readelf -S build/defect_suite_g 2>/dev/null | grep -ic debug; readelf -S build/defect_suite_g 2>/dev/null | grep -i debug | head -6; } \
    || echo "mcc -g FAILED（报错即边界证据）"
  echo; echo "== M1 shim 复用编译（内核 0 改动验证；.mu 后缀=M-07）=="
  mcc -O3 -std=c++17 --offload-arch=mp_22 -I"$HERE" \
      -DMUSA_HAS_PRINTF_FIFO_LIMIT=1 -DMUSA_PRINTF_FIFO_LIMIT_ENUM=musaLimitPrintfFifoSize \
      "$HERE/printf_trap_mu.mu" -o "$HERE/build/printf_trap" 2>&1 \
    && echo "printf_trap BUILD OK" || echo "printf_trap BUILD FAILED"
  mcc -O3 -std=c++17 --offload-arch=mp_22 -I"$HERE" \
      -DMUSA_HAS_PRINTF_FIFO_LIMIT=1 -DMUSA_PRINTF_FIFO_LIMIT_ENUM=musaLimitPrintfFifoSize \
      "$HERE/defect_suite_mu.mu" -o "$HERE/build/defect_suite" 2>&1 \
    && echo "defect_suite BUILD OK" || echo "defect_suite BUILD FAILED"
  echo; echo "== M2 printf FIFO limit 枚举探测（P5 头文件已证 musaLimitPrintfFifoSize 存在，运行时验证）=="
  mcc -O3 -std=c++17 --offload-arch=mp_22 -DPROBE_CANDIDATE=1 "$HERE/musa_fifo_limit_probe.mu" -o "$HERE/build/fifo_probe" 2>&1 \
    && timeout 60 "$HERE/build/fifo_probe" 2>&1 | head -6 || echo "candidate 1 (musaLimitPrintfFifoSize) 编译或运行失败"
  echo; echo "== M3 运行（顺序/截断/残缺/代价 + 缺陷集）=="
  [ -x "$HERE/build/printf_trap" ] && {
    echo "-- order:"; timeout 120 "$HERE/build/printf_trap" --mode order 2>&1 | head -20
    echo "-- trunc default FIFO:"; timeout 300 "$HERE/build/printf_trap" --mode trunc --blocks 1024 --block 256 --per-thread 1 2>&1 | tee /tmp/ep11_musa_trunc.log | head -3
    echo "received=$(grep -c '^row ' /tmp/ep11_musa_trunc.log) emitted=262144"
    echo "-- trunc 16MB FIFO:"; timeout 300 "$HERE/build/printf_trap" --mode trunc --blocks 1024 --block 256 --per-thread 1 --fifo-mb 16 2>&1 | tee /tmp/ep11_musa_trunc16.log | head -3
    echo "received=$(grep -c '^row ' /tmp/ep11_musa_trunc16.log) emitted=262144"
    echo "-- frag:"; timeout 120 "$HERE/build/printf_trap" --mode frag 2>&1 | head -15
    echo "-- cost:"; timeout 300 "$HERE/build/printf_trap" --mode cost --blocks 32 --block 256 --iters 8 --warmup 3 --repeats 10 2>&1 | tail -3
  }
  [ -x "$HERE/build/defect_suite" ] && {
    for c in oob race uninit; do for v in good bad; do
      echo "-- $c/$v:"; timeout 120 "$HERE/build/defect_suite" --case $c --variant $v 2>&1 | head -3; echo "rc=$?"
    done; done
    timeout 60 "$HERE/build/defect_suite" --case sync --variant good; echo "sync/good rc=$?"
    timeout 30 "$HERE/build/defect_suite" --case sync --variant bad; echo "sync/bad rc=$?(124=死锁)"
  }
  echo; echo "== M4 sanitizer 类工具对缺陷集（若 P2 有发现）=="
  SAN_BIN=$(find /usr/local/musa -maxdepth 3 -iname '*sanit*' -type f 2>/dev/null | head -1)
  if [ -n "$SAN_BIN" ]; then
    timeout 300 "$SAN_BIN" --tool memcheck "$HERE/build/defect_suite" --case oob --variant bad 2>&1 | tail -8
  else
    echo "无 sanitizer 类工具（P2 count=0）——缺陷集在 MUSA 上无自动体检手段=平台边界实锤"
  fi
} | tee "$RES/musa-build-run.txt"
echo "[locked 实测完成] -> $RES/musa-build-run.txt（结果回填 EXPERIMENT_LOG §7 / 对照表 §2-4）"
