#!/usr/bin/env bash
# ============================================================================
# tool_probe.sh —— 第 10 期 MUSA 列：调试工具链盘点（只读，不跑 GPU 实验）。
#
# 探测方向（大纲"待实机确认可用性"条目）：
#   P1 /usr/local/musa/bin 全清单：有无 sanitizer/debugger 类工具
#      （musa-gdb? musa-sanitizer? compute-sanitizer 对应物?）
#   P2 全盘 find：bin/lib 下名字含 sanit|gdb|racecheck|debug|check 的文件
#   P3 mcc 的调试信息支持：-g 编译 + 产物调试格式（DWARF?）+ llvm-dwarfdump
#   P4 MUPTI 头文件里有无 debug/ instrumentation 类 API（ep02 已探 1.3.0
#      存在但无 CLI/Python；本期看 debug 面）
#   P5 musa_runtime.h / musa_device_runtime_api.h 里 printf/limit 相关原型
#      （printf FIFO 大小是否有任何可查/可配入口）
#   P6 libmusart.so 导出符号里 sanit* / debug* 面
# 本脚本只读（ls/find/nm/grep/mcc --version/-c），可在持锁前跑；编译+运行
# 缺陷集在 build_and_run.sh（需持锁）。
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../results"
mkdir -p "$RES"
OUT="$RES/musa-tool-probe.txt"

{
echo "##### MUSA 调试工具链盘点 $(date -u +%FT%TZ) #####"
echo; echo "== P0 基本盘 =="
uname -a
cat /etc/os-release 2>/dev/null | head -2
ls -d /usr/local/musa && ls /usr/local/musa
command -v mcc && mcc --version | head -3

echo; echo "== P1 /usr/local/musa/bin 全清单 =="
ls -la /usr/local/musa/bin 2>/dev/null

echo; echo "== P2 全盘找 sanitizer/debugger 类名字 =="
find /usr/local/musa -maxdepth 3 \( -iname '*sanit*' -o -iname '*gdb*' -o -iname '*racecheck*' \
    -o -iname '*memcheck*' -o -iname '*initcheck*' -o -iname '*synccheck*' -o -iname '*debug*' \
    -o -iname '*lldb*' \) 2>/dev/null | head -50
echo "(count: $(find /usr/local/musa -maxdepth 3 \( -iname '*sanit*' -o -iname '*gdb*' -o -iname '*racecheck*' -o -iname '*memcheck*' -o -iname '*initcheck*' -o -iname '*synccheck*' -o -iname '*debug*' -o -iname '*lldb*' \) 2>/dev/null | wc -l))"
echo "-- PATH 上（/usr/bin /usr/local/bin）--"
ls /usr/bin /usr/local/bin 2>/dev/null | grep -iE 'sanit|gdb|lldb|musa|mthreads' | sort -u | head -40
echo "-- gdb 本体是否在系统 --"
command -v gdb || echo "(no gdb)"

echo; echo "== P3 mcc -g 调试信息 =="
cat > /tmp/ep11_dbg_probe.cc <<'EOF'
__global__ void tiny(int* o) { *o = 42; }
int main() { return 0; }
EOF
mcc -g -c /tmp/ep11_dbg_probe.cc -o /tmp/ep11_dbg_probe.o 2>&1 && {
  echo "mcc -g -c OK; debug sections:"
  readelf -S /tmp/ep11_dbg_probe.o 2>/dev/null | grep -i debug
  command -v llvm-dwarfdump && llvm-dwarfdump --version 2>/dev/null | head -2
  command -v llvm-dwarfdump && llvm-dwarfdump --debug-info /tmp/ep11_dbg_probe.o 2>/dev/null | head -15
} || echo "mcc -g FAILED（记录报错）"
ls /usr/local/musa/bin | grep -i llvm | head -10

echo; echo "== P4 MUPTI debug 面 =="
ls /usr/local/musa/include 2>/dev/null | grep -i mupti
grep -l -iE 'debug|instrument' /usr/local/musa/include/mupti* 2>/dev/null
grep -iE 'MUPTI_ERROR_DEBUG|debug' /usr/local/musa/include/mupti_result.h 2>/dev/null | head -5

echo; echo "== P5 printf/FIFO limit 原型 =="
grep -rn -iE 'printf.*fifo|fifo.*printf|LimitPrintf|PRINTF_FIFO' /usr/local/musa/include/ 2>/dev/null | head -10
grep -n 'musaDeviceSetLimit\|musaDeviceGetLimit' /usr/local/musa/include/musa_runtime_api.h 2>/dev/null | head -5
grep -n -A12 'typedef enum.*musaLimit\b\|enum musaLimit' /usr/local/musa/include/musa_runtime_api.h 2>/dev/null | head -30

echo; echo "== P6 libmusart 导出符号 debug/sanit 面 =="
find /usr/local/musa -name 'libmusart.so*' 2>/dev/null | head -2
SO=$(find /usr/local/musa -name 'libmusart.so*' 2>/dev/null | head -1)
[ -n "$SO" ] && nm -D "$SO" 2>/dev/null | grep -icE 'sanit|debug' || true
[ -n "$SO" ] && nm -D "$SO" 2>/dev/null | grep -iE 'sanit|debug' | head -10

echo; echo "== P7 文档/工具杂项 =="
ls /usr/local/musa/doc* 2>/dev/null | head
ls /usr/local/musa/tools 2>/dev/null | head
} | tee "$OUT"

echo "[probe] 结果已写入 $OUT"
