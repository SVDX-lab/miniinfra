#!/usr/bin/env bash
# 第 19 期 MUSA 异步拷贝机制探测（大纲 §14 准入第 10 条"待实机确认"的实机答案）
#
# 探测矩阵（P0 文档级 → P1 寻址 → P2 4B 指令 → P3 16B 指令 → P4 intrinsics）：
#   P0 头文件/符号 grep：memcpy_async / pipeline / cp.async / cvta 在 SDK 头
#      与 mcc 内部表的出现情况（文档级证据）；
#   P1 __cvta_generic_to_shared 编译+运行（shared 地址换算，cp.async 前置）；
#   P2 cp.async.ca.shared.global 4B 内联 PTX 编译+运行+数据校验（cpasync1/2 的
#      最低要求——通过则 build.sh 启用 -DHAVE_MUSA_CP_ASYNC）；
#   P3 cp.async.cg 16B（v16 向量变体的要求；主链不依赖）；
#   P4 pipeline intrinsics 头（__pipeline_memcpy_async 家族）。
# 结论写 probe-result.json（build.sh 消费）；全部日志与报错全文留档——
# 编不过的报错文本本身就是平台边界证据（协议 §6）。
set -uo pipefail
cd "$(dirname "$0")"

LOG="probe-log.txt"
: > "$LOG"
ARCH_FLAGS="${MUSA_PROBE_ARCH:---offload-arch=mp_22}"

log() { echo "$@" | tee -a "$LOG"; }

log "== MUSA 异步拷贝机制探测 $(date -u +%FT%TZ) =="
log "== P0a: mcc 版本 =="
mcc --version 2>&1 | tee -a "$LOG"

log "== P0b: SDK 头文件符号 grep（memcpy_async/pipeline/cp.async/cvta）=="
{
  grep -rniE "memcpy_async|pipeline|cp\.async|cvta" /usr/local/musa/include 2>/dev/null | head -60
  echo "--- grep 命中数 ---"
  grep -rnilE "memcpy_async" /usr/local/musa/include 2>/dev/null | wc -l
  grep -rnilE "cp\.async" /usr/local/musa/include 2>/dev/null | wc -l
  grep -rnilE "cvta_generic_to_shared" /usr/local/musa/include 2>/dev/null | wc -l
} >> "$LOG" 2>&1
grep -rniE "memcpy_async|cp\.async|cvta" /usr/local/musa/include 2>/dev/null | head -40 | tee -a "$LOG"

log "== P0c: mcc 内部表（ISA/装配器）中 cp.async 出现情况 =="
{
  grep -rli "cp\.async" /usr/local/musa/bin /usr/local/musa/lib 2>/dev/null | head -10
  echo "---"
  ls /usr/local/musa/bin 2>/dev/null | head -30
} >> "$LOG" 2>&1
grep -rli "cp\.async" /usr/local/musa/bin /usr/local/musa/lib 2>/dev/null | head -10 | tee -a "$LOG"

probe_compile_run() {
  local probe_name="$1" probe_file="$2"
  log "== $probe_name: mcc -O3 -std=c++17 $ARCH_FLAGS -c+run $probe_file =="
  local start end rc
  start=$(date +%s.%N)
  # shellcheck disable=SC2086
  mcc -O3 -std=c++17 $ARCH_FLAGS -o "${probe_file%.mu}" "$probe_file" >> "$LOG" 2>&1
  rc=$?
  end=$(date +%s.%N)
  log "$probe_name 编译 rc=$rc 耗时 $(echo "$end $start" | awk '{printf "%.1f", $1-$2}')s"
  if [ "$rc" -ne 0 ]; then
    log "$probe_name 编译报错全文（平台边界证据）："
    mcc -O3 -std=c++17 $ARCH_FLAGS -o "${probe_file%.mu}" "$probe_file" 2>&1 | head -25 | sed 's/^/    | /' | tee -a "$LOG"
    return 1
  fi
  ./"${probe_file%.mu}" >> "$LOG" 2>&1
  rc=$?
  tail -3 "$LOG" | sed 's/^/    | /' | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then
    log "$probe_name 运行 rc=$rc（编译过但执行/数据校验失败）"
    return 2
  fi
  log "$probe_name 全链通过"
  return 0
}

P1=FAIL; P2=FAIL; P3=FAIL; P4=FAIL
probe_compile_run P1 probe_cvta.mu && P1=PASS || true
probe_compile_run P2 probe_cpasync4.mu && P2=PASS || true
probe_compile_run P3 probe_cpasync16.mu && P3=PASS || true
probe_compile_run P4 probe_intrinsic.mu && P4=PASS || true

CONCLUSION="cp.async 内联 PTX 不可用且无 pipeline intrinsics：平台边界（仅同步 load 迁移）"
BUILD_DECISION="sync-only"
if [ "$P2" = "PASS" ]; then
  CONCLUSION="cp.async PTX 全链可用（全部异步级可迁移）"
  BUILD_DECISION="-DHAVE_MUSA_CP_ASYNC"
elif [ "$P4" = "PASS" ]; then
  CONCLUSION="cp.async 内联 PTX 被 mcc 后端拒绝（inline asm 不支持），但 <musa_pipeline.h> __pipeline_memcpy_async intrinsics 编译+运行全链可用 → MUSA 的 cp.async 对应物存在（intrinsics 形态），异步级经 intrinsics 迁移"
  BUILD_DECISION="-DHAVE_MUSA_CP_ASYNC"
fi

cat > probe-result.json <<EOF
{
  "probe_time_utc": "$(date -u +%FT%TZ)",
  "arch_flags": "$ARCH_FLAGS",
  "P0_header_grep": "见 probe-log.txt",
  "P1_cvta_generic_to_shared": "$P1",
  "P2_cp_async_ca_4B": "$P2",
  "P3_cp_async_cg_16B": "$P3",
  "P4_pipeline_intrinsics": "$P4",
  "conclusion": "$CONCLUSION",
  "build_decision": "$BUILD_DECISION"
}
EOF
cat probe-result.json | tee -a "$LOG"
log "== 探测结论：$CONCLUSION =="
[ "$BUILD_DECISION" = "-DHAVE_MUSA_CP_ASYNC" ] && exit 0 || exit 1
