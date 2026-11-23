#!/usr/bin/env bash
# 第 10 期四件套体检（sanitizer suite）——第 29 期 AI 三二进制版。
# 输入：已构建的 ai_rmsnorm / ai_gemm / ai_decode_attn
# 输出：results/sanitizer/{op}_{tool}.log + results/sanitizer-summary.json
#
# 口径（ch11 结论固化的预期）：
#   - memcheck：越界/未初始化访问报告（精确到 "bytes after nearest allocation"）；
#   - racecheck：shared memory 竞态报告；
#   - initcheck：未初始化读溯源；
#   - synccheck：CUDA13/sm_89 已知沉默（退出型被 exit-arrival 宽容、死锁型无
#     报告）——沉默记"无发现"，不记"无缺陷"。
# 每个工具跑 --selftest（覆盖 kernel 全部执行路径的最小入口；selftest 的
# 自身缺陷不影响 sanitizer 的内存/竞态观察）。

set -uo pipefail
cd "$(dirname "$0")/.."

OUT="results/sanitizer"
mkdir -p "$OUT"

which compute-sanitizer >/dev/null 2>&1 || {
  echo "[sanitizer] compute-sanitizer 不在 PATH——记录平台边界后退出"
  echo '{"error": "compute-sanitizer not found"}' > results/sanitizer-summary.json
  exit 1
}

declare -A BIN=(
  [rmsnorm]=implementations/ai/rmsnorm/ai_rmsnorm
  [gemm]=implementations/ai/gemm/ai_gemm
  [attention]=implementations/ai/attention/ai_decode_attn
)

SUMMARY='{"schema_version": "kernel-chapter30.sanitizer.v1", "runs": []'

for op in rmsnorm gemm attention; do
  for tool in memcheck racecheck initcheck synccheck; do
    echo "== ${op} / ${tool} =="
    LOG="$OUT/${op}_${tool}.log"
    timeout 900 compute-sanitizer --tool "$tool" --error-exitcode 99 \
      "${BIN[$op]}" --selftest > "$LOG" 2>&1
    RC=$?
    ERR=$(grep -cE "ERROR SUMMARY: [1-9][0-9]* errors" "$LOG" || true)
    RACE=$(grep -cE "ERROR SUMMARY: [1-9][0-9]* errors" "$LOG" || true)
    SUMMARY="$SUMMARY, {\"op\": \"$op\", \"tool\": \"$tool\", \"exit_code\": $RC, \"log\": \"$LOG\"}"
    echo "[sanitizer] ${op}/${tool}: exit=$RC log=$LOG"
  done
done

SUMMARY="$SUMMARY}"
echo "$SUMMARY" > results/sanitizer-summary.json
echo "[sanitizer] 摘要 results/sanitizer-summary.json（逐工具判定由 EXPERIMENT_LOG 记录）"
