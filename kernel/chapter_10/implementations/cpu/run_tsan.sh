#!/usr/bin/env bash
# ============================================================================
# run_tsan.sh —— CPU 列 ThreadSanitizer 实测（本地 WSL2 即可，无需 GPU）。
#
# 四个口径：
#   A 裸跑（无 TSAN）：坏版丢更新但不崩溃（退出码 2）——竞态的"静默损坏"证据；
#   B TSAN 跑坏版：报告 data race + 关键行（留证 server-logs/tsan-*.log）；
#   C TSAN 跑修复版：0 报告（mutex = happens-before 边）；
#   D TSAN 关闭时同样跑修复版：结果正确。
# 记录：TSAN 版本即 gcc 版本；-O2 优化口径（竞态在优化后仍必须被检出）。
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$HERE/../../results"
mkdir -p "$LOGDIR"

echo "== gcc $(gcc -dumpversion), tsan build -O1 -g -fsanitize=thread =="

gcc -O1 -g -fsanitize=thread -pthread race_counter_bad.c   -o tsan_bad
gcc -O1 -g -fsanitize=thread -pthread race_counter_fixed.c -o tsan_fixed
gcc -O2 -pthread race_counter_bad.c   -o plain_bad
gcc -O2 -pthread race_counter_fixed.c -o plain_fixed

echo "== [A] plain bad (no sanitizer, silent corruption expected) =="

./plain_bad | tee "$LOGDIR/tsan-A-plain-bad.out"
echo "exit=${PIPESTATUS[0]}"

echo "== [B] tsan bad (data race report expected) =="
./tsan_bad >"$LOGDIR/tsan-B-bad.stdout" 2>"$LOGDIR/tsan-B-bad.report"
rc=$?
echo "exit=$rc stdout=$(cat "$LOGDIR/tsan-B-bad.stdout")"
echo "report-lines=$(wc -l <"$LOGDIR/tsan-B-bad.report")"
grep -m1 "SUMMARY: ThreadSanitizer" "$LOGDIR/tsan-B-bad.report" || echo "(no SUMMARY)"

echo "== [C] tsan fixed (clean expected) =="
./tsan_fixed >"$LOGDIR/tsan-C-fixed.stdout" 2>"$LOGDIR/tsan-C-fixed.report"
echo "exit=$? stdout=$(cat "$LOGDIR/tsan-C-fixed.stdout") report-lines=$(wc -l <"$LOGDIR/tsan-C-fixed.report")"

echo "== [D] plain fixed =="
./plain_fixed | tee "$LOGDIR/tsan-D-plain-fixed.out"
echo "exit=${PIPESTATUS[0]}"

echo "== [E] tsan bad 报告关键行摘录（前 40 行）=="
head -40 "$LOGDIR/tsan-B-bad.report" | tee "$LOGDIR/tsan-E-keylines.log"
