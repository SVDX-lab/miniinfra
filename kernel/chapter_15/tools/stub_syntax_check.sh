#!/usr/bin/env bash
# 第 15 期 C++ stub 语法检查：把 <<<...>>> 启动语法替换为 stub_launch 调用后
# 用 g++ -fsyntax-only 全文件类型检查（本地无 nvcc/mcc 的 QA 手段）。
set -uo pipefail
cd "$(dirname "$0")"

SRC="${1:-../implementations/cuda/int8_gemm.cu}"
OUT=$(mktemp /tmp/ep16_stub_XXXX.cpp)

perl -0777 -pe '
  # kernel-name<<<grid, block>>>(args);  →  stub_launch(kernel, grid, block)(args);
  s/([A-Za-z_][A-Za-z0-9_]*(?:<[^<>]*>)?)\s*<<<\s*([^<>]*?)\s*>>>\s*([^;]*?);/stub_launch($1, $2)$3;/gs;
' "$SRC" > "$OUT"

HITS=$(grep -c "stub_launch(" "$OUT")
LEFT=$(grep -c "<<<" "$OUT" || true)
echo "stub_launch 替换数: $HITS；残留 <<< : $LEFT"
if [ "$LEFT" -ne 0 ]; then
  echo "存在未替换的 launch 语法，检查失败"
  exit 1
fi

g++ -std=c++17 -fsyntax-only -Wall -Wno-unknown-pragmas -I stub_include "$OUT"
RC=$?
if [ "$RC" -eq 0 ]; then
  echo "STUB SYNTAX CHECK PASS（$HITS 个 launch 全部替换并类型通过）"
else
  echo "STUB SYNTAX CHECK FAIL rc=$RC（见上方编译器输出）"
fi
rm -f "$OUT"
exit "$RC"
