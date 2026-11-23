#!/usr/bin/env bash
# 第 15 期 dequant 物化三变体诊断（v1 size_t 取模 / v2 2D 网格无取模 / v3 32 位取模）
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=implementations/musa/int8_gemm_bench
for V in dequant dequant2 dequant3; do
  for KN in "1024 1024" "4096 1024" "4096 4096"; do
    set -- $KN
    OUT=$("$BIN" bench --m 1 --k "$1" --n "$2" --kernel "$V" --tile 16x16x16 --reg 2x2 \
        --warmup 10 --repeats 50 --seed 20261121 2>/dev/null | tail -1)
    echo "$OUT" | python3 -c '
import json, sys
e = json.loads(sys.stdin.read())
print("%-9s %5dx%-5d p50=%.4fms bw=%.1fGB/s" % (e["kernel"], e["k"], e["n"], e["p50_ms"], e["effective_gb_s_p50"]))
'
  done
done
