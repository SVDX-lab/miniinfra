#!/usr/bin/env bash
# 第 18 期 MUSA MMA 接口探测（大纲 §14 准入第 10 条："MUSA 侧待实机确认，缺失记平台边界"）。
# 三步：头文件 grep → 最小程序逐候选编译 → 结论 JSON。
# 产出（results/mma-probe/）：header-grep.txt、probe-results.txt、probe-verdict.json
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHAPTER_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR"
OUT="$CHAPTER_DIR/results/mma-probe"
mkdir -p "$OUT"

echo "== [1/3] /usr/local/musa/include 头文件符号 grep ==" | tee "$OUT/header-grep.txt"
echo "== 文件名层 ==" | tee -a "$OUT/header-grep.txt"
find /usr/local/musa/include \( -iname "*wmma*" -o -iname "*mma*" -o -iname "*tensor*" -o -iname "*fragment*" \) 2>/dev/null | tee -a "$OUT/header-grep.txt"
echo "== 符号层（wmma/mma_sync/fragment/matrix_a/mma 内建）==" | tee -a "$OUT/header-grep.txt"
grep -rlE "wmma|mma_sync|fragment|matrix_a" /usr/local/musa/include 2>/dev/null | head -20 | tee -a "$OUT/header-grep.txt"
echo "== 符号命中行（前 40 行）==" | tee -a "$OUT/header-grep.txt"
grep -rnE "\bwmma\b|mma_sync|fragment|matrix_a|__builtin_mma|mma_(m|u)[0-9]" /usr/local/musa/include 2>/dev/null | head -40 | tee -a "$OUT/header-grep.txt"

VERDICT="none_found"
RESULTS="$OUT/probe-results.txt"
: > "$RESULTS"

# ---- 候选 1：<mma.h> + nvcuda::wmma（MUSA 移植 CUDA 头的假设）----
cat > /tmp/ep19_probe1.mu << 'PROBE'
#include <musa_runtime.h>
#include <mma.h>
__global__ void probe1(const half* a, const half* b, float* c) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> fa;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> fb;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> fc;
    nvcuda::wmma::fill_fragment(fc, 0.0f);
    nvcuda::wmma::mma_sync(fc, fa, fb, fc);
    (void)a; (void)b; (void)c;
}
int main() { return 0; }
PROBE
echo "== [2/3] 候选1: <mma.h> + nvcuda::wmma ==" | tee -a "$RESULTS"
if mcc -O3 -std=c++17 -o /tmp/ep19_probe1 /tmp/ep19_probe1.mu >> "$RESULTS" 2>&1; then
  echo "PASS" | tee -a "$RESULTS"; VERDICT="mma_h_nvcuda"
else
  echo "FAIL（错误全文见上）" | tee -a "$RESULTS"; tail -6 "$RESULTS" | sed 's/^/    | /'
fi

# ---- 候选 2：<musa_wmma.h>（若存在）----
cat > /tmp/ep19_probe2.mu << 'PROBE'
#include <musa_runtime.h>
#include <musa_wmma.h>
__global__ void probe2(const half* a, const half* b, float* c) {
    musa::wmma::fragment<musa::wmma::matrix_a, 16, 16, 16, half, musa::wmma::row_major> fa;
    musa::wmma::fragment<musa::wmma::matrix_b, 16, 16, 16, half, musa::wmma::row_major> fb;
    musa::wmma::fragment<musa::wmma::accumulator, 16, 16, 16, float> fc;
    musa::wmma::fill_fragment(fc, 0.0f);
    musa::wmma::mma_sync(fc, fa, fb, fc);
    (void)a; (void)b; (void)c;
}
int main() { return 0; }
PROBE
echo "== [2/3] 候选2: <musa_wmma.h> + musa::wmma ==" | tee -a "$RESULTS"
if mcc -O3 -std=c++17 -o /tmp/ep19_probe2 /tmp/ep19_probe2.mu >> "$RESULTS" 2>&1; then
  echo "PASS" | tee -a "$RESULTS"; VERDICT="musa_wmma_h"
else
  echo "FAIL（错误全文见上）" | tee -a "$RESULTS"; tail -6 "$RESULTS" | sed 's/^/    | /'
fi

# ---- 候选 3：<mma.h> + musa::wmma（头存在但换命名空间）----
cat > /tmp/ep19_probe3.mu << 'PROBE'
#include <musa_runtime.h>
#include <mma.h>
__global__ void probe3(const half* a, const half* b, float* c) {
    musa::wmma::fragment<musa::wmma::matrix_a, 16, 16, 16, half, musa::wmma::row_major> fa;
    musa::wmma::fragment<musa::wmma::matrix_b, 16, 16, 16, half, musa::wmma::row_major> fb;
    musa::wmma::fragment<musa::wmma::accumulator, 16, 16, 16, float> fc;
    musa::wmma::fill_fragment(fc, 0.0f);
    musa::wmma::mma_sync(fc, fa, fb, fc);
    (void)a; (void)b; (void)c;
}
int main() { return 0; }
PROBE
echo "== [2/3] 候选3: <mma.h> + musa::wmma ==" | tee -a "$RESULTS"
if mcc -O3 -std=c++17 -o /tmp/ep19_probe3 /tmp/ep19_probe3.mu >> "$RESULTS" 2>&1; then
  echo "PASS" | tee -a "$RESULTS"; VERDICT="mma_h_musa"
else
  echo "FAIL（错误全文见上）" | tee -a "$RESULTS"; tail -6 "$RESULTS" | sed 's/^/    | /'
fi

# ---- 候选 4：<mma.h> + mtmusa::wmma + --offload-arch=mp_22（真机验证）----
cat > /tmp/ep19_probe4.mu << 'PROBE'
#include <musa_runtime.h>
#include <mma.h>
__global__ void probe4(const half* a, const half* b, float* c) {
    mtmusa::wmma::fragment<mtmusa::wmma::matrix_a, 16, 16, 16, half, mtmusa::wmma::row_major> fa;
    mtmusa::wmma::fragment<mtmusa::wmma::matrix_b, 16, 16, 16, half, mtmusa::wmma::row_major> fb;
    mtmusa::wmma::fragment<mtmusa::wmma::accumulator, 16, 16, 16, float> fc;
    mtmusa::wmma::fill_fragment(fc, 0.0f);
    mtmusa::wmma::load_matrix_sync(fa, a, 16);
    mtmusa::wmma::load_matrix_sync(fb, b, 16);
    mtmusa::wmma::mma_sync(fc, fa, fb, fc);
    mtmusa::wmma::store_matrix_sync(c, fc, 16, mtmusa::wmma::mem_row_major);
}
int main() { return 0; }
PROBE
echo "== [2/3] 候选4: <mma.h> + mtmusa::wmma + --offload-arch=mp_22 ==" | tee -a "$RESULTS"
if mcc --offload-arch=mp_22 -O3 -std=c++17 -o /tmp/ep19_probe4 /tmp/ep19_probe4.mu >> "$RESULTS" 2>&1; then
  echo "PASS" | tee -a "$RESULTS"; VERDICT="mma_h_mtmusa_offload_mp22"
else
  echo "FAIL（错误全文见上）" | tee -a "$RESULTS"; tail -6 "$RESULTS" | sed 's/^/    | /'
fi

# ---- 候选 4：driver 层 MMA 符号探测（libmusa / libmusa_driver 是否导出 MMA 相关 API）----
echo "== [2/3] 候选4: 运行库符号层 ==" | tee -a "$RESULTS"
for lib in /usr/local/musa/lib/libmusa.so /usr/local/musa/lib/libmusa_driver.so /usr/local/musa/lib/*.so; do
  [ -f "$lib" ] || continue
  hits=$(nm -D "$lib" 2>/dev/null | grep -iE "mma|wmma|tensor" | head -5)
  [ -n "$hits" ] && { echo "-- $lib:" | tee -a "$RESULTS"; echo "$hits" | tee -a "$RESULTS"; }
done | head -30

cat > "$OUT/probe-verdict.json" << EOF
{
  "schema_version": "kernel-chapter19.mma-probe.v1",
  "probe_date": "$(date -Iseconds)",
  "mcc": "$(mcc --version 2>&1 | tail -1)",
  "verdict": "$VERDICT",
  "candidates": {
    "mma_h_nvcuda": "$(grep -A1 '候选1' "$RESULTS" | grep -oE 'PASS|FAIL' | head -1)",
    "musa_wmma_h": "$(grep -A1 '候选2' "$RESULTS" | grep -oE 'PASS|FAIL' | head -1)",
    "mma_h_musa": "$(grep -A1 '候选3' "$RESULTS" | grep -oE 'PASS|FAIL' | head -1)",
    "mma_h_mtmusa_offload_mp22": "$(grep -A1 '候选4: <mma.h> + mtmusa' "$RESULTS" | grep -oE 'PASS|FAIL' | head -1)"
  },
  "note": "verdict=none_found 时，S4000 SDK 5.2.0 无公开用户态 WMMA/MMA fragment API（平台边界，最小复现见 probe-results.txt 与 header-grep.txt）"
}
EOF
echo "== [3/3] verdict: $VERDICT → $OUT/probe-verdict.json =="
cat "$OUT/probe-verdict.json"
