#!/usr/bin/env bash
# nsys graph trace 取证（CUDA 列）：4090 上对本目录 nsys_decode_target.py 做 trace。
# 用法：在 4090 服务器 chapter_09/ 根目录执行
#   bash implementations/cuda/run_nsys_decode.sh [输出目录=results]
# 尝试 --graph-traces=cudaGraph（图内 kernel 实例证据）；不支持则回退纯 cuda,nvtx。
set -uo pipefail
cd "$(dirname "$0")/../.."   # chapter_09 根目录

OUT=${1:-results}
mkdir -p "$OUT"
NSYS=nsys
BASE_OPTS="--trace=cuda,nvtx --sample=none --cpuctxsw=none"

echo "[nsys] 版本：$($NSYS --version 2>&1 | head -1)"
echo "[nsys] graph trace 选项支持性探测："
$NSYS profile --help 2>&1 | grep -i "graph-traces" || echo "  --graph-traces 不存在，回退基础选项"

if $NSYS profile --help 2>&1 | grep -q "graph-traces"; then
  REPORT="$OUT/nsys-decode-graph"
  echo "[nsys] 启用 graph trace：$REPORT"
  $NSYS profile $BASE_OPTS --graph-traces=cudaGraph -o "$REPORT" \
    python3 implementations/cuda/nsys_decode_target.py
else
  REPORT="$OUT/nsys-decode-graph"
  echo "[nsys] 回退基础 trace：$REPORT"
  $NSYS profile $BASE_OPTS -o "$REPORT" \
    python3 implementations/cuda/nsys_decode_target.py
fi

echo "[nsys] stats 导出（kernsum/apisum/gputrace 摘要）"
$NSYS stats --report cuda_gpu_kern_sum --report cuda_api_sum \
  --report cuda_gpu_mem_time_sum "$REPORT.nsysrep" \
  --output "$OUT/nsys-decode-graph" --force-export true 2>&1 | tail -40 || true
ls -la "$OUT" | grep nsys || true
