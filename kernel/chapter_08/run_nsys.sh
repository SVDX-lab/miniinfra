#!/usr/bin/env bash
# ============================================================================
# 第 08 期 Nsight Systems 时间线证据采集（CUDA 4090；ncu 计数器权限受限，nsys 可用）。
# 用法：bash run_nsys.sh   （在 chapter_08/ 目录执行）
# 产出（results/ 下）：
#   nsys-overlap.nsys-rep          时间线（大文件不回传仓库，§11 约定）
#   nsys-overlap-gpu-trace.csv     逐 GPU op 明细（check_overlap_trace.py 的输入）
#   nsys-overlap-kernsum.txt       GPU kernel 汇总
#   nsys-overlap-apisum.txt        CUDA API 汇总（MemcpyAsync/EventRecord 次数证据）
#   nsys-overlap-memops-sum.txt    Memcpy 汇总（带宽对账）
#   overlap-trace-cuda.json        并行区间证据（check_overlap_trace.py 产出）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results/server-logs

BACKEND="${BACKEND:-cuda}"
LIB="implementations/$BACKEND/build/libtransfer_ops_${BACKEND}.so"

echo "=== [1/3] nsys profile（串行 vs 重叠两相位，payload 64MB chunk 8MB K=8） ==="
nsys profile -o results/nsys-overlap --force-overwrite true --trace=cuda \
    python3 bench_overlap.py --backend "$BACKEND" --lib "$LIB" \
        --payload-mb 64 --chunk-mb 8 --out results/nsys-target.json --nsys-mode \
    > results/server-logs/nsys-profile.log 2>&1
echo "profile exit=$?（日志 results/server-logs/nsys-profile.log）"

echo "=== [2/3] nsys stats 提取 ==="
nsys stats --report cuda_gpu_trace --format csv --output results/nsys-overlap-gpu-trace \
    results/nsys-overlap.nsys-rep > results/server-logs/nsys-stats.log 2>&1
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report gpu_mem_time_sum \
    results/nsys-overlap.nsys-rep > results/nsys-overlap-stats.txt 2>&1
awk '/^cuda_gpu_kern_sum/{f=1} /^cuda_api_sum/{f=0} f' results/nsys-overlap-stats.txt > results/nsys-overlap-kernsum.txt
awk '/^cuda_api_sum/{f=1} /^gpu_mem_time_sum/{f=0} f' results/nsys-overlap-stats.txt > results/nsys-overlap-apisum.txt
awk '/^gpu_mem_time_sum/{f=1} f' results/nsys-overlap-stats.txt > results/nsys-overlap-memops-sum.txt
ls results/nsys-overlap-gpu-trace*.csv 2>/dev/null || echo "警告：gpu trace csv 未生成（见 nsys-stats.log）"

echo "=== [3/3] 并行区间证据解析 ==="
CSV=$(ls results/nsys-overlap-gpu-trace*.csv 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
    python3 check_overlap_trace.py --trace-csv "$CSV" --backend "$BACKEND" --out results/overlap-trace-"$BACKEND".json
else
    echo "跳过解析（无 CSV）"
fi
echo "=== 完成 ==="
