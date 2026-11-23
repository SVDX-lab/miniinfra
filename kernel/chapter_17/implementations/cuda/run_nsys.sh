#!/usr/bin/env bash
# 第 17 期 Nsight Systems 证据采集（CUDA 4090；ncu 计数器权限受限时的替代证据链，
# 方法沿用 chapter_01/performance/implementations/cuda/run_nsys.sh）。
# 在 chapter_17/ 目录执行：NSYS=1 ./run_all.sh 或直接 bash implementations/cuda/run_nsys.sh
# 产出（results/ 下，.nsys-rep 大文件不回传仓库）：
#   nsys-gemm-tiled-kernsum.txt     主配置 tiled kernel 汇总（1024³ fp32）
#   nsys-gemm-naive-kernsum.txt     零组织基线 kernel 汇总（访存对账）
#   nsys-gemm-epi-fused-kernsum.txt 融合 epilogue（1 kernel/样本）
#   nsys-gemm-epi-sep3-kernsum.txt  分离 epilogue（3 kernel/样本）
#   nsys-vendor-gemm-kernsum.txt    torch.matmul（cuBLAS）kernel 名与时长的交叉核对
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results

BIN="implementations/cuda/naive_gemm_bench"

profile_and_stats() {
  local name="$1"; shift
  echo "=== nsys profile: $name ==="
  nsys profile -o "results/nsys-$name" --force-overwrite true --trace=cuda \
    "$@" > "results/nsys-$name.log" 2>&1
  echo "profile rc=$?（日志 results/nsys-$name.log）"
  nsys stats --report cuda_gpu_kern_sum "results/nsys-$name.nsys-rep" \
    > "results/nsys-$name-stats.txt" 2>&1
  # 统计段以 " ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):" 起始（首列非 ^）
  awk '/CUDA GPU Kernel Summary/{f=1} f' "results/nsys-$name-stats.txt" \
    > "results/nsys-$name-kernsum.txt"
  head -15 "results/nsys-$name-kernsum.txt"
}

# 1) 主 tiled 配置（64x64x8_reg4x4，fp32，1024³，20 样本）
profile_and_stats gemm-tiled "$BIN" bench --dtype fp32 --kernel tiled \
  --tile 64x64x8 --reg 4x4 --m 1024 --n 1024 --k 1024 --warmup 3 --repeats 20 --seed 20261118

# 2) 零组织基线（无 tiling；有效带宽对账的 memory-bound 侧证据）
profile_and_stats gemm-naive "$BIN" bench --dtype fp32 --kernel naive \
  --m 1024 --n 1024 --k 1024 --warmup 2 --repeats 5 --seed 20261118

# 3) 融合 epilogue（GEMM+bias+GELU 单 kernel）
profile_and_stats gemm-epi-fused "$BIN" bench --dtype fp32 --kernel tiled \
  --tile 64x64x8 --reg 4x4 --epilogue 2 --m 1024 --n 1024 --k 1024 --warmup 3 --repeats 20 --seed 20261118

# 4) 分离 epilogue（GEMM → bias → act，3 kernel/样本）
profile_and_stats gemm-epi-sep3 "$BIN" bench --dtype fp32 --kernel tiled \
  --tile 64x64x8 --reg 4x4 --epilogue 2 --sep 3 --m 1024 --n 1024 --k 1024 --warmup 3 --repeats 20 --seed 20261118

# 5) 厂商库交叉核对（cuBLAS FP32/FP16 1024³，20 样本）
cat > /tmp/ep18_vendor_nsys.py << 'EOF'
import torch
A = torch.randn(1024, 1024, device="cuda")
B = torch.randn(1024, 1024, device="cuda")
for _ in range(5):
    torch.matmul(A, B)
torch.cuda.synchronize()
for _ in range(20):
    torch.matmul(A, B)
torch.cuda.synchronize()
A16 = A.half(); B16 = B.half()
for _ in range(5):
    torch.matmul(A16, B16)
torch.cuda.synchronize()
for _ in range(20):
    torch.matmul(A16, B16)
torch.cuda.synchronize()
EOF
profile_and_stats vendor-gemm python3 /tmp/ep18_vendor_nsys.py

echo "=== 完成 ==="
