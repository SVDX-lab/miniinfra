#!/usr/bin/env bash
# 第 18 期 nsys 证据：WMMA kernel 时长 + 同会话朴素对照 + 厂商库 kernel 名。
# 在 chapter_18/ 目录执行。
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results

BIN="implementations/cuda/naive_gemm_bench"

profile_and_stats() {
  local name="$1"; shift
  echo "=== nsys profile: $name ==="
  nsys profile -o "results/nsys-$name" --force-overwrite true --trace=cuda \
    "$@" > "results/nsys-$name.log" 2>&1
  echo "profile rc=$?"
  nsys stats --report cuda_gpu_kern_sum "results/nsys-$name.nsys-rep" \
    > "results/nsys-$name-stats.txt" 2>&1
  awk '/CUDA GPU Kernel Summary/{f=1} f' "results/nsys-$name-stats.txt" \
    > "results/nsys-$name-kernsum.txt"
  head -15 "results/nsys-$name-kernsum.txt"
}

# 1) WMMA 最优预期配置（wm64x64 fp32-acc，fp16 输入，1024³）
profile_and_stats wmma-tiled "$BIN" bench --dtype fp16 --kernel wmma \
  --tile 64x64x8 --acc fp32 --m 1024 --n 1024 --k 1024 --warmup 3 --repeats 20 --seed 20261118

# 2) 同会话朴素对照（朴素 tiled 最优 fp32 32x32x8_reg4x4 + naive fp16 同形状）
profile_and_stats naive-fp32 "$BIN" bench --dtype fp32 --kernel tiled \
  --tile 32x32x8 --reg 4x4 --m 1024 --n 1024 --k 1024 --warmup 3 --repeats 20 --seed 20261118

# 3) 厂商库交叉核对（cuBLAS FP16 1024³）
cat > /tmp/ep19_vendor_nsys.py << 'PYEOF'
import torch
A16 = torch.randn(1024, 1024, device="cuda").half()
B16 = torch.randn(1024, 1024, device="cuda").half()
for _ in range(5):
    torch.matmul(A16, B16)
torch.cuda.synchronize()
for _ in range(20):
    torch.matmul(A16, B16)
torch.cuda.synchronize()
PYEOF
profile_and_stats vendor-fp16 python3 /tmp/ep19_vendor_nsys.py

echo "=== 完成 ==="
