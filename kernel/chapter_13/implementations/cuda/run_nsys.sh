#!/usr/bin/env bash
# ============================================================================
# 第 13 期 nsys 证据采集（CUDA 侧；MUSA 无 nsys —— 平台边界沿用 ch02/ch03）。
#
# 目的：Triton kernel 在 nsys 里可见性验证 + launch 数证据
#   （Triton RMSNorm = 每 launch 1 kernel；融合 = 1 kernel；
#    torch eager 分离链 = 多 kernel —— 对照 ch04 的 launch 数契约）。
# 被测脚本由 bash 内联生成（profile_target.py），nsys profile → stats 三份。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."   # chapter_13 根（implementations/cuda -> chapter_13）

mkdir -p results/server-logs
rm -f results/server-logs/nsys_target.py
cat > results/server-logs/nsys_target.py <<'EOF'
import sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "implementations/cuda")
from rmsnorm_triton import rmsnorm
from fused_triton import fused_add_rmsnorm

dev = torch.device("cuda")
eps = 1e-6
x = torch.randn(2048, 1024, device=dev, dtype=torch.float16)
w = torch.randn(1024, device=dev, dtype=torch.float16)
r = torch.randn(2048, 1024, device=dev, dtype=torch.float16)
h = torch.randn(2048, 1024, device=dev, dtype=torch.float16)
for _ in range(10):
    y = rmsnorm(x, w, eps)
    y2, _ = fused_add_rmsnorm(r, h, w, eps)
    t = torch.add(r, h)
    var = t.float().pow(2).mean(-1, keepdim=True)
    y3 = w * (t.float() * torch.rsqrt(var + eps)).to(torch.float16)
torch.cuda.synchronize()
print("nsys target done")
EOF

nsys profile -o results/server-logs/ch14-triton --force-overwrite true \
  python3 results/server-logs/nsys_target.py 2>&1 | tail -5

nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
  results/server-logs/ch14-triton.nsys-rep > results/nsys-ch14-kernsum.txt 2>results/server-logs/nsys-stats-err.log
nsys stats --report cuda_api_sum --format csv \
  results/server-logs/ch14-triton.nsys-rep > results/nsys-ch14-apisum.txt 2>>results/server-logs/nsys-stats-err.log
nsys stats --report cuda_gpu_kern_sum results/server-logs/ch14-triton.nsys-rep \
  > results/server-logs/nsys-ch14-kernsum-pretty.txt 2>/dev/null
echo "[ch14] nsys 产物：results/nsys-ch14-{kernsum,apisum}.txt"
head -8 results/nsys-ch14-kernsum.txt
