#!/usr/bin/env bash
# ============================================================================
# 第 06 期 CUDA nsys 证据：三策略的 kernel 发射数与时长对照。
# ncu 被宿主 RmProfilingAdminOnly=1 禁用（KI-02-02 延续）→ nsys + 有效带宽对账。
#
# v1 教训（KI-07-06）：workload 的 sys.path 按"从 chapter_06 根运行"假设写死
# 但 nsys 从 implementations/cuda cwd 启动 → 无 kernel 数据；且
# `nsys profile | tail` 在 pipefail 下因 nsys 退出码非零静默中止后续步骤。
# v2 修正：统一从 chapter_06 根运行 + workload 三路径注入 + 禁 pipefail 依赖。
#
# 证据点（rows=2048 段，4090 实测见 results/server-logs/ep07-nsys-kern-sum.txt）：
#   策略 A = 1 kernel（rmsnorm 6.69µs）/ 策略 B = 2 kernel（layout copy 30.4µs
#   + rmsnorm 6.69µs）/ 策略 C = 1 kernel（strided 31.8µs）/ 陷阱 = 0 kernel。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # chapter_06 根

mkdir -p results/server-logs

cat > results/server-logs/nsys-workload.py <<'EOF'
import sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "implementations/cuda")
sys.path.insert(0, "implementations/cpu")
from build_ext import get_module
from stride_layout import RMS_NORM_EPS
m = get_module()
device = torch.device("cuda")
torch.cuda.nvtx.range_push("warmup")
x = torch.randn(2048, 1024, device=device)
w = torch.randn(1024, device=device)
tv = x.t().contiguous().t()
for _ in range(20):
    m.rmsnorm_policy_a(x, w, RMS_NORM_EPS, 256)
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("policy_A_contig")
for _ in range(10):
    m.rmsnorm_policy_a(x, w, RMS_NORM_EPS, 256)
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("policy_B_tv")
for _ in range(10):
    m.rmsnorm_policy_b(tv, w, RMS_NORM_EPS, 256)
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("policy_C_tv")
for _ in range(10):
    m.rmsnorm_policy_c(tv, w, RMS_NORM_EPS, 256)
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("trap_zero_kernel")
for _ in range(10):
    y = tv.t().contiguous()
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("workload done")
EOF

nsys profile -o results/server-logs/ep07-policies --force-overwrite true \
    -t cuda,nvtx python3 results/server-logs/nsys-workload.py || \
    echo "[nsys] profile 返回码非零（进度输出仍有效），继续 stats"

nsys stats --force-export=true --report cuda_gpu_kern_sum \
    results/server-logs/ep07-policies.nsys-rep \
    > results/server-logs/ep07-nsys-kern-sum.txt 2>&1 || true
echo "=== kernel summary ==="
sed -n '1,20p' results/server-logs/ep07-nsys-kern-sum.txt
echo "完成：详见 results/server-logs/ep07-nsys-kern-sum.txt"
