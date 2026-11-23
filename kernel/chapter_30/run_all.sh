#!/usr/bin/env bash
# 第 30 期一键验收（CUDA 4090 单平台；纸面部分无服务器步骤）
# 用法：BACKENDS="cuda" ./run_all.sh   （本期固定 cuda；无 MUSA/CPU 列，见合同 platform_scope）
set -u
cd "$(dirname "$0")"
BACKENDS="${BACKENDS:-cuda}"
PY=${PY:-python3}
mkdir -p results

echo "[ch31] environment snapshot"
$PY - <<'EOF'
import json, subprocess, sys, torch
from datetime import datetime, timezone
env = {
    "schema_version": "kernel-chapter31.environment.v1",
    "ts": datetime.now(timezone.utc).strftime("%FT%TZ"),
    "episode": 31,
    "platform": "cuda4090",
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "triton": __import__("triton").__version__,
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
}
for cmd in ("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", "nvcc --version"):
    p = subprocess.run(cmd.split(), capture_output=True, text=True)
    env[cmd.split()[0]] = (p.stdout or p.stderr).strip()[-200:]
print(json.dumps(env, indent=2, ensure_ascii=False))
with open("results/environment-cuda.json", "w") as f:
    json.dump(env, f, indent=2, ensure_ascii=False)
EOF

echo "[ch31] experiment 1: FA4 generational boundary probe + SDPA backend compare"
$PY implementations/cuda/fa4_generation_probe.py 2>&1 | tee results/session-fa4probe.log

echo "[ch31] experiment 2: ecosystem smoke (pip-wheel verify where possible)"
$PY implementations/cuda/ecosystem_smoke.py 2>&1 | tee results/session-ecosmoke.log

echo "[ch31] results listing"
ls -la results/
echo "[ch31] done. EXPERIMENT_LOG §2/§3 回填后即完成本期。"
