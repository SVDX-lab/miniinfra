#!/usr/bin/env bash
# ============================================================================
# 第 08 期一键验收（协议 §1 BACKENDS 环境变量风格）。
#   BACKENDS="cuda" ./run_all.sh   （4090 会话）
#   BACKENDS="musa" ./run_all.sh   （S4000 会话）
# 步骤：环境快照 → build → api_probe 运行 → smoke → correctness →
#       bandwidth → overlap → stream probe → 拓扑 → （cuda）nsys
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
BACKENDS="${BACKENDS:-cuda}"
mkdir -p results/server-logs

for BE in $BACKENDS; do
  echo "=================================================================="
  echo "  chapter_08 backend=$BE"
  echo "=================================================================="
  LIB="implementations/$BE/build/libtransfer_ops_${BE}.so"

  echo "--- [0/8] 环境快照 ---"
  python3 - "$BE" <<'EOF'
import sys
sys.path.insert(0, ".")
from experiment_utils import backend_environment_record, write_results_json
be = sys.argv[1]
write_results_json(f"results/environment-{be}.json", backend_environment_record(be))
print(f"environment-{be}.json OK")
EOF

  echo "--- [1/8] build ---"
  bash "implementations/$BE/build.sh" || { echo "BUILD FAILED"; exit 1; }

  echo "--- [2/8] api_probe（传输 API 面编译+运行证据） ---"
  "implementations/$BE/build/api_probe" | tee "results/server-logs/api-probe-$BE.log" || true

  echo "--- [3/8] smoke ---"
  python3 smoke_test.py --backend "$BE" --lib "$LIB" || { echo "SMOKE FAILED"; exit 1; }

  echo "--- [4/8] correctness ---"
  python3 compare_correctness.py --backend "$BE" --lib "$LIB" \
      --out "results/correctness-$BE.json" || { echo "CORRECTNESS FAILED"; exit 1; }

  echo "--- [5/8] bandwidth（实验 1+3） ---"
  python3 bench_bandwidth.py --backend "$BE" --lib "$LIB" \
      --out "results/bandwidth-$BE.json" || { echo "BANDWIDTH FAILED"; exit 1; }

  echo "--- [6/8] overlap（实验 2+4） ---"
  python3 bench_overlap.py --backend "$BE" --lib "$LIB" \
      --out "results/overlap-$BE.json" || { echo "OVERLAP FAILED"; exit 1; }

  echo "--- [7/8] stream probe（多流语义） ---"
  python3 stream_probe.py --backend "$BE" --lib "$LIB" \
      --out "results/stream-probe-$BE.json" || { echo "STREAM PROBE FAILED"; exit 1; }

  echo "--- [8/8] 拓扑 + Host 侧基准 + nsys(cuda) ---"
  if [ "$BE" = "cuda" ]; then
    nvidia-smi topo -m > results/server-logs/topo-cuda.txt 2>&1 || true
    nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
        --format=csv > results/server-logs/pcie-link-cuda.txt 2>&1 || true
    echo "topo-cuda.txt / pcie-link-cuda.txt OK"
    bash run_nsys.sh || { echo "NSYS FAILED"; exit 1; }
  else
    bash implementations/musa/musa_topo_probe.sh || true
  fi
  python3 implementations/cpu/host_mem_bench.py --label "$BE-host" \
      --out "results/host-mem-$BE-host.json" || true

  echo "backend=$BE 完成"
done
echo "run_all 完成：results/ 见 *.json"
