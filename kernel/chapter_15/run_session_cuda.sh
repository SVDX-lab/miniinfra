#!/usr/bin/env bash
# 第 15 期 CUDA 列一键会话链（4090；MUSA run_session.sh 的对偶）
# 门禁：torch==2.12.1+cu130（ep24 连带升级 2.14.0 后主 agent 降级回滚——未完成则 GATE-FAIL 退出）
# 用法（锁内）：bash run_session_cuda.sh 2>&1 | tee results/session-cuda.log
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results/cuda-logs
LOG=results/cuda-logs

echo "== [0] torch 门禁（锁内等待循环：每 120s 探一次，最多 30 轮 ≈60min；ep21 持锁等待先例）=="
GATE_OK=0
for i in $(seq 1 30); do
  TV=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null | tail -1)
  TS=$(date -u +%FT%TZ)
  echo "$TS [gate] round $i torch=$TV"
  if [[ "$TV" == 2.12.1+cu130* ]]; then
    python3 -c "import torch; assert torch.cuda.is_available()" || { echo "GATE-FAIL cuda 不可用"; exit 42; }
    GATE_OK=1
    break
  fi
  sleep 120
done
[ $GATE_OK -ne 1 ] && { echo "GATE-TIMEOUT torch 降级 60min 未落地，释放锁重试"; exit 42; }

echo "== [0b] 源码完整性（md5 对拍；KI-21-06 条款）=="
md5sum implementations/cuda/int8_gemm.cu fixtures/correctness.npz | tee "$LOG/md5.log"
grep -q "bbb7bc31d803c6734baeb7d4e3206ede" "$LOG/md5.log" || { echo "int8_gemm.cu md5 不符"; exit 1; }
grep -q "1d1d0d5c0515b91ddc22b24d7682be3b" "$LOG/md5.log" || { echo "fixtures md5 不符"; exit 1; }

echo "== [1] build（nvcc sm_89，三变体全在二进制内）=="
bash implementations/cuda/build.sh 2>&1 | tee "$LOG/build.log" || exit 1
BIN=implementations/cuda/int8_gemm_bench
[ -x "$BIN" ] || { echo "二进制缺失"; exit 1; }

echo "== [2] selftest =="
"$BIN" selftest 2>&1 | tee "$LOG/selftest.log"
[ ${PIPESTATUS[0]} -ne 0 ] && exit 1

echo "== [3] attrs =="
"$BIN" attrs > results/attrs-cuda.json 2> "$LOG/attrs.err"
head -c 600 results/attrs-cuda.json; echo

echo "== [4] correctness（误差分层 Lb/Lc，fixtures 同 MUSA 口径）=="
python3 compare_correctness.py --backend cuda --binary "$BIN" 2>&1 | tee "$LOG/compare.log" || exit 1

echo "== [5] benchmark（全 profile + dequant v1/v2/v3 跨平台对照）=="
python3 benchmark.py --backend cuda --binary "$BIN" 2>&1 | tee "$LOG/bench.log" || exit 1

echo "== [6] engine anchor（FP16 锚点 + FP8 A 批探测：e4m3 convert / torch._scaled_mm 复测）=="
python3 engine_anchor.py --backend cuda 2>&1 | tee "$LOG/anchor.log" || exit 1

echo "== [7] FP8 B 批：Triton FP8 dot 尝试 =="
python3 fp8_triton_dot.py 2>&1 | tee "$LOG/triton-fp8.log" || echo "triton fp8 部分失败（逐变体错误已留档，继续）"

echo "== [8] 环境记录 =="
python3 - <<'EOF'
import sys, json
sys.path.insert(0, '.')
from experiment_utils import device_environment_record
import torch
rec = device_environment_record(torch.device('cuda'))
rec['episode'] = 16
rec['note'] = 'RTX 4090 24GB；CUDA 列回填会话（MUSA 列同口径 fixtures）'
with open('environment-cuda.json', 'w') as f:
    json.dump(rec, f, ensure_ascii=False, indent=2); f.write('\n')
print(json.dumps(rec.get('gpu', {}), ensure_ascii=False))
EOF

echo "== run_session_cuda 全部完成 =="
