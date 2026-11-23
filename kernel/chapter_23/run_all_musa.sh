#!/bin/bash
# 第 23 期 MUSA 列服务器端执行链（在服务器上运行）
# 用法: setsid nohup bash run_all_musa.sh <phase> > session-<tag>.log 2>&1 < /dev/null &
# 阶段: probe|run|matrix|profile|envjson|all
# 开关:
#   EAGER=1           加 --enforce-eager（无 graph）
#   CGMODE=PIECEWISE  加 --cudagraph-mode PIECEWISE（绕 FULL 捕获 page-size bug）
#   ATTNBK=TRITON_ATTN 经 VLLM_ATTENTION_BACKEND 注入
#   TAG=suffix        结果文件后缀（默认 cg=引擎默认 / 其余按开关组合）
set -u
PHASE="${1:-all}"
EAGER="${EAGER:-0}"
CGMODE="${CGMODE:-}"
ATTNBK="${ATTNBK:-}"
TAG="${TAG:-}"
MODEL="../chapter_01/models/qwen3-0.6b"
PY="${PYTHON_BIN:-$(command -v python3)}"   # 4090 有 conda：必要时 PYTHON_BIN=/usr/bin/python3 固定解释器
EFLAG=""; CFLAG=""
[ "$EAGER" = "1" ] && EFLAG="--enforce-eager"
[ -n "$CGMODE" ] && CFLAG="--cudagraph-mode $CGMODE"
[ -n "$ATTNBK" ] && export VLLM_ATTENTION_BACKEND="$ATTNBK"
if [ -z "$TAG" ]; then
  TAG="def"
  [ "$EAGER" = "1" ] && TAG="eager"
  [ -n "$CGMODE" ] && TAG="cg${CGMODE}"
  [ -n "$ATTNBK" ] && TAG="${TAG}-$ATTNBK"
fi
SUFX="${PLATFORM:-musa}"   # 结果文件后缀：musa / cuda
echo "=== ep24 MUSA session start $(date -u +%FT%TZ) phase=$PHASE eager=$EAGER cg=$CGMODE attn=$ATTNBK tag=$TAG ==="
df -h / | tail -1

# 外置显存采样器：V1 引擎在子进程，父进程 allocator 恒 0；
# 以 5s 周期采整卡已用 MiB 高水位（GPU 本期独占）
# MUSA→mthreads-gmi；CUDA→nvidia-smi（自动探测）
sample_mem() {
  local tag="$1" pid="$2"
  (
    while kill -0 "$pid" 2>/dev/null; do
      if command -v mthreads-gmi >/dev/null 2>&1; then
        echo "$(date -u +%FT%TZ) $(mthreads-gmi 2>/dev/null | grep -E 'MiB\(' | head -1)"
      else
        echo "$(date -u +%FT%TZ) used_MiB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
      fi
      sleep 5
    done
  ) > "results/memsample-$tag.log" 2>&1 &
  SAMPLER_PID=$!
}
stop_sampler() { [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""; }

if [ "$PHASE" = "probe" ] || [ "$PHASE" = "all" ]; then
  echo "=== [probe] env probe ==="
  "$PY" engine_env_probe.py --out results/env-probe-$SUFX.json 2>&1 | tee results/env-probe-$SUFX.log | tail -5
fi

if [ "$PHASE" = "run" ] || [ "$PHASE" = "all" ]; then
  echo "=== [run] truth run (greedy 32 vs ch01) ==="
  sleep 5 & wait; sample_mem "$TAG" $$ ; sleep 1
  "$PY" engine_run_qwen3.py --model "$MODEL" --truth fixtures/truth-ch01.json $EFLAG $CFLAG \
      --out "results/engine-truth-run-$SUFX-$TAG.json" 2>&1 | tee "results/engine-truth-run-$SUFX-$TAG.log" | tail -8 &
  RUNPID=$!; wait $RUNPID; stop_sampler
fi

if [ "$PHASE" = "matrix" ] || [ "$PHASE" = "all" ]; then
  echo "=== [matrix] 3x2x2 measure matrix ==="
  sample_mem "$TAG" $$
  "$PY" engine_measure_matrix.py --model "$MODEL" --truth fixtures/truth-ch01.json $EFLAG $CFLAG \
      --out "results/engine-matrix-$SUFX-$TAG.json" 2>&1 | tee "results/engine-matrix-$SUFX-$TAG.log" | tail -16 &
  MPID=$!; wait $MPID; stop_sampler
fi

if [ "$PHASE" = "profile" ] || [ "$PHASE" = "all" ]; then
  echo "=== [profile] decode op sequence + rms share ==="
  sample_mem "$TAG" $$
  "$PY" engine_profile_decode.py --model "$MODEL" --truth fixtures/truth-ch01.json \
      --out "results/engine-profile-$SUFX-$TAG.json" 2>&1 | tee "results/engine-profile-$SUFX-$TAG.log" | tail -14 &
  PPID2=$!; wait $PPID2; stop_sampler
fi

if [ "$PHASE" = "envjson" ]; then
  echo "=== [envjson] environment snapshot ==="
  "$PY" - <<'EOF'
import json, torch, platform, subprocess, sys
sys.path.insert(0, ".")
from experiment_utils import device_environment_record
from engine24_common import dump_json, utcnow
rec = device_environment_record(torch.device(os.environ.get("PLATFORM", "musa") if os.environ.get("PLATFORM", "musa") != "cuda" else "cuda"))
rec["created_at_utc"] = utcnow()
rec["engine"] = {}
try:
    import vllm
    rec["engine"]["vllm"] = vllm.__version__
    rec["engine"]["vllm_path"] = vllm.__file__
except Exception as e:
    rec["engine"]["vllm"] = repr(e)
try:
    import vllm_omni
    rec["engine"]["vllm_omni"] = getattr(vllm_omni, "__version__", "no-version")
except Exception as e:
    rec["engine"]["vllm_omni"] = repr(e)
try:
    import torch_musa
    rec["engine"]["torch_musa"] = getattr(torch_musa, "__version__", "no-version")
except Exception as e:
    rec["engine"]["torch_musa"] = repr(e)
import os as _os
dump_json(rec, "results/environment-%s.json" % _os.environ.get("PLATFORM", "musa"))
print("environment-musa.json written")
EOF
fi

echo "=== ep24 session end $(date -u +%FT%TZ) ==="
ls -la results/ 2>/dev/null
df -h / | tail -1
