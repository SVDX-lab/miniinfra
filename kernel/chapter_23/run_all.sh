#!/bin/bash
# ============================================================================
# 第 23 期 run_all.sh —— 引擎实验的会话编排指引（非一键全自动）
#
# 豁免说明（协议 §7 / 对照 ch12 豁免先例）：
#   本期为会话式引擎实验，与 kernel 期的"编译→冒烟→正确性→基准"全自动链不同：
#   1) 引擎栈安装不可重复脚本化（MUSA 侧 vllm-musa 为实例预装 dist-packages；
#      CUDA 侧 pip install vllm 连带升降 torch/triton 整个栈，见 EXPERIMENT_LOG
#      §0.2 版本耦合矩阵——装机动作属于环境决策，不属实验步骤）；
#   2) 单次引擎 init 代价高且随模式剧变（MUSA PIECEWISE 2589s / eager 117s /
#      CUDA def 144s），且部分组合是**已声明的平台边界**（默认 FULL_AND_PIECEWISE
#      在 S4000 上 FULL 段崩溃=KI-24-01；TRITON_ATTN env 被 musa.py:537 屏蔽=
#      KI-24-10，均不再补测）；
#   3) 引擎跑在 V1 子进程，测量口径修正（前缀缓存关闭/外置显存采样/worker 侧
#      profiler）已固化在脚本与 EXPERIMENT_LOG §3/§5 的会话记录里。
#   因此本脚本定位是"两段式会话编排"：实际执行按 EXPERIMENT_LOG §0.3/§5.0 的
#   会话记录进行；本脚本提供 前置条件检查 + 命令序列引用 + 产物清单核对。
#
# 用法：
#   bash run_all.sh plan            # 打印两会话计划 + 双平台前置条件检查（无引擎）
#   bash run_all.sh musa            # 执行 MUSA 会话段（顺序前台，见段内注记）
#   bash run_all.sh cuda            # 执行 CUDA 会话段
#   bash run_all.sh all             # 只打印计划并退出（引擎实验不做无人值守全自动）
#
# 结果文件后缀开关与语义（与 run_all_musa.sh 一致）：
#   EAGER=1 / CGMODE=PIECEWISE|FULL / ATTNBK=… / PLATFORM=cuda|musa
# ============================================================================
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PLATFORM_="${PLATFORM:-}"
PY="${PYTHON_BIN:-$(command -v python3)}"
MODEL="../chapter_01/models/qwen3-0.6b"

hr() { echo "--------------------------------------------------------------"; }

# ---------- 前置条件检查（只读，无引擎 init） ----------
prereq() { # $1 = musa|cuda
  local p="$1" ok=1
  hr; echo "[前置条件检查] 平台=$1  $(date -u +%FT%TZ)"; hr
  echo "1) torch 可导入与版本："
  "$PY" - "$p" <<'EOF' || ok=0
import sys, torch
p = sys.argv[1]
print("   torch", torch.__version__)
if p == "musa":
    import torch_musa
    print("   torch_musa", getattr(torch_musa, "__version__", "?"),
          "| musa available:", torch.musa.is_available())
else:
    print("   cuda available:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
  echo "2) vllm 可导入与版本："
  "$PY" -c "import vllm; print('   vllm', vllm.__version__, '@', vllm.__file__)" || ok=0
  echo "3) GPU 空闲（应为 0 进程/0 MiB 占用，锁协议确保独占）："
  if command -v mthreads-gmi >/dev/null 2>&1; then
    mthreads-gmi 2>/dev/null | grep -E "MiB\(" | head -1
  else
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -3
  fi
  echo "4) 模型权重在位（ch01 只读）："
  ls -la "$MODEL/model.safetensors" 2>/dev/null || { echo "   缺失！"; ok=0; }
  echo "5) 对拍锚点在位："
  ls fixtures/truth-ch01.json >/dev/null 2>&1 && echo "   fixtures/truth-ch01.json OK" || { echo "   缺失！"; ok=0; }
  echo "   前置检查结论：$([ $ok = 1 ] && echo PASS || echo FAIL)"
  return $ok
}

# ---------- 产物清单核对 ----------
artifact_check() { # $1 = musa|cuda
  hr; echo "[本段产物清单核对] 平台=$1"; hr
  local need=(
    "results/env-probe-$1.json"
    "results/environment-$1.json"
    "results/engine-truth-run-$1-eager.json"
    "results/engine-matrix-$1-eager.json"
    "results/engine-profile-trace-$1.json"
  )
  local extra
  case "$1" in
    musa)  extra=("results/engine-truth-run-musa-cgPIECEWISE.json"
                  "results/engine-matrix-musa-cgPIECEWISE.json"
                  "results/engine-truth-run-musa-triton-full.log"
                  "results/memsample-eager.log") ;;
    cuda)  extra=("results/engine-truth-run-cuda-def.json"
                  "results/engine-matrix-cuda-def.json"
                  "results/memsample-def.log"
                  "results/pip-freeze-after-cuda.txt"
                  "results/pip-install-vllm-cuda.log") ;;
    # 注：CUDA 侧不做 eager 12 格矩阵——eager 口径由 truth 锚点（TPOT 17.54ms）
    # 与 def 矩阵覆盖，见 EXPERIMENT_LOG §3.4 对照注记 4；两段均已声明边界
    # （MUSA: KI-24-01/10；CUDA: 无）不设回填位。
  esac
  local missing=0 f
  for f in "${need[@]}" "${extra[@]}"; do
    if [ -f "$f" ]; then echo "  [有] $f"; else echo "  [缺] $f"; missing=$((missing+1)); fi
  done
  echo "  缺 $missing 项（缺项的会话命令见下方 plan）"
}

# ---------- 会话计划（引用已归档命令序列） ----------
plan_musa() {
  hr; echo "[MUSA 会话段] 前置：S4000 锁（locks/musa_s4000.lock）+ 前置条件检查 PASS"; hr
  cat <<'EOF'
  步骤（每步独立引擎进程；按 EXPERIMENT_LOG §0.3 的实测 init 代价预算时间）：
  S1 环境探针（静态，~2min）
     python3 engine_env_probe.py --out results/env-probe-musa.json
  S2 引擎跑通+真值对拍（先默认图模式——预期在 FULL 捕获段崩溃=KI-24-01，
     属已声明平台边界，崩溃即取证；随后 eager 与 PIECEWISE 两口径完成对拍）
     python3 engine_run_qwen3.py --model $MODEL --truth fixtures/truth-ch01.json \
         --out results/engine-truth-run-musa-def.log          # 预期崩，留日志
     EAGER=1        bash run_all_musa.sh run                   # eager 27/27
     CGMODE=PIECEWISE bash run_all_musa.sh run                 # PIECEWISE 27/27（init ~43min）
  S3 测量矩阵 3×2×2（TTFT/TPOT/吞吐；外置 mthreads-gmi 采样自动启动）
     EAGER=1        bash run_all_musa.sh matrix                # 冷口径正式列
     CGMODE=PIECEWISE bash run_all_musa.sh matrix              # 服务口径参考列（可选）
  S4 worker 侧 profiler（KI-24-07 修正版；父进程 profiler 抓不到子进程模型）
     PLATFORM=musa python3 engine_profile_vllm.py --model $MODEL \
         --truth fixtures/truth-ch01.json                     # b1+b8 双 trace
  S5 环境快照 + 已声明边界单点（不补测）
     bash run_all_musa.sh envjson
     ATTNBK=TRITON_ATTN CGMODE=FULL bash run_all_musa.sh run   # KI-24-10：env 被
                                                               # musa.py:537 屏蔽，
                                                               # 同因崩溃留档即可
  本段产物：见 artifact_check musa（TRITON 单点只留 log 不出 JSON=预期行为）
EOF
}

plan_cuda() {
  hr; echo "[CUDA 会话段] 前置：4090 锁（locks/cuda4090.lock）+ 前置条件检查 PASS"; hr
  cat <<'EOF'
  步骤（装机属环境决策不属本段：官方 vllm 0.34.0 已连带 torch→2.14.0，
  记录见 results/pip-install-vllm-cuda.log / pip-freeze-after-cuda.txt；
  fork 守卫修复已固化 engine24_common.py：VLLM_WORKER_MULTIPROC_METHOD=spawn）：
  C1 环境探针
     python3 engine_env_probe.py --out results/env-probe-cuda.json
  C2 引擎跑通+真值对拍（def 图与 eager 双口径均应 27/27；CUDA 上无 FULL 崩溃）
     PLATFORM=cuda bash run_all_musa.sh run                       # def 图
     EAGER=1 PLATFORM=cuda bash run_all_musa.sh run               # eager
  C3 测量矩阵 3×2×2（prefix cache off 冷口径；nvidia-smi 采样自动启动）
     PLATFORM=cuda bash run_all_musa.sh matrix
  C4 worker 侧 profiler（0.34 需 profiler_config；trace 为 gzip，解析已兜底）
     PLATFORM=cuda python3 engine_profile_vllm.py --model $MODEL \
         --truth fixtures/truth-ch01.json
  C5 环境快照
     PLATFORM=cuda bash run_all_musa.sh envjson
  本段产物：见 artifact_check cuda
EOF
}

case "${1:-plan}" in
  plan)
    plan_musa; plan_cuda
    prereq musa;  echo; prereq cuda
    artifact_check musa; echo; artifact_check cuda
    echo "双列收官状态与已声明边界（不补测）：EXPERIMENT_LOG 头部状态行 + KI-24-01/10。"
    ;;
  musa)
    prereq musa && { plan_musa; artifact_check musa; } || echo "前置 FAIL——按 plan 输出排查后再执行"
    ;;
  cuda)
    prereq cuda && { plan_cuda; artifact_check cuda; } || echo "前置 FAIL——按 plan 输出排查后再执行"
    ;;
  all)
    plan_musa; plan_cuda
    echo "refusing auto-run：引擎实验为会话式（见头部豁免说明）；"
    echo "请分别 'bash run_all.sh musa' 与 'bash run_all.sh cuda' 并按 EXPERIMENT_LOG 会话记录执行。"
    ;;
  *) echo "用法: bash run_all.sh [plan|musa|cuda|all]"; exit 2 ;;
esac
