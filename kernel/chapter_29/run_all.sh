#!/usr/bin/env bash
# 第 29 期一键验收（BACKENDS 环境变量风格，参照 ch01/ch03）
#
#   BACKENDS="cpu"  ./run_all.sh   # 本地：fixtures 对拍 + 静态预审 + 冒烟
#   BACKENDS="cuda" ./run_all.sh   # 4090：build → smoke → correctness →
#                                  #        benchmark → sanitizer → fast_p
#
# cuda 列必须先取得 cuda4090.lock（见 course/coordination/AGENT_PROTOCOL.md §4）。
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
BACKENDS="${BACKENDS:-cpu}"

run_step() {
  echo "== $1 =="
  shift
  "$@" || { echo "[run_all] 步骤失败: $1（继续执行后续步骤，退出码取最差）"; RC=1; return 1; }
}

RC=0
for backend in ${BACKENDS}; do
  case "${backend}" in
    cpu)
      run_step "fixtures SHA256 对拍" "$PY" verify_fixtures.py
      run_step "静态预审（stub+扫描+探针+kernel_sim 全量）" "$PY" static_precheck.py
      run_step "smoke (cpu)" "$PY" smoke_test.py --backend cpu
      run_step "correctness (cpu)" "$PY" compare_correctness.py --backend cpu
      ;;
    cuda)
      run_step "build ai_rmsnorm" bash implementations/ai/rmsnorm/build.sh
      run_step "build ai_gemm" bash implementations/ai/gemm/build.sh
      run_step "build ai_decode_attn" bash implementations/ai/attention/build.sh
      run_step "smoke (cuda)" "$PY" smoke_test.py --backend cuda
      run_step "correctness (cuda)" "$PY" compare_correctness.py --backend cuda
      run_step "benchmark (cuda)" "$PY" benchmark.py --backend cuda
      run_step "sanitizer 四件套" bash gpu_acceptance/sanitizer_suite.sh
      run_step "fast_p 归约" "$PY" fast_p.py
      ;;
    musa)
      echo "[run_all] musa 列：本期不执行（MUSA 起草为可选扩展，见 GPU_PENDING.md）"
      ;;
    *)
      echo "[run_all] 未知 backend: ${backend}"; RC=1
      ;;
  esac
done

exit "$RC"
