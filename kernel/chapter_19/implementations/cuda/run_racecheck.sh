#!/usr/bin/env bash
# 第 19 期 racecheck 体检（接第 10 期工具链；大纲点名实验）
#
# 对象：流水版 GEMM 的共享内存——
#   R1 sync        第 17 期同步基线（阳性底：应 0 errors）
#   R2 cpasync1    cp.async 单缓冲（async 写读 race 是否被跟踪？应 0 errors）
#   R3 cpasync2    cp.async 双缓冲（barrier 结构完整 → 应 0 errors）
#   R4 cpasync2_broken 双缓冲去尾屏障（WAR 竞态 → racecheck 报告形态记录；
#      若 racecheck 不跟踪 cp.async 写则可能沉默——沉默本身是发现，对照 R5）
#   R5 sync2_broken    同步 load 双缓冲去尾屏障（保守阳性对照，racecheck 必报）
#
# 口径：shape 128³ + tile 32x32x8_reg4x4（racecheck 可承受，ep11 经验），
# warmup=0/repeats=2（sanitizer 下计时无意义，只看报告）。
# 输出：results/racecheck-*.log（由调用方重定向）。
set -uo pipefail
cd "$(dirname "$0")"

SAN="${COMPUTE_SANITIZER:-compute-sanitizer}"
BIN="${BIN:-./async_gemm_bench}"
SHAPE="${SHAPE:---m 128 --n 128 --k 128}"
TILE="--tile 32x32x8 --reg 4x4"
COMMON="$SHAPE $TILE --warmup 0 --repeats 2 --seed 20261118"

run_rc() {
  local name="$1"; shift
  echo "== racecheck $name : $* =="
  "$SAN" --tool racecheck --racecheck-detect-level warn "$BIN" bench "$@" 2>&1 \
    | tee "racecheck-$name.log"
  echo "== racecheck $name rc=$? 摘要 =="
  grep -E "ERROR SUMMARY|race|Hazard|WAR|WAW|RAW" "racecheck-$name.log" | head -20 || true
}

run_rc sync        --kernel sync        $COMMON
run_rc cpasync1    --kernel cpasync1    $COMMON
run_rc cpasync2    --kernel cpasync2    $COMMON
run_rc cpasync2_broken --kernel cpasync2_broken $COMMON
run_rc sync2_broken    --kernel sync2_broken    $COMMON
echo "全部 racecheck 运行完成；日志 racecheck-*.log 与结果由上层收集"
