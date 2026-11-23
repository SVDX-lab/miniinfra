#!/usr/bin/env bash
# ============================================================================
# run_sanitizer_suite.sh —— 第 10 期 CUDA 列主体检脚本（RTX 4090 实机）。
#
# 产出（全部落 results/，键=文件名）：
#   E1 检测矩阵：4 缺陷 bad × 4 工具（+good 版对照）——"哪类错误用哪个工具"
#   E2 健康体检：ch03/ch04/ch08 六个 target × 四工具（干净报告=对照基线）
#      —— 含大纲点名实验：racecheck 对 ch08 atomic vs 确定性归约
#   E3 开销测量：rmsnorm_bench（ch03 原样）bare vs 四工具
#      （kernel P50 比值 + 全进程 wall 比值）
#   E4 printf 陷阱：order/trunc/frag/cost 四模式
#   E5 lineinfo 源级关联：有/无 -lineinfo 的 memcheck/racecheck 输出对照
#      + cuobjdump/nvdisasm 汇编行号标注
#
# 运行前提：先 build.sh；需 GPU（compute-sanitizer 不需要计数器权限，
# 与 ncu 不同——但 memcheck 需要注入权限，实测为准）。
# 用法：./run_sanitizer_suite.sh [阶段...]（缺省全跑；阶段名 E1 E2 E3 E4 E5）
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../results/sanitizer"
mkdir -p "$RES"
BUILD="$HERE/build"
SAN="${SANITIZER:-compute-sanitizer}"
ERREXIT="--error-exitcode 42"
TOOLS="memcheck racecheck initcheck synccheck"
STAGES="${*:-E1 E2 E3 E4 E5}"

sum="$RES/summary.tsv"
echo -e "stage\tcase\tvariant\ttool\texit\treport_lines\tkey_line" > "$sum"

run_one() {  # run_one <stage> <case> <variant> <tool> <timeout_s> <cmd...>
    local stage="$1" kase="$2" variant="$3" tool="$4" tmo="$5"; shift 5
    local log="$RES/${stage}-${kase}-${variant}-${tool}.log"
    local t0=$(date +%s.%N)
    timeout "$tmo" "$@" >"$log" 2>&1
    local rc=$?
    local t1=$(date +%s.%N)
    local lines
    lines=$(wc -l <"$log")
    local key
    key=$(grep -m1 -E "Out-of-bounds|Uninitialized|data race|barrier|Barrier|ERROR SUMMARY" "$log" | head -1 | cut -c1-150)
    echo -e "${stage}\t${kase}\t${variant}\t${tool}\t${rc}\t${lines}\t${key}" >> "$sum"
    echo "[run] $stage/$kase/$variant/$tool rc=$rc wall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")s"
}

# ---------------------------------------------------------------- E1 检测矩阵
if [[ " $STAGES " == *" E1 "* ]]; then
  echo "===== E1 缺陷检测矩阵（4 case bad/good × 4 工具）====="
  for kase in oob race uninit sync; do
    # sync bad 是真死锁形态：报告在 barrier 执行点即产生，进程由 timeout 截断
    # （60s 足够报告落盘；其余 case 150s）
    tmo=150
    [[ "$kase" == "sync" ]] && tmo=60
    for variant in good bad; do
      for tool in $TOOLS; do
        run_one E1 "$kase" "$variant" "$tool" "$tmo" \
          "$SAN" --tool "$tool" $ERREXIT "$BUILD/defect_suite" --case "$kase" --variant "$variant" \
                 --n 1024 --blocks 4 --block 256
      done
    done
  done
fi

# ------------------------------------------------------- E2 健康体检 + racecheck 对照
if [[ " $STAGES " == *" E2 "* ]]; then
  echo "===== E2 健康版体检（ch03/ch04/ch08 × 四工具）====="
  # ch03：期原样 rmsnorm_bench（rows=64 h=1024 快口径）
  for tool in $TOOLS; do
    run_one E2 ch03-det health "$tool" 300 \
      "$SAN" --tool "$tool" $ERREXIT "$BUILD/rmsnorm_bench" --rows 64 --h 1024 \
             --dtype fp32 --tiles 256 --warmup 3 --repeats 5 --check true --check-rows 4
  done
  # ch04 融合（in-place 体检口径 iters=1）
  for tool in $TOOLS; do
    run_one E2 ch04-fused health "$tool" 300 \
      "$SAN" --tool "$tool" $ERREXIT "$BUILD/health_ch04" --target ch04-fused \
             --rows 64 --h 1024 --tile 256 --iters 1
  done
  # ch08 三 target（大纲点名：racecheck 对 atomic vs det）
  for tgt in ch08-det ch08-atomic-shared ch08-atomic-global; do
    for tool in $TOOLS; do
      run_one E2 "$tgt" health "$tool" 300 \
        "$SAN" --tool "$tool" $ERREXIT "$BUILD/health_ch08" --target "$tgt" \
               --rows 64 --h 1024 --tile 256 --iters 2 --splits 8
    done
  done
  # racecheck 灵敏度口径补充：racecheck-report race（去噪）与 detect-level
  run_one E2 ch08-atomic-shared health rc-report-race 300 \
    "$SAN" --tool racecheck --racecheck-report race $ERREXIT \
    "$BUILD/health_ch08" --target ch08-atomic-shared --rows 16 --h 1024 --tile 256 --iters 1
  run_one E2 ch08-det health rc-report-race 300 \
    "$SAN" --tool racecheck --racecheck-report race $ERREXIT \
    "$BUILD/health_ch08" --target ch08-det --rows 16 --h 1024 --tile 256 --iters 1
fi

# ---------------------------------------------------------------- E3 开销测量
if [[ " $STAGES " == *" E3 "* ]]; then
  echo "===== E3 sanitizer 开销（ch03 bench 同一 kernel，bare vs 四工具）====="
  local_bench=("$BUILD/rmsnorm_bench" --rows 2048 --h 1024 --dtype fp32 --tiles 256 \
               --warmup 10 --repeats 100 --check false --json-file)
  # bare
  t0=$(date +%s.%N)
  "${local_bench[@]}" "$RES/E3-bench-bare.jsonl" >"$RES/E3-bench-bare.stdout" 2>&1
  t1=$(date +%s.%N)
  echo -e "E3\tbench\tbare\tnone\t0\t0\twall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")s" >> "$sum"
  echo "[run] E3/bare wall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")s"
  for tool in $TOOLS; do
    t0=$(date +%s.%N)
    timeout 1500 "$SAN" --tool "$tool" "${local_bench[@]}" \
      "$RES/E3-bench-$tool.jsonl" >"$RES/E3-bench-$tool.stdout" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    echo -e "E3\tbench\tsanitized\t$tool\t$rc\t0\twall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")s" >> "$sum"
    echo "[run] E3/$tool rc=$rc wall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")s"
  done
  python3 - "$RES" <<'PYEOF'
import json, sys, pathlib
res = pathlib.Path(sys.argv[1])
rows = []
for tag in ["bare", "memcheck", "racecheck", "initcheck", "synccheck"]:
    f = res / f"E3-bench-{tag}.jsonl"
    p50 = p95 = None
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            p50, p95 = d.get("p50_ms"), d.get("p95_ms")
    rows.append((tag, p50, p95))
base = next((p for t, p, _ in rows if t == "bare" and p), None)
out = {"kernel": "rmsnorm fp32 rows=2048 h=1024 tile=256 warmup=10 repeats=100",
       "rows": []}
for tag, p50, p95 in rows:
    out["rows"].append({"mode": tag, "p50_ms": p50, "p95_ms": p95,
                        "p50_ratio_vs_bare": (p50 / base) if (p50 and base) else None})
(res / "E3-overhead-summary.json").write_text(json.dumps(out, indent=2) + "\n")
print("[E3]", json.dumps(out["rows"]))
PYEOF
  grep -h "wall=" "$sum" | grep E3 || true
fi

# ---------------------------------------------------------------- E4 printf 陷阱
if [[ " $STAGES " == *" E4 "* ]]; then
  echo "===== E4 printf 陷阱（顺序/截断/残缺行/代价）====="
  "$BUILD/printf_trap" --mode order >"$RES/E4-order.log" 2>&1
  echo "order rc=$?"
  "$BUILD/printf_trap" --mode trunc --blocks 1024 --block 256 --per-thread 1 \
    >"$RES/E4-trunc-default.log" 2>&1
  echo "trunc-default rc=$?"
  "$BUILD/printf_trap" --mode trunc --blocks 1024 --block 256 --per-thread 1 --fifo-mb 16 \
    >"$RES/E4-trunc-16mb.log" 2>&1
  echo "trunc-16mb rc=$?"
  "$BUILD/printf_trap" --mode frag --blocks 4 --block 64 >"$RES/E4-frag.log" 2>&1
  echo "frag rc=$?"
  "$BUILD/printf_trap" --mode cost --blocks 32 --block 256 --iters 8 --warmup 3 --repeats 10 \
    >"$RES/E4-cost.log" 2>&1
  echo "cost rc=$?"
  {
    for f in E4-order E4-trunc-default E4-trunc-16mb E4-frag; do
      echo "$f lines=$(grep -cve '^---' -e '^\[trunc' "$RES/$f.log")"
    done
  } > "$RES/E4-linecounts.txt"
  cat "$RES/E4-linecounts.txt"
fi

# ---------------------------------------------------------------- E5 lineinfo
if [[ " $STAGES " == *" E5 "* ]]; then
  echo "===== E5 -lineinfo 源级关联对照 ====="
  for tool in memcheck racecheck; do
    run_one E5 oob bad "${tool}-nolineinfo" 150 \
      "$SAN" --tool "$tool" $ERREXIT "$BUILD/defect_suite" --case oob --variant bad \
             --n 1024 --blocks 4 --block 256
    run_one E5 oob bad "${tool}-lineinfo" 150 \
      "$SAN" --tool "$tool" $ERREXIT "$BUILD/defect_suite_lineinfo" --case oob --variant bad \
             --n 1024 --blocks 4 --block 256
    run_one E5 race bad "${tool}-nolineinfo" 150 \
      "$SAN" --tool "$tool" $ERREXIT "$BUILD/defect_suite" --case race --variant bad \
             --n 1024 --blocks 4 --block 256
    run_one E5 race bad "${tool}-lineinfo" 150 \
      "$SAN" --tool "$tool" $ERREXIT "$BUILD/defect_suite_lineinfo" --case race --variant bad \
             --n 1024 --blocks 4 --block 256
  done
  # 汇编层行号标注：-lineinfo 版 cubin 的 SASS 带 //## File/line 注释
  cuobjdump -sass "$BUILD/defect_suite_lineinfo" 2>/dev/null \
    | grep -m 8 -B2 '##File' > "$RES/E5-sass-lineinfo-sample.txt" || true
  cuobjdump -sass "$BUILD/defect_suite" 2>/dev/null \
    | grep -m 4 -B2 '##File' > "$RES/E5-sass-nolineinfo-sample.txt" || true
  echo "sass samples: $(wc -l <"$RES/E5-sass-lineinfo-sample.txt") lineinfo / $(wc -l <"$RES/E5-sass-nolineinfo-sample.txt") nolineinfo"
fi

echo "===== summary.tsv ====="
cat "$sum"
echo "[done] 全部日志在 $RES"
