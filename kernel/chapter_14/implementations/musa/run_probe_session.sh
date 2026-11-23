#!/usr/bin/env bash
# 第 14 期 MUSA 探测一键会话（目标 <=30min，S4000 短会话）。
# 用法（从仓库根目录）: bash kernel/chapter_14/implementations/musa/run_probe_session.sh
# 产物全部落在 chapter_14/results/，最后由调用方 scp 拉回。
set -uo pipefail
HERE_MUSA="$(cd "$(dirname "$0")" && pwd)"          # .../chapter_14/implementations/musa
CH15="$(cd "$HERE_MUSA/../.." && pwd)"              # .../chapter_14
PY="${PY:-python3}"
RES="$CH15/results"
mkdir -p "$RES"

echo "== [0/5] 文件版本校验（快照回退教训）"
md5sum "$HERE_MUSA/musa_compile_probe.py" "$HERE_MUSA/musa_bench_compile.py" 2>&1 | tee "$RES/server-md5-musa.txt"
grep -c "musa-compile-probe.v1" "$HERE_MUSA/musa_compile_probe.py" || true

echo "== [1/5] 环境快照"
"$PY" - <<'EOF' > "$RES/environment-musa.json"
import json, platform, torch
import torch_musa
import datetime
p = torch.musa.get_device_properties(0)
record = {
    "schema_version": "kernel-chapter15.environment.v1",
    "platform_type": "musa",
    "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_musa": getattr(torch_musa, "__version__", "unknown"),
    "device": {"name": p.name, "cc": f"{p.major}.{p.minor}",
               "multi_processor_count": p.multi_processor_count,
               "total_memory_bytes": p.total_memory},
}
try:
    import triton
    record["triton"] = triton.__version__
    from triton.backends import backends as tb
    record["triton_backends"] = list(tb)
except Exception as e:
    record["triton"] = f"import FAIL: {e}"
print(json.dumps(record, ensure_ascii=False, indent=2))
EOF
cat "$RES/environment-musa.json"

echo "== [2/5] torch_musa 包内 compile/inductor/triton 文件树清单"
TM_DIR="$("$PY" -c 'import torch_musa, pathlib; print(pathlib.Path(torch_musa.__file__).parent)')"
{
  echo "package_dir: $TM_DIR"
  echo "--- path hits (inductor/compile/triton) ---"
  find "$TM_DIR" \( -iname "*inductor*" -o -iname "*compile*" -o -iname "*triton*" \) 2>/dev/null | head -80
  echo "--- grep hits: dynamo/inductor/compile registration ---"
  grep -rln "torch.compile\|_dynamo\|inductor" "$TM_DIR" 2>/dev/null | head -40
} > "$RES/musa-tree-inventory.txt" 2>&1
wc -l "$RES/musa-tree-inventory.txt"

echo "== [3/5] torch.compile 行为矩阵探测（含全 traceback）"
"$PY" "$HERE_MUSA/musa_compile_probe.py" > "$RES/musa-probe.log" 2>&1
echo "probe exit=$?"; tail -30 "$RES/musa-probe.log"

echo "== [4/5] 若 inductor 可用：生成代码形态捕获（TORCH_LOGS=output_code）"
TORCH_LOGS=output_code "$PY" "$HERE_MUSA/musa_compile_probe.py" \
  > "$RES/musa-probe-outputcode.log" 2>&1 || true
echo "output_code capture lines: $(wc -l < "$RES/musa-probe-outputcode.log")"

echo "== [5/5] 条件基准（compile 可用才有意义）"
"$PY" "$HERE_MUSA/musa_bench_compile.py" > "$RES/musa-bench.log" 2>&1
echo "bench exit=$?"; tail -20 "$RES/musa-bench.log"

echo "== 完成，results/ 清单："
ls -la "$RES"
