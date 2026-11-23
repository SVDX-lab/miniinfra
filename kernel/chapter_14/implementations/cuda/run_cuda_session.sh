#!/usr/bin/env bash
# 第 14 期 CUDA 回填一键会话（在 4090 服务器执行）。
# 前置：torch 版本门禁 2.12.1+cu130（会话外已验证）；fixtures/correctness.npz 已上传。
set -uo pipefail
HERE_CUDA="$(cd "$(dirname "$0")" && pwd)"          # .../chapter_14/implementations/cuda
CH15="$(cd "$HERE_CUDA/../.." && pwd)"
PY="${PY:-python3}"
RES="$CH15/results"
IND="$RES/inductor-cuda"
mkdir -p "$IND"

echo "== [0/6] 版本门禁 + 文件版本校验"
"$PY" -c "import torch; v=torch.__version__; print('torch', v); assert v=='2.12.1+cu130', 'gate FAIL'"
"$PY" -c "import triton; print('triton', triton.__version__)"
md5sum "$HERE_CUDA/compile_bench_cuda.py" "$HERE_CUDA/correctness_cuda.py" \
      "$HERE_CUDA/export_inductor_triton.py" "$CH15/experiment_utils.py" \
      "$CH15/fixtures/correctness.npz" | tee "$RES/server-md5-cuda.txt"

echo "== [1/6] 环境快照"
"$PY" - <<'EOF' > "$RES/environment-cuda.json"
import json, platform, torch, datetime
p = torch.cuda.get_device_properties(0)
record = {
    "schema_version": "kernel-chapter15.environment.v1",
    "platform_type": "cuda",
    "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": {"name": p.name, "cc": f"{p.major}.{p.minor}",
               "multi_processor_count": p.multi_processor_count,
               "total_memory_bytes": p.total_memory},
}
try:
    import triton
    record["triton"] = triton.__version__
except Exception as e:
    record["triton"] = f"import FAIL: {e}"
print(json.dumps(record, ensure_ascii=False, indent=2))
EOF
cat "$RES/environment-cuda.json"

echo "== [2/6] ch14 手写 Triton 源码取证（对照基准原文）"
{
  echo "=== chapter_13/implementations/cuda/rmsnorm_triton.py ==="
  cat "$CH15/../chapter_13/implementations/cuda/rmsnorm_triton.py" 2>/dev/null || echo MISSING
  echo "=== chapter_13/implementations/cuda/fused_triton.py ==="
  cat "$CH15/../chapter_13/implementations/cuda/fused_triton.py" 2>/dev/null || echo MISSING
} > "$RES/inductor-cuda/ch14-handwritten-triton-source.txt" 2>&1
wc -l "$RES/inductor-cuda/ch14-handwritten-triton-source.txt"

echo "== [3/6] Inductor 生成 Triton 源码导出（TORCH_LOGS=output_code）"
TORCH_LOGS=output_code "$PY" "$HERE_CUDA/export_inductor_triton.py" \
  > "$IND/output_code-add_rmsnorm-rmsnorm.txt" 2>&1
echo "export exit=$?; lines: $(wc -l < "$IND/output_code-add_rmsnorm-rmsnorm.txt")"
ls "$IND"

echo "== [4/6] mode 阶梯延迟基准（eager/inductor/reduce-overhead × fp16/fp32 × 1/128/2048/32768）"
PYTHONPATH="$CH15" "$PY" "$HERE_CUDA/compile_bench_cuda.py" > "$RES/cuda-bench.log" 2>&1
echo "bench exit=$?"; tail -5 "$RES/cuda-bench.log"

echo "== [5/6] 数值正确性（compile vs eager vs FP32 真值）"
PYTHONPATH="$CH15" "$PY" "$HERE_CUDA/correctness_cuda.py" > "$RES/cuda-correctness.log" 2>&1
echo "correctness exit=$?"; tail -8 "$RES/cuda-correctness.log"

echo "== [6/6] Inductor triton 缓存源码收集（按 output_code 的 kernel path 精确提取）"
"$PY" - "$IND" <<'EOF'
from pathlib import Path
import re, shutil
import sys
out = Path(sys.argv[1])
n = 0
log = out / "output_code-add_rmsnorm-rmsnorm.txt"
paths = set()
if log.exists():
    for m in re.finditer(r"kernel path: (\S+\.py)", log.read_text(errors="replace")):
        paths.add(m.group(1))
cache = Path("/tmp/torchinductor_root")
if cache.exists():
    for src in cache.rglob("*.py"):
        text_head = src.name
        if src.parent.name == "triton" or text_head.startswith("triton_"):
            paths.add(str(src))
for p in sorted(paths):
    src = Path(p)
    if src.exists():
        target = out / f"cache-{src.parent.name}-{src.name}"
        if not target.exists():
            shutil.copy(src, target)
            n += 1
print(f"collected {n} sources; kernel paths seen: {len(paths)}")
EOF
ls "$IND" | head -20
echo "== 完成"
