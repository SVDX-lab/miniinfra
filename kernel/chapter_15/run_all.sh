#!/usr/bin/env bash
# 第 15 期一键验收（协议 §1：BACKENDS 环境变量风格，ch01/ch18 同款）
#   BACKENDS="cpu" bash run_all.sh           # 本地（无 GPU）
#   BACKENDS="cpu musa" bash run_all.sh      # S4000 会话
#   BACKENDS="cpu cuda" bash run_all.sh      # 4090 会话【待 4090 修复】
#
# 链路：smoke → fixtures 完整性校验 → 正确性（误差分层）→ 基准 → 环境记录。
# GPU 后端要求对应二进制已构建（implementations/<backend>/int8_gemm_bench）。
set -uo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"
PY="${PY:-python3}"
echo "== 第 15 期 run_all：BACKENDS=$BACKENDS，PY=$PY =="

echo "== [1/4] smoke test（边界行为，CPU-only）=="
"$PY" smoke_test.py || exit 1

echo "== [2/4] fixtures 完整性（SHA256 对拍；KI-21-06 教训条款）=="
"$PY" - <<'EOF'
import json, sys
import numpy as np
sys.path.insert(0, '.')
from experiment_utils import sha256_of_array
data = np.load('fixtures/correctness.npz')
manifest = json.load(open('fixtures/manifest.json'))
bad = []
for c in manifest['combos']:
    key = f"{c['combo']}_{c['m']}x{c['k']}x{c['n']}"
    for base, want in c['sha256'].items():
        if sha256_of_array(data[key + '__' + base]) != want:
            bad.append(key + '__' + base)
print('fixtures tensors:', len(data.files), 'mismatch:', bad)
assert not bad, 'fixtures 完整性校验失败'
EOF
[ $? -ne 0 ] && exit 1

for BE in $BACKENDS; do
  echo "== [3/4] 正确性（backend=$BE）=="
  if [ "$BE" = "cpu" ]; then
    "$PY" compare_correctness.py --backend cpu || exit 1
  else
    BIN="implementations/$BE/int8_gemm_bench"
    [ -x "$BIN" ] || { echo "二进制缺失：$BIN（先 build）"; exit 1; }
    "$PY" compare_correctness.py --backend "$BE" --binary "$BIN" || exit 1
    echo "== [4/4] 基准（backend=$BE）=="
    "$PY" benchmark.py --backend "$BE" --binary "$BIN" || exit 1
    "$PY" engine_anchor.py --backend "$BE" || exit 1
  fi
done

if [ "$BE" = "cpu" ]; then
  echo "== [4/4] 基准（backend=cpu）=="
  "$PY" benchmark.py --backend cpu || exit 1
fi

echo "== run_all 全部完成 =="
