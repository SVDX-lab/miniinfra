#!/usr/bin/env bash
# 第 15 期 S4000 一键会话链（build → selftest → attrs → correctness → benchmark
# → engine anchor → 环境记录；日志全部落 results/musa-logs/，结果落 results/）。
# 用法（锁内执行）：bash run_session.sh 2>&1 | tee results/session-musa.log
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results/musa-logs
LOG=results/musa-logs

echo "== [0] 完整性预检（KI-21-06 教训：先验 fixtures 再跑）=="
python3 - <<'EOF'
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

echo "== [1] build =="
bash implementations/musa/build.sh 2>&1 | tee "$LOG/build.log" || exit 1
BIN=implementations/musa/int8_gemm_bench
[ -x "$BIN" ] || { echo "二进制缺失"; exit 1; }

echo "== [2] selftest =="
"$BIN" selftest 2>&1 | tee "$LOG/selftest.log"
[ ${PIPESTATUS[0]} -ne 0 ] && exit 1

echo "== [3] attrs =="
"$BIN" attrs > results/attrs-musa.json 2> "$LOG/attrs.err"
cat results/attrs-musa.json | head -c 600; echo

echo "== [4] correctness（误差分层 Lb/Lc）=="
python3 compare_correctness.py --backend musa --binary "$BIN" 2>&1 | tee "$LOG/compare.log" || exit 1

echo "== [5] benchmark =="
python3 benchmark.py --backend musa --binary "$BIN" 2>&1 | tee "$LOG/bench.log" || exit 1

echo "== [6] engine anchor + INT8 路径探测 =="
python3 engine_anchor.py --backend musa 2>&1 | tee "$LOG/anchor.log" || exit 1

echo "== [7] 环境记录 =="
python3 - <<'EOF'
import sys, json
sys.path.insert(0, '.')
from experiment_utils import device_environment_record
import torch
rec = device_environment_record(torch.device('musa'))
rec['episode'] = 16
rec['note'] = 'MTT S4000 48GB；MUSA 列实机会话'
with open('environment-musa.json', 'w') as f:
    json.dump(rec, f, ensure_ascii=False, indent=2); f.write('\n')
print(json.dumps(rec.get('gpu', {}), ensure_ascii=False))
EOF

echo "== [8] vllm-musa 量化算子调研（只读 grep）=="
python3 survey_vllm_musa.py 2>&1 | tee "$LOG/vllm-survey.log" || true

echo "== run_session 全部完成 =="
