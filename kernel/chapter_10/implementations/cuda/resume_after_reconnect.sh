#!/usr/bin/env bash
# ============================================================================
# resume_after_reconnect.sh —— 4090 实例重启后的全量续跑（幂等）。
# 策略：不拼补——全量重跑 build + suite(E1-E5) + 冒烟 + 正确性 + 基准 +
#       fixtures 清单，然后打包 results 供一次拉回。
# 用法：bash resume_after_reconnect.sh
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CH="$HERE/../.."

echo "== R0 环境 =="
nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader
ls "$CH" 2>/dev/null | tr '\n' ' '; echo

echo; echo "== R1 构建（幂等）=="
(cd "$HERE" && ./build.sh) || exit 1

echo; echo "== R2 全量套件 E1-E5（约 25 分钟，sync-bad 4x60s 死锁截断）=="
(cd "$HERE" && ./run_sanitizer_suite.sh)

echo; echo "== R3 冒烟 + 正确性 + 基准 + fixtures 清单 =="
(cd "$CH" && /usr/bin/python3 smoke_test.py | tail -18)
(cd "$CH" && /usr/bin/python3 compare_correctness.py | tail -10)
(cd "$CH" && /usr/bin/python3 benchmark.py | tail -45)
(cd "$CH" && /usr/bin/python3 generate_fixtures.py --emit-source-manifest | tail -3)

echo; echo "== R4 打包 =="
cd "$CH" && tar czf /tmp/ch11_results.tgz results environment-cuda.json environment-cpu.json
ls -la /tmp/ch11_results.tgz
echo "[resume] 完成——本地拉回 /tmp/ch11_results.tgz"
