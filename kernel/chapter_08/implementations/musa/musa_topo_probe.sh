#!/usr/bin/env bash
# ============================================================================
# musa_topo_probe.sh —— MUSA 拓扑查询工具探测（实验 5 的 MUSA 侧）。
# 大纲要求：nvidia-smi topo -m 的 MUSA 对应物探测；没有则记平台边界 + 给出
# 替代证据（mthreads-gmi 全量输出 + PCI sysfs link 信息）。
# ============================================================================
set -uo pipefail
# 日志路径从脚本自身位置推导（修复：../../ 相对 cwd 误写到 chapter_08 外，
# 首跑产物只能从 run_all 的 stdout 日志恢复 —— 实操教训记 KNOWN_ISSUES）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOGS="$SCRIPT_DIR/../../results/server-logs"
mkdir -p "$LOGS"

echo "== mthreads-gmi 帮助面 ==" | tee "$LOGS/musa-topo-probe.txt"
mthreads-gmi --help 2>&1 | head -60 | tee -a "$LOGS/musa-topo-probe.txt" || true

echo "== 尝试 mthreads-gmi 可能的 topo 类旗标 ==" | tee -a "$LOGS/musa-topo-probe.txt"
for f in -t -T --topo -m --topology -e -q -a; do
    echo "--- mthreads-gmi $f ---" | tee -a "$LOGS/musa-topo-probe.txt"
    timeout 10 mthreads-gmi $f 2>&1 | head -30 | tee -a "$LOGS/musa-topo-probe.txt" || true
done

echo "== PCI sysfs 链路信息（替代证据） ==" | tee -a "$LOGS/musa-topo-probe.txt"
for dev in /sys/bus/pci/devices/*; do
    class=$(cat "$dev/class" 2>/dev/null || true)
    # 显示控制器 class 0x03 / 加速器 0x12 都看
    case "$class" in
        0x03*|0x12*)
            bdf=$(basename "$dev")
            echo "--- $bdf (class=$class) ---" | tee -a "$LOGS/musa-topo-probe.txt"
            for f in vendor device current_link_speed max_link_speed current_link_width max_link_width numa_node; do
                v=$(cat "$dev/$f" 2>/dev/null || echo "N/A")
                echo "  $f = $v" | tee -a "$LOGS/musa-topo-probe.txt"
            done
            ;;
    esac
done

echo "== lspci 可用性 ==" | tee -a "$LOGS/musa-topo-probe.txt"
if command -v lspci >/dev/null; then
    lspci -nn 2>/dev/null | grep -i "mthreads\|Moore\|vga\|3d\|accel" | tee -a "$LOGS/musa-topo-probe.txt" || true
else
    echo "lspci 不可用（容器边界）" | tee -a "$LOGS/musa-topo-probe.txt"
fi

echo "== Host NUMA/CPU 亲和 ==" | tee -a "$LOGS/musa-topo-probe.txt"
lscpu | grep -i "numa\|socket" | tee -a "$LOGS/musa-topo-probe.txt" || true

echo "== 结论模板（由实验日志汇总） =="
echo "（1）mthreads-gmi 有无 topo -m 等价输出：见上方各旗标尝试结果；" | tee -a "$LOGS/musa-topo-probe.txt"
echo "（2）若无 → 平台边界：拓扑以 PCI sysfs link speed/width + NUMA 亲和为准；" | tee -a "$LOGS/musa-topo-probe.txt"
echo "（3）本文件为完整证据留档。" | tee -a "$LOGS/musa-topo-probe.txt"
