// ============================================================================
// printf_trap_mu.cc —— 第 10 期 MUSA 列：printf 陷阱演示入口（shim 单点复用）；.mu 后缀 = mcc 设备编译模式（M-07，.cc/.cu 为纯 host 模式无 SDK include 路径）。
//
// 共享 ../cuda/printf_trap.cu（内核 0 改动）。探测点：
//   1 设备端 printf 是否工作（行是否完整）；
//   2 输出顺序行为（block/线程调度顺序是否与 CUDA 同样不可靠）；
//   3 FIFO 截断行为（默认缓冲多大？环形覆盖还是丢弃？）；
//   4 musaLimitPrintfFifoSize 是否存在（shim 里未映射——编译期探测）；
//   5 printf 代价（FIFO 序列化比值）。
// ============================================================================

#include "musa_compat.h"
#include "../cuda/printf_trap.cu"
