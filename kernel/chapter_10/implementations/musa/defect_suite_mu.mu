// ============================================================================
// defect_suite_mu.cc —— 第 10 期 MUSA 列：缺陷集合入口（shim 单点复用）；.mu 后缀 = mcc 设备编译模式（M-07，.cc/.cu 为纯 host 模式无 SDK include 路径）。
//
// 与 CUDA 版共享同一缺陷 Kernel 源（../cuda/defect_suite.cu 逐字节复用，
// 内核 0 改动）：cuda* API 由 musa_compat.h 单点映射，kernel 语法同构。
// 若编译失败，失败行即工具链边界证据（照实记录，不做兜底改写）。
// ============================================================================

#include "musa_compat.h"
#include "../cuda/defect_suite.cu"
