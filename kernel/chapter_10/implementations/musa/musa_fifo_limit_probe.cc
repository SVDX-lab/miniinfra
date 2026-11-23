// ============================================================================
// musa_fifo_limit_probe.cc —— MUSA printf FIFO limit 枚举编译探测。
//
// 三候选逐个试编译（-c 即可，不需要 GPU）：
//   候选 1: musaLimitPrintfFifoSize（CUDA 命名映射习惯）
//   候选 2: MUSA_LIMIT_PRINTF_FIFO_SIZE（MUSA 枚举常量大写习惯）
//   候选 3: musaLimitPrintfFifo 大小写变体
// 哪个通过 = shim 的 MUSA_PRINTF_FIFO_LIMIT_ENUM 实值；全败 = 平台边界。
// 同文件顺带运行时探测 musaDeviceGetLimit/musaDeviceSetLimit 是否存在。
// ============================================================================

#include <cstdio>
#include <musa_runtime.h>

#if PROBE_CANDIDATE == 1
#define CAND musaLimitPrintfFifoSize
#elif PROBE_CANDIDATE == 2
#define CAND MUSA_LIMIT_PRINTF_FIFO_SIZE
#else
#define CAND musaLimitPrintfFifo
#endif

int main() {
    size_t value = 0;
    const musaError_t g = musaDeviceGetLimit(&value, CAND);
    std::printf("GetLimit(%s) -> %s, value=%zu\n", "candidate", musaGetErrorString(g), value);
    if (g == musaSuccess) {
        const musaError_t s = musaDeviceSetLimit(CAND, 16ull * 1024 * 1024);
        std::printf("SetLimit(16MB) -> %s\n", musaGetErrorString(s));
        size_t after = 0;
        musaDeviceGetLimit(&after, CAND);
        std::printf("GetLimit after set -> %zu\n", after);
    }
    return (g == musaSuccess) ? 0 : 1;
}
