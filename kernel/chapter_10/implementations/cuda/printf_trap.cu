// ============================================================================
// printf_trap.cu —— 第 10 期 printf 调试陷阱演示（CUDA / MUSA 同源双编译）。
//
// 四个模式：
//   mode=order   4 block × 32 线程各打一行"B%02d.T%02d"：展示 block 间/线程间
//                输出顺序=调度顺序（不按编号）；连跑 3 次对比（顺序逐次漂移）。
//   mode=trunc   大规模打印（默认 1024 block × 256 线程 = 262144 行）：
//                超过 printf FIFO（默认 1MB）→ 环形缓冲吞旧行 → 收到的行
//                数 << 发出数、且开头缺失；再演示 cudaDeviceSetLimit 调大
//                FIFO 后完整接收（"修复"口径）。
//   mode=frag    每线程两次 printf 拼一行（无换行的前半 + 带换行的后半）：
//                FIFO 按记录交错 → 残缺行/拼接行（第 06 期 KI-07-04 的
//                kernel 版复现：日志行不再可信）。
//   mode=cost    printf kernel vs 静默 kernel 的耗时对照（FIFO 序列化代价：
//                为什么 printf 是"最后手段"）。
//
// MUSA 侧：同文件用 mcc 编译（printf 语义待实测——缓冲上限、截断行为、
// deviceSetLimit 是否提供同名 limit 都是本期探测点）。
// ============================================================================

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d (%s)\n",              \
                         cudaGetErrorString(_err), __FILE__, __LINE__, #expr); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__global__ void k_order() {
    // 一线程一记录：整行一个 printf（记录级原子，行内不被拆）
    printf("B%02d.T%02d\n", blockIdx.x, threadIdx.x);
}

__global__ void k_trunc(int per_thread) {
    const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int k = 0; k < per_thread; ++k) {
        printf("row %lld.%d val=%08d\n", gid, k, static_cast<int>(gid * 31 + k));
    }
}

__global__ void k_frag() {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("a=%04d ", gid);          // 前半行（无换行）
    printf("b=%04d x2=%04d\n", gid, gid * 2);  // 后半行——两条记录间可被任意交错
}

__global__ void k_print_cost(int iters) {
    for (int i = 0; i < iters; ++i) {
        printf("p %d %d\n", blockIdx.x, threadIdx.x);
    }
}

__global__ void k_silent_cost(int iters) {
    float acc = 0.0f;
    for (int i = 0; i < iters; ++i) {
        acc += sinf(static_cast<float>(i));  // 等量算术，防整段被优化掉
    }
    if (acc == 1.234567f) printf("%.6f\n", acc);  // 永假分支保活（无打印路径）
}

static int mode_order() {
    for (int rep = 0; rep < 3; ++rep) {
        std::printf("--- rep %d begin ---\n", rep);
        std::fflush(stdout);
        k_order<<<4, 32>>>();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());  // 同步后 FIFO 才整体吐出
        std::printf("--- rep %d end ---\n", rep);
        std::fflush(stdout);
    }
    return 0;
}

static int mode_trunc(int blocks, int block, int per_thread, int fifo_mb) {
    if (fifo_mb > 0) {
        const size_t bytes = static_cast<size_t>(fifo_mb) * 1024 * 1024;
        const cudaError_t e = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, bytes);
        if (e != cudaSuccess) {
            std::printf("[trunc] setLimit(%dMB) FAILED: %s\n", fifo_mb, cudaGetErrorString(e));
        } else {
            size_t got = 0;
            cudaDeviceGetLimit(&got, cudaLimitPrintfFifoSize);
            std::printf("[trunc] printf FIFO set to %.2f MB\n", static_cast<double>(got) / 1048576.0);
        }
    } else {
        size_t got = 0;
        cudaDeviceGetLimit(&got, cudaLimitPrintfFifoSize);
        std::printf("[trunc] printf FIFO default = %.2f MB\n", static_cast<double>(got) / 1048576.0);
    }
    const long long emitted = static_cast<long long>(blocks) * block * per_thread;
    std::printf("[trunc] emitted=%lld blocks=%d block=%d per_thread=%d\n",
                emitted, blocks, block, per_thread);
    std::fflush(stdout);
    k_trunc<<<blocks, block>>>(per_thread);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("[trunc] kernel done（host 收到的行数由外层 wc -l 统计）\n");
    return 0;
}

static int mode_frag(int blocks, int block) {
    k_frag<<<blocks, block>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

static int mode_cost(int blocks, int block, int iters, int warmup, int repeats) {
    std::vector<cudaEvent_t> st(repeats), sp(repeats);
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventCreate(&st[i]));
        CUDA_CHECK(cudaEventCreate(&sp[i]));
    }
    auto timed = [&](bool do_print, double* ms_out) {
        for (int i = 0; i < warmup; ++i) {
            if (do_print) k_print_cost<<<blocks, block>>>(iters);
            else          k_silent_cost<<<blocks, block>>>(iters);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < repeats; ++i) {
            CUDA_CHECK(cudaEventRecord(st[i]));
            if (do_print) k_print_cost<<<blocks, block>>>(iters);
            else          k_silent_cost<<<blocks, block>>>(iters);
            CUDA_CHECK(cudaEventRecord(sp[i]));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double total = 0.0;
        for (int i = 0; i < repeats; ++i) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, st[i], sp[i]));
            total += ms;
        }
        *ms_out = total / repeats;
    };
    double silent_ms = 0.0, print_ms = 0.0;
    timed(false, &silent_ms);
    timed(true, &print_ms);
    std::printf("[cost] blocks=%d block=%d iters=%d warmup=%d repeats=%d "
                "silent=%.4fms print=%.4fms ratio=%.1fx emitted_lines=%lld\n",
                blocks, block, iters, warmup, repeats, silent_ms, print_ms,
                silent_ms > 0 ? print_ms / silent_ms : 0.0,
                static_cast<long long>(blocks) * block * iters);
    for (int i = 0; i < repeats; ++i) {
        cudaEventDestroy(st[i]);
        cudaEventDestroy(sp[i]);
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = "order";
    int blocks = 4, block = 32, per_thread = 1, fifo_mb = 0, iters = 4, warmup = 3, repeats = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--mode") mode = next();
        else if (a == "--blocks") blocks = std::stoi(next());
        else if (a == "--block") block = std::stoi(next());
        else if (a == "--per-thread") per_thread = std::stoi(next());
        else if (a == "--fifo-mb") fifo_mb = std::stoi(next());
        else if (a == "--iters") iters = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--repeats") repeats = std::stoi(next());
        else { std::fprintf(stderr, "未知参数 %s\n", a.c_str()); return 2; }
    }
    if (mode == "order") return mode_order();
    if (mode == "trunc") return mode_trunc(blocks, block, per_thread, fifo_mb);
    if (mode == "frag")  return mode_frag(blocks, block);
    if (mode == "cost")  return mode_cost(blocks, block, iters, warmup, repeats);
    std::fprintf(stderr, "用法: %s --mode order|trunc|frag|cost [options]\n", argv[0]);
    return 2;
}
