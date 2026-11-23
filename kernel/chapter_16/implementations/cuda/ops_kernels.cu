// 第 16 期五个手撕算子：CUDA 实现（单翻译单元，纯 Kernel 启动，无 Runtime API 调用）。
//
// 设计约定：
// - 所有显存分配/同步/计时在 Python 侧用 torch 完成（experiment_utils 口径），
//   本文件只做 Kernel 启动（默认流），extern "C" 导出 C ABI，ctypes 直接加载；
//   这样 MUSA 迁移版（ops_kernels.mu）可以做到逐行对应，迁移差异全部集中在
//   编译器与启动器库（nvcc/cudart vs mcc/libmusart），不掺杂绑定层差异。
// - 归约原语（warp/block reduce）是算子 2 的主体，同时被 softmax/scan 复用，
//   体现"归约封装"思想。
// - shape 合同：rows ∈ {1,128,2048} × cols ∈ {127,1024,4096}；cols ≤ 4096
//   是 scan 共享内存容量与 softmax 寄存器驻留（每线程 ≤4 元素）的前提。
// - 全部 FP32（与 CPU FP32 真值锚点同 dtype，正确性归因不含 dtype 转换）。

// ---------------------------------------------------------------------------
// 归约原语：warp shuffle 二分归约 + 跨 warp 共享内存二次归约
// ---------------------------------------------------------------------------

// warp 内二分归约（sum）：蝴蝶形交换，5 步覆盖 32 lane。
// __shfl_xor_sync：lane i 与 lane i^offset 互换，每步活跃对折 → O(log32) 步。
__inline__ __device__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// warp 内二分归约（max）：同上，算子换成 fmaxf。
__inline__ __device__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

// block 级归约：warp 归约 → 每 warp lane0 写共享内存 → warp0 归约 → 广播。
// shared 容量要求 ≥ (blockDim.x + 31) / 32 个 float（调用方各给一块）。
__inline__ __device__ float block_reduce_sum(float value, float *shared) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  int warp_count = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float partial = (lane < warp_count) ? shared[lane] : 0.0f;
    partial = warp_reduce_sum(partial);
    if (lane == 0) {
      shared[0] = partial;
    }
  }
  __syncthreads();
  return shared[0];
}

__inline__ __device__ float block_reduce_max(float value, float *shared) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  value = warp_reduce_max(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  int warp_count = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float partial = (lane < warp_count) ? shared[lane] : -INFINITY;
    partial = warp_reduce_max(partial);
    if (lane == 0) {
      shared[0] = partial;
    }
  }
  __syncthreads();
  return shared[0];
}

// host + device 双侧可用（extern "C" 启动器与 Kernel 内都要用）
__inline__ __host__ __device__ int next_pow2(int value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

// ---------------------------------------------------------------------------
// 1. softmax（数值稳定版，一行一个 block，寄存器驻留单次读入）
// ---------------------------------------------------------------------------

// 每线程最多持有 ceil(4096/1024)=4 个元素：max/sum 两阶段都从寄存器读，
// 全局显存只读一遍（2N 字节），这是与"两 Kernel 三遍读"版本的本质区别。
#define SOFTMAX_MAX_CHUNK 4

__global__ void softmax_rows_kernel(const float *x, float *y, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int threads = blockDim.x;
  int chunk = (cols + threads - 1) / threads;

  const float *row_x = x + (size_t)row * cols;
  float local[SOFTMAX_MAX_CHUNK];
  float local_max = -INFINITY;
  for (int k = 0; k < chunk; ++k) {
    int col = tid + k * threads;
    float value = (col < cols) ? row_x[col] : -INFINITY;
    local[k] = value;
    local_max = fmaxf(local_max, value);
  }

  __shared__ float reduce_scratch[32];
  float row_max = block_reduce_max(local_max, reduce_scratch);

  float local_sum = 0.0f;
  for (int k = 0; k < chunk; ++k) {
    local_sum += expf(local[k] - row_max);  // padding 处 exp(-inf)=0，自动屏蔽
  }
  float row_sum = block_reduce_sum(local_sum, reduce_scratch);

  for (int k = 0; k < chunk; ++k) {
    int col = tid + k * threads;
    if (col < cols) {
      y[(size_t)row * cols + col] = expf(local[k] - row_max) / row_sum;
    }
  }
}

extern "C" void softmax_rows(const float *x, float *y, int rows, int cols) {
  int threads = max(32, min(1024, next_pow2(cols)));
  softmax_rows_kernel<<<rows, threads>>>(x, y, rows, cols);
}

// ---------------------------------------------------------------------------
// 2. per-row 归约：sum / max / sum+max 融合（一次读入完成两个归约）
// ---------------------------------------------------------------------------

__global__ void reduce_sum_rows_kernel(const float *x, float *out, int rows, int cols) {
  int row = blockIdx.x;
  const float *row_x = x + (size_t)row * cols;
  float local = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local += row_x[col];
  }
  __shared__ float reduce_scratch[32];
  float total = block_reduce_sum(local, reduce_scratch);
  if (threadIdx.x == 0) {
    out[row] = total;
  }
}

__global__ void reduce_max_rows_kernel(const float *x, float *out, int rows, int cols) {
  int row = blockIdx.x;
  const float *row_x = x + (size_t)row * cols;
  float local = -INFINITY;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local = fmaxf(local, row_x[col]);
  }
  __shared__ float reduce_scratch[32];
  float total = block_reduce_max(local, reduce_scratch);
  if (threadIdx.x == 0) {
    out[row] = total;
  }
}

// 融合版：同一遍读入，sum 与 max 两个归约同时算——读字节数减半的对照样本。
__global__ void reduce_sum_max_rows_kernel(const float *x, float *sum_out, float *max_out,
                                           int rows, int cols) {
  int row = blockIdx.x;
  const float *row_x = x + (size_t)row * cols;
  float local_sum = 0.0f;
  float local_max = -INFINITY;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = row_x[col];
    local_sum += value;
    local_max = fmaxf(local_max, value);
  }
  __shared__ float sum_scratch[32];
  __shared__ float max_scratch[32];
  float total_sum = block_reduce_sum(local_sum, sum_scratch);
  float total_max = block_reduce_max(local_max, max_scratch);
  if (threadIdx.x == 0) {
    sum_out[row] = total_sum;
    max_out[row] = total_max;
  }
}

extern "C" void reduce_sum_rows(const float *x, float *out, int rows, int cols) {
  int threads = max(32, min(1024, next_pow2(cols)));
  reduce_sum_rows_kernel<<<rows, threads>>>(x, out, rows, cols);
}

extern "C" void reduce_max_rows(const float *x, float *out, int rows, int cols) {
  int threads = max(32, min(1024, next_pow2(cols)));
  reduce_max_rows_kernel<<<rows, threads>>>(x, out, rows, cols);
}

extern "C" void reduce_sum_max_rows(const float *x, float *sum_out, float *max_out, int rows,
                                    int cols) {
  int threads = max(32, min(1024, next_pow2(cols)));
  reduce_sum_max_rows_kernel<<<rows, threads>>>(x, sum_out, max_out, rows, cols);
}

// ---------------------------------------------------------------------------
// 3. transpose：朴素版 vs 片上 tile 版（pad / 不 pad 对照 bank conflict）
// ---------------------------------------------------------------------------

// 朴素版：按输出线性顺序遍历（写合并，读以 cols 为步长跨行跳跃）。
__global__ void transpose_naive_kernel(const float *x, float *y, int rows, int cols) {
  long long total = (long long)rows * cols;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long out_index = (long long)blockIdx.x * blockDim.x + threadIdx.x; out_index < total;
       out_index += stride) {
    int col = (int)(out_index / rows);  // 输出行 = 输入列
    int row = (int)(out_index % rows);  // 输出列 = 输入行
    y[out_index] = x[(size_t)row * cols + col];
  }
}

// tile 版：TILE×TILE 片上暂存，读/写两个方向都合并。
// 写阶段从共享内存"转置读" tile[tx][ty]：
//   - PAD=1（stride=33）：warp 各 lane bank = (tx+ty)%32，互不相同 → 无 conflict；
//   - PAD=0（stride=32）：warp 各 lane bank 全部 = ty → 32 路 conflict。
// 用 PAD 模板参数生成两个实例，实测对照 bank conflict 代价。
#define TRANS_TILE 32

template <int PAD>
__global__ void transpose_tiled_kernel(const float *x, float *y, int rows, int cols) {
  __shared__ float tile[TRANS_TILE][TRANS_TILE + PAD];
  int col_base = blockIdx.x * TRANS_TILE;
  int row_base = blockIdx.y * TRANS_TILE;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = col_base + tx;
  int row = row_base + ty;
  if (row < rows && col < cols) {
    tile[ty][tx] = x[(size_t)row * cols + col];  // 读合并：warp 内 col 连续
  }
  __syncthreads();
  int out_row = col_base + ty;  // 输出行 = 输入列
  int out_col = row_base + tx;  // 输出列 = 输入行
  if (out_row < cols && out_col < rows) {
    y[(size_t)out_row * rows + out_col] = tile[tx][ty];  // 写合并 + 共享内存转置读
  }
}

extern "C" void transpose_naive(const float *x, float *y, int rows, int cols) {
  long long total = (long long)rows * cols;
  int block = 256;
  long long blocks_needed = (total + block - 1) / block;
  int blocks = (int)min(blocks_needed, (long long)65535);
  transpose_naive_kernel<<<blocks, block>>>(x, y, rows, cols);
}

extern "C" void transpose_tiled_padded(const float *x, float *y, int rows, int cols) {
  dim3 block(TRANS_TILE, TRANS_TILE);
  dim3 grid((cols + TRANS_TILE - 1) / TRANS_TILE, (rows + TRANS_TILE - 1) / TRANS_TILE);
  transpose_tiled_kernel<1><<<grid, block>>>(x, y, rows, cols);
}

extern "C" void transpose_tiled_unpadded(const float *x, float *y, int rows, int cols) {
  dim3 block(TRANS_TILE, TRANS_TILE);
  dim3 grid((cols + TRANS_TILE - 1) / TRANS_TILE, (rows + TRANS_TILE - 1) / TRANS_TILE);
  transpose_tiled_kernel<0><<<grid, block>>>(x, y, rows, cols);
}

// ---------------------------------------------------------------------------
// 4. inclusive prefix sum / scan：Hillis-Steele（延迟最优）vs Blelloch（工作最优）
//    一行一个 block；cols ≤ 4096 → 补齐到 ≤4096 的 2 的幂，动态共享内存。
// ---------------------------------------------------------------------------

// Hillis-Steele：双缓冲，log2(N) 步，每步全部 N 个槽位做一次加法 → O(N log N) 工作。
__global__ void scan_hillis_kernel(const float *x, float *y, int rows, int cols, int padded) {
  extern __shared__ float smem[];
  float *buf_a = smem;
  float *buf_b = smem + padded;
  int row = blockIdx.x;
  const float *row_x = x + (size_t)row * cols;
  for (int col = threadIdx.x; col < padded; col += blockDim.x) {
    buf_a[col] = (col < cols) ? row_x[col] : 0.0f;  // 加法单位元填充
  }
  __syncthreads();
  float *src = buf_a;
  float *dst = buf_b;
  for (int offset = 1; offset < padded; offset <<= 1) {
    for (int col = threadIdx.x; col < padded; col += blockDim.x) {
      dst[col] = src[col] + ((col >= offset) ? src[col - offset] : 0.0f);
    }
    __syncthreads();
    float *tmp = src;
    src = dst;
    dst = tmp;
  }
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    y[(size_t)row * cols + col] = src[col];
  }
}

// Blelloch：up-sweep 归约到根 → 根清零 → down-sweep 下推，得排他 scan；
// 共 2*log2(N) 步、约 2N 次加法 → O(N) 工作。排他结果加原值转 inclusive。
// 每线程寄存器保留自己 load 过的原始值（≤4 个），避免第二遍全局读。
#define SCAN_MAX_CHUNK 4

__global__ void scan_blelloch_kernel(const float *x, float *y, int rows, int cols, int padded) {
  extern __shared__ float buf[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  const float *row_x = x + (size_t)row * cols;
  float original[SCAN_MAX_CHUNK];
  int chunk = (cols + blockDim.x - 1) / blockDim.x;
  if (chunk > SCAN_MAX_CHUNK) {
    chunk = SCAN_MAX_CHUNK;  // 合同 cols ≤ 4096、threads ≥ 1024 时不会触发
  }
  for (int k = 0; k < chunk; ++k) {
    int col = tid + k * blockDim.x;
    float value = 0.0f;
    if (col < cols) {
      value = row_x[col];
    }
    original[k] = value;
    if (col < padded) {
      buf[col] = value;
    }
  }
  // up-sweep
  for (int d = 1; d < padded; d <<= 1) {
    __syncthreads();
    int slots = padded / (2 * d);
    for (int s = tid; s < slots; s += blockDim.x) {
      int i = s * (2 * d);
      buf[i + 2 * d - 1] += buf[i + d - 1];
    }
  }
  __syncthreads();
  if (tid == 0) {
    buf[padded - 1] = 0.0f;
  }
  // down-sweep
  for (int d = padded >> 1; d >= 1; d >>= 1) {
    __syncthreads();
    int slots = padded / (2 * d);
    for (int s = tid; s < slots; s += blockDim.x) {
      int i = s * (2 * d);
      float t = buf[i + d - 1];
      buf[i + d - 1] = buf[i + 2 * d - 1];
      buf[i + 2 * d - 1] += t;
    }
  }
  __syncthreads();
  for (int k = 0; k < chunk; ++k) {
    int col = tid + k * blockDim.x;
    if (col < cols) {
      y[(size_t)row * cols + col] = buf[col] + original[k];  // 排他 + 原值 = inclusive
    }
  }
}

extern "C" void scan_hillis(const float *x, float *y, int rows, int cols) {
  int padded = next_pow2(cols);
  int threads = min(1024, padded);
  size_t shared_bytes = (size_t)2 * padded * sizeof(float);
  scan_hillis_kernel<<<rows, threads, shared_bytes>>>(x, y, rows, cols, padded);
}

extern "C" void scan_blelloch(const float *x, float *y, int rows, int cols) {
  int padded = next_pow2(cols);
  int threads = min(1024, padded);
  size_t shared_bytes = (size_t)padded * sizeof(float);
  scan_blelloch_kernel<<<rows, threads, shared_bytes>>>(x, y, rows, cols, padded);
}

// ---------------------------------------------------------------------------
// 5. SiLU + Gate Multiply：分离两 Kernel vs 融合单 Kernel vs 融合 float4
//    语义：y[i] = silu(gate[i]) * up[i]，silu(g) = g / (1 + exp(-g))
// ---------------------------------------------------------------------------

__global__ void silu_only_kernel(const float *gate, float *tmp, long long n) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    float g = gate[i];
    tmp[i] = g / (1.0f + expf(-g));
  }
}

__global__ void mul_only_kernel(const float *tmp, const float *up, float *y, long long n) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    y[i] = tmp[i] * up[i];
  }
}

__global__ void silu_gate_fused_kernel(const float *gate, const float *up, float *y,
                                       long long n) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    float g = gate[i];
    y[i] = (g / (1.0f + expf(-g))) * up[i];
  }
}

// float4 向量化版：n 为 4 的倍数且指针 16B 对齐时启用（torch 分配天然 256B 对齐）。
__global__ void silu_gate_fused_v4_kernel(const float4 *gate, const float4 *up, float4 *y,
                                          long long vec4_count) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < vec4_count;
       i += stride) {
    float4 g = gate[i];
    float4 u = up[i];
    float4 r;
    r.x = (g.x / (1.0f + expf(-g.x))) * u.x;
    r.y = (g.y / (1.0f + expf(-g.y))) * u.y;
    r.z = (g.z / (1.0f + expf(-g.z))) * u.z;
    r.w = (g.w / (1.0f + expf(-g.w))) * u.w;
    y[i] = r;
  }
}

static int elementwise_grid(long long n, int block) {
  long long needed = (n + block - 1) / block;
  return (int)min(needed, (long long)65535);
}

extern "C" void silu_only(const float *gate, float *tmp, long long n) {
  silu_only_kernel<<<elementwise_grid(n, 256), 256>>>(gate, tmp, n);
}

extern "C" void mul_only(const float *tmp, const float *up, float *y, long long n) {
  mul_only_kernel<<<elementwise_grid(n, 256), 256>>>(tmp, up, y, n);
}

extern "C" void silu_gate_fused(const float *gate, const float *up, float *y, long long n) {
  silu_gate_fused_kernel<<<elementwise_grid(n, 256), 256>>>(gate, up, y, n);
}

extern "C" int silu_gate_fused_v4(const float *gate, const float *up, float *y, long long n) {
  if (n % 4 != 0) {
    return -1;  // 让 Python 侧回退标量版并记录（127 列的 shape 不满足对齐）
  }
  long long vec4_count = n / 4;
  silu_gate_fused_v4_kernel<<<elementwise_grid(vec4_count, 256), 256>>>(
      (const float4 *)gate, (const float4 *)up, (float4 *)y, vec4_count);
  return 0;
}
