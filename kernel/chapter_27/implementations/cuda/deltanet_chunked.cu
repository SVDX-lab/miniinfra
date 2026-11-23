// 第 27 期 CUDA chunked Gated DeltaNet 核实现（回填版）。
// 设计：deltanet_chunked_design.md（K1/K2/K3 分相位——nsys 分相位证据的前提）。
// 算法基准：implementations/cpu/deltanet_chunked.py（标量门控主线）与
// kernel_sim.py（(H,NC) 块粒度仿真 12/12 位级）。
// 口径：fp32 接口 + fp32 计算（fp16io = 输入以 fp16 存储、host cast 后进核，
// 与 FLA naive 参考的 cast-to-fp32 行为一致；见 EXPERIMENT_LOG §5）。
// 数值安全（设计文档 §2 全部落实）：
//   - 所有 expf 参数 ≤ 0（G 单调不增 + tril 掩码语义），无上溢路径；
//   - scale(1/√K) 只在输出处乘（o 对 q_scaled 线性，可提取）；
//   - padding 行零填充（q/k/v/β/g 越界读 0）——padding 语义中性；
//   - 无 atomics，块内串行深度 (C−1)(K2) + NC(K3)。
#include <torch/extension.h>
#include <cuda_runtime.h>

#define KDIM 128
#define VDIM 128
#define NTHREADS 256
#define HALF_COLS (VDIM / 2)

// ---------------------------------------------------------------- K1 intra
// grid=(NC, H) block=256；每 CTA：块内 cumsum → M1（严格下三角取负）+ 两 payload。
// smem：k_blk[C][K] + g[C]（动态）。越界行读 0（padding 中性）。
__global__ void gdn_k1_intra(
    const float* __restrict__ q,       // [N,H,K]（未缩放；scale 不进 K1）
    const float* __restrict__ k,       // [N,H,K]
    const float* __restrict__ v,       // [N,H,V]
    const float* __restrict__ beta,    // [N,H]
    const float* __restrict__ g_log,   // [N,H]
    float* __restrict__ m1,            // [NC,H,C,C]
    float* __restrict__ wpay,          // [NC,H,C,V]   βv
    float* __restrict__ kpay,          // [NC,H,C,K]   kβe^G
    const int n, const int nc, const int c)
{
    const int ci = blockIdx.x;
    const int hh = blockIdx.y;
    extern __shared__ float smem[];
    float* k_blk = smem;               // [c][KDIM]
    float* g = smem + (size_t)c * KDIM; // [c]

    const int tid = threadIdx.x;
    const int base = ci * c;

    // 载入 k 块（越界 0）与 g（越界 0）
    for (int idx = tid; idx < c * KDIM; idx += NTHREADS) {
        int j = idx / KDIM, kk = idx % KDIM;
        int row = base + j;
        k_blk[idx] = (row < n) ? k[(size_t)row * gridDim.y * KDIM + hh * KDIM + kk] : 0.0f;
    }
    if (tid < c) {
        int row = base + tid;
        g[tid] = (row < n) ? g_log[(size_t)row * gridDim.y + hh] : 0.0f;
    }
    __syncthreads();

    // 块内 cumsum（串行 C 步，线程 0）
    if (tid == 0) {
        float acc = 0.0f;
        for (int j = 0; j < c; ++j) { acc += g[j]; g[j] = acc; }
    }
    __syncthreads();

    // M1：thread 每 (i,j) 对，j<i：−β_i·(k_i·k_j)·e^{G_i−G_j}；其余 0
    const float* beta_h = beta + hh;
    for (int idx = tid; idx < c * c; idx += NTHREADS) {
        int i = idx / c, j = idx % c;
        float val = 0.0f;
        if (j < i) {
            const float* ki = k_blk + (size_t)i * KDIM;
            const float* kj = k_blk + (size_t)j * KDIM;
            float dot = 0.0f;
            #pragma unroll 4
            for (int kk = 0; kk < KDIM; ++kk) dot += ki[kk] * kj[kk];
            int row = base + i;
            float bi = (row < n) ? beta_h[(size_t)row * gridDim.y] : 0.0f;
            val = -bi * dot * expf(g[i] - g[j]);   // 指数 ≤ 0
        }
        m1[((size_t)ci * gridDim.y + hh) * c * c + idx] = val;
    }

    // wpay = βv；kpay = kβe^G（越界行 β=0 → 全 0）
    for (int idx = tid; idx < c * VDIM; idx += NTHREADS) {
        int j = idx / VDIM, col = idx % VDIM;
        int row = base + j;
        float b = (row < n) ? beta_h[(size_t)row * gridDim.y] : 0.0f;
        float vv = (row < n) ? v[(size_t)row * gridDim.y * VDIM + hh * VDIM + col] : 0.0f;
        wpay[((size_t)ci * gridDim.y + hh) * c * VDIM + idx] = b * vv;
    }
    for (int idx = tid; idx < c * KDIM; idx += NTHREADS) {
        int j = idx / KDIM, kk = idx % KDIM;
        int row = base + j;
        float b = (row < n) ? beta_h[(size_t)row * gridDim.y] : 0.0f;
        kpay[((size_t)ci * gridDim.y + hh) * c * KDIM + idx] =
            b * k_blk[idx] * expf(g[j]);
    }
}

// ---------------------------------------------------------------- K2 ut_solve
// grid=(NC, H) block=256；前向替换 X_i = P_i + Σ_{j<i} m1[i][j]·X_j（就地）。
// 线程 t<128 解 w 的列 t（payload βv），t≥128 解 kd 的列 t−128（payload kβe^G）。
// X 驻留 global（L2 热）；smem：m1 行 [C]。
__global__ void gdn_k2_ut(
    float* __restrict__ w,             // [NC,H,C,V] in βv → out T(βv)
    float* __restrict__ kd,            // [NC,H,C,K] in kβe^G → out T(kβe^G)
    const float* __restrict__ m1,      // [NC,H,C,C]
    const int nc, const int c)
{
    const int ci = blockIdx.x;
    const int hh = blockIdx.y;
    const int tid = threadIdx.x;
    extern __shared__ float negrow[];  // [c]

    const bool is_w = tid < VDIM;
    const int col = is_w ? tid : (tid - VDIM);
    float* X = is_w
        ? w + ((size_t)ci * gridDim.y + hh) * c * VDIM
        : kd + ((size_t)ci * gridDim.y + hh) * c * KDIM;
    const int stride = is_w ? VDIM : KDIM;
    const float* M1 = m1 + ((size_t)ci * gridDim.y + hh) * c * c;

    for (int i = 1; i < c; ++i) {
        // m1 行 i 载入 smem（本线程串行段内广播）
        for (int j = tid; j < c; j += NTHREADS) negrow[j] = M1[(size_t)i * c + j];
        __syncthreads();
        float acc = X[(size_t)i * stride + col];           // P_i
        for (int j = 0; j < i; ++j) {
            float nij = negrow[j];
            if (nij != 0.0f) acc += nij * X[(size_t)j * stride + col];
        }
        X[(size_t)i * stride + col] = acc;
        __syncthreads();
    }
}

// ---------------------------------------------------------------- K3 carry
// grid=(H, VDIM/2) block=256；串行 NC 步。scheme A：state 驻留 smem；
// scheme B：state 驻留 global（state_out 就地演化，L2 热）。
// 每步 5 个 pass（块内 sync 分隔）：cumsum → attn → vnew → o → state 更新。
// scale 只在 o 写出时乘。attn/vnew 经 global scratch（L2 热）。
__global__ void gdn_k3_carry(
    const float* __restrict__ q,       // [N,H,K]
    const float* __restrict__ k,       // [N,H,K]
    const float* __restrict__ g_log,   // [N,H]
    const float* __restrict__ w,       // [NC,H,C,V]  已解
    const float* __restrict__ kd,      // [NC,H,C,K]  已解
    float* __restrict__ o,             // [N,H,V]
    float* __restrict__ state,         // [H,K,V]     进=初始 出=最终
    float* __restrict__ vnew_scr,      // [H,C,V]
    float* __restrict__ attn_scr,      // [H,C,C]
    const int n, const int nc, const int c,
    const float scale, const int state_in_smem)
{
    const int hh = blockIdx.x;
    const int half = blockIdx.y;
    const int tid = threadIdx.x;
    const int col0 = half * HALF_COLS;

    extern __shared__ float smem[];
    float* st = smem;                  // scheme A: [K][HALF_COLS]；scheme B: g[C]
    float* g = smem + (state_in_smem ? (size_t)KDIM * HALF_COLS : 0); // [c]

    // 状态载入（A：smem；B：就地 global）
    if (state_in_smem) {
        for (int idx = tid; idx < KDIM * HALF_COLS; idx += NTHREADS) {
            int kk = idx / HALF_COLS, col = idx % HALF_COLS;
            st[idx] = state[((size_t)hh * KDIM + kk) * VDIM + col0 + col];
        }
    }
    __syncthreads();

    for (int ci = 0; ci < nc; ++ci) {
        const int base = ci * c;
        // pass 0：g 载入 + cumsum（线程 0 串行）
        if (tid < c) {
            int row = base + tid;
            g[tid] = (row < n) ? g_log[(size_t)row * gridDim.x + hh] : 0.0f;
        }
        __syncthreads();
        if (tid == 0) {
            float acc = 0.0f;
            for (int j = 0; j < c; ++j) { acc += g[j]; g[j] = acc; }
        }
        __syncthreads();
        const float g_last = g[c - 1];

        // pass 1：attn_raw[c][i] = Σ_k q[c][k]·e^{G_c−G_i}·k_i[k]，i≤c；i>c 置 0
        for (int idx = tid; idx < c * c; idx += NTHREADS) {
            int i = idx / c, j = idx % c;
            float val = 0.0f;
            if (j <= i) {
                int rowc = base + i, rowj = base + j;
                const float* qi = q + (size_t)rowc * gridDim.x * KDIM + hh * KDIM;
                const float* kj = k + (size_t)rowj * gridDim.x * KDIM + hh * KDIM;
                float dot = 0.0f;
                for (int kk = 0; kk < KDIM; ++kk) {
                    float qv = (rowc < n) ? qi[kk] : 0.0f;
                    float kv = (rowj < n) ? kj[kk] : 0.0f;
                    dot += qv * kv;
                }
                val = dot * expf(g[i] - g[j]);   // i≥j → 指数 ≤ 0
            }
            attn_scr[(((size_t)hh * 2 + half) * c) * c + idx] = val;
        }
        __syncthreads();

        // pass 2：v_new[c][col] = w[c][col] − Σ_k kd[c][k]·S[k][col]
        const float* kd_c = kd + ((size_t)ci * gridDim.x + hh) * c * KDIM;
        const float* w_c = w + ((size_t)ci * gridDim.x + hh) * c * VDIM;
        for (int idx = tid; idx < c * HALF_COLS; idx += NTHREADS) {
            int i = idx / HALF_COLS, col = idx % HALF_COLS;
            int row = base + i;
            float acc = (row < n) ? w_c[(size_t)i * VDIM + col0 + col] : 0.0f;
            for (int kk = 0; kk < KDIM; ++kk) {
                float s = state_in_smem ? st[(size_t)kk * HALF_COLS + col]
                                        : state[((size_t)hh * KDIM + kk) * VDIM + col0 + col];
                acc -= kd_c[(size_t)i * KDIM + kk] * s;
            }
            vnew_scr[((size_t)hh * c) * VDIM + (size_t)i * VDIM + col0 + col] = acc;
        }
        __syncthreads();

        // pass 3：o = scale·( (q⊙e^G)ᵀS + Σ_{i≤c} attn[c][i]·v_new[i][col] )
        for (int idx = tid; idx < c * HALF_COLS; idx += NTHREADS) {
            int i = idx / HALF_COLS, col = idx % HALF_COLS;
            int row = base + i;
            if (row < n) {
                float o_inter = 0.0f;
                const float* qi = q + (size_t)row * gridDim.x * KDIM + hh * KDIM;
                float eg = expf(g[i]);
                for (int kk = 0; kk < KDIM; ++kk) {
                    float s = state_in_smem ? st[(size_t)kk * HALF_COLS + col]
                                            : state[((size_t)hh * KDIM + kk) * VDIM + col0 + col];
                    o_inter += qi[kk] * eg * s;
                }
                float acc = o_inter;
                const float* attn_row = attn_scr + ((size_t)hh * 2 + half) * c * c + (size_t)i * c;
                for (int j = 0; j <= i; ++j)
                    acc += attn_row[j] * vnew_scr[((size_t)hh * c) * VDIM + (size_t)j * VDIM + col0 + col];
                o[((size_t)row * gridDim.x + hh) * VDIM + col0 + col] = scale * acc;
            }
        }
        __syncthreads();

        // pass 4：S[k][col] = S·e^{G_last} + Σ_c k[c][k]·e^{G_last−G_c}·v_new[c][col]
        for (int idx = tid; idx < KDIM * HALF_COLS; idx += NTHREADS) {
            int kk = idx / HALF_COLS, col = idx % HALF_COLS;
            float acc = 0.0f;
            for (int i = 0; i < c; ++i) {
                int row = base + i;
                float kv = (row < n) ? k[(size_t)row * gridDim.x * KDIM + hh * KDIM + kk] : 0.0f;
                acc += kv * expf(g_last - g[i]) * vnew_scr[((size_t)hh * c) * VDIM + (size_t)i * VDIM + col0 + col];
            }
            float decayed;
            if (state_in_smem) {
                decayed = st[idx] * expf(g_last);
                st[idx] = decayed + acc;
            } else {
                size_t off = ((size_t)hh * KDIM + kk) * VDIM + col0 + col;
                decayed = state[off] * expf(g_last);
                state[off] = decayed + acc;
            }
        }
        __syncthreads();
    }

    if (state_in_smem) {
        for (int idx = tid; idx < KDIM * HALF_COLS; idx += NTHREADS) {
            int kk = idx / HALF_COLS, col = idx % HALF_COLS;
            state[((size_t)hh * KDIM + kk) * VDIM + col0 + col] = st[idx];
        }
    }
}

// ---------------------------------------------------------------- host
std::vector<torch::Tensor> gdn_chunked_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor beta, torch::Tensor g_log,
    int64_t chunk_size, double scale,
    std::optional<torch::Tensor> initial_state,
    int64_t scheme)   // 0=smem state（方案A）, 1=global state（方案B）
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && beta.is_cuda() && g_log.is_cuda(),
                "inputs must be cuda");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "fp32 interface（fp16io 由 host cast）");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
                && beta.is_contiguous() && g_log.is_contiguous(), "contiguous required");
    const int n = (int)q.size(0);
    const int h = (int)q.size(1);
    TORCH_CHECK((int)q.size(2) == KDIM, "head_k_dim must be ", KDIM);
    TORCH_CHECK((int)v.size(2) == VDIM, "head_v_dim must be ", VDIM);
    TORCH_CHECK((int)k.size(0) == n && (int)k.size(1) == h, "k shape mismatch");
    TORCH_CHECK((int)v.size(0) == n && (int)v.size(1) == h, "v shape mismatch");
    TORCH_CHECK((int)beta.size(0) == n && (int)beta.size(1) == h, "beta shape mismatch");
    TORCH_CHECK((int)g_log.size(0) == n && (int)g_log.size(1) == h, "g shape mismatch");
    const int c = (int)chunk_size;
    TORCH_CHECK(c >= 1 && c <= 64, "chunk_size 1..64");

    const int nc = (n + c - 1) / c;
    auto opts = q.options();
    auto o = torch::empty({n, (int64_t)h, VDIM}, opts);
    auto state_out = initial_state.has_value() && initial_state->defined()
        ? initial_state->to(opts).clone()
        : torch::zeros({(int64_t)h, KDIM, VDIM}, opts);
    auto m1 = torch::empty({(int64_t)nc, (int64_t)h, c, c}, opts);
    auto wpay = torch::empty({(int64_t)nc, (int64_t)h, c, VDIM}, opts);
    auto kpay = torch::empty({(int64_t)nc, (int64_t)h, c, KDIM}, opts);
    auto vnew_scr = torch::empty({(int64_t)h, c, VDIM}, opts);
    // attn scratch 每 half 独立——同头两个 half 块无互同步，共享 [H][C][C]
    // 会被块间 chunk 进度漂移互相覆盖（N=2048 间歇性 o 错误的根因）
    auto attn_scr = torch::empty({(int64_t)h, 2, c, c}, opts);

    dim3 grid1(nc, h);
    size_t smem1 = (size_t)c * KDIM * 4 + (size_t)c * 4;
    gdn_k1_intra<<<grid1, NTHREADS, smem1>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        beta.data_ptr<float>(), g_log.data_ptr<float>(),
        m1.data_ptr<float>(), wpay.data_ptr<float>(), kpay.data_ptr<float>(),
        n, nc, c);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "K1 launch failed: ", cudaGetErrorString(err));

    dim3 grid2(nc, h);
    size_t smem2 = (size_t)c * 4;
    gdn_k2_ut<<<grid2, NTHREADS, smem2>>>(
        wpay.data_ptr<float>(), kpay.data_ptr<float>(), m1.data_ptr<float>(),
        nc, c);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "K2 launch failed: ", cudaGetErrorString(err));

    dim3 grid3(h, VDIM / HALF_COLS);   // 2 个 half 块（不是 64！）
    const bool smem_state = (scheme == 0);
    size_t smem3 = smem_state ? (size_t)KDIM * HALF_COLS * 4 + (size_t)c * 4
                              : (size_t)c * 4;
    gdn_k3_carry<<<grid3, NTHREADS, smem3>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), g_log.data_ptr<float>(),
        wpay.data_ptr<float>(), kpay.data_ptr<float>(),
        o.data_ptr<float>(), state_out.data_ptr<float>(),
        vnew_scr.data_ptr<float>(), attn_scr.data_ptr<float>(),
        n, nc, c, (float)scale, smem_state ? 1 : 0);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "K3 launch failed: ", cudaGetErrorString(err));

    return {o, state_out, m1, wpay, kpay, vnew_scr, attn_scr};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_chunked_forward", &gdn_chunked_forward,
          "chunked gated deltanet (K1/K2/K3, scheme 0=smem/1=global)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("beta"),
          py::arg("g_log"), py::arg("chunk_size"), py::arg("scale"),
          py::arg("initial_state") = std::nullopt,
          py::arg("scheme") = 0);
}
