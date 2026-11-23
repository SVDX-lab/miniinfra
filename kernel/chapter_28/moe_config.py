"""第 28 期 MoE 配置单源：公开 config 出处 + 扫描矩阵 + 合成设定声明。

所有公开参数在 2026-11-21 从 HuggingFace 抓取（URL 见各 dict 的 source 字段），
合成/派生设定显式标 SYNTHETIC，不做静默混用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEED_BASE = 20260329  # 本期 fixtures/权重固定 seed（沿用 ep26/28 约定格式）

# ---------------------------------------------------------------------------
# 公开 MoE config（性能/估算校核用；抓取 2026-11-21）
# ---------------------------------------------------------------------------


@dataclass
class PublicMoEConfig:
    name: str
    d_model: int
    d_ff_moe: int                 # moe_intermediate_size（单专家 FFN 中间维）
    n_experts: int                # 路由专家数（不含共享专家）
    topk: int                     # 每 token 激活专家数 num_experts_per_tok
    n_layers: int
    n_shared: int = 0
    first_k_dense_replace: int = 0  # 前 k 层稠密 FFN（V2/V3 口径）
    n_moe_layers: int = 0           # 显式给定优先；否则 = n_layers - first_k_dense_replace
    dense_ff: int = 0             # 稠密层 intermediate_size（V2-Lite 第一层）
    norm_topk_prob: bool = True
    # 注意力（激活参数估算用，MLA 与 MHA 两口径）
    attn_kind: str = "mha"        # "mha" | "mla"
    q_heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    q_lora_rank: int | None = None
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    vocab: int = 0
    tie_word_embeddings: bool = False
    # 已公开的总参/激活参数（B；来源=模型卡公开数字，用于估算流程校核）
    published_total_b: float | None = None
    published_active_b: float | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def moe_layers(self) -> int:
        if self.n_moe_layers:
            return self.n_moe_layers
        return self.n_layers - self.first_k_dense_replace


QWEN3_30B_A3B = PublicMoEConfig(
    name="Qwen3-30B-A3B-Instruct-2507",
    d_model=2048, d_ff_moe=768, n_experts=128, topk=8,
    n_layers=48, norm_topk_prob=True,
    attn_kind="mha", q_heads=32, kv_heads=4, head_dim=128,
    vocab=151936, tie_word_embeddings=False,
    published_total_b=30.5, published_active_b=3.3,
    source="https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json"
           "（num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768,"
           " decoder_sparse_step=1, mlp_only_layers=[]——无稠密 FFN 层；抓取 2026-11-21）",
    notes=["mlp_only_layers=[] + decoder_sparse_step=1 ⇒ n_moe_layers = n_layers = 48"],
)

DS_V2_LITE = PublicMoEConfig(
    name="DeepSeek-V2-Lite",
    d_model=2048, d_ff_moe=1408, n_experts=64, topk=6,
    n_layers=27, first_k_dense_replace=1, n_moe_layers=26, n_shared=2,
    dense_ff=10944, norm_topk_prob=False,
    attn_kind="mla", q_heads=16, kv_heads=16, head_dim=128,
    q_lora_rank=None, kv_lora_rank=512, qk_nope_head_dim=128,
    qk_rope_head_dim=64, v_head_dim=128,
    vocab=102400, tie_word_embeddings=False,
    published_total_b=15.7, published_active_b=2.4,
    source="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json"
           "（n_routed_experts=64, num_experts_per_tok=6, moe_intermediate_size=1408,"
           " n_shared_experts=2, first_k_dense_replace=1, norm_topk_prob=false,"
           " topk_method=greedy；抓取 2026-11-21）",
    notes=["norm_topk_prob=false + topk_method=greedy + sigmoid scoring（V2 系口径）："
           "权重不做 top-k 内归一化——本期参考实现提供 norm 开关（见 moe_reference.route_topk），"
           "主链按 Qwen3 口径 norm=true，V2-Lite 口径在合同中作变体声明",
           "总参/激活公开数字 15.7B/2.4B = DeepSeek-V2-Lite 模型卡口径"],
)

MIXTRAL_8X7B = PublicMoEConfig(
    name="Mixtral-8x7B-v0.1",
    d_model=4096, d_ff_moe=14336, n_experts=8, topk=2,
    n_layers=32, norm_topk_prob=False,
    attn_kind="mha", q_heads=32, kv_heads=8, head_dim=128,
    vocab=32000, tie_word_embeddings=False,
    published_total_b=46.7, published_active_b=12.9,
    source="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json"
           "（num_local_experts=8, num_experts_per_tok=2, intermediate_size=14336；抓取 2026-11-21）",
    notes=["E=8/k=2 是本扫描矩阵 experts=8 列的公开结构锚点"],
)

DS_V3 = PublicMoEConfig(
    name="DeepSeek-V3",
    d_model=7168, d_ff_moe=2048, n_experts=256, topk=8,
    n_layers=61, first_k_dense_replace=3, n_moe_layers=58, n_shared=1,
    dense_ff=18432, norm_topk_prob=True,
    attn_kind="mla", q_heads=128, kv_heads=128, head_dim=128,
    q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
    qk_rope_head_dim=64, v_head_dim=128,
    vocab=129280, tie_word_embeddings=False,
    published_total_b=671.0, published_active_b=37.0,
    source="https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json"
           "（n_routed_experts=256, num_experts_per_tok=8, moe_intermediate_size=2048,"
           " n_shared_experts=1, first_k_dense_replace=3, routed_scaling_factor=2.5；抓取 2026-11-21）",
    notes=["V3 激活参数的公开口径 37B 作为附录 B 题 11 估算流程的第三校核点"],
)

KIMI_K2 = PublicMoEConfig(
    name="Kimi-K2-Instruct",
    d_model=7168, d_ff_moe=2048, n_experts=384, topk=8,
    n_layers=61, first_k_dense_replace=1, n_moe_layers=60, n_shared=1,
    norm_topk_prob=True,
    attn_kind="mla", q_heads=64, kv_heads=64, head_dim=128,
    q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
    qk_rope_head_dim=64, v_head_dim=128,
    vocab=163840, tie_word_embeddings=False,
    published_total_b=1000.0, published_active_b=32.0,
    source="https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json"
           "（n_routed_experts=384, num_experts_per_tok=8, moe_intermediate_size=2048,"
           " first_k_dense_replace=1, q_heads=64；总参/激活 1T/32B = 模型卡口径；抓取 2026-11-21）",
    notes=["K2 是 V4-Pro 同类（万亿级、激活比 ~3%）的最近公开参照"],
)

PUBLIC_CONFIGS = [QWEN3_30B_A3B, DS_V2_LITE, MIXTRAL_8X7B, DS_V3, KIMI_K2]

# ---------------------------------------------------------------------------
# 实验扫描 config（性能矩阵）——SYNTHETIC 设定显式声明
# ---------------------------------------------------------------------------

# 主扫描（实验 2 路由开销矩阵 / 实验 3 负载不均）：tokens × experts × topk
SCAN_TOKENS = [1, 8, 64, 256, 2048]
SCAN_EXPERTS = [8, 64]
SCAN_TOPK = [2, 6, 8]
# 扫描矩阵的 FFN 维统一取 V2-Lite 口径（d_ff=1408）以便 experts/topk 单变量扫描：
# E=8/k=2 结构锚点 = Mixtral-8x7B（E=8,k=2），E=64/k=6 = V2-Lite，k=8 = Qwen3；
# d_model=2048 为两公开 config 公共值。d_ff 不随 E 切换（SYNTHETIC 声明）。
SCAN_D_MODEL = 2048
SCAN_D_FF = 1408

# 正确性/负载不均主 config（E=64 = V2-Lite 专家数；topk=6 = V2-Lite 口径）
MAIN_E = 64
MAIN_TOPK = 6
MAIN_N_TOKENS_BENCH = 2048

# 实验 3 负载不均（Zipf 倾斜）构造参数（ep26 planted 注入同思想）
ZIPF_HOT_FRAC = 0.2    # 20% 专家为热专家
ZIPF_TOKEN_FRAC = 0.8  # 80% token 的 top-1 落在热集
ZIPF_ALPHA = 1.0       # 热集内 Zipf 指数（p_i ∝ 1/(i+1)^alpha）

# 实验 5 权重读取字节（dtype 扫描）
TRAFFIC_DTYPES = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}


def scan_ffn_bytes(dtype_bytes: float) -> int:
    """单专家权重字节 = 3 × d_model × d_ff × dtype_bytes（gate/up/down 三投影）。"""
    return round(3 * SCAN_D_MODEL * SCAN_D_FF * dtype_bytes)


def expert_params(d_model: int, d_ff: int) -> int:
    """单专家参数 = 3 × d_model × d_ff（gate + up + down 投影，无 bias——主流 MoE 口径）。"""
    return 3 * d_model * d_ff
