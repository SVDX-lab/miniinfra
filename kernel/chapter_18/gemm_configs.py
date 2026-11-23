"""第 17 期 tile 配置表（Python 侧单源，与 naive_gemm.cu 的 config_table 严格一致）。

第 18 期 WMMA 对照沿用同一张表（同 tile 配置、同形状、同计时口径）。
"""

from __future__ import annotations

# (name, TILE_M, TILE_N, TILE_K, REG_M, REG_N, epi_variants)
# block = (TILE_N/REG_N, TILE_M/REG_M)（x=列快维，y=行）
# smem  = (TM*(TK+1) + TK*(TN+1)) * 4 字节（+1 padding 消 bank conflict）
CONFIGS = [
    {"name": "16x16x16_reg2x2", "tm": 16, "tn": 16, "tk": 16, "rm": 2, "rn": 2, "epi": False},
    {"name": "16x16x16_reg4x4", "tm": 16, "tn": 16, "tk": 16, "rm": 4, "rn": 4, "epi": False},
    {"name": "32x32x8_reg2x2", "tm": 32, "tn": 32, "tk": 8, "rm": 2, "rn": 2, "epi": True},
    {"name": "32x32x8_reg4x4", "tm": 32, "tn": 32, "tk": 8, "rm": 4, "rn": 4, "epi": False},
    {"name": "32x64x8_reg2x2", "tm": 32, "tn": 64, "tk": 8, "rm": 2, "rn": 2, "epi": False},
    {"name": "32x64x8_reg4x4", "tm": 32, "tn": 64, "tk": 8, "rm": 4, "rn": 4, "epi": False},
    {"name": "64x32x8_reg2x2", "tm": 64, "tn": 32, "tk": 8, "rm": 2, "rn": 2, "epi": False},
    {"name": "64x32x8_reg4x4", "tm": 64, "tn": 32, "tk": 8, "rm": 4, "rn": 4, "epi": False},
    {"name": "64x64x8_reg2x2", "tm": 64, "tn": 64, "tk": 8, "rm": 2, "rn": 2, "epi": False},
    {"name": "64x64x8_reg4x4", "tm": 64, "tn": 64, "tk": 8, "rm": 4, "rn": 4, "epi": True},
    {"name": "64x64x8_reg8x8", "tm": 64, "tn": 64, "tk": 8, "rm": 8, "rn": 8, "epi": False},
]


def parse_config(name: str) -> dict:
    for config in CONFIGS:
        if config["name"] == name:
            return dict(config)
    raise ValueError(f"未知配置: {name}（可选: {[c['name'] for c in CONFIGS]}）")


def block_dims(config: dict) -> tuple[int, int]:
    """(block.x, block.y) = (TN/RN, TM/RM)。"""
    return config["tn"] // config["rn"], config["tm"] // config["rm"]


def grid_dims(config: dict, m: int, n: int) -> tuple[int, int]:
    """(grid.x, grid.y) = (ceil(N/TN), ceil(M/TM))。"""
    return -(-n // config["tn"]), -(-m // config["tm"])


def smem_bytes(config: dict) -> int:
    """静态共享内存字节数（与 kernel 中 As[TM][TK+1] + Bs[TK][TN+1] 一致）。"""
    return (config["tm"] * (config["tk"] + 1) + config["tk"] * (config["tn"] + 1)) * 4


def accumulator_count(config: dict) -> int:
    """每线程寄存器累加器数（寄存器分块深度）。"""
    return config["rm"] * config["rn"]


def epilogue_code(epilogue: str) -> int:
    """epilogue 名称 → 模板参数。0=无 1=bias 2=bias_gelu 3=bias_silu。"""
    return {"none": 0, "bias": 1, "bias_gelu": 2, "bias_silu": 3}[epilogue]


def supports_epilogue(name: str, code: int) -> bool:
    """该配置是否实例化了对应 epilogue 变体（0 恒可）。"""
    config = parse_config(name)
    if code == 0:
        return True
    return bool(config["epi"])


# ---------------------------------------------------------------------------
# 第 18 期：WMMA 映射（同 TILE_M/TILE_N，计算单元换 fragment 16×16×16）
# 复制自 chapter_17/gemm_configs.py（第 17 期合同指定为配置单源），追加 WMMA 段；
# 两文件内容漂移时以 chapter_17 为准并同步。
# ---------------------------------------------------------------------------
# KSTEP 恒为 16：WMMA fragment 的 k 维固定 16；第 17 期 TILE_K=8 的配置在
# WMMA 版映射为一次加载 16 行 k（=2 个 k-fragment），记 wm16 映射。
WMMA_KSTEP = 16

WMMA_CONFIGS = [
    {"name": "wm16x16", "tm": 16, "tn": 16, "kstep": WMMA_KSTEP},
    {"name": "wm32x32", "tm": 32, "tn": 32, "kstep": WMMA_KSTEP},
    {"name": "wm32x64", "tm": 32, "tn": 64, "kstep": WMMA_KSTEP},
    {"name": "wm64x32", "tm": 64, "tn": 32, "kstep": WMMA_KSTEP},
    {"name": "wm64x64", "tm": 64, "tn": 64, "kstep": WMMA_KSTEP},
]

# 第 17 期 tile → WMMA 名映射（同 TILE_M/TILE_N 对照用）
WMMA_MAP_FROM_EP18 = {c["name"]: None for c in CONFIGS}
for _c in WMMA_CONFIGS:
    for _ep in CONFIGS:
        if _ep["tm"] == _c["tm"] and _ep["tn"] == _c["tn"]:
            WMMA_MAP_FROM_EP18[_ep["name"]] = _c["name"]


def parse_wmma_config(name: str) -> dict:
    for config in WMMA_CONFIGS:
        if config["name"] == name:
            return dict(config)
    raise ValueError(f"未知 WMMA 配置: {name}")


def wmma_block_threads(config: dict) -> int:
    """warp 布局 = (TM/16)×(TN/16)，每 warp 1 个 16×16 累加 fragment。"""
    return (config["tm"] // 16) * (config["tn"] // 16) * 32


def wmma_warps(config: dict) -> tuple[int, int]:
    return config["tm"] // 16, config["tn"] // 16


def wmma_fragments_per_warp() -> int:
    """每 warp 累加 fragment 数（本期布局恒为 1；每 fragment = 16×16 输出）。"""
    return 1


def wmma_smem_bytes(config: dict, acc_float: bool = True) -> int:
    """WMMA kernel 静态 smem：As[TM][KSTEP+8] + Bs[KSTEP][TN+8]（half）+ 暂存。

    fp32 累加：暂存 Cs_f[TM][TN]（float）；fp16 累加：Cs_h[TM][TN+8]（half）。
    """
    as_mem = config["tm"] * (config["kstep"] + 8) * 2
    bs_mem = config["kstep"] * (config["tn"] + 8) * 2
    if acc_float:
        staging = config["tm"] * config["tn"] * 4
    else:
        staging = config["tm"] * (config["tn"] + 8) * 2
    return as_mem + bs_mem + staging


def wmma_mma_sync_count(m: int, n: int, k: int, config: dict) -> int:
    """整个 GEMM 的 mma_sync 次数 = grid × warps × K/KSTEP（每条 8192 FLOP）。"""
    gx = -(-n // config["tn"])
    gy = -(-m // config["tm"])
    warps = (config["tm"] // 16) * (config["tn"] // 16)
    return gx * gy * warps * (-(-k // config["kstep"]))
