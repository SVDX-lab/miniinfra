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
