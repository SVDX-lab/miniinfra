#!/usr/bin/env python3
"""生成第 24 期 MLA fixtures（固定 seed 的合成权重、输入与真值输出）。

内容：
- 合成权重 6 张（W_QA/W_QB/W_KVA/W_UK/W_UV/W_O，FP32，逐张量独立 seed）；
- decode query x [1,2048]（全部 N 共用一份）；
- 每个合同 N 的 KV cache：c_kv [N,512]、k_rope_raw [N,64]、k_rope_rot [N,64]
  （rot = apply_rope(raw, positions)，即"已逐位置旋转的 rope cache"——真实 MLA
  cache 里存的就是旋转后的）、positions [N]；
- 真值输出（每 N）：out_unabs_fp32（未吸收 FP32 链）、out_abs_fp32（吸收 FP32
  链）、out_truth_fp64（未吸收 FP64 链，代数等价的位级锚点）、
  scores/attn head0 探针。

seed 规则：torch.Generatormanual_seed，逐张量独立（权重 = SEED_BASE + 序号；
x = SEED_X；N 组 = SEED_BASE + 1000 + N）。manifest 记录逐张量 SHA256，
两台服务器共用同一份文件；--verify-only 可在服务器上免上传重建核对
（要求相同 torch 版本，manifest 记录）。

性能基准的输入不进 fixtures：benchmark.py 用同一 seed 规则运行时生成
（性能对照只依赖 shape/dtype/数值分布，不依赖具体数值跨平台一致）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementations" / "cpu"))
sys.path.insert(0, str(HERE))

import mla_config as cfg  # noqa: E402
from mla_reference import (  # noqa: E402
    WEIGHT_KEYS,
    apply_rope,
    mla_decode_absorbed,
    mla_decode_unabsorbed,
    rope_freqs,
)

SEED_BASE = 20260325
SEED_X = SEED_BASE + 1
KV_LENS = [1, 3, 33, 128, 2048]        # 正确性 N（含 N=1 边界与 2048 长上下文）
BENCH_KV_LENS = [1, 128, 2048, 32768]  # 性能扫描 N（不入 fixtures，运行时生成）


def tensor_seed(name: str) -> int:
    return SEED_BASE + (sorted(WEIGHT_KEYS).index(name) + 1)


def sha256_of_tensor(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def make_weights() -> dict[str, torch.Tensor]:
    """合成权重：randn × He 初始化缩放（std = 1/sqrt(fan_in)）。

    缩放的理由（本地开发期实测教训）：N(0,1) 权重下激活逐层放大
    （q_nope ~ 1.8e3、score ~ 2.6e4），softmax(score×1/sqrt(192)) 退化为
    one-hot（除 argmax 外全部下溢为 0）——加权求和/吸收 W_UV 路径几乎不被
    执行，等价性验证失去代表性。He 缩放后各层激活 ~O(1)、score×scale ~O(1)，
    softmax 分布非退化，两条形式的每一段都真实参与。
    """
    weights = {}
    for name in sorted(WEIGHT_KEYS):
        shape = cfg.WEIGHT_SHAPES[name]
        gen = torch.Generator().manual_seed(tensor_seed(name))
        scale = 1.0 / (shape[0] ** 0.5)  # fan_in = 输入维度
        weights[name] = torch.randn(*shape, generator=gen, dtype=torch.float32) * scale
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fixtures")
    parser.add_argument("--verify-only", action="store_true",
                        help="重建全部张量并与 manifest 逐张量核对 SHA256，不写文件")
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    freqs = rope_freqs()

    arrays: dict[str, torch.Tensor] = {}
    weights = make_weights()
    for name, tensor in weights.items():
        arrays[f"weight_{name}"] = tensor

    x_seed_gen = torch.Generator().manual_seed(SEED_X)
    x = torch.randn(1, cfg.D_MODEL, generator=x_seed_gen, dtype=torch.float32)
    arrays["x"] = x

    for n in KV_LENS:
        gen = torch.Generator().manual_seed(SEED_BASE + 1000 + n)
        c_kv = torch.randn(n, cfg.KV_LORA_RANK, generator=gen, dtype=torch.float32)
        k_rope_raw = torch.randn(n, cfg.QK_ROPE_HEAD_DIM, generator=gen, dtype=torch.float32)
        positions = torch.arange(n, dtype=torch.float32)
        k_rope_rot = apply_rope(k_rope_raw, positions, freqs)
        pos_q = n
        truth_unabs = mla_decode_unabsorbed(x, weights, c_kv, k_rope_rot, pos_q, freqs,
                                            want_probes=True)
        truth_abs = mla_decode_absorbed(x, weights, c_kv, k_rope_rot, pos_q, freqs)
        # FP64 代数锚点（吸收/未吸收应位级一致，见 quick_selfcheck [8]）
        truth64 = mla_decode_unabsorbed(
            x.double(), {k: v.double() for k, v in weights.items()},
            c_kv.double(), k_rope_rot.double(), pos_q, freqs.double())
        prefix = f"n{n}_"
        arrays[prefix + "c_kv"] = c_kv
        arrays[prefix + "k_rope_raw"] = k_rope_raw
        arrays[prefix + "k_rope_rot"] = k_rope_rot
        arrays[prefix + "positions"] = positions
        arrays[prefix + "out_unabs_fp32"] = truth_unabs["out"]
        arrays[prefix + "out_abs_fp32"] = truth_abs["out"]
        arrays[prefix + "out_truth_fp64"] = truth64["out"]
        arrays[prefix + "scores_head0"] = truth_unabs["scores"][0]
        arrays[prefix + "attn_head0"] = truth_unabs["attn"][0]

    manifest: dict = {
        "schema_version": "kernel-chapter25.fixtures.v1",
        "seed_base": SEED_BASE,
        "seed_x": SEED_X,
        "seed_rule": "torch.Generator.manual_seed；权重=SEED_BASE+权重名字典序序号；"
                     "x=SEED_X；N 组=SEED_BASE+1000+N",
        "rng": "torch.randn (Mersenne Twister via torch.Generator, CPU) × He 缩放 "
               "(std=1/sqrt(fan_in)，fan_in=shape[0]；理由见 make_weights docstring："
               "N(0,1) 会使 softmax 退化为 one-hot)",
        "torch_version": torch.__version__,
        "portability_note": "portable=true 的张量是纯 RNG 输出，跨 torch 版本重建"
            "SHA256 一致；portable=false 的真值/探针张量依赖生成机器的 torch CPU "
            "kernel（einsum 分块/线程），跨版本会有 FP32 舍入级差异——verify-only "
            "只核对 portable 张量，真值以随仓库分发的 npz 文件本身为准",
        "portable_keys_rule": "weight_* / x / n*_c_kv / n*_k_rope_raw / n*_positions",
        "config": {
            "d_model": cfg.D_MODEL, "n_heads": cfg.N_HEADS,
            "q_lora_rank": cfg.Q_LORA_RANK, "kv_lora_rank": cfg.KV_LORA_RANK,
            "qk_nope_head_dim": cfg.QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": cfg.QK_ROPE_HEAD_DIM,
            "v_head_dim": cfg.V_HEAD_DIM,
            "softmax_scale": cfg.SOFTMAX_SCALE, "rope_theta": cfg.ROPE_THETA,
        },
        "config_source": cfg.CONFIG_SOURCE,
        "kv_lens": KV_LENS,
        "bench_kv_lens_runtime_only": BENCH_KV_LENS,
        "truth_implementation": "mla_decode_unabsorbed (FP32 CPU) + FP64 锚点 "
                                "mla_decode_unabsorbed(float64)；吸收形式 FP32 同时记录",
        "rope_convention": "Llama/HF 半分约定（x1=x[..,:32]），freqs=theta^(-2i/64)。",
        "scale_note": "softmax_scale=1/sqrt(192)=官方 modeling_deepseek.py L745 q_head_dim**-0.5；"
                      "两种形式共用，等价性与 scale 取值无关",
        "tensors": {},
    }
    for name, tensor in arrays.items():
        is_portable = name.startswith("weight_") or name == "x" or \
            any(key in name for key in ("c_kv", "k_rope_raw", "positions"))
        manifest["tensors"][name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": sha256_of_tensor(tensor),
            "portable": is_portable,
        }

    if arguments.verify_only:
        manifest_path = out_dir / "manifest.json"
        if not manifest_path.is_file():
            print(f"[verify] 找不到 {manifest_path}，无法核对")
            return 1
        with manifest_path.open("r", encoding="utf-8") as file:
            expected = json.load(file)
        bad = []
        checked = 0
        for name, entry in expected["tensors"].items():
            if not entry.get("portable", False):
                continue  # 真值/探针依赖生成机 torch CPU kernel，不跨版本核对
            if name not in arrays:
                bad.append(f"{name}: 本地重建缺失")
                continue
            checked += 1
            actual_sha = sha256_of_tensor(arrays[name])
            if actual_sha != entry["sha256"]:
                bad.append(f"{name}: sha256 不一致")
        if bad:
            print("[verify] 失败：")
            for line in bad:
                print("  -", line)
            return 1
        print(f"[verify] 通过：{checked}/{len(expected['tensors'])} 张量（portable RNG 张量）"
              f"SHA256 一致（本机 torch {torch.__version__}；真值张量以 npz 文件为准，"
              f"见 manifest.portability_note）")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    npz_payload = {name: tensor.numpy() for name, tensor in arrays.items()}
    import numpy as np

    np.savez(out_dir / "correctness.npz", **npz_payload)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    elapsed = time.perf_counter() - started
    total_bytes = sum(t.numel() * t.element_size() for t in arrays.values())
    print(f"[fixtures] 写出 {out_dir}/correctness.npz（{len(arrays)} 张量, "
          f"{total_bytes / 1024 / 1024:.1f} MB, 生成+写盘 {elapsed:.1f}s）")
    print(f"[fixtures] manifest.json：seed_base={SEED_BASE}, KV_LENS={KV_LENS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
