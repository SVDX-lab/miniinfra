#!/usr/bin/env python3
"""第 22 期真实权重单层验证（Qwen3-0.6B 第 0 层 q/k 投影 → RoPE）。

数据源（与 chapter_01 真值体系同源）：
- config.json：rope_theta/head_dim/HQ/HKV（实读，不抄常数）；
- truth/probes.npz 的 embedding 探针（ch01 FP32 CPU 参考路径的 embedding
  输出，prompt 20 token——跳过 safetensors 的 embed_tokens 复算，直接用
  ch01 权威中间值）；
- model.safetensors 第 0 层：q_proj/k_proj 权重 + q_norm/k_norm 权重。

链路：
  embedding 探针 → q_proj → view(T,HQ,D) → q_norm(ch01 RMSNorm 语义 fp32)
  → RoPE：
     R1 HF 语义参考（vendored，torch fp32；无 torch 时 numpy fp32）；
     R2 本期 compact 参考（FP64 锚点）；
     R3 GPU 核（standalone binary apply 桥：fp32 bin ↔ kernel ↔ fp32 bin，
        prefill T=20 pos0 与 decode pos=19 两形态 + fp16 io/cache 变体）。
  全部对照 + 确定性（同 desc 跑两次位级比对）。

--apply-selfcheck：无模型依赖的桥自检（随机数据，验证 binary apply 通路）。

用法（S4000 上）：
  python3 verify_qwen3_layer.py --binary implementations/musa/rope_bench \
      --model-dir ../../chapter_01/models/qwen3-0.6b \
      --work-dir /tmp/rope_verify --out results/qwen3-layer-verification.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import rope_config as cfg  # noqa: E402
import rope_reference as ref  # noqa: E402


# ---------------------------------------------------------------------------
# safetensors 裸解析（无 safetensors 依赖时的 fallback；有则直接用库）
# ---------------------------------------------------------------------------

def load_safetensors_tensors(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    data_start = 8 + header_len
    out = {}
    dtypes = {"BF16": None, "F32": np.float32, "F16": np.float16}
    with path.open("rb") as f:
        for name in names:
            info = header[name]
            dt = info["dtype"]
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            if dtypes.get(dt) is None:
                # BF16 → fp32：2 字节尾数扩展（手工，避免 bfloat16 依赖）
                u16 = np.frombuffer(raw, dtype="<u2")
                u32 = (u16.astype(np.uint32) << 16)
                arr = u32.view(np.float32) if u32.dtype.byteorder in ("<", "=") else u32.byteswap().view(np.float32)
            else:
                arr = np.frombuffer(raw, dtype=dtypes[dt])
            out[name] = arr.reshape(info["shape"])
    return out


def rms_norm_ch01(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """chapter_01 RMSNorm 语义：fp32 平方和均值 + rsqrt + weight。"""
    xf = x.astype(np.float32)
    variance = (xf * xf).mean(axis=-1, keepdims=True)
    normalized = xf * (1.0 / np.sqrt(variance + eps))
    return (weight.astype(np.float32) * normalized).astype(np.float32)


def hf_rope_np(q: np.ndarray, theta: float, head_dim: int, positions: np.ndarray) -> np.ndarray:
    """HF 语义 numpy fp32（torch 缺席时的 R1 路径）。"""
    dims = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = (1.0 / np.power(np.float32(theta), dims / np.float32(head_dim))).astype(np.float32)
    freqs = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32)[:, None, :]
    sin = np.sin(emb).astype(np.float32)[:, None, :]
    half = head_dim // 2
    return q * cos + np.concatenate([-q[..., half:], q[..., :half]], axis=-1) * sin


def write_bin(path: Path, arr: np.ndarray) -> None:
    arr.astype("<f4").tofile(path)


def read_bin(path: Path, expect_elems: int) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f4")
    assert arr.size == expect_elems, f"{path}: {arr.size} != {expect_elems}"
    return arr


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    a = actual.astype(np.float64).reshape(-1)
    e = expected.astype(np.float64).reshape(-1)
    d = np.abs(a - e)
    denom = np.maximum(np.abs(e), 1e-6)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "cosine": float(np.sum(a * e) / (np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(e * e)) + 1e-30)),
        "nan": int(np.isnan(a).sum()),
    }


def run_binary(binary: str, desc_path: Path, cwd: Path) -> None:
    binary_abs = str(Path(binary).resolve())   # cwd 切到 work 目录，binary 必须绝对路径
    r = subprocess.run([binary_abs, "apply", str(desc_path.resolve())],
                       capture_output=True, text=True, cwd=cwd.resolve())
    if r.returncode != 0:
        raise RuntimeError(f"binary apply 失败 rc={r.returncode}: {r.stdout} {r.stderr}")
    print(r.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--work-dir", default="/tmp/rope_verify")
    parser.add_argument("--out", default="results/qwen3-layer-verification.json")
    parser.add_argument("--apply-selfcheck", action="store_true",
                        help="无模型桥自检（随机数据端到端）")
    parser.add_argument("--probes-path", default="",
                        help="truth/probes.npz 显式路径（缺省按 model_dir 邻近布局探测）")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema_version": "kernel-chapter23.qwen3layer.v1",
                    "binary": args.binary, "work_dir": str(work)}

    if args.apply_selfcheck:
        rng = np.random.default_rng(99)
        T = 12
        q = rng.standard_normal((T, cfg.HQ, cfg.HEAD_DIM)).astype(np.float32)
        k = rng.standard_normal((T, cfg.HKV, cfg.HEAD_DIM)).astype(np.float32)
        cos, sin = ref.build_cache(T, cfg.ROPE_THETA, cfg.HEAD_DIM, "float32")
        write_bin(work / "q.bin", q)
        write_bin(work / "k.bin", k)
        write_bin(work / "cos.bin", cos)
        write_bin(work / "sin.bin", sin)
        qr, kr = ref.apply_prefill(q, k, cos, sin, 0)
        desc = {"form": "prefill", "io_dtype": "fp32", "cache_dtype": "fp32",
                "num_tokens": T, "hq": cfg.HQ, "hkv": cfg.HKV, "seq_offset": 0,
                "head_dim": cfg.HEAD_DIM, "max_seq": T,
                "input_q": str(work / "q.bin"), "input_k": str(work / "k.bin"),
                "cos": str(work / "cos.bin"), "sin": str(work / "sin.bin"),
                "output_q": str(work / "qout.bin"), "output_k": str(work / "kout.bin")}
        (work / "desc.json").write_text(json.dumps(desc))
        run_binary(args.binary, work / "desc.json", work)
        qo = read_bin(work / "qout.bin", qr.size)
        m = metrics(qo, qr)
        report["apply_selfcheck"] = m
        ok = m["max_abs"] <= 5e-4 and m["nan"] == 0
        report["apply_selfcheck_pass"] = bool(ok)
        print(f"[apply-selfcheck] max_abs={m['max_abs']:.3e} pass={ok}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0 if ok else 1

    if not args.model_dir:
        print("需要 --model-dir（真实权重模式）", file=sys.stderr)
        return 2

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    theta = float(config["rope_theta"])
    head_dim = int(config["head_dim"])
    hq = int(config["num_attention_heads"])
    hkv = int(config["num_key_value_heads"])
    eps = float(config["rms_norm_eps"])
    assert head_dim == cfg.HEAD_DIM and hq == cfg.HQ and hkv == cfg.HKV
    report["config_read"] = {"rope_theta": theta, "head_dim": head_dim,
                             "hq": hq, "hkv": hkv, "rms_norm_eps": eps,
                             "rope_scaling": config.get("rope_scaling"),
                             "source": str(model_dir / "config.json")}

    # embedding 探针（ch01 truth 体系；S4000 布局 chapter_01/truth/probes.npz）
    probes_candidates = ([Path(args.probes_path)] if args.probes_path else []) + [
        model_dir.parent.parent / "truth" / "probes.npz",
        model_dir.parent / "truth" / "probes.npz",
        model_dir / "truth" / "probes.npz"]
    probes_path = next((p for p in probes_candidates if p.is_file()), None)
    if probes_path is None:
        print(f"probes.npz 未找到（ tried {[str(p) for p in probes_candidates]}）", file=sys.stderr)
        return 2
    probes = np.load(probes_path)
    report["probes_path"] = str(probes_path)
    emb = probes["embedding"]          # [T, hidden] fp32
    T = emb.shape[0]
    report["embedding_probe"] = {"shape": list(emb.shape), "dtype": str(emb.dtype)}

    # 第 0 层权重
    names = ["model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.self_attn.k_proj.weight",
             "model.layers.0.self_attn.q_norm.weight",
             "model.layers.0.self_attn.k_norm.weight"]
    tensors = load_safetensors_tensors(model_dir / "model.safetensors", names)
    wq = tensors[names[0]].astype(np.float32)   # [HQ*D, hidden]
    wk = tensors[names[1]].astype(np.float32)   # [HKV*D, hidden]
    qw = tensors[names[2]].astype(np.float32)   # [D]
    kw = tensors[names[3]].astype(np.float32)

    # q/k 投影 + 归一化（ch01 顺序：q_proj → view → q_norm）
    q = (emb @ wq.T).reshape(T, hq, head_dim)
    k = (emb @ wk.T).reshape(T, hkv, head_dim)
    qn = rms_norm_ch01(q, qw, eps)
    kn = rms_norm_ch01(k, kw, eps)
    report["qk_stats"] = {"q_rms_per_token_first": float(np.sqrt((qn[0] ** 2).mean())),
                          "k_rms_per_token_first": float(np.sqrt((kn[0] ** 2).mean()))}

    # 参考输出
    positions = np.arange(T)
    cos32, sin32 = ref.build_cache(T, theta, head_dim, "float32")
    cos64, sin64 = ref.build_cache(T, theta, head_dim, "float64")
    try:
        import torch
        # R1：torch HF 语义（vendored chapter_01）
        dims = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (dims / head_dim))
        freqs = torch.from_numpy(positions).float().unsqueeze(-1) * inv_freq
        embt = torch.cat((freqs, freqs), dim=-1)
        cos_hf = embt.cos().numpy()[:, None, :]
        sin_hf = embt.sin().numpy()[:, None, :]
        half = head_dim // 2
        qt = torch.from_numpy(qn)
        q_hf = (qt * torch.from_numpy(cos_hf)
                + torch.cat((-qt[..., half:], qt[..., :half]), dim=-1) * torch.from_numpy(sin_hf)).numpy()
        report["r1_engine"] = f"torch {torch.__version__}"
    except ImportError:
        q_hf = hf_rope_np(qn, theta, head_dim, positions)
        report["r1_engine"] = "numpy-fallback"
    q_ref64, _ = ref.apply_prefill(qn.astype(np.float64), kn.astype(np.float64), cos64, sin64, 0)

    # R3：GPU 核（apply 桥）
    write_bin(work / "q.bin", qn)
    write_bin(work / "k.bin", kn)
    write_bin(work / "cos.bin", cos32)
    write_bin(work / "sin.bin", sin32)

    def kernel_case(form: str, io_dtype: str, cache_dtype: str, seq_offset: int = 0):
        if form == "prefill":
            n_tok = T
        else:
            n_tok = 1
            write_bin(work / "q1.bin", qn[-1])
            write_bin(work / "k1.bin", kn[-1])
        desc = {"form": form, "io_dtype": io_dtype, "cache_dtype": cache_dtype,
                "num_tokens": n_tok, "hq": hq, "hkv": hkv, "seq_offset": seq_offset,
                "head_dim": head_dim, "max_seq": T,
                "input_q": str(work / ("q.bin" if form == "prefill" else "q1.bin")),
                "input_k": str(work / ("k.bin" if form == "prefill" else "k1.bin")),
                "cos": str(work / "cos.bin"), "sin": str(work / "sin.bin"),
                "output_q": str(work / "qout.bin"), "output_k": str(work / "kout.bin")}
        (work / "desc.json").write_text(json.dumps(desc))
        run_binary(args.binary, work / "desc.json", work)
        elems_q = n_tok * hq * head_dim
        elems_k = n_tok * hkv * head_dim
        qo = read_bin(work / "qout.bin", elems_q).reshape(-1)
        ko = read_bin(work / "kout.bin", elems_k).reshape(-1)
        return np.concatenate([qo, ko])

    cases = {}
    for form, offset in (("prefill", 0), ("decode", T - 1)):
        for io_dtype in ("fp32", "fp16"):
            got = kernel_case(form, io_dtype, "fp32", offset)
            io = np.float16 if io_dtype == "fp16" else None
            if form == "prefill":
                q32o, k32o = ref.apply_prefill(qn, kn, cos32, sin32, 0, io_dtype=io)
            else:
                q32o, k32o = ref.apply_decode(qn[-1], kn[-1], cos32, sin32, offset, io_dtype=io)
            refv = np.concatenate([q32o.reshape(-1), k32o.reshape(-1)])
            entry = {"kernel_vs_ref_same_dtype": metrics(got, refv)}
            if form == "prefill" and io_dtype == "fp32":
                ref64_full = np.concatenate([
                    q_ref64.reshape(-1),
                    ref.apply_prefill(kn.astype(np.float64), kn.astype(np.float64),
                                      cos64, sin64, 0)[1].reshape(-1)])
                entry["kernel_vs_fp64_anchor"] = metrics(got, ref64_full.astype(np.float32))
                # HF 语义参考 vs compact 参考（同输入，语义双表达对照）
                hf_full = np.concatenate([q_hf.reshape(-1),
                                          hf_rope_np(kn, theta, head_dim, positions).reshape(-1)])
                mine32 = ref.apply_prefill(qn, kn, cos32, sin32, 0)
                entry["hf_vs_compact_ref"] = metrics(
                    hf_full, np.concatenate([mine32[0].reshape(-1), mine32[1].reshape(-1)]))
            cases[f"{form}_{io_dtype}"] = entry
    report["kernel_cases"] = cases

    # 确定性：prefill fp32 连跑两次位级
    got1 = kernel_case("prefill", "fp32", "fp32", 0)
    got2 = kernel_case("prefill", "fp32", "fp32", 0)
    report["determinism_bitwise"] = bool(np.array_equal(got1, got2))

    report["checkpoint"] = checksum_report(report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["kernel_cases"], indent=2))
    print(f"[qwen3-layer] determinism={report['determinism_bitwise']} → {args.out}")
    return 0


def checksum_report(report: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
