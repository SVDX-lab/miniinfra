"""第 26 期驱动脚本共享引导：sys.path、兼容层、模型加载与权重校验。

所有驱动脚本（smoke_test / compare_correctness / benchmark）先调用
bootstrap()，再用 load_verified_model() 取得与第 01 期真值锚定的目标模型。

兼容层（平台边界，KNOWN_ISSUES 同步）：
- 本地 venv 无 safetensors 包 → implementations/cpu/safetensors_compat
  （仅本地生效；服务器有官方包时不插入）；
- 本地 venv 无 regex 包 → 插入系统用户 site-packages 的 regex（cp312 同 ABI）。
"""

from __future__ import annotations

import json
import site
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CPU_DIR = ROOT / "implementations" / "cpu"

# 真值与权重引用（chapter_01 只读）。
TRUTH_MANIFEST = ROOT.parent / "chapter_01" / "truth" / "truth-manifest.json"


def default_model_dir() -> str:
    local = ROOT / "models" / "qwen3-0.6b"
    if (local / "model.safetensors").is_file():
        return str(local)
    return str(ROOT.parent / "chapter_01" / "models" / "qwen3-0.6b")


def bootstrap() -> None:
    """统一 sys.path 与兼容层注入（幂等）。"""
    if str(CPU_DIR) not in sys.path:
        sys.path.insert(0, str(CPU_DIR))
    try:
        import safetensors  # noqa: F401

        has_safetensors = True
    except ModuleNotFoundError:
        has_safetensors = False
    if not has_safetensors:
        compat = str(CPU_DIR / "safetensors_compat")
        if compat not in sys.path:
            sys.path.insert(0, compat)
    try:
        import regex  # noqa: F401
    except ModuleNotFoundError:
        candidate = site.getusersitepackages()
        if Path(candidate).exists() and candidate not in sys.path:
            sys.path.append(candidate)


def load_fixtures():
    fixtures_path = ROOT / "fixtures" / "prompts.json"
    with fixtures_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_truth_manifest():
    with TRUTH_MANIFEST.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_verified_model(model_dir: str, dtype=torch.float32):
    """加载 FP32 CPU 目标模型，并用 ch01 真值 prompt 的首 Token logits 对拍。

    校验两层：
    1. SHA256 位级对拍（本地 torch 2.6.0+cpu 实测与 ch01 真值逐位一致；
       权重加载器任何字节错位都会在此暴露）；
    2. 跨机器回退口径：不同 torch 版本/BLAS 的 CPU FP32 归约顺序不同
       （S4000 torch 2.11 实测 SHA 不同但 argmax/top5/logit 值一致——
       KNOWN_ISSUES KI-27-07），此时退化要求 首 Token argmax 一致 +
       top-5 集合一致 + top-5 logit 值偏差 < 1e-2；仍失败才报错。
    """
    import torch

    from experiment_utils import sha256_of_tensor
    from qwen3_model import load_handwritten_model, resolve_model_directory

    directory = resolve_model_directory(model_dir)
    model = load_handwritten_model(directory, torch.device("cpu"), dtype=dtype)
    model.eval()

    truth = load_truth_manifest()
    input_ids = torch.tensor([truth["prompt_token_ids"]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)
    first_logits = logits[0, -1, :].float().cpu()
    sha = sha256_of_tensor(first_logits)
    expected = truth["first_token_logits_sha256"]
    if sha == expected:
        print("[verify] 首 Token logits SHA256 位级一致（本地口径）")
        return model

    # 跨 torch 版本回退口径（声明于 EXPERIMENT_LOG §0.2）。
    argmax_id = int(first_logits.argmax().item())
    top5 = torch.topk(first_logits, 5)
    top5_ids = sorted(int(v) for v in top5.indices.tolist())
    truth_argmax = truth["first_token_id"]
    truth_top5_ids = sorted(item["id"] for item in truth["first_token_top5"])
    truth_top5_logits = {item["id"]: item["logit"] for item in truth["first_token_top5"]}
    value_deviation = max(
        abs(float(first_logits[tid]) - value) for tid, value in truth_top5_logits.items()
    )
    portable_ok = (
        argmax_id == truth_argmax
        and top5_ids == truth_top5_ids
        and value_deviation < 1e-2
    )
    if not portable_ok:
        raise RuntimeError(
            "权重完整性校验失败：SHA256 " + sha + " != 真值 " + expected
            + "，且回退口径不一致（argmax=" + str(argmax_id) + " vs " + str(truth_argmax)
            + "，top5=" + str(top5_ids) + " vs " + str(truth_top5_ids)
            + "，logit 偏差 " + str(value_deviation) + "）"
        )
    print(
        "[verify] SHA256 与本地口径不同（跨 torch/BLAS 归约顺序，KI-27-07）；"
        "回退口径通过：argmax/top5 一致，top-5 logit 最大偏差 "
        + str(round(value_deviation, 8))
    )
    return model
