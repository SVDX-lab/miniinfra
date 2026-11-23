#!/usr/bin/env python3
"""生成第 01 期模型级真值（CPU FP32 手写 Qwen3 参考路径）。

产物（写入 truth/ 目录）：
- truth/probes.npz：Prompt 前向的逐层隐藏状态探针（28 个 Decoder Layer
  输出 + embedding + final_norm，FP32）、首 Token Logits（FP32，151936 维）；
- truth/truth-manifest.json：Prompt 与 Token IDs、Greedy 32 Token、
  首 Token Top-5、逐探针 SHA256、环境记录。

真值只生成一次（在任一机器的 CPU 上），跨平台由 compare_correctness.py
消费同一份文件，保证三后端对照共享同一锚点。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiment_utils import device_environment_record, sha256_of_tensor, write_results_json
from qwen3_model import load_handwritten_model, resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer

DEFAULT_PROMPT = "用一句话解释 RMSNorm 的作用。"
DEFAULT_MAX_NEW_TOKENS = 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/qwen3-0.6b")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--dtype", default="float32", choices=["float32"])
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="固定 Prompt（non-thinking chat template）"
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--out-dir", default="truth")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    if arguments.dtype != "float32" or device.type != "cpu":
        raise ValueError("真值路径固定为 CPU FP32，参数不得更改")

    model_directory = resolve_model_directory(arguments.model_dir)
    tokenizer = Qwen3Tokenizer(model_directory)
    model = load_handwritten_model(model_directory, device, dtype=torch.float32)
    model.eval()

    prompt_ids = tokenizer.encode_chat_prompt(arguments.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 1) Prompt 前向（带探针）：逐层隐藏状态与首 Token Logits。
    with torch.no_grad():
        logits, probes = model(input_ids, probe=True)
    first_token_logits = logits[0, -1, :].float().cpu()
    top5 = torch.topk(first_token_logits, 5)
    first_token_top5 = [
        {"id": int(top5.indices[i]), "logit": round(float(top5.values[i]), 6)}
        for i in range(5)
    ]
    first_token_id = int(top5.indices[0])

    # 2) Greedy 解码 32 Token（每步重算整段序列，与第 01 期"无优化"边界一致）。
    generated, finished = model.greedy_generate(
        input_ids,
        max_new_tokens=arguments.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = generated[0].tolist()
    new_token_ids = generated_ids[len(prompt_ids):]

    # 3) 探针与摘要写入。
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_hashes = {}
    arrays = {"first_token_logits": first_token_logits.numpy()}
    for name, tensor in probes.items():
        arrays[name] = tensor[0].cpu().numpy()  # [sequence, hidden]
        probe_hashes[name] = sha256_of_tensor(tensor)

    probes_path = out_dir / "probes.npz"
    np.savez_compressed(probes_path, **arrays)

    manifest = {
        "schema_version": "kernel-chapter01.truth.v1",
        "prompt": arguments.prompt,
        "max_new_tokens": arguments.max_new_tokens,
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "generated_token_ids": new_token_ids,
        "generated_token_count": len(new_token_ids),
        "finished_by_eos": finished,
        "decoded_text": tokenizer.decode(new_token_ids, skip_special_tokens=True),
        "eos_token_id": tokenizer.eos_token_id,
        "first_token_id": first_token_id,
        "first_token_top5": first_token_top5,
        "first_token_logits_sha256": sha256_of_tensor(first_token_logits),
        "probe_sha256": probe_hashes,
        "probes_npz_sha256": hashlib_sha256(probes_path),
        "environment": device_environment_record(device),
    }
    write_results_json(out_dir / "truth-manifest.json", manifest)
    print(f"[done] greedy={len(new_token_ids)} tokens, finished_by_eos={finished}")
    print(f"[decoded] {manifest['decoded_text'][:120]}")
    return 0


def hashlib_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
