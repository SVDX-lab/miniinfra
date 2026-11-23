#!/usr/bin/env python3
"""第 09 期 fixtures：固定 seed 生成 + SHA256 清单（协议 §2 约定）。

本期无 CPU 列（期首声明），fixtures 用途是"图化正确性对账的输入冻结"：
  1. spot-rmsnorm-2048x1024.pt —— normal-shape 抽查的 (r, h, w) FP16 张量；
  2. decode-step-inputs.json —— decode 步输入 token ids（来自 ch01 truth 冻结）
     与 graph 捕获 shape（seq=46）及期望 next token；
  3. launch-bound 场景输入为确定性 arange（无需落盘，ch02 契约 v2 原样）。

seed 规则：20261121 + rows*100000 + hidden（与 ch05 家族一致）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TRUTH_DEFAULT = HERE.parent / "chapter_01" / "truth"


def sha256_of_tensor(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    fixtures = HERE / "fixtures"
    fixtures.mkdir(exist_ok=True)
    manifest: dict[str, dict] = {}

    rows, hidden = 2048, 1024
    seed = 20261121 + rows * 100000 + hidden
    generator = torch.Generator().manual_seed(seed)
    r = torch.randn(rows, hidden, generator=generator).to(torch.float16)
    h = torch.randn(rows, hidden, generator=generator).to(torch.float16)
    w = torch.randn(hidden, generator=generator).to(torch.float16)
    torch.save({"r": r, "h": h, "w": w, "seed": seed},
               fixtures / "spot-rmsnorm-2048x1024.pt")
    manifest["spot-rmsnorm-2048x1024.pt"] = {
        "seed": seed,
        "sha256_r": sha256_of_tensor(r),
        "sha256_h": sha256_of_tensor(h),
        "sha256_w": sha256_of_tensor(w),
    }

    truth = json.loads((TRUTH_DEFAULT / "truth-manifest.json").read_text(encoding="utf-8"))
    prompt_ids = truth["prompt_token_ids"]
    generated_ids = truth["generated_token_ids"]
    generated_index = 26
    seq = len(prompt_ids) + generated_index
    decode_inputs = {
        "source": "chapter_01/truth/truth-manifest.json（冻结输入）",
        "prompt_token_ids": prompt_ids,
        "generated_token_ids_prefix": generated_ids[:generated_index],
        "graph_capture_seq": seq,
        "expected_next_token": generated_ids[generated_index],
        "eos_token_id": truth["eos_token_id"],
    }
    payload = json.dumps(decode_inputs, sort_keys=True, ensure_ascii=False).encode("utf-8")
    decode_inputs["inputs_sha256"] = hashlib.sha256(payload).hexdigest()
    (fixtures / "decode-step-inputs.json").write_text(
        json.dumps(decode_inputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["decode-step-inputs.json"] = {"inputs_sha256": decode_inputs["inputs_sha256"]}

    (fixtures / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[fixtures] 已写入 {fixtures}（2 组 + 清单）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
