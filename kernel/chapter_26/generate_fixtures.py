#!/usr/bin/env python3
"""第 26 期 fixtures：prompt 集（ch01 真值锚点 + 变长/变域扩展）+ draft 种子清单。

第 26 期的"输入数据"是 prompt（投机解码的对照真值是第 01 期的 Greedy 32
Token 真值 manifest，本脚本不重新生成真值，只记录 prompt 的 token 编码与
SHA256，保证三平台共享同一份输入）。P01 与 chapter_01 真值 prompt 完全一致。

运行（本地 venv）：/home/aiuser/venvs/ml/bin/python generate_fixtures.py --model-dir models/qwen3-0.6b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ep27_common  # noqa: E402

ep27_common.bootstrap()

# 从 implementations/cpu 引用 tokenizer（协议 §2：复制复用，不 import ch01）。
from qwen3_tokenizer import Qwen3Tokenizer  # noqa: E402

SCHEMA_VERSION = "kernel-chapter27.fixtures.v1"

# P01 = ch01 真值锚点 prompt；P02-P06 为变长/变域扩展（代数推理、代码域、
# 诗意文本、极短事实、长 technical——draft 接受率对不同领域的敏感度即
# "接受率与领域难度的关系"实验的自变量）。
PROMPTS = [
    {"id": "P01", "note": "ch01 真值锚点（RMSNorm 解释）", "text": "用一句话解释 RMSNorm 的作用。"},
    {"id": "P02", "note": "本域概念（投机解码）", "text": "简单介绍投机解码的基本思想。"},
    {"id": "P03", "note": "推理链（访存解释，长输出倾向）", "text": "为什么 GPU 上 decode 比 prefill 慢？请从访存角度回答。"},
    {"id": "P04", "note": "诗意文本（重复模式多）", "text": "写一句关于秋天的短诗。"},
    {"id": "P05", "note": "极短事实问答", "text": "1 加 1 等于几？"},
    {"id": "P06", "note": "长 technical 问答", "text": "请详细解释多头注意力中 Q、K、V 各自的作用，并说明为什么注意力分数要除以根号 d。"},
    {"id": "P07", "note": "重复指令域（lookup/copy draft 的预期高接受域）", "text": "请把'今天天气很好'这句话重复三遍。"},
    {"id": "P08", "note": "列举计数域（强规律输出）", "text": "从 1 数到 10，用中文写出每个数字，用顿号分隔。"},
]

# random draft 的初始化种子（复现性清单；每种子一份独立权重）。
DRAFT_SEEDS = {"random_main": 20271121, "random_alt": 20271122}

K_SWEEP = [1, 2, 3, 4, 6, 8]


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=ep27_common.default_model_dir())
    parser.add_argument("--out-dir", default="fixtures")
    arguments = parser.parse_args()

    model_directory = Path(arguments.model_dir)
    tokenizer = Qwen3Tokenizer(str(model_directory))

    prompt_records = []
    for entry in PROMPTS:
        prompt_ids = tokenizer.encode_chat_prompt(entry["text"])
        prompt_records.append(
            {
                "id": entry["id"],
                "note": entry["note"],
                "text": entry["text"],
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "eos_token_id": tokenizer.eos_token_id,
            }
        )
        print(f"[fixtures] {entry['id']}: {len(prompt_ids)} prompt tokens")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_for_episode": 27,
        "truth_reference": "chapter_01/truth/truth-manifest.json（只读引用；P01 与之同 prompt 同 max_new_tokens）",
        "max_new_tokens": 32,
        "chat_template": "enable_thinking=False（non-thinking，与 ch01 合同一致）",
        "k_sweep": K_SWEEP,
        "draft_seeds": DRAFT_SEEDS,
        "draft_modes": ["reuse", "reuse_hidden", "random", "copy", "lookup", "lookup2"],
        "prompts": prompt_records,
    }

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prompts.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    tokenizer_sha = sha256_of_bytes((model_directory / "tokenizer.json").read_bytes())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": {
            "prompts.json": sha256_of_bytes(out_path.read_bytes()),
        },
        "tokenizer_json_sha256": tokenizer_sha,
        "prompt_count": len(prompt_records),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"[fixtures] 已写入 {out_path} 与 manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
