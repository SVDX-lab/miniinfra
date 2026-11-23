#!/usr/bin/env python3
"""第 15 期平台调研：vllm-musa 内置量化算子探测（只读，纸面+文件证据）。

目标路径：/data/feihu/vllm-musa-029（S4000 服务器上的 vLLM 0.29.1.dev0 musa
自编译版，servers.local.md 记录）。探测内容（大纲第 15 期实验清单第 5 项）：
1. 量化方法注册表：vllm/model_executor/layers/quantization/ 下的方法族
   （w8a8/w8a16/gptq/awq/fp8/compressed-tensors...）；
2. INT8 weight-only 路径：dequant 融合 kernel 的落点（csrc/kernels 与
   musa 平台补丁目录）；
3. musa 平台适配层对量化 kernel 的替换/禁用记录（哪些方法在 MUSA 上
   被路由到哪条实现——对照自研 kernel 的位置）；
4. FP8 在 MUSA 侧的支持状态（预期无硬件路径——平台边界证据）。

产出：results/vllm-survey/ 证据文件 + results/vllm-survey-musa.json。
只读操作：find/grep/ls/head，不修改任何被探测文件。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

DEFAULT_ROOT = "/data/feihu/vllm-musa-029"

# 探测目标：目录 → 说明
QUANT_DIRS = [
    "vllm/model_executor/layers/quantization",
    "vllm/model_executor/layers",
    "csrc/quantization",
    "csrc",
]

KEYWORDS = {
    "int8_weight_only": ["w8a16", "weight_only", "weight-only", "dequant_weights", "dequantize"],
    "int8_w8a8": ["w8a8", "int8_gemm", "_int_mm", "per_token_group"],
    "gptq_awq": ["gptq", "awq", "marlin", "machete", "exlnda", "exllama"],
    "fp8": ["fp8", "e4m3", "scaled_mm"],
    "musa_platform": ["musa"],
}

GREP_EXCLUDE = ["*.so", "*.bin", "*.a", "*.pyc"]


def sh(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default="results")
    arguments = parser.parse_args()

    root = Path(arguments.root)
    out_dir = Path(arguments.out_dir)
    survey_dir = out_dir / "vllm-survey"
    survey_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "kernel-chapter16.vllmsurvey.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "exists": root.exists(),
        "sections": {},
    }
    if not root.exists():
        print(f"[survey] 路径不存在：{root}")
        (out_dir / "vllm-survey-musa.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return 1

    # 0. 顶层结构快照
    rc, out = sh(["ls", str(root)])
    payload["sections"]["top_level"] = out.strip()[:2000]
    print(f"[survey] 顶层：{out.strip()[:300]}")

    # 1. 版本与平台补丁定位
    for probe in ["vllm/platform/musa", "vllm_musa", "vllm/platforms/musa"]:
        p = root / probe
        if p.exists():
            rc, out = sh(["ls", str(p)])
            payload["sections"]["musa_platform_dir"] = {
                "path": str(p),
                "listing": out.strip()[:3000],
            }
            print(f"[survey] MUSA 平台目录：{p}")
            (survey_dir / "musa-platform-listing.txt").write_text(out, encoding="utf-8")

    # 2. 量化方法目录清单
    quant_dir = None
    for cand in QUANT_DIRS:
        p = root / cand
        if p.exists() and "quantization" in cand:
            quant_dir = p
            break
    if quant_dir is None:
        rc, out = sh(["find", str(root), "-type", "d", "-name", "quantization", "-maxdepth", "6"])
        hits = [line for line in out.strip().splitlines() if line.strip()]
        if hits:
            quant_dir = Path(hits[0])
    if quant_dir is not None:
        rc, out = sh(["ls", str(quant_dir)])
        payload["sections"]["quantization_dir"] = {"path": str(quant_dir), "listing": out.strip()[:5000]}
        (survey_dir / "quantization-dir-listing.txt").write_text(out, encoding="utf-8")
        print(f"[survey] 量化方法目录：{quant_dir}（{len(out.strip().splitlines())} 项）")

    # 3. 关键词扫描（文件级计数 + 证据片段）
    for section, kws in KEYWORDS.items():
        section_result = {}
        for kw in kws:
            excludes = []
            for pat in GREP_EXCLUDE:
                excludes += ["--exclude", pat]
            rc, out = sh(
                ["grep", "-rn", "-i", "--include=*.py", "--include=*.cu", "--include=*.cpp",
                 "--include=*.h", "--include=*.cuh", "--include=*.mu", *excludes, kw, str(root)]
            )
            lines = [line for line in out.strip().splitlines() if line.strip()]
            # grep 失败（rc=1）= 无命中；rc>1 = 真错误
            section_result[kw] = {
                "match_files": len({line.split(":")[0] for line in lines if ":" in line}),
                "match_lines": len(lines),
                "sample": lines[:25],
            }
            print(f"[survey] {section}/{kw}: {section_result[kw]['match_lines']} 行 / "
                  f"{section_result[kw]['match_files']} 文件")
        payload["sections"][section] = section_result
        (survey_dir / f"grep-{section}.json").write_text(
            json.dumps(section_result, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    # 4. musa 侧 kernel 注册证据：平台分发里量化方法的选择逻辑
    rc, out = sh(
        ["grep", "-rn", "-i", "-E", "(int8|w8a16|weight_only|gptq|awq|marlin|machete)",
         "--include=*.py", str(root / "vllm/platform")] if (root / "vllm/platform").exists()
        else ["true"]
    )
    lines = [line for line in out.strip().splitlines() if line.strip() and "Binary" not in line]
    payload["sections"]["platform_dispatch_quant"] = {"match_lines": len(lines), "sample": lines[:40]}
    (survey_dir / "platform-dispatch-quant.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[survey] platform 层量化分发命中：{len(lines)} 行")

    out_path = out_dir / "vllm-survey-musa.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[survey] 证据与 JSON 已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
