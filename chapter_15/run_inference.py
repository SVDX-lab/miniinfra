"""使用自然语言 Prompt 演示 monolithic 与跨进程 P/D 分离。"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from experiment_utils import add_model_arguments, write_json
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer


def main():
    parser = argparse.ArgumentParser(description="第 15 期自然语言推理演示")
    add_model_arguments(parser)
    parser.add_argument("--prompt", default="请用一句话解释 Prefill 和 Decode 的区别。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--worker-warmup", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(directory)
    token_ids = tokenizer.encode_chat_prompt(args.prompt)
    validator = Path(__file__).with_name("validate_disaggregated.py")
    with tempfile.TemporaryDirectory(prefix="chapter15-demo-") as temp:
        temp = Path(temp)
        tokens_path = temp / "tokens.json"
        result_path = temp / "result.json"
        tokens_path.write_text(json.dumps(token_ids), encoding="utf-8")
        command = [
            sys.executable, str(validator),
            "--model", args.model, "--revision", args.revision,
            "--device", args.device, "--dtype", args.dtype,
            "--token-ids-file", str(tokens_path),
            "--max-new-tokens", str(args.max_new_tokens),
            "--block-size", str(args.block_size),
            "--token-budget", str(args.token_budget),
            "--worker-warmup", str(args.worker_warmup),
            "--output", str(result_path),
        ]
        subprocess.run(command, check=True)
        result = json.loads(result_path.read_text(encoding="utf-8"))
    baseline_tokens = result["baseline"]["new_token_ids"]
    split_tokens = result["split"]["workers"]["decode"]["new_token_ids"]
    print("Prompt Token 数:", len(token_ids))
    print("monolithic:", tokenizer.decode(baseline_tokens, skip_special_tokens=True))
    print("disaggregated:", tokenizer.decode(split_tokens, skip_special_tokens=True))
    print("输出 Token 一致:", baseline_tokens == split_tokens)
    result["config"]["token_ids_file"] = "<temporary-token-file>"
    result["demo"] = {
        "prompt": args.prompt,
        "prompt_tokens": len(token_ids),
        "monolithic_text": tokenizer.decode(
            baseline_tokens, skip_special_tokens=True
        ),
        "disaggregated_text": tokenizer.decode(
            split_tokens, skip_special_tokens=True
        ),
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
