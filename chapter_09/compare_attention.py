"""手写 FlashAttention 与 Eager Reference 的 CUDA 数值正确性测试。"""

import argparse
import json

import torch

from flash_attention import flash_attention_forward
from qwen3_model import eager_attention_forward


def parse_args():
    parser = argparse.ArgumentParser(description="FlashAttention Kernel 正确性测试")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def build_case(name, batch, query_length, key_length, causal_offset, padding):
    device = torch.device("cuda")
    query = torch.randn(
        batch, 16, query_length, 128, device=device, dtype=torch.bfloat16
    ) * 0.25
    key = torch.randn(
        batch, 16, key_length, 128, device=device, dtype=torch.bfloat16
    ) * 0.25
    value = torch.randn_like(key) * 0.25
    key_valid = torch.ones(batch, key_length, dtype=torch.bool, device=device)
    query_valid = torch.ones(batch, query_length, dtype=torch.bool, device=device)
    if padding:
        # 第二行模拟左 Padding 的 Chunk；前缀有效，当前 Chunk 的 Padding 无效。
        prefix_length = causal_offset
        pad_count = min(11, query_length - 1)
        key_valid[1, prefix_length:prefix_length + pad_count] = False
        query_valid[1, :pad_count] = False
    return {
        "name": name,
        "query": query,
        "key": key,
        "value": value,
        "key_valid": key_valid,
        "query_valid": query_valid,
        "causal_offset": causal_offset,
    }


def run_case(case):
    with torch.inference_mode():
        reference = eager_attention_forward(
            case["query"], case["key"], case["value"],
            case["key_valid"], case["query_valid"], case["causal_offset"],
        )
        actual = flash_attention_forward(
            case["query"], case["key"], case["value"],
            case["key_valid"], case["query_valid"], case["causal_offset"],
        )
    torch.cuda.synchronize()
    valid = case["query_valid"][:, None, :, None].expand_as(actual)
    difference = (reference.float() - actual.float()).abs()[valid]
    maximum = difference.max().item()
    mean = difference.mean().item()
    passed = torch.allclose(reference, actual, atol=2e-2, rtol=2e-2)
    if not passed:
        raise AssertionError(
            "%s 数值误差超限: max=%g mean=%g" % (case["name"], maximum, mean)
        )
    return {
        "case": case["name"],
        "batch": case["query"].shape[0],
        "query_length": case["query"].shape[2],
        "key_length": case["key"].shape[2],
        "causal_offset": case["causal_offset"],
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "passed": passed,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("正确性测试需要 NVIDIA GPU")
    torch.manual_seed(args.seed)
    cases = [
        build_case("full-prefill-128", 1, 128, 128, 0, False),
        build_case("full-prefill-tail-257", 1, 257, 257, 0, False),
        build_case("chunked-prefill", 1, 64, 320, 256, False),
        build_case("left-padded-chunk-batch", 2, 64, 320, 256, True),
        build_case("decode", 2, 1, 1025, 1024, False),
    ]
    results = [run_case(case) for case in cases]
    for row in results:
        print(
            "%(case)s: Q=%(query_length)d K=%(key_length)d "
            "max=%(max_abs_error).6f mean=%(mean_abs_error).6f PASS" % row
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump({"results": results}, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

