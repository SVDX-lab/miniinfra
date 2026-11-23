"""Read a fixed Qwen3 config and the prefix of its Safetensors file."""

import json
from math import prod
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen


REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
REPO = f"https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/{REVISION}"
MODEL_BYTES = 1_192_135_096
PREFIX_BYTES = 65_536


def read_remote(request, limit, content_range=None):
    for attempt in range(3):
        try:
            with urlopen(request, timeout=15) as response:
                if content_range is not None and (
                    response.status != 206 or response.headers.get("Content-Range") != content_range
                ):
                    raise ValueError("expected the declared range from the fixed-size weight file")
                data = response.read(limit + 1)
                if len(data) > limit or (content_range is not None and len(data) != limit):
                    raise ValueError("unexpected response size")
                return data
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            sleep(attempt + 1)
    raise RuntimeError("request retry loop ended unexpectedly")


def expected_shapes(config):
    vocab = config["vocab_size"]
    width = config["hidden_size"]
    middle = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    q_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    if (config["model_type"] != "qwen3" or config["attention_bias"]
            or not config["tie_word_embeddings"] or config["hidden_act"] != "silu"):
        raise ValueError("expected the bias-free, tied Qwen3 Base configuration")
    q_width = q_heads * head_dim
    kv_width = kv_heads * head_dim
    one_layer = {
        "input_layernorm.weight": (width,),
        "self_attn.q_proj.weight": (q_width, width),
        "self_attn.k_proj.weight": (kv_width, width),
        "self_attn.v_proj.weight": (kv_width, width),
        "self_attn.o_proj.weight": (width, q_width),
        "self_attn.q_norm.weight": (head_dim,),
        "self_attn.k_norm.weight": (head_dim,),
        "post_attention_layernorm.weight": (width,),
        "mlp.gate_proj.weight": (middle, width),
        "mlp.up_proj.weight": (middle, width),
        "mlp.down_proj.weight": (width, middle),
    }
    shapes = {"model.embed_tokens.weight": (vocab, width), "model.norm.weight": (width,)}
    for layer in range(layers):
        for suffix, shape in one_layer.items():
            shapes[f"model.layers.{layer}.{suffix}"] = shape
    return shapes, one_layer


def parse_header(prefix):
    header_size = int.from_bytes(prefix[:8], "little")
    if not 2 <= header_size <= len(prefix) - 8:
        raise ValueError("header length does not fit in the bounded prefix")
    entries = json.loads(prefix[8:8 + header_size])
    entries.pop("__metadata__", None)
    return header_size, entries


def verify(entries, shapes, payload_size):
    if entries.keys() != shapes.keys():
        raise ValueError("checkpoint keys disagree with config-derived keys")
    by_offset = sorted(entries, key=lambda name: entries[name]["data_offsets"][0])
    cursor = 0
    for name in by_offset:
        entry = entries[name]
        begin, end = entry["data_offsets"]
        if (entry["dtype"] != "BF16" or tuple(entry["shape"]) != shapes[name]
                or begin != cursor or end - begin != 2 * prod(shapes[name])):
            raise ValueError(f"invalid BF16 shape or offsets for {name}")
        cursor = end
    if cursor != payload_size:
        raise ValueError("tensor bytes do not match the declared file size")


def main():
    config = json.loads(read_remote(Request(f"{REPO}/config.json"), 8192))
    stop = PREFIX_BYTES - 1
    request = Request(
        f"{REPO}/model.safetensors?lesson20=header",
        headers={"Range": f"bytes=0-{stop}"},
    )
    prefix = read_remote(request, PREFIX_BYTES, f"bytes 0-{stop}/{MODEL_BYTES}")
    header_size, entries = parse_header(prefix)
    shapes, one_layer = expected_shapes(config)
    payload_size = MODEL_BYTES - 8 - header_size
    verify(entries, shapes, payload_size)

    print(f"revision: {REVISION}")
    print(f"layers/vocab/hidden/intermediate: {config['num_hidden_layers']}/"
          f"{config['vocab_size']}/{config['hidden_size']}/{config['intermediate_size']}")
    print(f"Q/KV heads and head_dim: {config['num_attention_heads']}/"
          f"{config['num_key_value_heads']}, {config['head_dim']}")
    print(f"header/payload/file bytes: {header_size}/{payload_size}/{MODEL_BYTES}")
    for name in ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
                 "model.layers.0.self_attn.k_norm.weight", "model.layers.0.mlp.down_proj.weight"):
        print(f"{name}: {shapes[name]}, {entries[name]['dtype']}")
    print(f"tied LM Head: {'lm_head.weight' not in entries}")
    print(f"per-layer parameters: {sum(prod(shape) for shape in one_layer.values()):,}")
    print(f"stored tensors: {len(entries)}")
    print(f"total parameters: {sum(prod(shape) for shape in shapes.values()):,}")


if __name__ == "__main__":
    main()
