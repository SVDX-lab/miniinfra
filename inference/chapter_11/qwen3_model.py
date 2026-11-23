"""第 11 期独立使用的 Qwen3 推理模型与预分配 KV Cache。

本文件不引用其他期代码。实现支持单请求 BF16/FP16/FP32 推理、PyTorch SDPA、
非分片和 Safetensors 分片权重，以及可以按逻辑长度回滚的连续 KV Cache。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn


MODEL_REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
}
DEFAULT_TARGET_MODEL_ID = "Qwen/Qwen3-4B"
DEFAULT_DRAFT_MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    torch_dtype: str
    tie_word_embeddings: bool

    @classmethod
    def from_model_directory(cls, model_directory):
        with (Path(model_directory) / "config.json").open(
            "r", encoding="utf-8"
        ) as file:
            raw = json.load(file)
        config = cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            torch_dtype=raw.get("torch_dtype", "bfloat16"),
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
        )
        config.validate()
        return config

    def validate(self):
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("Attention 头数必须能被 KV 头数整除")
        if not self.tie_word_embeddings:
            raise ValueError("本期实现只支持共享 Embedding 和 LM Head 的 Qwen3")


class ContiguousKVCache:
    """单请求预分配 KV Cache；rollback 只移动逻辑长度。"""

    def __init__(self, config, max_length, device, dtype):
        if max_length < 1:
            raise ValueError("max_length 必须大于 0")
        shape = (
            1,
            config.num_key_value_heads,
            max_length,
            config.head_dim,
        )
        self.keys = [
            torch.empty(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]
        self.values = [
            torch.empty(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]
        self.max_length = max_length
        self.length = 0

    @property
    def bytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.keys + self.values
        )

    def reset(self):
        self.length = 0

    def checkpoint(self):
        return self.length

    def rollback(self, length):
        if not 0 <= length <= self.length:
            raise ValueError(
                "只能回滚到 [0, %d]，收到 %d" % (self.length, length)
            )
        self.length = length

    def write(self, layer_index, start, key, value):
        query_length = key.shape[2]
        end = start + query_length
        if start != self.length:
            raise RuntimeError(
                "KV 写入起点必须等于逻辑长度: start=%d, length=%d"
                % (start, self.length)
            )
        if end > self.max_length:
            raise RuntimeError(
                "KV Cache 容量不足: need=%d, capacity=%d"
                % (end, self.max_length)
            )
        expected = self.keys[layer_index][:, :, start:end, :].shape
        if key.shape != expected or value.shape != expected:
            raise ValueError(
                "KV 形状错误: key=%s, value=%s, expected=%s"
                % (tuple(key.shape), tuple(value.shape), tuple(expected))
            )
        self.keys[layer_index][:, :, start:end, :].copy_(key)
        self.values[layer_index][:, :, start:end, :].copy_(value)
        return (
            self.keys[layer_index][:, :, :end, :],
            self.values[layer_index][:, :, :end, :],
        )

    def commit(self, expected_start, query_length):
        if expected_start != self.length:
            raise RuntimeError("KV Cache 在一次前向中被意外修改")
        self.length += query_length


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        float_states = hidden_states.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rope_theta):
        super().__init__()
        dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (dimensions / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, output_dtype):
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq
        embedding = torch.cat((freqs, freqs), dim=-1)
        return embedding.cos().to(output_dtype), embedding.sin().to(output_dtype)


def rotate_half(hidden_states):
    half = hidden_states.shape[-1] // 2
    return torch.cat(
        (-hidden_states[..., half:], hidden_states[..., :half]), dim=-1
    )


def apply_rotary_embedding(query, key, cosine, sine):
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


def repeat_key_value(hidden_states, repeat_count):
    if repeat_count == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeat_count, dim=1)


def sdpa_attention(query, key, value, start_position):
    query_length = query.shape[2]
    key_length = key.shape[2]
    query_positions = start_position + torch.arange(
        query_length, device=query.device
    )
    key_positions = torch.arange(key_length, device=query.device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=allowed.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
    )


class Qwen3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        query_size = config.num_attention_heads * config.head_dim
        key_value_size = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, query_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.o_proj = nn.Linear(query_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, cosine, sine, cache, layer_index, start):
        batch_size, query_length, _ = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).view(
                batch_size, query_length, self.num_attention_heads, self.head_dim
            )
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch_size, query_length, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query, key = apply_rotary_embedding(query, key, cosine, sine)
        key, value = cache.write(layer_index, start, key, value)
        key = repeat_key_value(key, self.num_key_value_groups)
        value = repeat_key_value(value, self.num_key_value_groups)
        output = sdpa_attention(query, key, value, start)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        return self.o_proj(output)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, cosine, sine, cache, layer_index, start):
        residual = hidden_states
        hidden_states = residual + self.self_attn(
            self.input_layernorm(hidden_states),
            cosine,
            sine,
            cache,
            layer_index,
            start,
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, input_ids, cache):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("本期模型只接受 [1, sequence] 非空 input_ids")
        start = cache.length
        query_length = input_ids.shape[1]
        end = start + query_length
        if end > cache.max_length:
            raise RuntimeError("输入将超过 KV Cache 容量")
        position_ids = torch.arange(start, end, device=input_ids.device).unsqueeze(0)
        hidden_states = self.embed_tokens(input_ids)
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                cosine,
                sine,
                cache,
                layer_index,
                start,
            )
        cache.commit(start, query_length)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def new_cache(self, max_length):
        parameter = next(self.parameters())
        return ContiguousKVCache(
            self.config, max_length, parameter.device, parameter.dtype
        )

    def forward(self, input_ids, cache):
        return self.lm_head(self.model(input_ids, cache))


def resolve_revision(model_name_or_path, revision=None):
    if revision is not None:
        return revision
    return MODEL_REVISIONS.get(model_name_or_path)


def resolve_model_directory(model_name_or_path, revision=None):
    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())
    resolved_revision = resolve_revision(model_name_or_path, revision)
    if resolved_revision is None:
        raise ValueError(
            "远端模型必须显式提供固定 revision；已内置 Qwen3-0.6B/1.7B/4B"
        )
    return snapshot_download(
        repo_id=model_name_or_path,
        revision=resolved_revision,
        max_workers=2,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ],
    )


def model_dtype_from_config(config):
    supported = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if config.torch_dtype not in supported:
        raise ValueError("不支持的模型 dtype: " + config.torch_dtype)
    return supported[config.torch_dtype]


def weight_files(model_directory):
    directory = Path(model_directory)
    single = directory / "model.safetensors"
    if single.is_file():
        return [single]
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError("没有找到 Safetensors 权重或分片索引")
    with index_path.open("r", encoding="utf-8") as file:
        index = json.load(file)
    names = sorted(set(index["weight_map"].values()))
    files = [directory / name for name in names]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少权重分片: " + ", ".join(missing))
    return files


def load_handwritten_model(model_directory, device, dtype=None):
    config = Qwen3Config.from_model_directory(model_directory)
    target_dtype = model_dtype_from_config(config) if dtype is None else dtype
    if target_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("不支持的模型 dtype: " + str(target_dtype))

    model = Qwen3ForCausalLM(config).to(dtype=target_dtype)
    expected = set(model.state_dict()) - {"lm_head.weight"}
    loaded = set()
    unexpected = set()
    for path in weight_files(model_directory):
        shard = load_file(str(path), device="cpu")
        result = model.load_state_dict(shard, strict=False)
        unexpected.update(result.unexpected_keys)
        loaded.update(shard)
        del shard
    missing = expected - loaded
    if missing or unexpected:
        raise RuntimeError(
            "权重名称不匹配，missing=%s, unexpected=%s"
            % (sorted(missing), sorted(unexpected))
        )
    model.tie_weights()
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    model.eval()
    return model
