"""第 08 期独立使用的 Qwen3-0.6B 推理模型。

只依赖 PyTorch、huggingface-hub 和 safetensors，不引用其他期代码。
模型显式接收二维 Attention Mask 与 Position IDs，以支持左 Padding、
Batched Prefill 和分页 Cache 的权重计算组件。
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


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
            raise ValueError("本期实现只支持共享 Embedding 和 LM Head 权重的模型")


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


def cache_size_bytes(past_key_values):
    if past_key_values is None:
        return 0
    return sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in past_key_values
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

    def forward(
        self,
        hidden_states,
        cosine,
        sine,
        attention_mask,
        past_key_value=None,
        use_cache=False,
    ):
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
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        present_key_value = (key, value) if use_cache else None

        attention_key = repeat_key_value(key, self.num_key_value_groups)
        attention_value = repeat_key_value(value, self.num_key_value_groups)
        scores = torch.matmul(query, attention_key.transpose(2, 3))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + attention_mask
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, attention_value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, -1
        )
        return self.o_proj(output), present_key_value


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

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
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        cosine,
        sine,
        attention_mask,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        attention_output, present = self.self_attn(
            self.input_layernorm(hidden_states),
            cosine,
            sine,
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present


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

    def validate_past_key_values(self, past_key_values, batch_size):
        if past_key_values is None:
            return 0
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values 层数与模型层数不一致")
        past_length = past_key_values[0][0].shape[2]
        expected_prefix = (batch_size, self.config.num_key_value_heads)
        for layer_index, (key, value) in enumerate(past_key_values):
            expected = expected_prefix + (past_length, self.config.head_dim)
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(
                    "第 %d 层 KV Cache 形状错误: key=%s, value=%s, expected=%s"
                    % (layer_index, tuple(key.shape), tuple(value.shape), expected)
                )
        return past_length

    def prepare_inputs(
        self, input_ids, attention_mask, position_ids, past_key_values
    ):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids 必须是 [batch, sequence] 非空 Tensor")
        batch_size, query_length = input_ids.shape
        past_length = self.validate_past_key_values(past_key_values, batch_size)
        key_value_length = past_length + query_length

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, key_value_length),
                dtype=torch.bool,
                device=input_ids.device,
            )
        if tuple(attention_mask.shape) != (batch_size, key_value_length):
            raise ValueError(
                "attention_mask 应为 %s，实际为 %s"
                % ((batch_size, key_value_length), tuple(attention_mask.shape))
            )
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, key_value_length, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        if tuple(position_ids.shape) != (batch_size, query_length):
            raise ValueError(
                "position_ids 应为 %s，实际为 %s"
                % ((batch_size, query_length), tuple(position_ids.shape))
            )
        position_ids = position_ids.to(device=input_ids.device, dtype=torch.long)

        query_physical_positions = torch.arange(
            past_length, key_value_length, device=input_ids.device
        )
        key_physical_positions = torch.arange(
            key_value_length, device=input_ids.device
        )
        future = key_physical_positions.unsqueeze(0) > query_physical_positions.unsqueeze(1)
        allowed = (~future).unsqueeze(0) & attention_mask.unsqueeze(1)
        return position_ids, allowed.unsqueeze(1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        if past_key_values is not None and not use_cache:
            raise ValueError("传入 past_key_values 时必须设置 use_cache=True")
        hidden_states = self.embed_tokens(input_ids)
        position_ids, allowed = self.prepare_inputs(
            input_ids, attention_mask, position_ids, past_key_values
        )
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        additive_mask = torch.zeros(
            allowed.shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        additive_mask.masked_fill_(~allowed, torch.finfo(hidden_states.dtype).min)

        present_key_values = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            layer_past = (
                None if past_key_values is None else past_key_values[layer_index]
            )
            hidden_states, present = layer(
                hidden_states,
                cosine,
                sine,
                additive_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                present_key_values.append(present)
        return self.norm(hidden_states), present_key_values


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        hidden_states, present = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return self.lm_head(hidden_states), present


def resolve_model_directory(model_name_or_path, revision=DEFAULT_MODEL_REVISION):
    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())
    return snapshot_download(
        repo_id=model_name_or_path,
        revision=revision,
        max_workers=2,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
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


def load_handwritten_model(model_directory, device, dtype=None):
    config = Qwen3Config.from_model_directory(model_directory)
    model = Qwen3ForCausalLM(config)
    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))
    state_dict = load_file(str(weight_path), device="cpu")
    result = model.load_state_dict(state_dict, strict=False)
    unexpected_missing = set(result.missing_keys) - {"lm_head.weight"}
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            "权重名称不匹配，missing=%s, unexpected=%s"
            % (sorted(unexpected_missing), sorted(result.unexpected_keys))
        )
    model.tie_weights()
    target_dtype = model_dtype_from_config(config) if dtype is None else dtype
    if target_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("不支持的模型 dtype: " + str(target_dtype))
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=target_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    model.eval()
    return model
