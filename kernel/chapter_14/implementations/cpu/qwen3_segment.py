"""第 14 期小 Qwen3 前向段：复用第 01 期模型代码，构造可编译的模型段。

复用声明（协议：不跨期 import，复制改造自 chapter_01/qwen3_model.py）：
- RMSNorm / RotaryEmbedding / Qwen3Attention / Qwen3MLP / Qwen3DecoderLayer
  / Qwen3Model 的 forward 数学与第 01 期逐字一致（仅去掉 probe 参数与
  文档位置调整）；
- "段"的定义：embed_tokens → 2 × DecoderLayer → final norm（不含 lm_head，
  避免词表投影把测量变成 matmul 吞吐实验；段的最后一层 RMSNorm 恰好是
  本期第 1 档被编译对象在真实模型里的位置）；
- 配置取 Qwen3-0.6B 的 hidden 口径（hidden=1024 / 16 头 / 8 KV 头 /
  head_dim=128 / intermediate=3072 / eps=1e-6 / rope_theta=1e6），层数减为
  2、权重用固定 seed 随机初始化 FP32（本地无模型权重；编译器融合行为
  与权重数值无关，随机初始化即可，如实记录）。

被编译时 dynamo 面对的是"整段 Python 前向"：图捕获会跨 RMSNorm/Attention/
MLP/Residual，Inductor 决定哪些子链融合成 kernel、哪些 matmul 留给
ATen/MKL、softmax 等如何降级——这是第 3 档实验的素材。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SegmentConfig:
    """小 Qwen3 段配置（hidden 口径 = Qwen3-0.6B config.json）。"""

    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 2
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    def validate(self):
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("Attention 头数必须能被 KV 头数整除")


class RMSNorm(nn.Module):
    """Qwen3 使用的均方根归一化（平方和求均值固定 float32）——第 01 期原样。"""

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
    """RoPE 正余弦位置编码——第 01 期原样。"""

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
    """把向量的两半交换，并对原来的后半部分取负——第 01 期原样。"""
    half = hidden_states.shape[-1] // 2
    first_half = hidden_states[..., :half]
    second_half = hidden_states[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_embedding(query, key, cosine, sine):
    """RoPE 同时应用到 Query 与 Key——第 01 期原样。"""
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotated_query = query * cosine + rotate_half(query) * sine
    rotated_key = key * cosine + rotate_half(key) * sine
    return rotated_query, rotated_key


def repeat_key_value(hidden_states, repeat_count):
    """把较少的 KV 头复制到与 Query 头数相同——第 01 期原样。"""
    if repeat_count == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeat_count, dim=1)


class Qwen3Attention(nn.Module):
    """不带 KV Cache 的 Grouped Query Attention——第 01 期原样。"""

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

    def forward(self, hidden_states, cosine, sine, causal_mask):
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        query = query.view(
            batch_size, sequence_length, self.num_attention_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_proj(hidden_states)
        key = key.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        key = self.k_norm(key).transpose(1, 2)
        value = self.v_proj(hidden_states)
        value = value.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value = value.transpose(1, 2)
        query, key = apply_rotary_embedding(query, key, cosine, sine)
        key = repeat_key_value(key, self.num_key_value_groups)
        value = repeat_key_value(value, self.num_key_value_groups)
        attention_scores = torch.matmul(query, key.transpose(2, 3))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_scores = attention_scores + causal_mask
        attention_probs = F.softmax(attention_scores.float(), dim=-1).to(query.dtype)
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, sequence_length, -1)
        return self.o_proj(attention_output)


class Qwen3MLP(nn.Module):
    """SwiGLU 前馈网络——第 01 期原样。"""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen3DecoderLayer(nn.Module):
    """Pre-Norm Decoder Layer——第 01 期原样。"""

    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, cosine, sine, causal_mask):
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(normalized, cosine, sine, causal_mask)
        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normalized)
        return hidden_states


class Qwen3Segment(nn.Module):
    """embed → 2×DecoderLayer → final norm 的前向段（不含 lm_head）。"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or SegmentConfig()
        self.config.validate()
        cfg = self.config
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    def make_attention_mask(self, sequence_length, dtype, device):
        smallest_value = torch.finfo(dtype).min
        mask = torch.full(
            (sequence_length, sequence_length), smallest_value, dtype=dtype, device=device
        )
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        batch_size, sequence_length = input_ids.shape
        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        causal_mask = self.make_attention_mask(
            sequence_length, hidden_states.dtype, hidden_states.device
        )
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, cosine, sine, causal_mask)
        return self.norm(hidden_states)


def build_segment(seed: int = 20260415) -> Qwen3Segment:
    """固定 seed 构建随机初始化 FP32 段（编译器行为与权重数值无关）。"""
    generator = torch.Generator().manual_seed(seed)
    segment = Qwen3Segment()
    for name, parameter in segment.named_parameters():
        if "embed_tokens" in name:
            torch.nn.init.normal_(parameter, mean=0.0, std=0.02, generator=generator)
        else:
            fan_in = max(1, parameter.shape[-1])
            torch.nn.init.normal_(parameter, mean=0.0, std=fan_in ** -0.5, generator=generator)
    return segment
