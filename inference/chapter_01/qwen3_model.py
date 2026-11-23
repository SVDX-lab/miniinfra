"""使用基础 PyTorch 组件实现 Qwen3-0.6B。

本文件故意不使用 Transformers 中的模型实现，也不加入 KV Cache、
FlashAttention 等优化，目的是清楚展示一次完整的模型前向计算。
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


# 固定模型版本可以避免上游权重更新后，实验结果悄悄发生变化。
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass
class Qwen3Config:
    """手写实现真正需要使用的模型配置。"""

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
        config_path = Path(model_directory) / "config.json"
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = json.load(file)

        config = cls(
            vocab_size=raw_config["vocab_size"],
            hidden_size=raw_config["hidden_size"],
            intermediate_size=raw_config["intermediate_size"],
            num_hidden_layers=raw_config["num_hidden_layers"],
            num_attention_heads=raw_config["num_attention_heads"],
            num_key_value_heads=raw_config["num_key_value_heads"],
            head_dim=raw_config["head_dim"],
            rms_norm_eps=raw_config["rms_norm_eps"],
            rope_theta=raw_config["rope_theta"],
            torch_dtype=raw_config.get("torch_dtype", "bfloat16"),
            tie_word_embeddings=raw_config.get("tie_word_embeddings", True),
        )
        config.validate()
        return config

    def validate(self):
        """尽早拒绝本期尚未覆盖的模型变体，避免静默地产生错误结果。"""

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("Attention 头数必须能被 KV 头数整除")
        if not self.tie_word_embeddings:
            raise ValueError("本期实现只支持共享 Embedding 和 LM Head 权重的模型")


class RMSNorm(nn.Module):
    """Qwen3 使用的均方根归一化。"""

    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # 平方和求均值使用 float32，防止 bfloat16 精度不足。
        input_dtype = hidden_states.dtype
        float_states = hidden_states.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """生成 RoPE 所需的正弦和余弦位置编码。"""

    def __init__(self, head_dim, rope_theta):
        super().__init__()

        # 每两个维度共享一个旋转频率。inv_freq 不参与训练，所以注册为 buffer。
        dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (dimensions / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, output_dtype):
        # freqs: [batch, sequence_length, head_dim / 2]
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq
        # Qwen3 将同一组频率分别应用到向量的前半部分和后半部分。
        embedding = torch.cat((freqs, freqs), dim=-1)
        return embedding.cos().to(output_dtype), embedding.sin().to(output_dtype)


def rotate_half(hidden_states):
    """把向量的两半交换，并对原来的后半部分取负。"""

    half = hidden_states.shape[-1] // 2
    first_half = hidden_states[..., :half]
    second_half = hidden_states[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_embedding(query, key, cosine, sine):
    """把 RoPE 同时应用到 Query 和 Key。"""

    # Query/Key 多一个 attention head 维度，需要在这里补一个维度用于广播。
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotated_query = query * cosine + rotate_half(query) * sine
    rotated_key = key * cosine + rotate_half(key) * sine
    return rotated_query, rotated_key


def repeat_key_value(hidden_states, repeat_count):
    """把较少的 KV 头复制到与 Query 头数相同。"""

    if repeat_count == 1:
        return hidden_states
    # 这里选择直观的 repeat_interleave；Paged Attention 等优化留到后续章节。
    return hidden_states.repeat_interleave(repeat_count, dim=1)


class Qwen3Attention(nn.Module):
    """不带 KV Cache 的 Grouped Query Attention。"""

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

        # Qwen3 会在每个 attention head 内单独归一化 Query 和 Key。
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, cosine, sine, causal_mask):
        batch_size, sequence_length, _ = hidden_states.shape

        # 投影后拆分 head，形状变为 [batch, heads, sequence, head_dim]。
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

        # causal_mask 保证当前位置只能看到自己和之前的 Token。
        attention_scores = torch.matmul(query, key.transpose(2, 3))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_scores = attention_scores + causal_mask

        # Softmax 使用 float32 计算，再转回模型 dtype，与参考实现保持一致。
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_probs = attention_probs.to(query.dtype)
        attention_output = torch.matmul(attention_probs, value)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, sequence_length, -1)
        return self.o_proj(attention_output)


class Qwen3MLP(nn.Module):
    """Qwen3 的 SwiGLU 前馈网络。"""

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
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen3DecoderLayer(nn.Module):
    """一个完整的 Qwen3 Decoder Layer。"""

    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden_states, cosine, sine, causal_mask):
        # Attention 子层：Pre-Norm -> Attention -> Residual。
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            normalized, cosine, sine, causal_mask
        )

        # MLP 子层：Pre-Norm -> MLP -> Residual。
        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normalized)
        return hidden_states


class Qwen3Model(nn.Module):
    """由 Embedding、多个 Decoder Layer 和最终 Norm 组成的主干网络。"""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def make_causal_mask(self, sequence_length, dtype, device):
        """创建上三角为极小值、其余位置为 0 的 Attention Mask。"""

        smallest_value = torch.finfo(dtype).min
        mask = torch.full(
            (sequence_length, sequence_length),
            smallest_value,
            dtype=dtype,
            device=device,
        )
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        batch_size, sequence_length = input_ids.shape

        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        causal_mask = self.make_causal_mask(
            sequence_length, hidden_states.dtype, hidden_states.device
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, cosine, sine, causal_mask
            )

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 主干网络加上把隐藏状态映射为词表 Logits 的输出层。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        # Qwen3-0.6B 的输入 Embedding 与输出 LM Head 使用同一份权重。
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)


def resolve_model_directory(model_name_or_path, revision=DEFAULT_MODEL_REVISION):
    """返回本地模型目录；模型不存在时从 Hugging Face 下载固定版本。"""

    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())

    return snapshot_download(
        repo_id=model_name_or_path,
        revision=revision,
        # 测试服务器到 Hugging Face 的连接偶尔会被重置。降低并发数可以
        # 提高首次下载的稳定性，且不会影响模型加载和推理性能。
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
    """把配置文件中的 dtype 名称转换为 PyTorch dtype。"""

    supported_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if config.torch_dtype not in supported_dtypes:
        raise ValueError("不支持的模型 dtype: " + config.torch_dtype)
    return supported_dtypes[config.torch_dtype]


def load_handwritten_model(model_directory, device):
    """从 safetensors 权重构造手写 Qwen3 模型。"""

    config = Qwen3Config.from_model_directory(model_directory)
    model = Qwen3ForCausalLM(config)

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    state_dict = load_file(str(weight_path), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)

    # 共享权重只在 safetensors 中保存一份，所以 lm_head.weight 允许缺失。
    allowed_missing_keys = {"lm_head.weight"}
    unexpected_missing = set(load_result.missing_keys) - allowed_missing_keys
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "权重名称不匹配，missing="
            + str(sorted(unexpected_missing))
            + ", unexpected="
            + str(sorted(load_result.unexpected_keys))
        )

    model.tie_weights()
    model_dtype = model_dtype_from_config(config)

    # model.to(dtype=...) 会连 float buffer 一起转换。RoPE 的频率必须保留
    # float32 精度，因此先保存一份，再在模型移动完成后放回目标设备。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    model.eval()
    return model

