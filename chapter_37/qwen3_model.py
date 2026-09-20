"""使用基础 PyTorch 组件实现可训练的 Qwen3-0.6B-Base。

本文件故意不使用 Transformers 中的模型实现，也不加入 KV Cache、
FlashAttention 等优化，目的是清楚展示一次完整的模型前向和反向计算。

Chapter 37 additions (everything else is the unchanged Chapter 33/20/21 copy):
``embed_factory`` / ``lm_head_factory`` let the model build the vocab-parallel
embedding and LM head instead of the replicated pair, and
``load_handwritten_model_vocab_tp`` loads a model whose embedding, attention
AND MLP weights are all read as rank-local shards straight from safetensors.
Qwen3-0.6B ties the embedding and LM head weights, so the two vocab shards
are literally the same Parameter — one row block of the embedding matrix per
rank, consumed from below (lookup) and from above (logits).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mlp_tp import GATE_KEY_SUFFIX, DOWN_KEY_SUFFIX, UP_KEY_SUFFIX, ParallelMLP, shard_error


# 固定模型版本可以避免上游权重更新后，实验结果悄悄发生变化。
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


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

    def __init__(self, config, mlp_factory=None, attn_factory=None):
        super().__init__()
        # attn_factory=None keeps the replicated Qwen3Attention; chapter 21
        # passes a ParallelAttention factory so only the attention weight
        # layout changes (the chapter-35 mlp_factory stays optional).
        self.self_attn = attn_factory(config) if attn_factory is not None else Qwen3Attention(config)
        self.mlp = mlp_factory(config) if mlp_factory is not None else Qwen3MLP(config)
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

    def __init__(self, config, mlp_factory=None, attn_factory=None, embed_factory=None):
        super().__init__()
        # embed_factory=None keeps the replicated nn.Embedding; chapter 22
        # passes a VocabParallelEmbedding factory so only the vocab dim layout
        # changes.
        self.embed_tokens = embed_factory(config) if embed_factory is not None else nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, mlp_factory=mlp_factory, attn_factory=attn_factory)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def make_attention_mask(self, attention_mask, sequence_length, dtype, device):
        """同时屏蔽未来 Token 和作为 Key 的 Padding Token。"""

        smallest_value = torch.finfo(dtype).min
        mask = torch.full(
            (sequence_length, sequence_length),
            smallest_value,
            dtype=dtype,
            device=device,
        )
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is None:
            return mask
        if attention_mask.shape[1] != sequence_length:
            raise ValueError("attention_mask 与 input_ids 的序列长度不一致")
        mask = mask.expand(attention_mask.shape[0], 1, sequence_length, sequence_length).clone()
        key_is_padding = attention_mask[:, None, None, :] == 0
        return mask.masked_fill(key_is_padding, smallest_value)

    def forward(self, input_ids, attention_mask=None, position_ids=None, attention_mask_4d=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 [batch, sequence] 的二维 Tensor")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask 的 Shape 必须与 input_ids 相同")
        hidden_states = self.embed_tokens(input_ids)
        batch_size, sequence_length = input_ids.shape

        # Sequence Packing 需要 Pack 内每个样本的位置从 0 重置；默认行为不变。
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape != input_ids.shape:
            raise ValueError("position_ids 的 Shape 必须与 input_ids 相同")
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        if attention_mask_4d is not None:
            expected = (batch_size, 1, sequence_length, sequence_length)
            if tuple(attention_mask_4d.shape) != expected:
                raise ValueError(f"attention_mask_4d 的 Shape 必须是 {expected}")
            causal_mask = attention_mask_4d.to(hidden_states.dtype)
        else:
            causal_mask = self.make_attention_mask(
                attention_mask, sequence_length, hidden_states.dtype, hidden_states.device
            )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, cosine, sine, causal_mask
            )

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 主干网络加上把隐藏状态映射为词表 Logits 的输出层。"""

    def __init__(self, config, mlp_factory=None, attn_factory=None, embed_factory=None, lm_head_factory=None):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, mlp_factory=mlp_factory, attn_factory=attn_factory, embed_factory=embed_factory)
        if embed_factory is not None and lm_head_factory is None:
            raise ValueError(
                "a vocab-parallel embedding needs a matching vocab-parallel lm_head_factory: "
                "tie_weights shares one sharded Parameter between lookup and logits"
            )
        self.lm_head = lm_head_factory(config) if lm_head_factory is not None else nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.tie_weights()

    def tie_weights(self):
        # Qwen3-0.6B 的输入 Embedding 与输出 LM Head 使用同一份权重。
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None, attention_mask_4d=None):
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            attention_mask_4d=attention_mask_4d,
        )
        return self.lm_head(hidden_states)


def resolve_model_directory(model_name_or_path, revision=DEFAULT_MODEL_REVISION):
    """返回本地模型目录；模型不存在时从 Hugging Face 下载固定版本。"""

    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())

    from huggingface_hub import snapshot_download

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


def load_handwritten_model(model_directory, device, dtype=None):
    """从 safetensors 权重构造手写 Qwen3 模型。"""

    config = Qwen3Config.from_model_directory(model_directory)
    model = Qwen3ForCausalLM(config)

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    from safetensors.torch import load_file

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
    model_dtype = dtype or model_dtype_from_config(config)

    # model.to(dtype=...) 会连 float buffer 一起转换。RoPE 的频率必须保留
    # float32 精度，因此先保存一份，再在模型移动完成后放回目标设备。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    return model


def load_handwritten_model_tp(model_directory, device, tp_rank, tp_size, dtype=None, group=None):
    """Build the TP model: full non-MLP weights, rank-local MLP weight shards.

    Non-MLP weights are read completely on every rank.  MLP weights are NOT
    materialized in full and then split: the safetensors ``PySafeSlice`` API
    reads only this rank's ``[tp_rank*local:(tp_rank+1)*local]`` window, so a
    rank never holds the other ranks' MLP weight bytes in host memory.  A
    full-then-split fallback would double the peak host memory of the MLP for
    no benefit; the sliced read is only a few lines more.
    """

    config = Qwen3Config.from_model_directory(model_directory)
    if config.intermediate_size % tp_size != 0:
        raise shard_error(
            f"model intermediate size {config.intermediate_size} is not divisible by tp_size {tp_size}",
            (config.intermediate_size, config.hidden_size),
            torch.float32,
            torch.device("cpu"),
            tp_rank,
            tp_size,
        )
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")

    def mlp_factory(layer_config):
        return ParallelMLP(
            layer_config.hidden_size,
            layer_config.intermediate_size,
            tp_rank,
            tp_size,
            group=group,
        )

    model = Qwen3ForCausalLM(config, mlp_factory=mlp_factory)

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    from safetensors import safe_open

    local_intermediate = config.intermediate_size // tp_size
    shard_begin = tp_rank * local_intermediate
    shard_end = shard_begin + local_intermediate
    state_dict = {}
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            source = handle.get_slice(key)
            if key.endswith(GATE_KEY_SUFFIX) or key.endswith(UP_KEY_SUFFIX):
                # Column parallel: full [intermediate, hidden] -> shard rows.
                shape = source.get_shape()
                if shape[0] != config.intermediate_size:
                    raise shard_error(
                        f"unexpected gate/up shape for {key!r} (expected dim0 {config.intermediate_size})",
                        shape,
                        source.get_dtype(),
                        "cpu",
                        tp_rank,
                        tp_size,
                    )
                state_dict[key] = source[shard_begin:shard_end, :]
            elif key.endswith(DOWN_KEY_SUFFIX):
                # Row parallel: full [hidden, intermediate] -> shard columns.
                shape = source.get_shape()
                if shape[1] != config.intermediate_size:
                    raise shard_error(
                        f"unexpected down shape for {key!r} (expected dim1 {config.intermediate_size})",
                        shape,
                        source.get_dtype(),
                        "cpu",
                        tp_rank,
                        tp_size,
                    )
                state_dict[key] = source[:, shard_begin:shard_end]
            else:
                state_dict[key] = source[:]

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
    model_dtype = dtype or model_dtype_from_config(config)

    # model.to(dtype=...) 会连 float buffer 一起转换。RoPE 的频率必须保留
    # float32 精度，因此先保存一份，再在模型移动完成后放回目标设备。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    return model


def load_handwritten_model_atp(model_directory, device, tp_rank, tp_size, dtype=None, group=None):
    """Build the TP model with BOTH attention and MLP weights rank-sharded.

    Attention heads split along the head dim (q/k/v row blocks, o column
    blocks — each head occupies ``head_dim`` consecutive rows, so contiguous
    slices are whole heads); MLP shards follow chapter 20; norms, embedding
    and LM head stay fully replicated on every rank.  As in
    ``load_handwritten_model_tp``, every shard is read through the
    safetensors ``PySafeSlice`` API so a rank never materializes weights that
    belong to other ranks.  Head divisibility (num_attention_heads and
    num_key_value_heads vs tp_size) fails loudly via ``gqa_head_layout``.
    """

    # Imported here (not at module level) to keep the import direction
    # qwen3_model <- attention_tp acyclic.
    from attention_tp import (
        K_KEY_SUFFIX,
        O_KEY_SUFFIX,
        ParallelAttention,
        Q_KEY_SUFFIX,
        V_KEY_SUFFIX,
        gqa_head_layout,
    )

    config = Qwen3Config.from_model_directory(model_directory)
    if config.intermediate_size % tp_size != 0:
        raise shard_error(
            f"model intermediate size {config.intermediate_size} is not divisible by tp_size {tp_size}",
            (config.intermediate_size, config.hidden_size),
            torch.float32,
            torch.device("cpu"),
            tp_rank,
            tp_size,
        )
    # Raises shard_error with full context when head counts do not divide;
    # called here so a bad tp_size dies before any weight byte is read.
    layout = gqa_head_layout(tp_rank, tp_size, config.num_attention_heads, config.num_key_value_heads)
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")

    def mlp_factory(layer_config):
        return ParallelMLP(
            layer_config.hidden_size,
            layer_config.intermediate_size,
            tp_rank,
            tp_size,
            group=group,
        )

    def attn_factory(layer_config):
        return ParallelAttention(
            layer_config.hidden_size,
            layer_config.num_attention_heads,
            layer_config.num_key_value_heads,
            layer_config.head_dim,
            layer_config.rms_norm_eps,
            tp_rank,
            tp_size,
            group=group,
        )

    model = Qwen3ForCausalLM(config, mlp_factory=mlp_factory, attn_factory=attn_factory)

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    from safetensors import safe_open

    local_q_rows = layout["local_q_heads"] * config.head_dim
    local_kv_rows = layout["local_kv_heads"] * config.head_dim
    q_begin = tp_rank * local_q_rows
    kv_begin = tp_rank * local_kv_rows
    q_end = q_begin + local_q_rows
    kv_end = kv_begin + local_kv_rows
    local_intermediate = config.intermediate_size // tp_size
    shard_begin = tp_rank * local_intermediate
    shard_end = shard_begin + local_intermediate
    state_dict = {}
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            source = handle.get_slice(key)
            if key.endswith(Q_KEY_SUFFIX) or key.endswith(O_KEY_SUFFIX):
                # Q rows / o columns follow the Q-head layout.
                shape = source.get_shape()
                expected = config.num_attention_heads * config.head_dim
                if shape[0 if key.endswith(Q_KEY_SUFFIX) else 1] != expected:
                    raise shard_error(
                        f"unexpected attention shape for {key!r} (expected Q dim {expected})",
                        shape,
                        source.get_dtype(),
                        "cpu",
                        tp_rank,
                        tp_size,
                    )
                state_dict[key] = source[q_begin:q_end, :] if key.endswith(Q_KEY_SUFFIX) else source[:, q_begin:q_end]
            elif key.endswith(K_KEY_SUFFIX) or key.endswith(V_KEY_SUFFIX):
                shape = source.get_shape()
                expected = config.num_key_value_heads * config.head_dim
                if shape[0] != expected:
                    raise shard_error(
                        f"unexpected KV shape for {key!r} (expected dim0 {expected})",
                        shape,
                        source.get_dtype(),
                        "cpu",
                        tp_rank,
                        tp_size,
                    )
                state_dict[key] = source[kv_begin:kv_end, :]
            elif key.endswith(GATE_KEY_SUFFIX) or key.endswith(UP_KEY_SUFFIX):
                state_dict[key] = source[shard_begin:shard_end, :]
            elif key.endswith(DOWN_KEY_SUFFIX):
                state_dict[key] = source[:, shard_begin:shard_end]
            else:
                state_dict[key] = source[:]

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
    model_dtype = dtype or model_dtype_from_config(config)

    # model.to(dtype=...) 会连 float buffer 一起转换。RoPE 的频率必须保留
    # float32 精度，因此先保存一份，再在模型移动完成后放回目标设备。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    return model


def load_handwritten_model_vocab_tp(model_directory, device, tp_rank, tp_size, dtype=None, group=None):
    """Build the TP model with vocab, attention AND MLP weights rank-sharded.

    The embedding matrix is split along the vocab dim: rank r owns rows
    ``[r*local_vocab, (r+1)*local_vocab)``.  Because Qwen3-0.6B ties the LM
    head to the embedding, the SAME sharded Parameter produces the logits —
    ``lm_head.weight`` stays absent from the safetensors and is re-tied after
    loading.  Attention and MLP shards follow chapters 21/20; only the
    LayerNorms are fully replicated.  As in the earlier loaders, every shard
    is read through the safetensors ``PySafeSlice`` API so a rank never
    materializes another rank's rows.
    """

    # Imported here (not at module level) to keep the import direction
    # qwen3_model <- attention_tp / vocab_tp acyclic.
    from attention_tp import (
        K_KEY_SUFFIX,
        O_KEY_SUFFIX,
        ParallelAttention,
        Q_KEY_SUFFIX,
        V_KEY_SUFFIX,
        gqa_head_layout,
    )
    from vocab_tp import EMBED_KEY_SUFFIX, VocabParallelEmbedding, VocabParallelLMHead

    config = Qwen3Config.from_model_directory(model_directory)
    for value, name in (
        (config.vocab_size, "vocab size"),
        (config.intermediate_size, "intermediate size"),
    ):
        if value % tp_size != 0:
            raise shard_error(
                f"model {name} {value} is not divisible by tp_size {tp_size}",
                (value, config.hidden_size),
                torch.float32,
                torch.device("cpu"),
                tp_rank,
                tp_size,
            )
    layout = gqa_head_layout(tp_rank, tp_size, config.num_attention_heads, config.num_key_value_heads)
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")

    def mlp_factory(layer_config):
        return ParallelMLP(
            layer_config.hidden_size,
            layer_config.intermediate_size,
            tp_rank,
            tp_size,
            group=group,
        )

    def attn_factory(layer_config):
        return ParallelAttention(
            layer_config.hidden_size,
            layer_config.num_attention_heads,
            layer_config.num_key_value_heads,
            layer_config.head_dim,
            layer_config.rms_norm_eps,
            tp_rank,
            tp_size,
            group=group,
        )

    def embed_factory(layer_config):
        return VocabParallelEmbedding(
            layer_config.vocab_size,
            layer_config.hidden_size,
            tp_rank,
            tp_size,
            group=group,
        )

    def lm_head_factory(layer_config):
        return VocabParallelLMHead(
            layer_config.vocab_size,
            layer_config.hidden_size,
            tp_rank,
            tp_size,
            group=group,
        )

    model = Qwen3ForCausalLM(
        config, mlp_factory=mlp_factory, attn_factory=attn_factory,
        embed_factory=embed_factory, lm_head_factory=lm_head_factory,
    )

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    from safetensors import safe_open

    local_vocab = config.vocab_size // tp_size
    vocab_begin = tp_rank * local_vocab
    vocab_end = vocab_begin + local_vocab
    local_q_rows = layout["local_q_heads"] * config.head_dim
    local_kv_rows = layout["local_kv_heads"] * config.head_dim
    q_begin = tp_rank * local_q_rows
    kv_begin = tp_rank * local_kv_rows
    q_end = q_begin + local_q_rows
    kv_end = kv_begin + local_kv_rows
    local_intermediate = config.intermediate_size // tp_size
    shard_begin = tp_rank * local_intermediate
    shard_end = shard_begin + local_intermediate
    state_dict = {}
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            source = handle.get_slice(key)
            if key.endswith(EMBED_KEY_SUFFIX):
                shape = source.get_shape()
                if shape[0] != config.vocab_size:
                    raise shard_error(
                        f"unexpected embedding shape for {key!r} (expected dim0 {config.vocab_size})",
                        shape,
                        source.get_dtype(),
                        "cpu",
                        tp_rank,
                        tp_size,
                    )
                state_dict[key] = source[vocab_begin:vocab_end, :]
            elif key.endswith(Q_KEY_SUFFIX) or key.endswith(O_KEY_SUFFIX):
                shape = source.get_shape()
                expected = config.num_attention_heads * config.head_dim
                if shape[0 if key.endswith(Q_KEY_SUFFIX) else 1] != expected:
                    raise shard_error(
                        f"unexpected attention shape for {key!r} (expected Q dim {expected})",
                        shape,
                        source.get_dtype(),
                        "cpu",
                        tp_rank,
                        tp_size,
                    )
                state_dict[key] = source[q_begin:q_end, :] if key.endswith(Q_KEY_SUFFIX) else source[:, q_begin:q_end]
            elif key.endswith(K_KEY_SUFFIX) or key.endswith(V_KEY_SUFFIX):
                shape = source.get_shape()
                expected = config.num_key_value_heads * config.head_dim
                if shape[0] != expected:
                    raise shard_error(
                        f"unexpected KV shape for {key!r} (expected dim0 {expected})",
                        shape,
                        source.get_dtype(),
                        "cpu",
                        tp_rank,
                        tp_size,
                    )
                state_dict[key] = source[kv_begin:kv_end, :]
            elif key.endswith(GATE_KEY_SUFFIX) or key.endswith(UP_KEY_SUFFIX):
                state_dict[key] = source[shard_begin:shard_end, :]
            elif key.endswith(DOWN_KEY_SUFFIX):
                state_dict[key] = source[:, shard_begin:shard_end]
            else:
                state_dict[key] = source[:]

    load_result = model.load_state_dict(state_dict, strict=False)

    # lm_head.weight 与 embed_tokens.weight 共享分片，safetensors 只保存一份。
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
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise RuntimeError("tie_weights failed: the vocab shard is not shared between embedding and lm_head")
    model_dtype = dtype or model_dtype_from_config(config)

    # model.to(dtype=...) 会连 float buffer 一起转换。RoPE 的频率必须保留
    # float32 精度，因此先保存一份，再在模型移动完成后放回目标设备。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    return model
