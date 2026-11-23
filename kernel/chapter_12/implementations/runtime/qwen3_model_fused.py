"""第 04 期修改版手写 Qwen3：原生组合表达式 vs 自定义融合算子（可切换）。

本文件复制自 chapter_01/qwen3_model.py（已验证实现，协议允许复制不跨期
import），本章改动**只有一处主题**：把"相邻 Residual Add 与 RMSNorm"的
原生组合表达式替换为自定义融合算子的可切换路径。与 chapter_01 的逐项差异：

1. Qwen3Model / Qwen3ForCausalLM 增加构造参数 `fused_ops=None`
   （MiniInfraOps 协议对象，见 implementations/ops_router.py）；
   None = 原生路径（与 chapter_01 逐字相同）；非 None = 融合路径；
2. Qwen3Model.forward 增加融合分支（vLLM 式重构，见下方注释）；
3. load_handwritten_model 增加 fused_ops 透传；
4. 其余（config/RoPE/Attention/MLP/RMSNorm 模块/Greedy 解码）零改动。

融合分支的数据流重构（生产引擎同款，vLLM ResidualConnect 模式）：

    原生：  h = embed;  for L: n=L.in_ln(h); h = h + attn(n);
                          n=L.post_ln(h); h = h + mlp(n);   h = norm(h)
    融合：  residual = embed（residual 流全程原位更新，不再重assign）;
            for L:
              [边界 op]  residual += 上一层 mlp_out → 用 L.in_ln.weight 归一化
                         （layer 0 无前置 add，保留原生 input_layernorm 模块）
              attn_out = attn(normalized)
              [层内 op]  residual += attn_out → 用 L.post_ln.weight 归一化，
                         归一化结果直接写进 attn_out buffer
              mlp_out = mlp(normalized_attn)
            [终局 op]  residual += mlp_out → 用 model.norm.weight 归一化
            hidden = mlp_out buffer（终局 op 把它改写为最终 norm 输出）

    op 调用数 = 层数 × 2（28 层模型 = 56 次）：layer 0 的 input_layernorm
    无前置 Residual Add（embed 即初始 residual，保留原生模块——
    "相邻 Add+Norm 对"集合的例外项）；其余每层 2 次（跨层边界 op 覆盖
    上一层 MLP 之后的 Add + 本层 input_layernorm；层内 op 覆盖 Attention
    之后的 Add + post_attention_layernorm）+ 终局 op（最后 Add + final
    norm）。原生路径全部 56 次 Residual Add 与 57 次 RMSNorm 中，56 对
    "相邻 Add+Norm"全部进入融合 op。

数值语义（正确性链容差的依据，合同 numerical_tolerance）：
    - 加法：融合 kernel 与 torch 的 fp16/fp32 加法同为"精确和一次舍入"
      （fp16 加法精确和可被 fp32 表示），位级一致；
    - 归一化：融合 kernel 全程 FP32、输出单次舍入；原生 RMSNorm 模块是
      "FP32 归一化 → 舍回 T → T 乘 weight"两次舍入 —— 数学定义一致，
      数值容差内一致（第 01 期容差链：单层误差/逐层累积/最终 Token）。

probe 语义保持与 chapter_01 完全一致（probes.npz 逐键可比）：
    embedding  = embed 输出；
    layer_i    = 第 i 个 Decoder Layer 完整输出（= residual 流在该层
                 mlp_out 加完后，即下一层边界 op 之后 / 终局 op 之后的
                 residual buffer 内容，FP32 克隆）；
    final_norm = 最终 RMSNorm 输出（终局 op 写进 mlp_out buffer 的值）。
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# 固定模型版本（与 chapter_01 相同）。
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION = "PENDING-DOWNLOAD"


@dataclass
class Qwen3Config:
    """手写实现真正需要使用的模型配置（与 chapter_01 逐字相同）。"""

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
            head_dim=raw_config.get("head_dim", raw_config["hidden_size"] // raw_config["num_attention_heads"]),
            rms_norm_eps=raw_config["rms_norm_eps"],
            rope_theta=raw_config["rope_theta"],
            torch_dtype=raw_config.get("torch_dtype", "bfloat16"),
            tie_word_embeddings=raw_config.get("tie_word_embeddings", True),
        )
        config.validate()
        return config

    def validate(self):
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("Attention 头数必须能被 KV 头数整除")
        if not self.tie_word_embeddings:
            raise ValueError("本期实现只支持共享 Embedding 和 LM Head 权重的模型")


class RMSNorm(nn.Module):
    """Qwen3 使用的均方根归一化（平方和求均值固定使用 float32）。
    与 chapter_01 逐字相同——原生路径的模块本体；融合路径只用它的 weight。"""

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
    """生成 RoPE 所需的正弦和余弦位置编码（与 chapter_01 逐字相同）。"""

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
    first_half = hidden_states[..., :half]
    second_half = hidden_states[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_embedding(query, key, cosine, sine):
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotated_query = query * cosine + rotate_half(query) * sine
    rotated_key = key * cosine + rotate_half(key) * sine
    return rotated_query, rotated_key


def repeat_key_value(hidden_states, repeat_count):
    if repeat_count == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeat_count, dim=1)


class Qwen3Attention(nn.Module):
    """不带 KV Cache 的 Grouped Query Attention（与 chapter_01 逐字相同）。"""

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
    """Qwen3 的 SwiGLU 前馈网络（与 chapter_01 逐字相同）。"""

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
    """一个完整的 Qwen3 Decoder Layer（模块结构零改动；融合重构在
    Qwen3Model.forward 层做——保证 native/fused 共用同一套层模块与权重）。"""

    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden_states, cosine, sine, causal_mask):
        # 原生路径：Pre-Norm -> Attention -> Residual -> Pre-Norm -> MLP -> Residual。
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            normalized, cosine, sine, causal_mask
        )

        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normalized)
        return hidden_states


class Qwen3Model(nn.Module):
    """由 Embedding、多个 Decoder Layer 和最终 Norm 组成的主干网络。

    fused_ops（第 04 期新增）：MiniInfraOps 协议对象或 None。
    None → 原生组合表达式路径（与 chapter_01 逐字相同）。
    非 None → 融合算子路径（fused_ops.fused_add_rmsnorm(input, residual,
    weight, eps)，双 buffer 原地改写）。
    """

    def __init__(self, config, fused_ops=None):
        super().__init__()
        self.fused_ops = fused_ops
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def make_attention_mask(self, attention_mask, sequence_length, dtype, device):
        """屏蔽未来 Token 和作为 Key 的 Padding Token（与 chapter_01 逐字相同）。"""

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

    def _forward_native(self, input_ids, attention_mask, probe):
        """原生组合表达式路径：与 chapter_01 的 forward 逐字相同。"""
        probes: dict[str, torch.Tensor] = {}
        hidden_states = self.embed_tokens(input_ids)
        if probe:
            probes["embedding"] = hidden_states.detach().float().clone()
        batch_size, sequence_length = input_ids.shape

        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        causal_mask = self.make_attention_mask(
            attention_mask, sequence_length, hidden_states.dtype, hidden_states.device
        )

        for layer_index, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states, cosine, sine, causal_mask
            )
            if probe:
                probes[f"layer_{layer_index}"] = hidden_states.detach().float().clone()

        hidden_states = self.norm(hidden_states)
        if probe:
            probes["final_norm"] = hidden_states.detach().float().clone()
        return hidden_states, probes

    def _forward_fused(self, input_ids, attention_mask, probe):
        """融合算子路径（第 04 期主题）：相邻 Residual Add + RMSNorm 全部
        替换为 fused_ops.fused_add_rmsnorm（56 次调用覆盖全部 56 个 Add 与
        56 个 Norm；layer 0 的 input_layernorm 保留原生模块）。

        op 契约：fused_add_rmsnorm(input, residual, weight, eps)
          residual ← residual + input；input ← RMSNorm(residual) * weight
          （双 buffer 原地改写，零新分配——vLLM fused_add_rms_norm 同款）。
        """
        op = self.fused_ops.fused_add_rmsnorm
        eps = float(self.norm.eps)
        probes: dict[str, torch.Tensor] = {}

        residual = self.embed_tokens(input_ids)
        if probe:
            probes["embedding"] = residual.detach().float().clone()
        batch_size, sequence_length = input_ids.shape

        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cosine, sine = self.rotary_emb(position_ids, residual.dtype)
        causal_mask = self.make_attention_mask(
            attention_mask, sequence_length, residual.dtype, residual.device
        )

        carry = None  # 上一层 MLP 输出 buffer（会被下一层边界 op 原地改写为归一化输入）
        for layer_index, decoder_layer in enumerate(self.layers):
            if layer_index == 0:
                # layer 0 的 input_layernorm 前没有 Residual Add（embed 即初始
                # residual），保留原生模块——这是"相邻 Add+Norm"集合的例外项。
                normalized = decoder_layer.input_layernorm(residual)
            else:
                # 跨层边界 op：residual += 上一层 mlp_out；归一化结果写进 carry。
                op(carry, residual, decoder_layer.input_layernorm.weight, eps)
                normalized = carry
                if probe:
                    probes[f"layer_{layer_index - 1}"] = residual.detach().float().clone()

            attn_out = decoder_layer.self_attn(normalized, cosine, sine, causal_mask)
            # 层内 op：residual += attn_out；归一化结果写进 attn_out buffer
            # （attn_out 在 residual 相加后即死值，原地覆盖零分配）。
            op(attn_out, residual, decoder_layer.post_attention_layernorm.weight, eps)
            carry = decoder_layer.mlp(attn_out)

        # 终局 op：residual += 最后层 mlp_out；model.norm 的归一化写进 carry。
        op(carry, residual, self.norm.weight, eps)
        if probe:
            probes[f"layer_{len(self.layers) - 1}"] = residual.detach().float().clone()
            probes["final_norm"] = carry.detach().float().clone()
        return carry, probes

    def forward(self, input_ids, attention_mask=None, probe=False):
        """前向计算（native/fused 按 fused_ops 切换；probe 键集合两者一致）。"""

        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 [batch, sequence] 的二维 Tensor")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask 的 Shape 必须与 input_ids 相同")
        if self.fused_ops is None:
            hidden_states, probes = self._forward_native(input_ids, attention_mask, probe)
        else:
            hidden_states, probes = self._forward_fused(input_ids, attention_mask, probe)
        if probe:
            return hidden_states, probes
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 主干网络加上把隐藏状态映射为词表 Logits 的输出层。"""

    def __init__(self, config, fused_ops=None):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, fused_ops=fused_ops)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        # Qwen3-0.6B 的输入 Embedding 与输出 LM Head 使用同一份权重。
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, probe=False):
        if probe:
            hidden_states, probes = self.model(input_ids, attention_mask=attention_mask, probe=True)
            return self.lm_head(hidden_states), probes
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids,
        max_new_tokens=32,
        eos_token_id=None,
        probe=False,
    ):
        """无 KV Cache 的朴素 Greedy 解码（与 chapter_01 逐字相同）。"""

        generated = input_ids
        collected_probes: dict[str, dict[str, torch.Tensor]] = {}
        finished = False
        for step in range(max_new_tokens):
            logits = self(generated, probe=probe)
            if probe:
                logits, step_probes = logits
                collected_probes[f"step_{step}"] = step_probes
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token_id), dim=1)
            if eos_token_id is not None and bool((next_token_id == eos_token_id).all()):
                finished = True
                break
        if probe:
            return generated, collected_probes, finished
        return generated, finished


def resolve_model_directory(model_name_or_path):
    """返回本地模型目录。本期不做远端下载；下载由 download_model.py 负责。"""

    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())
    raise FileNotFoundError(
        "本地模型目录不存在："
        + str(local_path)
        + "。请先运行 python download_model.py 下载固定 revision。"
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


def load_handwritten_model(model_directory, device, dtype=None, fused_ops=None):
    """从 safetensors 权重构造手写 Qwen3 模型（fused_ops 透传，第 04 期新增）。

    safetensors 延迟 import：模型结构/管线测试（随机小配置）不需要加载权重，
    本地 venv 无 safetensors 也可运行（服务器侧加载权重时才 import）。
    """
    from safetensors.torch import load_file

    config = Qwen3Config.from_model_directory(model_directory)
    model = Qwen3ForCausalLM(config, fused_ops=fused_ops)

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
    model_dtype = dtype or model_dtype_from_config(config)

    # model.to(dtype=...) 会连 float buffer 一起转换。RoPE 的频率必须保留
    # float32 精度，因此先保存一份，再在模型移动完成后放回目标设备。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    return model
