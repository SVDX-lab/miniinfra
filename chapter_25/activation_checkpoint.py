"""Chapter 25 core: activation checkpointing with a hand-written autograd hook.

The mechanism: instead of keeping every intermediate activation alive until
Backward, a checkpointed segment runs its Forward under ``no_grad``, keeps
only the segment input (plus the RNG state), and in Backward re-runs the
segment with gradients enabled to rebuild a local graph, then propagates the
incoming output gradients through it.  Parameter gradients accumulate during
that recompute exactly as they would in a normal graph; the segment input
gradient is returned to autograd.

``RecomputeCheckpoint`` is deliberately small (~50 lines) so every step of
that trade is readable.  RNG preservation is implemented and *switchable*:
the smoke test proves that with dropout>0 a checkpoint that forgets the RNG
state produces different gradients than the un-checkpointed reference, which
is why the machinery must not rely on Qwen3's dropout being zero.

``CheckpointedQwen3Model`` regroups the 28 decoder layers into segments of
``layers_per_segment`` and checkpoints each segment.  ``layers_per_segment=1``
is the finest granularity; ``=28`` recompute the whole stack from one saved
activation.
"""

from __future__ import annotations

import torch
from torch import nn

from qwen3_model import Qwen3Config, Qwen3Model


def _rng_states(device: torch.device):
    if device.type == "cuda":
        return ("cuda", torch.cuda.get_rng_state(device), None)
    return ("cpu", torch.get_rng_state(), None)


def _set_rng_states(state, device: torch.device) -> None:
    kind, payload, _extra = state
    if kind == "cuda":
        torch.cuda.set_rng_state(payload, device)
    else:
        torch.set_rng_state(payload)


class RecomputeCheckpoint(torch.autograd.Function):
    """Save the input, recompute the segment in Backward.

    forward: run ``segment`` under no_grad, store its input and the RNG state
    captured *before* the run, and return the outputs detached from any graph.

    backward: restore the captured RNG state, rebuild the graph from detached
    input copies, back-propagate the incoming gradients through it (module
    parameter gradients accumulate as a side effect), restore the RNG state
    the backward started with, and hand the input gradients back to autograd.
    """

    @staticmethod
    def forward(ctx, segment, preserve_rng, *inputs):
        if not any(torch.is_tensor(t) and t.requires_grad for t in inputs):
            raise RuntimeError(
                "checkpointed segment received no input with requires_grad=True; "
                "backward would have nothing to propagate into"
            )
        device = next(t.device for t in inputs if torch.is_tensor(t))
        # segment 与 preserve_rng 存在 ctx 上而不是类属性上：一条前向里有多段
        # checkpoint 时，Backward 必须各自拿到自己的重算函数。
        ctx.fn = segment
        ctx.preserve_rng = preserve_rng
        ctx.device = device
        if preserve_rng:
            ctx.rng_state = _rng_states(device)
        tensor_inputs = [t for t in inputs if torch.is_tensor(t)]
        ctx.save_for_backward(*tensor_inputs)
        with torch.no_grad():
            outputs = segment(*inputs)
        if not isinstance(outputs, torch.Tensor):
            raise ValueError("checkpointed segment must return a single Tensor")
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        (hidden_states, *rest) = ctx.saved_tensors
        state_at_backward = _rng_states(ctx.device) if ctx.preserve_rng else None
        if ctx.preserve_rng:
            _set_rng_states(ctx.rng_state, ctx.device)
        try:
            detached = hidden_states.detach()
            detached.requires_grad_(hidden_states.requires_grad)
            # 自定义 Function 的 backward 运行在 grad 关闭的上下文里；
            # 重算必须显式重新打开，否则建不出局部计算图。
            with torch.enable_grad():
                recomputed = ctx.fn(detached, *rest)
                torch.autograd.backward(recomputed, grad_output)
            input_grad = detached.grad
        finally:
            if ctx.preserve_rng and state_at_backward is not None:
                _set_rng_states(state_at_backward, ctx.device)
        # forward 的入参是 (segment, preserve_rng, *tensor_inputs)，backward
        # 必须逐一返回同样数量的梯度；只有段输入 hidden 需要梯度。
        extras = [None] * (len(ctx.saved_tensors) - 1)
        return (None, None, input_grad, *extras)


def checkpoint_segment(segment, preserve_rng: bool, *inputs):
    """Run one segment under the checkpoint; same signature in and out."""

    return RecomputeCheckpoint.apply(segment, preserve_rng, *inputs)


class CheckpointedQwen3Model(Qwen3Model):
    """Qwen3Model whose decoder stack runs in ``layers_per_segment`` chunks."""

    def __init__(self, config, layers_per_segment: int = 1, preserve_rng: bool = True):
        super().__init__(config)
        if layers_per_segment < 1 or config.num_hidden_layers % layers_per_segment:
            raise ValueError(
                "layers_per_segment must divide the layer count"
            )
        self.layers_per_segment = layers_per_segment
        self.preserve_rng = preserve_rng

    def forward(self, input_ids, attention_mask=None, position_ids=None, attention_mask_4d=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 [batch, sequence] 的二维 Tensor")
        hidden_states = self.embed_tokens(input_ids)
        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
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

        layers = list(self.layers)
        step = self.layers_per_segment
        for start in range(0, len(layers), step):
            segment_layers = layers[start : start + step]

            def run_segment(current_hidden, _layers=segment_layers, _cos=cosine, _sin=sine, _mask=causal_mask):
                for layer in _layers:
                    current_hidden = layer(current_hidden, _cos, _sin, _mask)
                return current_hidden

            hidden_states = checkpoint_segment(run_segment, self.preserve_rng, hidden_states)
        return self.norm(hidden_states)


class CheckpointedQwen3ForCausalLM(nn.Module):
    """ForCausalLM wrapper whose backbone checkpoints segments of layers.

    Parameter names match ``Qwen3ForCausalLM`` exactly, so the published
    safetensors load with the same ``strict=False`` rule as the plain model
    (``lm_head.weight`` is tied and absent from the file).
    """

    def __init__(self, config: Qwen3Config, layers_per_segment: int = 1, preserve_rng: bool = True):
        super().__init__()
        self.config = config
        self.model = CheckpointedQwen3Model(
            config, layers_per_segment=layers_per_segment, preserve_rng=preserve_rng
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None, attention_mask_4d=None):
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            attention_mask_4d=attention_mask_4d,
        )
        return self.lm_head(hidden_states)


def load_checkpointed_model(
    directory,
    device,
    layers_per_segment: int,
    preserve_rng: bool = True,
    dtype=torch.float32,
) -> CheckpointedQwen3ForCausalLM:
    """Load the published weights into the checkpointed model."""

    from pathlib import Path

    from safetensors.torch import load_file

    config = Qwen3Config.from_model_directory(directory)
    model = CheckpointedQwen3ForCausalLM(config, layers_per_segment, preserve_rng)
    weight_path = Path(directory) / "model.safetensors"
    state_dict = load_file(str(weight_path), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)
    allowed_missing_keys = {"lm_head.weight"}
    unexpected_missing = set(load_result.missing_keys) - allowed_missing_keys
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "权重名称不匹配，missing=" + str(sorted(unexpected_missing))
            + ", unexpected=" + str(sorted(load_result.unexpected_keys))
        )
    model.tie_weights()
    # 与 load_handwritten_model 相同的迁移规则：RoPE 频率保持 float32。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    return model
