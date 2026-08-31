"""第 09 期手写 Triton FlashAttention 前向 Kernel。

实现边界：CUDA、bfloat16、forward-only、head_dim=128。输入已经按 Query Head
展开 GQA 的 K/V，因此 Eager 和 Flash 路径接收完全相同的稠密 Q/K/V。
"""

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU Smoke Test 不要求安装 Triton。
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _flash_attention_forward_kernel(
        query,
        key,
        value,
        key_valid,
        query_valid,
        output,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_kmb,
        stride_kmn,
        stride_qmb,
        stride_qmm,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        query_length,
        key_length,
        head_count: tl.constexpr,
        head_dim: tl.constexpr,
        causal_offset,
        softmax_scale_log2,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // head_count
        head = batch_head % head_count

        query_offsets = query_block * block_m + tl.arange(0, block_m)
        key_offsets_base = tl.arange(0, block_n)
        dim_offsets = tl.arange(0, head_dim)
        query_in_bounds = query_offsets < query_length
        query_is_valid = tl.load(
            query_valid + batch * stride_qmb + query_offsets * stride_qmm,
            mask=query_in_bounds,
            other=0,
        ).to(tl.int1)

        query_pointers = (
            query
            + batch * stride_qb
            + head * stride_qh
            + query_offsets[:, None] * stride_qm
            + dim_offsets[None, :] * stride_qd
        )
        query_tile = tl.load(
            query_pointers,
            mask=query_in_bounds[:, None],
            other=0.0,
        )

        row_max = tl.where(query_is_valid, -float("inf"), 0.0)
        row_sum = tl.zeros((block_m,), dtype=tl.float32)
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)

        for key_start in range(0, key_length, block_n):
            key_offsets = key_start + key_offsets_base
            key_in_bounds = key_offsets < key_length
            key_is_valid = tl.load(
                key_valid + batch * stride_kmb + key_offsets * stride_kmn,
                mask=key_in_bounds,
                other=0,
            ).to(tl.int1)

            key_pointers = (
                key
                + batch * stride_kb
                + head * stride_kh
                + key_offsets[None, :] * stride_kn
                + dim_offsets[:, None] * stride_kd
            )
            key_tile = tl.load(
                key_pointers,
                mask=key_in_bounds[None, :],
                other=0.0,
            )
            scores = tl.dot(query_tile, key_tile) * softmax_scale_log2
            allowed = (
                query_is_valid[:, None]
                & key_is_valid[None, :]
                & (key_offsets[None, :] <= causal_offset + query_offsets[:, None])
            )
            scores = tl.where(allowed, scores, -float("inf"))

            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, tile_max)
            correction = tl.exp2(row_max - new_max)
            probabilities = tl.exp2(scores - new_max[:, None])
            probabilities = tl.where(allowed, probabilities, 0.0)
            new_sum = row_sum * correction + tl.sum(probabilities, axis=1)

            value_pointers = (
                value
                + batch * stride_vb
                + head * stride_vh
                + key_offsets[:, None] * stride_vn
                + dim_offsets[None, :] * stride_vd
            )
            value_tile = tl.load(
                value_pointers,
                mask=key_in_bounds[:, None],
                other=0.0,
            )
            accumulator = accumulator * correction[:, None]
            accumulator += tl.dot(probabilities.to(tl.bfloat16), value_tile)
            row_max = new_max
            row_sum = new_sum

        normalized = accumulator / row_sum[:, None]
        normalized = tl.where(query_is_valid[:, None], normalized, 0.0)
        output_pointers = (
            output
            + batch * stride_ob
            + head * stride_oh
            + query_offsets[:, None] * stride_om
            + dim_offsets[None, :] * stride_od
        )
        tl.store(
            output_pointers,
            normalized,
            mask=query_in_bounds[:, None],
        )


def _validate_inputs(query, key, value, key_valid, query_valid, causal_offset):
    if triton is None:
        raise RuntimeError("FlashAttention 路径需要安装 Triton")
    if not query.is_cuda:
        raise RuntimeError("手写 FlashAttention 只支持 CUDA Tensor")
    if query.dtype != torch.bfloat16:
        raise ValueError("手写 FlashAttention 当前只支持 bfloat16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Q/K/V dtype 必须一致")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q/K/V 必须是 [batch, heads, sequence, head_dim]")
    if key.shape != value.shape:
        raise ValueError("K/V 形状必须一致")
    if query.shape[:2] != key.shape[:2] or query.shape[3] != key.shape[3]:
        raise ValueError("Q/K/V 的 batch、head 和 head_dim 必须一致")
    if query.shape[-1] != 128:
        raise ValueError("手写 FlashAttention 当前只支持 head_dim=128")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("Q/K/V 最后一维必须连续")
    expected_key_mask = (query.shape[0], key.shape[2])
    expected_query_mask = (query.shape[0], query.shape[2])
    if tuple(key_valid.shape) != expected_key_mask:
        raise ValueError("key_valid 形状应为 %s" % (expected_key_mask,))
    if tuple(query_valid.shape) != expected_query_mask:
        raise ValueError("query_valid 形状应为 %s" % (expected_query_mask,))
    if key_valid.device != query.device or query_valid.device != query.device:
        raise ValueError("Mask 与 Q/K/V 必须位于同一设备")
    if causal_offset < 0:
        raise ValueError("causal_offset 不能小于 0")


def flash_attention_forward(
    query,
    key,
    value,
    key_valid,
    query_valid,
    causal_offset,
    softmax_scale=None,
):
    """执行手写 FlashAttention，并返回与 query 同形状的结果。"""

    _validate_inputs(
        query, key, value, key_valid, query_valid, int(causal_offset)
    )
    batch_size, head_count, query_length, head_dim = query.shape
    key_length = key.shape[2]
    output = torch.empty_like(query)
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    # exp2 比 exp 更适合 Triton；把自然指数的 scale 转换到以 2 为底。
    scale_log2 = float(scale) * math.log2(math.e)
    block_m = 16
    block_n = 32
    grid = (triton.cdiv(query_length, block_m), batch_size * head_count)
    _flash_attention_forward_kernel[grid](
        query,
        key,
        value,
        key_valid,
        query_valid,
        output,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *key_valid.stride(),
        *query_valid.stride(),
        *output.stride(),
        query_length,
        key_length,
        head_count=head_count,
        head_dim=head_dim,
        causal_offset=int(causal_offset),
        softmax_scale_log2=scale_log2,
        block_m=block_m,
        block_n=block_n,
        num_warps=4,
        num_stages=2,
    )
    return output

