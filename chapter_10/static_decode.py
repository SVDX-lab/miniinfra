"""第 10 期：可由 CUDA Graph 捕获的静态 Paged Decode 执行器。

核心对照 ``static_eager`` 与 ``cuda_graph`` 调用完全相同的 ``_decode_step``。
两者使用相同的静态 Batch、Context Bucket、Block Table、KV Pool 和 Attention；
唯一差别是前者逐个提交 CUDA Kernel，后者重放已经捕获的 CUDA Graph。
"""

import math
import time

import torch

from qwen3_model import (
    apply_rotary_embedding,
    dense_attention_forward,
    repeat_key_value,
)


class StaticPagedKVCache:
    """为静态 Decode Bucket 预留物理块和固定宽度 Block Table。

    教学实现仍通过 Block Table 间接寻址物理 KV Block，但为了保证 Graph 捕获期间
    Tensor 地址和形状稳定，会提前为每个 Slot 预留整个 Context Bucket。它不是
    生产级动态 Block 分配器，这一预留代价会在实验中单独报告。
    """

    def __init__(self, config, capacity, context_bucket, block_size, device, dtype):
        if capacity < 1:
            raise ValueError("capacity 必须大于 0")
        if context_bucket < 1:
            raise ValueError("context_bucket 必须大于 0")
        if block_size < 1:
            raise ValueError("block_size 必须大于 0")
        self.config = config
        self.capacity = int(capacity)
        self.context_bucket = int(context_bucket)
        self.block_size = int(block_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.blocks_per_slot = math.ceil(context_bucket / block_size)
        self.data_block_count = self.capacity * self.blocks_per_slot
        self.sink_block_count = self.capacity
        self.total_block_count = self.data_block_count + self.sink_block_count

        self.blocks = torch.zeros(
            (
                config.num_hidden_layers,
                2,
                self.total_block_count,
                config.num_key_value_heads,
                block_size,
                config.head_dim,
            ),
            dtype=dtype,
            device=self.device,
        )
        self.block_table = torch.arange(
            self.data_block_count, dtype=torch.long, device=self.device
        ).reshape(self.capacity, self.blocks_per_slot)
        self.sink_block_ids = torch.arange(
            self.data_block_count,
            self.total_block_count,
            dtype=torch.long,
            device=self.device,
        )

    @property
    def bytes_per_block(self):
        values = (
            self.config.num_hidden_layers
            * 2
            * self.config.num_key_value_heads
            * self.block_size
            * self.config.head_dim
        )
        return values * torch.empty((), dtype=self.dtype).element_size()

    @property
    def pool_bytes(self):
        return self.total_block_count * self.bytes_per_block

    @property
    def reserved_token_slots(self):
        return self.data_block_count * self.block_size

    def store_prompt(self, slot, prompt_cache):
        if not 0 <= slot < self.capacity:
            raise IndexError("slot 超出范围")
        if len(prompt_cache) != self.config.num_hidden_layers:
            raise ValueError("Prompt KV 层数与模型配置不一致")
        prompt_length = prompt_cache[0][0].shape[2]
        if prompt_length > self.context_bucket:
            raise ValueError("Prompt 长度超过 Context Bucket")
        table = self.block_table[slot]
        for layer_index, (key, value) in enumerate(prompt_cache):
            expected = (
                1,
                self.config.num_key_value_heads,
                prompt_length,
                self.config.head_dim,
            )
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError("Prompt KV 形状错误")
            for logical_block in range(math.ceil(prompt_length / self.block_size)):
                start = logical_block * self.block_size
                end = min(prompt_length, start + self.block_size)
                count = end - start
                block_id = int(table[logical_block].item())
                self.blocks[
                    layer_index, 0, block_id, :, :count, :
                ].copy_(key[0, :, start:end, :])
                self.blocks[
                    layer_index, 1, block_id, :, :count, :
                ].copy_(value[0, :, start:end, :])

    def dense_slot_cache(self, slot, logical_length):
        block_count = math.ceil(logical_length / self.block_size)
        block_ids = self.block_table[slot:slot + 1, :block_count]
        result = []
        for layer_index in range(self.config.num_hidden_layers):
            key = self.blocks[layer_index, 0, block_ids]
            value = self.blocks[layer_index, 1, block_ids]
            key = key.permute(0, 2, 1, 3, 4).reshape(
                1,
                self.config.num_key_value_heads,
                block_count * self.block_size,
                self.config.head_dim,
            )[:, :, :logical_length, :]
            value = value.permute(0, 2, 1, 3, 4).reshape(
                1,
                self.config.num_key_value_heads,
                block_count * self.block_size,
                self.config.head_dim,
            )[:, :, :logical_length, :]
            result.append((key, value))
        return result


class StaticDecodeRunner:
    """固定 Batch/Context Bucket 的 Eager 与 CUDA Graph 共用执行器。"""

    MODES = ("static_eager", "cuda_graph")

    def __init__(
        self,
        model,
        capacity,
        context_bucket,
        block_size=16,
        pad_token_id=0,
    ):
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.capacity = int(capacity)
        self.context_bucket = int(context_bucket)
        self.block_size = int(block_size)
        self.pad_token_id = int(pad_token_id)
        if not 0 <= self.pad_token_id < self.config.vocab_size:
            raise ValueError("pad_token_id 超出词表范围")

        self.cache = StaticPagedKVCache(
            self.config,
            self.capacity,
            self.context_bucket,
            self.block_size,
            self.device,
            self.dtype,
        )
        self.input_ids = torch.full(
            (self.capacity, 1),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self.position_ids = torch.zeros(
            (self.capacity, 1), dtype=torch.long, device=self.device
        )
        self.sequence_lengths = torch.zeros(
            self.capacity, dtype=torch.long, device=self.device
        )
        self.active_mask = torch.zeros(
            self.capacity, dtype=torch.bool, device=self.device
        )
        self.slot_indices = torch.arange(
            self.capacity, dtype=torch.long, device=self.device
        )
        self.key_positions = torch.arange(
            self.context_bucket, dtype=torch.long, device=self.device
        )
        self.output_tokens = torch.full(
            (self.capacity,),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self.last_logits = torch.zeros(
            (self.capacity, self.config.vocab_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.graph = None
        self.capture_ms = 0.0
        self.capture_allocated_bytes = 0
        self.capture_reserved_bytes = 0
        self.active_count = 0
        self.prompt_first_tokens = []
        self.remaining_decode_steps = 0

    @property
    def static_buffer_bytes(self):
        tensors = (
            self.input_ids,
            self.position_ids,
            self.sequence_lengths,
            self.active_mask,
            self.slot_indices,
            self.key_positions,
            self.output_tokens,
            self.last_logits,
            self.cache.block_table,
            self.cache.sink_block_ids,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _write_layer_token(self, layer_index, key, value):
        logical_blocks = torch.div(
            self.sequence_lengths, self.block_size, rounding_mode="floor"
        )
        offsets = torch.remainder(self.sequence_lengths, self.block_size)
        physical_blocks = self.cache.block_table[
            self.slot_indices, logical_blocks
        ]
        write_blocks = torch.where(
            self.active_mask, physical_blocks, self.cache.sink_block_ids
        )
        write_offsets = torch.where(
            self.active_mask, offsets, torch.zeros_like(offsets)
        )
        self.cache.blocks[
            layer_index, 0, write_blocks, :, write_offsets, :
        ] = key[:, :, 0, :]
        self.cache.blocks[
            layer_index, 1, write_blocks, :, write_offsets, :
        ] = value[:, :, 0, :]

    def _read_layer_bucket(self, layer_index):
        block_ids = self.cache.block_table
        key = self.cache.blocks[layer_index, 0, block_ids]
        value = self.cache.blocks[layer_index, 1, block_ids]
        key = key.permute(0, 2, 1, 3, 4).reshape(
            self.capacity,
            self.config.num_key_value_heads,
            self.cache.blocks_per_slot * self.block_size,
            self.config.head_dim,
        )[:, :, :self.context_bucket, :]
        value = value.permute(0, 2, 1, 3, 4).reshape(
            self.capacity,
            self.config.num_key_value_heads,
            self.cache.blocks_per_slot * self.block_size,
            self.config.head_dim,
        )[:, :, :self.context_bucket, :]
        return key, value

    def _decode_step(self):
        hidden_states = self.model.model.embed_tokens(self.input_ids)
        cosine, sine = self.model.model.rotary_emb(
            self.position_ids, hidden_states.dtype
        )
        key_valid = self.key_positions.unsqueeze(0) < (
            self.sequence_lengths + 1
        ).unsqueeze(1)
        key_valid = key_valid & self.active_mask.unsqueeze(1)
        query_valid = self.active_mask.unsqueeze(1)

        for layer_index, layer in enumerate(self.model.model.layers):
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            attention = layer.self_attn
            query = attention.q_norm(
                attention.q_proj(normalized).view(
                    self.capacity,
                    1,
                    attention.num_attention_heads,
                    attention.head_dim,
                )
            ).transpose(1, 2)
            key = attention.k_norm(
                attention.k_proj(normalized).view(
                    self.capacity,
                    1,
                    attention.num_key_value_heads,
                    attention.head_dim,
                )
            ).transpose(1, 2)
            value = attention.v_proj(normalized).view(
                self.capacity,
                1,
                attention.num_key_value_heads,
                attention.head_dim,
            ).transpose(1, 2)
            query, key = apply_rotary_embedding(query, key, cosine, sine)
            self._write_layer_token(layer_index, key, value)

            bucket_key, bucket_value = self._read_layer_bucket(layer_index)
            bucket_key = repeat_key_value(
                bucket_key, attention.num_key_value_groups
            )
            bucket_value = repeat_key_value(
                bucket_value, attention.num_key_value_groups
            )
            attention_output = dense_attention_forward(
                attention.attention_backend,
                query,
                bucket_key,
                bucket_value,
                key_valid,
                query_valid,
                self.context_bucket - 1,
            )
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                self.capacity, 1, -1
            )
            hidden_states = residual + attention.o_proj(attention_output)
            residual = hidden_states
            hidden_states = residual + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1)
        next_tokens = torch.where(
            self.active_mask,
            next_tokens,
            torch.full_like(next_tokens, self.pad_token_id),
        )
        self.last_logits.copy_(logits)
        self.output_tokens.copy_(next_tokens)
        self.input_ids[:, 0].copy_(next_tokens)
        self.sequence_lengths.add_(self.active_mask.to(torch.long))
        self.position_ids[:, 0].copy_(
            torch.where(
                self.active_mask,
                self.sequence_lengths,
                torch.zeros_like(self.sequence_lengths),
            )
        )

    @torch.inference_mode()
    def prepare_prompts(self, prompt_token_ids):
        if len(prompt_token_ids) < 1:
            raise ValueError("至少需要一条 Prompt")
        if len(prompt_token_ids) > self.capacity:
            raise ValueError("Prompt 数量超过 Batch Capacity")
        self.input_ids.fill_(self.pad_token_id)
        self.position_ids.zero_()
        self.sequence_lengths.zero_()
        self.active_mask.zero_()
        first_tokens = []
        for slot, raw_tokens in enumerate(prompt_token_ids):
            tokens = [int(token) for token in raw_tokens]
            if not tokens:
                raise ValueError("Prompt 不能为空")
            if len(tokens) >= self.context_bucket:
                raise ValueError("Prompt 必须为后续 Decode 留出至少一个 Token")
            input_ids = torch.tensor(
                tokens, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            positions = torch.arange(
                len(tokens), dtype=torch.long, device=self.device
            ).unsqueeze(0)
            mask = torch.ones(
                (1, len(tokens)), dtype=torch.bool, device=self.device
            )
            logits, prompt_cache = self.model(
                input_ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
            )
            self.cache.store_prompt(slot, prompt_cache)
            token = torch.argmax(logits[0, -1, :], dim=-1)
            self.input_ids[slot, 0].copy_(token)
            self.position_ids[slot, 0] = len(tokens)
            self.sequence_lengths[slot] = len(tokens)
            self.active_mask[slot] = True
            first_tokens.append(int(token.item()))
            del input_ids, positions, mask, logits, prompt_cache
        self.active_count = len(prompt_token_ids)
        self.prompt_first_tokens = first_tokens
        self.remaining_decode_steps = min(
            self.context_bucket - len(tokens) for tokens in prompt_token_ids
        )
        return list(first_tokens)

    @torch.inference_mode()
    def capture(self, warmup_steps=3):
        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graph 只能在 CUDA 设备上捕获")
        if warmup_steps < 1:
            raise ValueError("warmup_steps 必须大于 0")

        self.input_ids.fill_(self.pad_token_id)
        self.position_ids.zero_()
        self.sequence_lengths.zero_()
        self.active_mask.zero_()
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup_steps):
                self._decode_step()
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        allocated_before = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        start = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._decode_step()
        torch.cuda.synchronize(self.device)
        self.capture_ms = (time.perf_counter() - start) * 1000
        self.capture_allocated_bytes = max(
            0, torch.cuda.memory_allocated(self.device) - allocated_before
        )
        self.capture_reserved_bytes = max(
            0, torch.cuda.memory_reserved(self.device) - reserved_before
        )
        return {
            "capture_ms": self.capture_ms,
            "capture_allocated_bytes": self.capture_allocated_bytes,
            "capture_reserved_bytes": self.capture_reserved_bytes,
        }

    @torch.inference_mode()
    def step(self, mode):
        if mode not in self.MODES:
            raise ValueError("mode 必须是 static_eager 或 cuda_graph")
        if self.remaining_decode_steps < 1:
            raise RuntimeError("Decode 将超过 Context Bucket")
        if mode == "static_eager":
            self._decode_step()
        else:
            if self.graph is None:
                raise RuntimeError("cuda_graph 模式必须先调用 capture()")
            self.graph.replay()
        self.remaining_decode_steps -= 1
        return self.output_tokens

    def snapshot(self):
        return {
            "capacity": self.capacity,
            "active_count": self.active_count,
            "context_bucket": self.context_bucket,
            "block_size": self.block_size,
            "blocks_per_slot": self.cache.blocks_per_slot,
            "pool_blocks": self.cache.total_block_count,
            "pool_bytes": self.cache.pool_bytes,
            "reserved_token_slots": self.cache.reserved_token_slots,
            "static_buffer_bytes": self.static_buffer_bytes,
            "capture_ms": self.capture_ms,
            "capture_allocated_bytes": self.capture_allocated_bytes,
            "capture_reserved_bytes": self.capture_reserved_bytes,
        }
