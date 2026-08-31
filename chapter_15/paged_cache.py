"""第 15 期独立使用的 Paged KV Cache。

Prefill Worker 导出逻辑连续 KV，Decode Worker 在自己的池中重新申请 Physical
Block。跨进程协议永远不传递只在单个 Worker 内有效的 Block ID。
"""

import heapq
import math

import torch


DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
NAME_DTYPES = {value: key for key, value in DTYPE_NAMES.items()}


class PagedKVCache:
    def __init__(self, config, block_size, max_blocks, device, dtype):
        if block_size < 1 or max_blocks < 1:
            raise ValueError("block_size 和 max_blocks 必须大于 0")
        if dtype not in DTYPE_NAMES:
            raise ValueError("不支持的 KV dtype: %s" % dtype)
        self.config = config
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.device = torch.device(device)
        self.dtype = dtype
        self.blocks = torch.empty(
            (
                config.num_hidden_layers,
                2,
                max_blocks,
                config.num_key_value_heads,
                block_size,
                config.head_dim,
            ),
            dtype=dtype,
            device=self.device,
        )
        self.free_block_ids = list(range(max_blocks))
        self.block_tables = {}
        self.sequence_lengths = {}
        self.peak_used_blocks = 0

    @property
    def bytes_per_token(self):
        return (
            self.config.num_hidden_layers
            * 2
            * self.config.num_key_value_heads
            * self.config.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )

    @property
    def bytes_per_block(self):
        return self.bytes_per_token * self.block_size

    @property
    def used_blocks(self):
        return self.max_blocks - len(self.free_block_ids)

    def create_request(self, request_id):
        if request_id in self.block_tables:
            raise ValueError("请求已经存在: " + request_id)
        self.block_tables[request_id] = []
        self.sequence_lengths[request_id] = 0

    def _ensure_blocks(self, request_id, token_count):
        required = math.ceil(token_count / self.block_size)
        table = self.block_tables[request_id]
        while len(table) < required:
            if not self.free_block_ids:
                raise RuntimeError("Paged KV Cache 物理 Block 已耗尽")
            table.append(heapq.heappop(self.free_block_ids))
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)

    def append(self, request_id, current_cache):
        """追加只包含本轮新 Token 的逐层 K/V。"""
        if not current_cache:
            raise ValueError("current_cache 不能为空")
        count = current_cache[0][0].shape[2]
        if count < 1:
            raise ValueError("追加 Token 数必须大于 0")
        start = self.sequence_lengths[request_id]
        end = start + count
        self._ensure_blocks(request_id, end)
        table = self.block_tables[request_id]
        expected = (
            1,
            self.config.num_key_value_heads,
            count,
            self.config.head_dim,
        )
        for layer, (key, value) in enumerate(current_cache):
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError("追加的 KV Cache 形状错误")
            source = 0
            position = start
            while position < end:
                block_id = table[position // self.block_size]
                offset = position % self.block_size
                length = min(end - position, self.block_size - offset)
                self.blocks[layer, 0, block_id, :, offset:offset + length, :].copy_(
                    key[0, :, source:source + length, :]
                )
                self.blocks[layer, 1, block_id, :, offset:offset + length, :].copy_(
                    value[0, :, source:source + length, :]
                )
                source += length
                position += length
        self.sequence_lengths[request_id] = end

    def read_layer(self, layer, request_id, length=None):
        if length is None:
            length = self.sequence_lengths[request_id]
        if length == 0:
            return None
        block_count = math.ceil(length / self.block_size)
        ids = torch.as_tensor(
            self.block_tables[request_id][:block_count],
            dtype=torch.long,
            device=self.device,
        )
        key = self.blocks[layer, 0, ids].permute(1, 0, 2, 3).reshape(
            1,
            self.config.num_key_value_heads,
            block_count * self.block_size,
            self.config.head_dim,
        )[:, :, :length, :]
        value = self.blocks[layer, 1, ids].permute(1, 0, 2, 3).reshape(
            1,
            self.config.num_key_value_heads,
            block_count * self.block_size,
            self.config.head_dim,
        )[:, :, :length, :]
        return key, value

    def dense(self, request_id):
        length = self.sequence_lengths[request_id]
        if length == 0:
            return None
        return [
            self.read_layer(layer, request_id, length)
            for layer in range(self.config.num_hidden_layers)
        ]

    def external_shape(self, token_count):
        return (
            self.config.num_hidden_layers,
            2,
            self.config.num_key_value_heads,
            token_count,
            self.config.head_dim,
        )

    def export_chunk(self, request_id, start, token_count):
        end = start + token_count
        if start < 0 or end > self.sequence_lengths[request_id]:
            raise ValueError("导出的 Token 区间超出请求 Cache")
        output = torch.empty(
            self.external_shape(token_count), dtype=self.dtype, device="cpu"
        )
        for layer in range(self.config.num_hidden_layers):
            key, value = self.read_layer(layer, request_id, end)
            output[layer, 0].copy_(key[0, :, start:end, :])
            output[layer, 1].copy_(value[0, :, start:end, :])
        return output.contiguous().view(torch.uint8).numpy().tobytes()

    def import_chunk(self, request_id, payload, token_count):
        shape = self.external_shape(token_count)
        expected_bytes = math.prod(shape) * torch.empty(
            (), dtype=self.dtype
        ).element_size()
        if len(payload) != expected_bytes:
            raise ValueError(
                "外部 KV Payload 字节数错误: expected=%d actual=%d"
                % (expected_bytes, len(payload))
            )
        byte_tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        source = byte_tensor.view(self.dtype).reshape(shape)
        current_cache = [
            (
                source[layer, 0].unsqueeze(0).to(self.device),
                source[layer, 1].unsqueeze(0).to(self.device),
            )
            for layer in range(self.config.num_hidden_layers)
        ]
        self.append(request_id, current_cache)

    def release(self, request_id):
        table = self.block_tables.pop(request_id)
        self.sequence_lengths.pop(request_id)
        for block_id in table:
            heapq.heappush(self.free_block_ids, block_id)

    def snapshot(self):
        return {
            "block_size": self.block_size,
            "max_blocks": self.max_blocks,
            "used_blocks": self.used_blocks,
            "peak_used_blocks": self.peak_used_blocks,
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_block": self.bytes_per_block,
            "pool_bytes": self.max_blocks * self.bytes_per_block,
            "sequence_lengths": dict(self.sequence_lengths),
            "block_tables": {
                key: list(value) for key, value in self.block_tables.items()
            },
        }
