"""第 14 期外部 KV Cache 的进程间协议与 Token Chunk 标识。

协议刻意保持简单：TCP 只承载长度前缀 JSON 和可选原始字节。CacheKey 由模型
命名空间、父 Chunk 摘要和当前 Chunk Token IDs 共同决定，不能使用进程内 Block ID。
"""

import hashlib
import json
import socket
import struct
from dataclasses import dataclass


PROTOCOL_VERSION = 1
HEADER = struct.Struct("!Q")


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def recv_exact(connection, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("连接在消息接收完成前关闭")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(connection, metadata, payload=b""):
    metadata = dict(metadata)
    metadata["protocol_version"] = PROTOCOL_VERSION
    metadata["payload_bytes"] = len(payload)
    header = _canonical_json(metadata)
    connection.sendall(HEADER.pack(len(header)))
    connection.sendall(header)
    if payload:
        connection.sendall(payload)


def receive_message(connection, max_header_bytes=1024 * 1024):
    header_size = HEADER.unpack(recv_exact(connection, HEADER.size))[0]
    if header_size < 2 or header_size > max_header_bytes:
        raise ValueError("协议 Header 长度非法: %d" % header_size)
    metadata = json.loads(recv_exact(connection, header_size).decode("utf-8"))
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("外部 KV Cache 协议版本不匹配")
    payload_size = int(metadata.get("payload_bytes", 0))
    if payload_size < 0:
        raise ValueError("协议 Payload 长度不能为负数")
    return metadata, recv_exact(connection, payload_size) if payload_size else b""


@dataclass(frozen=True)
class ChunkIdentity:
    digest: str
    parent_digest: str
    token_start: int
    token_count: int
    token_ids: tuple


class TokenChunker:
    """生成与引擎物理 Block ID 无关的链式 CacheKey。"""

    def __init__(self, namespace, chunk_size):
        if chunk_size < 1:
            raise ValueError("external_chunk_size 必须大于 0")
        self.namespace = dict(namespace)
        self.chunk_size = int(chunk_size)
        namespace_with_protocol = dict(self.namespace)
        namespace_with_protocol["external_chunk_size"] = self.chunk_size
        namespace_with_protocol["format_version"] = 1
        self.namespace_bytes = _canonical_json(namespace_with_protocol)

    @property
    def namespace_digest(self):
        return hashlib.sha256(self.namespace_bytes).hexdigest()

    def identities(self, token_ids, leave_last_token=True):
        reusable = len(token_ids) - 1 if leave_last_token else len(token_ids)
        full_chunks = max(0, reusable // self.chunk_size)
        parent = ""
        result = []
        for index in range(full_chunks):
            start = index * self.chunk_size
            values = tuple(int(value) for value in token_ids[start:start + self.chunk_size])
            hasher = hashlib.sha256()
            hasher.update(self.namespace_bytes)
            hasher.update(bytes.fromhex(parent) if parent else b"")
            for value in values:
                hasher.update(value.to_bytes(8, "little", signed=True))
            digest = hasher.hexdigest()
            result.append(ChunkIdentity(
                digest=digest,
                parent_digest=parent,
                token_start=start,
                token_count=self.chunk_size,
                token_ids=values,
            ))
            parent = digest
        return result


class ExternalCacheClient:
    def __init__(self, host="127.0.0.1", port=65432, timeout=120.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def _request(self, metadata, payload=b""):
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        ) as connection:
            connection.settimeout(self.timeout)
            send_message(connection, metadata, payload)
            response, response_payload = receive_message(connection)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "外部 KV Cache 请求失败"))
        return response, response_payload

    def ping(self):
        response, _ = self._request({"op": "ping"})
        return response

    def lookup(self, identities):
        response, _ = self._request({
            "op": "lookup",
            "entries": [
                {
                    "digest": item.digest,
                    "parent_digest": item.parent_digest,
                    "token_count": item.token_count,
                    "token_ids": list(item.token_ids),
                }
                for item in identities
            ],
        })
        return int(response["hit_chunks"])

    def load(self, identities):
        response, payload = self._request({
            "op": "load",
            "entries": [
                {
                    "digest": item.digest,
                    "parent_digest": item.parent_digest,
                    "token_count": item.token_count,
                    "token_ids": list(item.token_ids),
                }
                for item in identities
            ],
        })
        sizes = [int(value) for value in response["entry_sizes"]]
        checksums = response["checksums"]
        if sum(sizes) != len(payload):
            raise RuntimeError("Load 返回的 Payload 长度与元数据不一致")
        chunks = []
        offset = 0
        for size, checksum in zip(sizes, checksums):
            value = payload[offset:offset + size]
            offset += size
            if hashlib.sha256(value).hexdigest() != checksum:
                raise RuntimeError("Load Payload 校验和不一致")
            chunks.append(value)
        return chunks

    def store(self, identity, payload, namespace_digest, dtype, shape, ancestors):
        response, _ = self._request({
            "op": "store",
            "digest": identity.digest,
            "parent_digest": identity.parent_digest,
            "token_start": identity.token_start,
            "token_count": identity.token_count,
            "token_ids": list(identity.token_ids),
            "namespace_digest": namespace_digest,
            "dtype": dtype,
            "shape": list(shape),
            "checksum": hashlib.sha256(payload).hexdigest(),
            "protected_ancestors": list(ancestors),
        }, payload)
        return response

    def stats(self):
        response, _ = self._request({"op": "stats"})
        return response["stats"]

    def clear(self):
        response, _ = self._request({"op": "clear"})
        return response

    def shutdown(self):
        response, _ = self._request({"op": "shutdown"})
        return response
