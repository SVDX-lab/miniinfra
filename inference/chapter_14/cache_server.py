"""独立 CPU 外部 KV Cache Service。

服务进程只认识不可变 Chunk、字节容量和父子依赖，不导入模型或 CUDA。固定使用
leaf-LRU，淘汰策略不是本期实验变量。
"""

import argparse
import hashlib
import socketserver
import threading
from dataclasses import dataclass

from cache_protocol import receive_message, send_message


@dataclass
class CacheEntry:
    digest: str
    parent_digest: str
    token_start: int
    token_count: int
    token_ids: tuple
    namespace_digest: str
    dtype: str
    shape: tuple
    checksum: str
    payload: bytes
    last_access: int


class CacheStore:
    def __init__(self, capacity_bytes):
        if capacity_bytes < 1:
            raise ValueError("capacity_bytes 必须大于 0")
        self.capacity_bytes = int(capacity_bytes)
        self.entries = {}
        self.children = {}
        self.clock = 0
        self.used_bytes = 0
        self.lookup_count = 0
        self.hit_chunks = 0
        self.load_count = 0
        self.store_count = 0
        self.duplicate_store_count = 0
        self.eviction_count = 0
        self.rejected_store_count = 0
        self.lock = threading.Lock()

    def _touch(self, entry):
        self.clock += 1
        entry.last_access = self.clock

    def _remove(self, digest):
        entry = self.entries.pop(digest)
        self.used_bytes -= len(entry.payload)
        self.children.pop(digest, None)
        if entry.parent_digest:
            siblings = self.children.get(entry.parent_digest)
            if siblings is not None:
                siblings.discard(digest)
        self.eviction_count += 1

    def _make_room(self, required, protected):
        if required > self.capacity_bytes:
            return False
        while self.used_bytes + required > self.capacity_bytes:
            leaves = [
                entry for digest, entry in self.entries.items()
                if not self.children.get(digest) and digest not in protected
            ]
            if not leaves:
                return False
            self._remove(min(leaves, key=lambda entry: entry.last_access).digest)
        return True

    @staticmethod
    def _matches(entry, requested):
        return (
            entry.digest == requested["digest"]
            and entry.parent_digest == requested.get("parent_digest", "")
            and entry.token_count == int(requested["token_count"])
            and (
                "token_ids" not in requested
                or entry.token_ids == tuple(int(value) for value in requested["token_ids"])
            )
        )

    def lookup(self, requested):
        with self.lock:
            self.lookup_count += 1
            parent = ""
            count = 0
            for item in requested:
                entry = self.entries.get(item["digest"])
                if (
                    entry is None or not self._matches(entry, item)
                    or entry.parent_digest != parent
                ):
                    break
                self._touch(entry)
                parent = entry.digest
                count += 1
            self.hit_chunks += count
            return count

    def load(self, requested):
        with self.lock:
            entries = []
            parent = ""
            for item in requested:
                entry = self.entries.get(item["digest"])
                if (
                    entry is None or not self._matches(entry, item)
                    or entry.parent_digest != parent
                ):
                    raise KeyError("请求的 Chunk 链在 Load 前失效")
                if hashlib.sha256(entry.payload).hexdigest() != entry.checksum:
                    raise RuntimeError("服务端 Payload 校验和不一致")
                self._touch(entry)
                entries.append(entry)
                parent = entry.digest
            self.load_count += len(entries)
            return entries

    def store(self, request, payload):
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != request["checksum"]:
            raise ValueError("Store Payload 校验和不一致")
        digest = request["digest"]
        parent = request.get("parent_digest", "")
        with self.lock:
            existing = self.entries.get(digest)
            if existing is not None:
                if (
                    existing.checksum != checksum
                    or existing.parent_digest != parent
                    or existing.token_ids != tuple(request["token_ids"])
                    or existing.namespace_digest != request["namespace_digest"]
                    or existing.dtype != request["dtype"]
                    or existing.shape != tuple(request["shape"])
                ):
                    raise RuntimeError("相同 CacheKey 对应了不同内容")
                self._touch(existing)
                self.duplicate_store_count += 1
                return "exists"
            if parent and parent not in self.entries:
                raise ValueError("父 Chunk 尚未存在，拒绝发布不可达子 Chunk")
            protected = set(request.get("protected_ancestors", []))
            if parent:
                protected.add(parent)
            if not self._make_room(len(payload), protected):
                self.rejected_store_count += 1
                return "no_space"
            self.clock += 1
            entry = CacheEntry(
                digest=digest,
                parent_digest=parent,
                token_start=int(request["token_start"]),
                token_count=int(request["token_count"]),
                token_ids=tuple(int(value) for value in request["token_ids"]),
                namespace_digest=request["namespace_digest"],
                dtype=request["dtype"],
                shape=tuple(int(value) for value in request["shape"]),
                checksum=checksum,
                payload=payload,
                last_access=self.clock,
            )
            self.entries[digest] = entry
            self.children.setdefault(digest, set())
            if parent:
                self.children.setdefault(parent, set()).add(digest)
            self.used_bytes += len(payload)
            self.store_count += 1
            return "stored"

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.children.clear()
            self.used_bytes = 0

    def stats(self):
        with self.lock:
            return {
                "capacity_bytes": self.capacity_bytes,
                "used_bytes": self.used_bytes,
                "entry_count": len(self.entries),
                "lookup_count": self.lookup_count,
                "hit_chunks": self.hit_chunks,
                "load_count": self.load_count,
                "store_count": self.store_count,
                "duplicate_store_count": self.duplicate_store_count,
                "eviction_count": self.eviction_count,
                "rejected_store_count": self.rejected_store_count,
                "digests": list(self.entries),
            }


class CacheRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            request, payload = receive_message(self.request)
            operation = request.get("op")
            store = self.server.cache_store
            if operation == "ping":
                response, output = {"ok": True, "status": "ready"}, b""
            elif operation == "lookup":
                count = store.lookup(request.get("entries", []))
                response, output = {"ok": True, "hit_chunks": count}, b""
            elif operation == "load":
                entries = store.load(request.get("entries", []))
                response = {
                    "ok": True,
                    "entry_sizes": [len(entry.payload) for entry in entries],
                    "checksums": [entry.checksum for entry in entries],
                }
                output = b"".join(entry.payload for entry in entries)
            elif operation == "store":
                status = store.store(request, payload)
                response, output = {"ok": True, "status": status}, b""
            elif operation == "stats":
                response, output = {"ok": True, "stats": store.stats()}, b""
            elif operation == "clear":
                store.clear()
                response, output = {"ok": True}, b""
            elif operation == "shutdown":
                response, output = {"ok": True}, b""
                threading.Thread(
                    target=self.server.shutdown, daemon=True
                ).start()
            else:
                raise ValueError("未知操作: %r" % operation)
            send_message(self.request, response, output)
        except Exception as error:
            send_message(self.request, {
                "ok": False,
                "error": "%s: %s" % (type(error).__name__, error),
            })


class ThreadedCacheServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, capacity_bytes):
        super().__init__(address, CacheRequestHandler)
        self.cache_store = CacheStore(capacity_bytes)


def main():
    parser = argparse.ArgumentParser(description="第 14 期外部 KV Cache Service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--capacity-mib", type=float, default=1024.0)
    args = parser.parse_args()
    capacity = int(args.capacity_mib * 1024 * 1024)
    with ThreadedCacheServer((args.host, args.port), capacity) as server:
        print(
            "cache_server_ready host=%s port=%d capacity_bytes=%d"
            % (args.host, args.port, capacity),
            flush=True,
        )
        server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    main()
