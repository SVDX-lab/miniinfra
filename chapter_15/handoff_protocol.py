"""请求级 KV Handoff 的长度前缀 TCP 协议与客户端。"""

import hashlib
import json
import socket
import struct


MAX_HEADER_BYTES = 4 * 1024 * 1024
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 * 1024


def payload_digest(payload):
    return hashlib.sha256(payload).hexdigest()


def namespace_digest(namespace):
    encoded = json.dumps(
        namespace, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recv_exact(sock, length):
    chunks = []
    received = 0
    while received < length:
        chunk = sock.recv(min(1024 * 1024, length - received))
        if not chunk:
            raise ConnectionError("连接在消息接收完成前关闭")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def send_message(sock, header, payload=b""):
    header = dict(header)
    header["payload_bytes"] = len(payload)
    encoded = json.dumps(
        header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("协议消息超过教学实现的大小限制")
    sock.sendall(struct.pack("!I", len(encoded)))
    sock.sendall(encoded)
    if payload:
        sock.sendall(payload)


def recv_message(sock):
    header_length = struct.unpack("!I", recv_exact(sock, 4))[0]
    if header_length < 2 or header_length > MAX_HEADER_BYTES:
        raise ValueError("非法 Header 长度")
    header = json.loads(recv_exact(sock, header_length).decode("utf-8"))
    payload_length = int(header.get("payload_bytes", 0))
    if payload_length < 0 or payload_length > MAX_PAYLOAD_BYTES:
        raise ValueError("非法 Payload 长度")
    return header, recv_exact(sock, payload_length) if payload_length else b""


class HandoffClient:
    def __init__(self, host="127.0.0.1", port=0, timeout=120.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def _request(self, header, payload=b""):
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        ) as sock:
            sock.settimeout(self.timeout)
            send_message(sock, header, payload)
            response, response_payload = recv_message(sock)
        if response.get("status") == "error":
            raise RuntimeError(response.get("error", "Handoff Service 错误"))
        return response, response_payload

    def ping(self):
        return self._request({"op": "ping"})[0]

    def publish(self, manifest, payload):
        return self._request(
            {"op": "publish", "manifest": manifest}, payload
        )[0]

    def status(self, request_id, attempt_id):
        return self._request({
            "op": "status", "request_id": request_id,
            "attempt_id": attempt_id,
        })[0]

    def receive(self, request_id, attempt_id):
        return self._request({
            "op": "receive", "request_id": request_id,
            "attempt_id": attempt_id,
        })

    def acknowledge(self, request_id, attempt_id, accepted, reason=None):
        return self._request({
            "op": "ack", "request_id": request_id,
            "attempt_id": attempt_id, "accepted": bool(accepted),
            "reason": reason,
        })[0]

    def abort(self, request_id, attempt_id, reason):
        return self._request({
            "op": "abort", "request_id": request_id,
            "attempt_id": attempt_id, "reason": reason,
        })[0]

    def release(self, request_id, attempt_id):
        return self._request({
            "op": "release", "request_id": request_id,
            "attempt_id": attempt_id,
        })[0]

    def stats(self):
        return self._request({"op": "stats"})[0]

    def shutdown(self):
        return self._request({"op": "shutdown"})[0]
