#!/usr/bin/env python3
"""下载固定 revision 的 Qwen/Qwen3-0.6B 模型文件并生成校验清单。

不依赖 huggingface_hub：直接经 HTTP 从官方端点或镜像（如 hf-mirror.com）
下载固定文件清单，解析 revision 的 commit sha 并写入 MANIFEST.json，
供 experiment-contract.yaml 与报告回填引用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
]


def http_get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mini-infra-kernel/01"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def resolve_revision(endpoint: str, repo: str, revision: str) -> str:
    """把分支名/引用解析为固定 commit sha；本身就是 sha 时原样返回。"""

    if len(revision) == 40 and all(character in "0123456789abcdef" for character in revision):
        return revision
    api_url = f"{endpoint}/api/models/{repo}/revision/{revision}"
    payload = json.loads(http_get(api_url).decode("utf-8"))
    sha = payload.get("sha")
    if not sha:
        raise RuntimeError("revision 解析失败: " + api_url)
    return sha


def download_file(endpoint: str, repo: str, revision: str, filename: Path, target: Path) -> None:
    url = f"{endpoint}/{repo}/resolve/{revision}/{filename}"
    if target.is_file() and target.stat().st_size > 0:
        print(f"[skip] {filename} 已存在（{target.stat().st_size} 字节）")
        return
    print(f"[down] {filename} <- {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "mini-infra-kernel/01"})
    with urllib.request.urlopen(request, timeout=600) as response, target.open("wb") as file:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            file.write(chunk)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", default="models/qwen3-0.6b")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="官方为 https://huggingface.co；国内常用镜像 https://hf-mirror.com",
    )
    arguments = parser.parse_args()

    target_directory = Path(arguments.out)
    target_directory.mkdir(parents=True, exist_ok=True)

    revision = arguments.revision
    try:
        revision = resolve_revision(arguments.endpoint, arguments.repo, arguments.revision)
    except Exception as error:  # noqa: BLE001 —— ModelScope 等端点没有 HF 风格 revision API
        print(
            f"[warn] revision API 解析失败（{error}），回退到 '{revision}' 分支；"
            "内容以 MANIFEST.json 的文件 SHA256 固定"
        )
    print(f"[revision] {arguments.repo}@{revision}")

    manifest = {
        "repo": arguments.repo,
        "revision": revision,
        "endpoint": arguments.endpoint,
        "files": {},
    }
    for filename in FILES:
        target = target_directory / filename
        download_file(arguments.endpoint, arguments.repo, revision, filename, target)
        manifest["files"][filename] = {
            "size": target.stat().st_size,
            "sha256": sha256_of_file(target),
        }

    manifest_path = target_directory / "MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"[manifest] {manifest_path}")
    print(f"[done] revision={revision}（请回填 experiment-contract.yaml 与 README）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
