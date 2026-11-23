"""结果序列化：写入章节 results_local/（Git 忽略），保留逐次原始数据。"""
from __future__ import annotations

import datetime
import json
import os

from . import env


def write_result(chapter_dir: str, experiment: str, data: dict, extra_meta: dict | None = None) -> str:
    out_dir = os.path.join(chapter_dir, "results_local")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = os.path.join(out_dir, f"{experiment}.{stamp}.local.json")
    doc = {
        "experiment": experiment,
        "meta": env.collect_env_metadata(extra=extra_meta),
        "data": data,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
    return path
