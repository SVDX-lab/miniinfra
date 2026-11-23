"""safetensors 兼容 shim 包：仅当环境缺少官方 safetensors 时由
ep27_common.bootstrap() 插入 sys.path。子模块 torch 提供 load_file。"""

from . import torch as torch  # noqa: F401
