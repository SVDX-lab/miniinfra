"""第 12 期统一运行时包（runtime 包初始化：公开 Backend 协议与入口）。"""

from .backend import (  # noqa: F401
    Backend,
    BackendNotAvailableError,
    CapabilityError,
    CapabilityMissingError,
    CapabilityReport,
    DtypeCapability,
    ExplicitFallbackError,
    create_backend,
    list_backends,
)
from .qwen3_runtime import Qwen3Runtime, GenerateResult, load_truth  # noqa: F401
