"""第 08 期 ctypes 绑定层：加载 libtransfer_ops_{cuda,musa}.so，直传指针。

设计（复用 chapter_16 ctypes 直传模式，M-17）：
- Python 经 ctypes 把 Host/设备指针直接传进库；分配、传输、流、事件、
  RMSNorm 启动全部经本类单一出口；
- 库内返回错误码（0=成功），本层 assert 后抛 RuntimeError（带 t_last_error），
  绝不静默吞错；
- T9_HAVE_* 是库的编译期开关（MUSA 版由实机符号探测决定），本层用
  hasattr 探测对应导出函数是否存在 —— 平台能力面在 Python 侧可见。
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

KIND_H2D = 0
KIND_D2H = 1
KIND_D2D = 2

_CVP = ctypes.c_void_p
_SZ = ctypes.c_size_t
_IP = ctypes.c_int
_FP = ctypes.c_float
_PP = ctypes.POINTER


class TransferOpsError(RuntimeError):
    pass


class TransferOps:
    """extern "C" 传输/流/算子库的 ctypes 封装。"""

    def __init__(self, lib_path: str | Path):
        self.lib_path = str(lib_path)
        self.lib = ctypes.CDLL(self.lib_path)
        lib = self.lib
        # --- 签名注册 -----------------------------------------------------
        lib.t_init.argtypes = []
        lib.t_init.restype = _IP
        lib.t_dev_info.argtypes = [ctypes.c_char_p, _IP]
        lib.t_dev_info.restype = _IP
        lib.t_device_count.argtypes = []
        lib.t_device_count.restype = _IP
        lib.t_malloc_device.argtypes = [_SZ]
        lib.t_malloc_device.restype = _CVP
        lib.t_free_device.argtypes = [_CVP]
        lib.t_free_device.restype = _IP
        lib.t_malloc_host_pageable.argtypes = [_SZ]
        lib.t_malloc_host_pageable.restype = _CVP
        lib.t_free_host_pageable.argtypes = [_CVP]
        lib.t_free_host_pageable.restype = None
        lib.t_malloc_host_pinned_mallochost.argtypes = [_SZ]
        lib.t_malloc_host_pinned_mallochost.restype = _CVP
        lib.t_malloc_host_pinned_hostalloc.argtypes = [_SZ, ctypes.c_uint]
        lib.t_malloc_host_pinned_hostalloc.restype = _CVP
        lib.t_malloc_host_pinned_default.argtypes = [_SZ]
        lib.t_malloc_host_pinned_default.restype = _CVP
        # T9_HAVE_* 守卫函数（MUSA 侧可能不存在——用 hasattr 探测后再注册）：
        # 全部返回指针，restype 必须显式 c_void_p（默认 c_int 会截断 64 位指针
        # —— 4090/S4000 两端实测抓出：截断句柄导致 query 段错误 / destroy
        # "invalid device context" / memcpy 驱动层挂死，见 KNOWN_ISSUES KI-09-05）
        if hasattr(lib, "t_malloc_host_pinned_portable"):
            lib.t_malloc_host_pinned_portable.argtypes = [_SZ]
            lib.t_malloc_host_pinned_portable.restype = _CVP
        if hasattr(lib, "t_malloc_host_pinned_mapped"):
            lib.t_malloc_host_pinned_mapped.argtypes = [_SZ]
            lib.t_malloc_host_pinned_mapped.restype = _CVP
        if hasattr(lib, "t_host_get_device_pointer"):
            lib.t_host_get_device_pointer.argtypes = [_CVP]
            lib.t_host_get_device_pointer.restype = _CVP
        if hasattr(lib, "t_stream_create_nonblocking"):
            lib.t_stream_create_nonblocking.argtypes = []
            lib.t_stream_create_nonblocking.restype = _CVP
        if hasattr(lib, "t_host_register"):
            lib.t_host_register.argtypes = [_CVP, _SZ]
            lib.t_host_register.restype = _IP
        if hasattr(lib, "t_host_unregister"):
            lib.t_host_unregister.argtypes = [_CVP]
            lib.t_host_unregister.restype = _IP
        lib.t_free_host_pinned.argtypes = [_CVP]
        lib.t_free_host_pinned.restype = _IP
        lib.t_memcpy_sync.argtypes = [_CVP, _CVP, _SZ, _IP]
        lib.t_memcpy_sync.restype = _IP
        lib.t_memcpy_async.argtypes = [_CVP, _CVP, _SZ, _IP, _CVP]
        lib.t_memcpy_async.restype = _IP
        lib.t_stream_create.argtypes = []
        lib.t_stream_create.restype = _CVP
        lib.t_stream_destroy.argtypes = [_CVP]
        lib.t_stream_destroy.restype = _IP
        lib.t_stream_sync.argtypes = [_CVP]
        lib.t_stream_sync.restype = _IP
        lib.t_stream_query.argtypes = [_CVP]
        lib.t_stream_query.restype = _IP
        lib.t_event_create.argtypes = []
        lib.t_event_create.restype = _CVP
        lib.t_event_destroy.argtypes = [_CVP]
        lib.t_event_destroy.restype = _IP
        lib.t_event_record.argtypes = [_CVP, _CVP]
        lib.t_event_record.restype = _IP
        lib.t_stream_wait_event.argtypes = [_CVP, _CVP]
        lib.t_stream_wait_event.restype = _IP
        lib.t_event_elapsed.argtypes = [_CVP, _CVP, _PP(_FP)]
        lib.t_event_elapsed.restype = _IP
        lib.t_device_sync.argtypes = []
        lib.t_device_sync.restype = _IP
        lib.t_rmsnorm.argtypes = [_CVP, _CVP, _CVP, ctypes.c_longlong, ctypes.c_longlong,
                                  ctypes.c_double, ctypes.c_longlong, _IP, _CVP]
        lib.t_rmsnorm.restype = _IP
        lib.t_last_error.argtypes = []
        lib.t_last_error.restype = ctypes.c_char_p
        lib.t_clear_last_error.argtypes = []
        lib.t_clear_last_error.restype = _IP
        lib.t_api_names.argtypes = []
        lib.t_api_names.restype = ctypes.c_char_p
        lib.t_pci_bus_id.argtypes = [ctypes.c_char_p, _IP]
        lib.t_pci_bus_id.restype = _IP

        # --- T9_HAVE_* 平台能力面（库编译期探测结果的 Python 可见形式） ----
        self.features: dict[str, bool] = {}
        for name in [
            "t_malloc_host_pinned_portable", "t_malloc_host_pinned_mapped",
            "t_host_get_device_pointer", "t_host_register", "t_host_unregister",
            "t_stream_create_nonblocking",
        ]:
            self.features[name] = hasattr(lib, name)

    # -- 错误处理 ----------------------------------------------------------

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            err = self.lib.t_last_error().decode("utf-8", "replace")
            raise TransferOpsError(f"{what} 失败 rc={rc}: {err}")

    def clear_last_error(self) -> None:
        """清除运行时 sticky last-error（测试段隔离用；S4000 实测发现陈旧错误
        会污染后续 kernel 的 GetLastError 检查）。"""
        self.lib.t_clear_last_error()

    # -- 生命周期 ----------------------------------------------------------

    def init(self) -> int:
        """返回 t_init 的原始返回码（非 0 = flags 设置失败，调用方记录）。"""
        return int(self.lib.t_init())

    def dev_info(self) -> dict:
        buf = ctypes.create_string_buffer(1024)
        self._check(self.lib.t_dev_info(buf, 1024), "t_dev_info")
        return json.loads(buf.value.decode("utf-8"))

    def pci_bus_id(self) -> str:
        buf = ctypes.create_string_buffer(64)
        rc = self.lib.t_pci_bus_id(buf, 64)
        if rc != 0:
            return ""
        return buf.value.decode("utf-8", "replace")

    def api_names(self) -> list[str]:
        return self.lib.t_api_names().decode("utf-8", "replace").split(",")

    # -- 内存 ---------------------------------------------------------------

    def malloc_device(self, nbytes: int) -> int:
        p = self.lib.t_malloc_device(_SZ(nbytes))
        if not p:
            raise TransferOpsError(f"t_malloc_device({nbytes}) 失败: {self.lib.t_last_error().decode()}")
        return int(p)

    def free_device(self, p: int) -> None:
        self._check(self.lib.t_free_device(_CVP(p)), "t_free_device")

    def malloc_pageable(self, nbytes: int) -> int:
        p = self.lib.t_malloc_host_pageable(_SZ(nbytes))
        if not p:
            raise TransferOpsError(f"malloc({nbytes}) 失败")
        return int(p)

    def free_pageable(self, p: int) -> None:
        self.lib.t_free_host_pageable(_CVP(p))

    def malloc_pinned_mallochost(self, nbytes: int) -> int:
        p = self.lib.t_malloc_host_pinned_mallochost(_SZ(nbytes))
        if not p:
            raise TransferOpsError(f"cudaMallocHost({nbytes}) 失败: {self.lib.t_last_error().decode()}")
        return int(p)

    def malloc_pinned_default(self, nbytes: int) -> int:
        p = self.lib.t_malloc_host_pinned_default(_SZ(nbytes))
        if not p:
            raise TransferOpsError(f"cudaHostAlloc(Default,{nbytes}) 失败: {self.lib.t_last_error().decode()}")
        return int(p)

    def malloc_pinned_portable(self, nbytes: int) -> int:
        if not self.features["t_malloc_host_pinned_portable"]:
            raise TransferOpsError("平台无 Portable pinned（T9_HAVE_HOSTALLOC_PORTABLE 未注入）")
        p = self.lib.t_malloc_host_pinned_portable(_SZ(nbytes))
        if not p:
            raise TransferOpsError(f"cudaHostAlloc(Portable,{nbytes}) 失败: {self.lib.t_last_error().decode()}")
        return int(p)

    def malloc_pinned_mapped(self, nbytes: int) -> int:
        if not self.features["t_malloc_host_pinned_mapped"]:
            raise TransferOpsError("平台无 Mapped pinned（T9_HAVE_HOSTALLOC_MAPPED 未注入）")
        p = self.lib.t_malloc_host_pinned_mapped(_SZ(nbytes))
        if not p:
            raise TransferOpsError(f"cudaHostAlloc(Mapped,{nbytes}) 失败: {self.lib.t_last_error().decode()}")
        return int(p)

    def host_get_device_pointer(self, p: int) -> int:
        if not self.features["t_host_get_device_pointer"]:
            raise TransferOpsError("平台无 t_host_get_device_pointer")
        dp = self.lib.t_host_get_device_pointer(_CVP(p))
        if not dp:
            raise TransferOpsError(f"cudaHostGetDevicePointer 失败: {self.lib.t_last_error().decode()}")
        return int(dp)

    def host_register(self, p: int, nbytes: int) -> None:
        if not self.features["t_host_register"]:
            raise TransferOpsError("平台无 t_host_register")
        self._check(self.lib.t_host_register(_CVP(p), _SZ(nbytes)), "t_host_register")

    def host_unregister(self, p: int) -> None:
        if not self.features["t_host_unregister"]:
            raise TransferOpsError("平台无 t_host_unregister")
        self._check(self.lib.t_host_unregister(_CVP(p)), "t_host_unregister")

    def free_pinned(self, p: int) -> None:
        self._check(self.lib.t_free_host_pinned(_CVP(p)), "t_free_host_pinned")

    # -- 传输 ----------------------------------------------------------------

    def memcpy_sync(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        self._check(self.lib.t_memcpy_sync(_CVP(dst), _CVP(src), _SZ(nbytes), _IP(kind)), "t_memcpy_sync")

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
        self._check(
            self.lib.t_memcpy_async(_CVP(dst), _CVP(src), _SZ(nbytes), _IP(kind), _CVP(stream)),
            "t_memcpy_async")

    # -- stream / event ------------------------------------------------------

    def stream_create(self) -> int:
        s = self.lib.t_stream_create()
        if not s:
            raise TransferOpsError(f"t_stream_create 失败: {self.lib.t_last_error().decode()}")
        return int(s)

    def stream_create_nonblocking(self) -> int:
        if not self.features["t_stream_create_nonblocking"]:
            raise TransferOpsError("平台无 t_stream_create_nonblocking")
        s = self.lib.t_stream_create_nonblocking()
        if not s:
            raise TransferOpsError(f"t_stream_create_nonblocking 失败: {self.lib.t_last_error().decode()}")
        return int(s)

    def stream_destroy(self, s: int) -> None:
        self._check(self.lib.t_stream_destroy(_CVP(s)), "t_stream_destroy")

    def stream_sync(self, s: int) -> None:
        self._check(self.lib.t_stream_sync(_CVP(s)), "t_stream_sync")

    def stream_query(self, s: int) -> int:
        return int(self.lib.t_stream_query(_CVP(s)))

    def event_create(self) -> int:
        e = self.lib.t_event_create()
        if not e:
            raise TransferOpsError(f"t_event_create 失败: {self.lib.t_last_error().decode()}")
        return int(e)

    def event_destroy(self, e: int) -> None:
        self._check(self.lib.t_event_destroy(_CVP(e)), "t_event_destroy")

    def event_record(self, e: int, stream: int) -> None:
        self._check(self.lib.t_event_record(_CVP(e), _CVP(stream)), "t_event_record")

    def stream_wait_event(self, stream: int, e: int) -> None:
        self._check(self.lib.t_stream_wait_event(_CVP(stream), _CVP(e)), "t_stream_wait_event")

    def event_elapsed_ms(self, a: int, b: int) -> float:
        ms = _FP(-1.0)
        self._check(self.lib.t_event_elapsed(_CVP(a), _CVP(b), ctypes.byref(ms)), "t_event_elapsed")
        return float(ms.value)

    def device_sync(self) -> None:
        self._check(self.lib.t_device_sync(), "t_device_sync")

    # -- 计算负载（第 02 期 RMSNorm） ------------------------------------------

    def rmsnorm(self, x: int, w: int, y: int, rows: int, h: int, eps: float,
                tile: int, stream: int, is_fp16: bool = False) -> None:
        rc = self.lib.t_rmsnorm(_CVP(x), _CVP(w), _CVP(y), ctypes.c_longlong(rows),
                                ctypes.c_longlong(h), ctypes.c_double(eps),
                                ctypes.c_longlong(tile), _IP(1 if is_fp16 else 0), _CVP(stream))
        if rc != 0:
            raise TransferOpsError(f"t_rmsnorm 失败 rc={rc}: {self.lib.t_last_error().decode()}")


def memcmp_equal(a: int, b: int, nbytes: int) -> bool:
    """libc memcmp 位级比较（零拷贝到 Python，大缓冲安全）。"""
    libc = ctypes.CDLL(None)
    libc.memcmp.argtypes = [_CVP, _CVP, _SZ]
    libc.memcmp.restype = _IP
    return libc.memcmp(_CVP(a), _CVP(b), _SZ(nbytes)) == 0
