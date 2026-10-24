"""Device worker: loads PTX and launches kernels for a client over newline-delimited JSON.

Stdlib only, because it is sent to a bare GPU host as ``python3 -u -c <source>``.
Each request is one JSON object per line on stdin; each response is one line on stdout.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import itertools
import json
import re
import sys
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import IO

    Holder = ctypes.c_uint64 | ctypes.c_int64 | ctypes.c_double

Kind = Literal["ptr", "i64", "f64"]

_ENTRY = re.compile(r"\.entry\s+([A-Za-z_$%][\w$]*)\s*(?:\(([^)]*)\))?")
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1

# A driver error during launch or synchronize leaves the context unusable, so
# the worker reports it and exits instead of serving further requests.
_STICKY_OPS = frozenset({"launch"})


class BadRequestError(Exception):
    """The request is malformed or refers to something that does not exist."""


class CudaError(Exception):
    """A CUDA driver call failed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _param_kind(decl: str) -> Kind:
    match decl.split():
        case [".param", ".u64", *modifiers, _name] if ".ptr" in modifiers:
            return "ptr"
        case [".param", ".u64" | ".s64" | ".b64", _name]:
            return "i64"
        case [".param", ".f64", _name]:
            return "f64"
    msg = f"unsupported kernel parameter: {decl.strip()}"
    raise ValueError(msg)


def parse_entries(ptx: str) -> dict[str, list[Kind]]:
    """Map each ``.entry`` kernel in ``ptx`` to the kinds of its parameters, in order."""
    entries: dict[str, list[Kind]] = {}
    for match in _ENTRY.finditer(ptx):
        params = (match.group(2) or "").strip()
        entries[match.group(1)] = [_param_kind(p) for p in params.split(",")] if params else []
    return entries


class Driver(Protocol):
    def load(self, ptx: str) -> int: ...
    def function(self, module: int, name: str) -> int: ...
    def alloc(self, size: int) -> int: ...
    def write(self, pointer: int, data: bytes) -> None: ...
    def read(self, pointer: int, size: int) -> bytes: ...
    def free(self, pointer: int) -> None: ...
    def launch(
        self,
        function: int,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[Holder],
    ) -> None: ...


_CU_JIT_ERROR_LOG_BUFFER = 5
_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
_JIT_LOG_BYTES = 16384

_SIGNATURES: dict[str, list[object]] = {
    "cuInit": [ctypes.c_uint],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
    "cuCtxSetCurrent": [ctypes.c_void_p],
    "cuModuleLoadDataEx": [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "cuModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
    "cuMemAlloc_v2": [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
    "cuMemsetD8_v2": [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t],
    "cuMemcpyHtoD_v2": [ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t],
    "cuMemcpyDtoH_v2": [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
    "cuMemFree_v2": [ctypes.c_uint64],
    "cuLaunchKernel": [
        ctypes.c_void_p,
        *[ctypes.c_uint] * 7,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "cuCtxSynchronize": [],
    "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "cuGetErrorString": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
}


class CudaDriver:
    """The CUDA driver API on device 0's primary context, through ctypes."""

    def __init__(self, library: str = "libcuda.so.1") -> None:
        self._cu = ctypes.CDLL(library)
        for name, argtypes in _SIGNATURES.items():
            fn = getattr(self._cu, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
        self._check(self._cu.cuInit(0))
        device = ctypes.c_int()
        self._check(self._cu.cuDeviceGet(ctypes.byref(device), 0))
        context = ctypes.c_void_p()
        self._check(self._cu.cuDevicePrimaryCtxRetain(ctypes.byref(context), device))
        self._check(self._cu.cuCtxSetCurrent(context))

    def _check(self, result: int, detail: str = "") -> None:
        if result == 0:
            return
        name = ctypes.c_char_p()
        text = ctypes.c_char_p()
        self._cu.cuGetErrorName(result, ctypes.byref(name))
        self._cu.cuGetErrorString(result, ctypes.byref(text))
        code = name.value.decode() if name.value else f"CUDA_ERROR_{result}"
        message = text.value.decode() if text.value else "unknown CUDA error"
        raise CudaError(code, f"{message}\n{detail}" if detail else message)

    def load(self, ptx: str) -> int:
        log = ctypes.create_string_buffer(_JIT_LOG_BYTES)
        options = (ctypes.c_int * 2)(_CU_JIT_ERROR_LOG_BUFFER, _CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
        values = (ctypes.c_void_p * 2)(ctypes.addressof(log), _JIT_LOG_BYTES)
        module = ctypes.c_void_p()
        result = self._cu.cuModuleLoadDataEx(ctypes.byref(module), ptx.encode(), 2, options, values)
        self._check(result, log.value.decode(errors="replace"))
        return module.value or 0

    def function(self, module: int, name: str) -> int:
        function = ctypes.c_void_p()
        self._check(self._cu.cuModuleGetFunction(ctypes.byref(function), module, name.encode()))
        return function.value or 0

    def alloc(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self._check(self._cu.cuMemAlloc_v2(ctypes.byref(pointer), size))
        try:
            self._check(self._cu.cuMemsetD8_v2(pointer, 0, size))
        except CudaError:
            self._cu.cuMemFree_v2(pointer)
            raise
        return pointer.value

    def write(self, pointer: int, data: bytes) -> None:
        self._check(self._cu.cuMemcpyHtoD_v2(pointer, data, len(data)))

    def read(self, pointer: int, size: int) -> bytes:
        out = ctypes.create_string_buffer(size)
        self._check(self._cu.cuMemcpyDtoH_v2(ctypes.addressof(out), pointer, size))
        return out.raw

    def free(self, pointer: int) -> None:
        self._check(self._cu.cuMemFree_v2(pointer))

    def launch(
        self,
        function: int,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[Holder],
    ) -> None:
        params = (ctypes.c_void_p * len(args))(*(ctypes.addressof(a) for a in args))
        self._check(self._cu.cuLaunchKernel(function, *grid, *block, 0, None, params, None))
        self._check(self._cu.cuCtxSynchronize())


def _field(request: dict, name: str) -> object:
    if name not in request:
        msg = f"missing field {name!r}"
        raise BadRequestError(msg)
    return request[name]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(request: dict, name: str) -> int:
    value = _field(request, name)
    if not _is_int(value) or value <= 0:
        msg = f"{name} must be a positive integer, got {value!r}"
        raise BadRequestError(msg)
    return value


def _dims(request: dict, name: str) -> tuple[int, int, int]:
    match _field(request, name):
        case [x, y, z] if all(_is_int(v) and v > 0 for v in (x, y, z)):
            return x, y, z
        case value:
            msg = f"{name} must be three positive integers, got {value!r}"
            raise BadRequestError(msg)


class Worker:
    """Owns the modules and buffers created through one driver and serves requests."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver
        self._ids = itertools.count(1)
        self._kernels: dict[int, dict[str, tuple[int, list[Kind]]]] = {}
        self._buffers: dict[int, tuple[int, int]] = {}
        self._ops: dict[str, Callable[[dict], dict]] = {
            "load": self._load,
            "alloc": self._alloc,
            "write": self._write,
            "read": self._read,
            "free": self._free,
            "launch": self._launch,
        }

    def handle(self, request: dict) -> dict:
        op = request.get("op")
        handler = self._ops.get(op) if isinstance(op, str) else None
        if handler is None:
            msg = f"unknown op {op!r}"
            raise BadRequestError(msg)
        return handler(request)

    def _buffer(self, request: dict, name: str = "buffer") -> tuple[int, int]:
        handle = _field(request, name)
        if not _is_int(handle) or handle not in self._buffers:
            msg = f"unknown buffer {handle!r}"
            raise BadRequestError(msg)
        return self._buffers[handle]

    def _load(self, request: dict) -> dict:
        ptx = _field(request, "ptx")
        if not isinstance(ptx, str):
            msg = "ptx must be a string"
            raise BadRequestError(msg)
        try:
            entries = parse_entries(ptx)
        except ValueError as e:
            raise BadRequestError(str(e)) from e
        module = self._driver.load(ptx)
        handle = next(self._ids)
        self._kernels[handle] = {
            name: (self._driver.function(module, name), kinds) for name, kinds in entries.items()
        }
        return {"module": handle, "kernels": entries}

    def _alloc(self, request: dict) -> dict:
        size = _positive_int(request, "size")
        pointer = self._driver.alloc(size)
        handle = next(self._ids)
        self._buffers[handle] = (pointer, size)
        return {"buffer": handle}

    def _write(self, request: dict) -> dict:
        pointer, size = self._buffer(request)
        encoded = _field(request, "data")
        if not isinstance(encoded, str):
            msg = "data must be a base64 string"
            raise BadRequestError(msg)
        try:
            data = base64.b64decode(encoded, validate=True)
        except binascii.Error as e:
            msg = f"data is not valid base64: {e}"
            raise BadRequestError(msg) from e
        if len(data) > size:
            msg = f"data is {len(data)} bytes but the buffer holds {size}"
            raise BadRequestError(msg)
        self._driver.write(pointer, data)
        return {}

    def _read(self, request: dict) -> dict:
        pointer, size = self._buffer(request)
        return {"data": base64.b64encode(self._driver.read(pointer, size)).decode()}

    def _free(self, request: dict) -> dict:
        pointer, _size = self._buffer(request)
        self._driver.free(pointer)
        del self._buffers[request["buffer"]]
        return {}

    def _launch(self, request: dict) -> dict:
        module = _field(request, "module")
        if not _is_int(module) or module not in self._kernels:
            msg = f"unknown module {module!r}"
            raise BadRequestError(msg)
        name = _field(request, "kernel")
        kernels = self._kernels[module]
        if not isinstance(name, str) or name not in kernels:
            msg = f"module {module} has no kernel {name!r}; it has {sorted(kernels)}"
            raise BadRequestError(msg)
        function, kinds = kernels[name]
        grid = _dims(request, "grid")
        block = _dims(request, "block")
        args = _field(request, "args")
        if not isinstance(args, list) or len(args) != len(kinds):
            count = len(args) if isinstance(args, list) else args
            msg = f"kernel {name!r} takes {len(kinds)} arguments ({kinds}), got {count!r}"
            raise BadRequestError(msg)
        holders = [
            self._pack(i, kind, arg) for i, (kind, arg) in enumerate(zip(kinds, args, strict=True))
        ]
        self._driver.launch(function, grid, block, holders)
        return {}

    def _pack(self, index: int, kind: Kind, arg: object) -> Holder:
        if kind == "ptr":
            if isinstance(arg, dict) and "buffer" in arg:
                return ctypes.c_uint64(self._buffer(arg)[0])
        elif isinstance(arg, dict) and "value" in arg:
            value = arg["value"]
            if kind == "i64" and _is_int(value) and _I64_MIN <= value <= _I64_MAX:
                return ctypes.c_int64(value)
            if kind == "f64" and (_is_int(value) or isinstance(value, float)):
                try:
                    return ctypes.c_double(float(value))
                except OverflowError:
                    pass
        expected = {"ptr": '{"buffer": id}', "i64": "a 64-bit integer", "f64": "a number"}[kind]
        msg = f"argument {index} is {kind} and needs {expected}, got {arg!r}"
        raise BadRequestError(msg)


def _error(request_id: object, code: str, message: str, *, fatal: bool) -> dict:
    return {
        "id": request_id,
        "ok": False,
        "error": {"code": code, "message": message},
        "fatal": fatal,
    }


def _parse_request(line: str) -> dict:
    try:
        request = json.loads(line)
    except json.JSONDecodeError as e:
        msg = f"malformed JSON: {e}"
        raise BadRequestError(msg) from e
    if not isinstance(request, dict) or "id" not in request:
        msg = "request must be a JSON object with an id"
        raise BadRequestError(msg)
    return request


def serve(driver: Driver, requests: Iterable[str], responses: IO[str]) -> int:
    """Answer each request line until input ends or a sticky error; return the exit status."""
    worker = Worker(driver)
    for line in requests:
        if not line.strip():
            continue
        request_id = None
        fatal = False
        try:
            request = _parse_request(line)
            request_id = request["id"]
            response = {"id": request_id, "ok": True, "result": worker.handle(request)}
        except BadRequestError as e:
            response = _error(request_id, "BAD_REQUEST", str(e), fatal=False)
        except CudaError as e:
            fatal = request.get("op") in _STICKY_OPS
            response = _error(request_id, e.code, e.message, fatal=fatal)
        responses.write(json.dumps(response) + "\n")
        responses.flush()
        if fatal:
            return 1
    return 0


def main() -> int:
    try:
        driver = CudaDriver()
    except (OSError, CudaError) as e:
        sys.stderr.write(f"warpforth worker: cannot initialize CUDA: {e}\n")
        return 2
    return serve(driver, sys.stdin, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
