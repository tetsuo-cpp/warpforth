"""Execute a PTX kernel from a JSON request on stdin."""

from __future__ import annotations

import base64
import binascii
import ctypes
import importlib
import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import SimpleNamespace

DIMENSION_COUNT = 3
SINGLE_VALUE_RESULT_LENGTH = 2
SUPPORTED_TYPES = ("i64", "f64", "i64[]", "f64[]")


class RequestError(ValueError):
    """Raised when the input request does not match the runner protocol."""


class CudaError(RuntimeError):
    """Raised when a CUDA Driver API call fails."""


@dataclass
class Param:
    """A kernel parameter and any associated host/device storage."""

    type_name: str
    host_value: object
    device_ptr: object | None = None


@dataclass
class RunnerRequest:
    """Validated runner input."""

    ptx: bytes
    kernel: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    params: list[Param]
    outputs: list[tuple[int, int]]


def _require_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        message = f"{name} must be an object"
        raise RequestError(message)
    return value


def _parse_dims(request: dict[str, object], name: str) -> tuple[int, int, int]:
    value = request.get(name, [1, 1, 1])
    if (
        not isinstance(value, list)
        or len(value) != DIMENSION_COUNT
        or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in value)
    ):
        message = f"{name} must contain three positive integers"
        raise RequestError(message)
    return value[0], value[1], value[2]


def _parse_params(request: dict[str, object]) -> list[Param]:
    specs = request.get("params")
    if not isinstance(specs, list) or not specs:
        message = "params must be a non-empty array"
        raise RequestError(message)
    return [_parse_param(raw_spec, index) for index, raw_spec in enumerate(specs)]


def _parse_param(raw_spec: object, index: int) -> Param:
    spec = _require_object(raw_spec, f"params[{index}]")
    type_name = spec.get("type")
    if type_name not in SUPPORTED_TYPES:
        message = f"params[{index}].type is unsupported: {type_name!r}"
        raise RequestError(message)

    type_name = cast("str", type_name)
    ctype = ctypes.c_double if type_name.startswith("f64") else ctypes.c_int64
    if type_name.endswith("[]"):
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            message = f"params[{index}].values must be a non-empty array"
            raise RequestError(message)
        try:
            return Param(type_name, (ctype * len(values))(*values))
        except (OverflowError, TypeError, ValueError) as exc:
            message = f"params[{index}].values do not match {type_name}"
            raise RequestError(message) from exc

    if "value" not in spec:
        message = f"params[{index}].value is required"
        raise RequestError(message)
    try:
        return Param(type_name, ctype(spec["value"]))
    except (OverflowError, TypeError, ValueError) as exc:
        message = f"params[{index}].value does not match {type_name}"
        raise RequestError(message) from exc


def _parse_outputs(request: dict[str, object], params: list[Param]) -> list[tuple[int, int]]:
    specs = request.get("outputs")
    if not isinstance(specs, list) or not specs:
        message = "outputs must be a non-empty array"
        raise RequestError(message)
    return [_parse_output(raw_spec, index, params) for index, raw_spec in enumerate(specs)]


def _parse_output(raw_spec: object, index: int, params: list[Param]) -> tuple[int, int]:
    spec = _require_object(raw_spec, f"outputs[{index}]")
    param_index = spec.get("param")
    if (
        not isinstance(param_index, int)
        or isinstance(param_index, bool)
        or param_index < 0
        or param_index >= len(params)
    ):
        message = f"outputs[{index}].param is out of range"
        raise RequestError(message)

    param = params[param_index]
    if not param.type_name.endswith("[]"):
        message = f"outputs[{index}].param refers to a scalar"
        raise RequestError(message)
    size = len(cast("ctypes.Array[object]", param.host_value))
    count = spec.get("count", size)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0 or count > size:
        message = f"outputs[{index}].count must be between 0 and {size}"
        raise RequestError(message)
    return param_index, count


def _parse_request(value: object) -> RunnerRequest:
    request = _require_object(value, "request")
    encoded_ptx = request.get("ptx_base64")
    if not isinstance(encoded_ptx, str) or not encoded_ptx:
        message = "ptx_base64 must be a non-empty string"
        raise RequestError(message)
    try:
        ptx = base64.b64decode(encoded_ptx, validate=True)
    except (binascii.Error, ValueError) as exc:
        message = "ptx_base64 is not valid base64"
        raise RequestError(message) from exc
    if not ptx:
        message = "ptx_base64 decodes to empty PTX"
        raise RequestError(message)

    kernel = request.get("kernel")
    if not isinstance(kernel, str) or not kernel:
        message = "kernel must be a non-empty string"
        raise RequestError(message)
    params = _parse_params(request)
    return RunnerRequest(
        ptx=ptx,
        kernel=kernel,
        grid=_parse_dims(request, "grid"),
        block=_parse_dims(request, "block"),
        params=params,
        outputs=_parse_outputs(request, params),
    )


def _load_cuda() -> object:
    try:
        return importlib.import_module("cuda.bindings.driver")
    except ImportError:
        try:
            return importlib.import_module("cuda.cuda")
        except ImportError as exc:
            message = "cuda-python is not installed"
            raise RuntimeError(message) from exc


def _cuda_check(cuda: object, result: tuple[object, ...]) -> object:
    cuda_module = cast("SimpleNamespace", cuda)
    success = cuda_module.CUresult.CUDA_SUCCESS
    error = result[0]
    if error != success:
        _, name = _cuda_result(cuda, "cuGetErrorName", error)
        _, description = _cuda_result(cuda, "cuGetErrorString", error)
        message = f"{_decode(name)}: {_decode(description)}"
        raise CudaError(message)
    if len(result) == 1:
        return None
    if len(result) == SINGLE_VALUE_RESULT_LENGTH:
        return result[1]
    return result[1:]


def _cuda_result(cuda: object, name: str, *args: object) -> tuple[object, ...]:
    function = cast("Callable[..., tuple[object, ...]]", getattr(cuda, name))
    return function(*args)


def _cuda_call(cuda: object, name: str, *args: object) -> object:
    return _cuda_check(cuda, _cuda_result(cuda, name, *args))


def _decode(value: object) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


class CudaSession:
    """Own the CUDA resources used to execute one request."""

    def __init__(self, cuda: object, request: RunnerRequest) -> None:
        self.cuda = cuda
        self.request = request
        self.context: object | None = None
        self.module: object | None = None

    def load_kernel(self) -> object:
        _cuda_call(self.cuda, "cuInit", 0)
        device = _cuda_call(self.cuda, "cuDeviceGet", 0)
        try:
            self.context = _cuda_call(self.cuda, "cuCtxCreate", None, 0, device)
        except TypeError:
            self.context = _cuda_call(self.cuda, "cuCtxCreate", 0, device)
        ptx_buffer = ctypes.create_string_buffer(self.request.ptx)
        self.module = _cuda_call(self.cuda, "cuModuleLoadData", ptx_buffer)
        return _cuda_call(
            self.cuda,
            "cuModuleGetFunction",
            self.module,
            self.request.kernel.encode(),
        )

    def allocate_params(self) -> tuple[tuple[object, ...], tuple[object, ...]]:
        values: list[object] = []
        types: list[object] = []
        for param in self.request.params:
            value, value_type = self._allocate_param(param)
            values.append(value)
            types.append(value_type)
        return tuple(values), tuple(types)

    def _allocate_param(self, param: Param) -> tuple[object, object]:
        if not param.type_name.endswith("[]"):
            scalar = cast("ctypes.c_int64 | ctypes.c_double", param.host_value)
            return scalar.value, type(scalar)
        size_bytes = ctypes.sizeof(param.host_value)
        param.device_ptr = _cuda_call(self.cuda, "cuMemAlloc", size_bytes)
        _cuda_call(
            self.cuda,
            "cuMemcpyHtoD",
            param.device_ptr,
            ctypes.addressof(param.host_value),
            size_bytes,
        )
        return param.device_ptr, ctypes.c_void_p

    def read_outputs(self) -> list[dict[str, object]]:
        outputs: list[dict[str, object]] = []
        for param_index, count in self.request.outputs:
            param = self.request.params[param_index]
            output_buffer = type(param.host_value)()
            _cuda_call(
                self.cuda,
                "cuMemcpyDtoH",
                ctypes.addressof(output_buffer),
                param.device_ptr,
                ctypes.sizeof(output_buffer),
            )
            outputs.append(
                {
                    "param": param_index,
                    "type": param.type_name,
                    "values": list(output_buffer)[:count],
                }
            )
        return outputs

    def cleanup(self) -> None:
        for param in self.request.params:
            if param.device_ptr is not None:
                _cuda_call(self.cuda, "cuMemFree", param.device_ptr)
        if self.module is not None:
            _cuda_call(self.cuda, "cuModuleUnload", self.module)
        if self.context is not None:
            _cuda_call(self.cuda, "cuCtxDestroy", self.context)


def execute(value: object, cuda: object | None = None) -> list[dict[str, object]]:
    """Validate and execute one runner request."""
    request = _parse_request(value)
    session = CudaSession(cuda or _load_cuda(), request)
    try:
        function = session.load_kernel()
        kernel_params = session.allocate_params()
        _cuda_call(
            session.cuda,
            "cuLaunchKernel",
            function,
            *request.grid,
            *request.block,
            0,
            0,
            kernel_params,
            0,
        )
        _cuda_call(session.cuda, "cuCtxSynchronize")
        return session.read_outputs()
    finally:
        session.cleanup()


def main() -> int:
    """Read one request and write one response."""
    try:
        response = {"status": "ok", "outputs": execute(json.load(sys.stdin))}
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        response = {"status": "error", "error": str(exc)}
    json.dump(response, sys.stdout, allow_nan=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
