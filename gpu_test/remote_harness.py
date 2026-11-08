"""JSON-lines CUDA harness for WarpForth correctness runs and paired benchmarks.

This script runs on the rented GPU host next to the driver. One JSON request per line on
stdin, one JSON response per line on stdout. Every response is `{"ok": true, ...}` or
`{"ok": false, "error": message}`; a failure never kills the stream.

Ops:

  {"op": "ping"}
    -> {"ok": true, "environment": {gpu_uuid, driver_version, sm_clock_mhz,
       memory_clock_mhz}}. Proves the driver is reachable and reports the device
       metadata the bench_stats result contract needs.

  {"op": "correctness", "ptx": str, "kernel": str, "params": [...],
   "grid": [x, y, z], "block": [x, y, z], "output_param": int,
   "output_count": int | null}
    -> {"ok": true, "values": [number, ...]}. Params mirror the warpforth-runner CLI:
       {"type": "i64"|"f64", "value": number} for scalars and
       {"type": "i64[]"|"f64[]", "values": [...]} for arrays, in kernel argument order.
       Arrays upload as given, the kernel runs, and the output array reads back:
       ints for i64[], floats for f64[]. A null output_count reads the whole array.

  {"op": "bench", ...}
    -> {"ok": true, "result": <one case in the bench_stats contract shape>,
        "input_digests": {name: sha256 of the uploaded bytes},
        "validation": {"base": bool, "head": bool},
        "batch_launches": {"base": int, "head": int}}.
    Loads both sides into one CUDA context, generates every array from its seeded spec,
    validates each side against the reference with the case tolerance BEFORE any timing,
    warms up, then times interleaved batches with CUDA events on one explicit stream.
    Sample times are per launch (batch elapsed divided by its launch count) and the side
    that runs first alternates per sample, so session drift lands on both sides. A side
    that fails validation yields result.correct = false with no samples.

    Bench request shape:
      "case_id": str, "case_version": int,
      "reference": a key of the REFERENCES table, "tolerance": [rtol, atol],
      "params": ordered kernel params:
        {"name": str, "kind": "scalar", "type": "i64"|"f64"}
        {"name": str, "kind": "array", "type": "i64"|"f64", "cells": int,
         "dtype": "i64"|"f64"|"f32" (default: the cell type)}.
        An f32 array packs two elements per i64 cell, matching the Forth headers.
      "scalars": {name: number} covering every scalar param,
      "inputs": {array name: {"kind": "uniform"|"int_uniform", "dtype": str,
                  "seed": int, "low": number, "high": number,
                  "length": int (0 fills the buffer)}}. Arrays without a spec stay zero.
      "output": name of the array param the kernel writes,
      "base" / "head":
        {"kind": "ptx", "ptx": str, "kernel": str,
         "launch": {"grid": [x, y, z], "block": [x, y, z]}}
        {"kind": "host_lib", "library": "cublas", "symbol": "cublasDgemm",
         "n": int, "a": name, "b": name, "c": name} for a row-major f64 C = A @ B,
      "timing": {"warmup": int, "samples": int, "sample_ms": number},
      "environment": {"base_commit": str, "head_commit": str}, both optional.

Only the output buffer is restored before each validation launch: the kernels treat
every other buffer as read-only, and the work per timed launch does not depend on the
values, so timed batches need no restores.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

CUDA_SUCCESS = 0

# The numeric ABI values below are stable across CUDA toolkit versions. Using the plain
# integers keeps the harness working with both the legacy (cuda.cuda) and the current
# (cuda.bindings.driver) cuda-python layouts.
CU_EVENT_DEFAULT = 0x0
CU_STREAM_NON_BLOCKING = 0x1
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CUBLAS_OP_N = 0

UINT32_RANGE = 0x100000000
MAX_BATCH_LAUNCHES = 100_000
MIN_CALIBRATION_US = 1.0
CELL_BYTES = 8
ELEMENT_BYTES = {"i64": 8, "f64": 8, "f32": 4}
CELL_DTYPES = {"i64": np.int64, "f64": np.float64}
ARG_CTYPES = {"ptr": None, "i64": ctypes.c_int64, "f64": ctypes.c_double}
DIMS = 3
TOLERANCE_PAIR = 2


class HarnessError(Exception):
    """A request cannot be served; becomes the "error" field of a failed response."""


@dataclass(frozen=True)
class InputSpec:
    """A seeded generator for one array, mirroring `benchmarks.manifest.InputSpec`."""

    kind: str
    seed: int
    low: float
    high: float
    length: int


@dataclass(frozen=True)
class ScalarParam:
    name: str
    type: str


@dataclass(frozen=True)
class ArrayParam:
    """`type` is the 8-byte cell type the header declares; `dtype` is what it holds."""

    name: str
    type: str
    dtype: str
    cells: int

    @property
    def elements(self) -> int:
        return self.cells * (CELL_BYTES // ELEMENT_BYTES[self.dtype])


Param = ScalarParam | ArrayParam


@dataclass(frozen=True)
class Launch:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


@dataclass(frozen=True)
class PtxSide:
    ptx: str
    kernel: str
    launch: Launch


@dataclass(frozen=True)
class HostLibSide:
    symbol: str
    n: int
    a: str
    b: str
    c: str


Side = PtxSide | HostLibSide


@dataclass(frozen=True)
class BenchRequest:
    case_id: str
    case_version: int
    reference: str
    tolerance: tuple[float, float]
    params: tuple[Param, ...]
    scalars: dict[str, float | int]
    inputs: dict[str, InputSpec]
    output_name: str
    base: Side
    head: Side
    warmup: int
    samples: int
    sample_ms: float
    base_commit: str
    head_commit: str


@dataclass(frozen=True)
class ScalarValue:
    type: str
    value: float | int


@dataclass(frozen=True)
class ArrayValue:
    type: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class CorrectnessRequest:
    ptx: str
    kernel: str
    params: tuple[ScalarValue | ArrayValue, ...]
    launch: Launch
    output_param: int
    output_count: int | None


class Gpu(Protocol):
    """The device surface the flows use. CudaGpu implements it for real, tests fake it."""

    def environment(self) -> dict[str, object]: ...
    def alloc(self, nbytes: int) -> object: ...
    def free(self, ptr: object) -> None: ...
    def upload(self, ptr: object, data: np.ndarray) -> None: ...
    def download(self, ptr: object, count: int, dtype: np.dtype) -> np.ndarray: ...
    def copy_device(self, dst: object, src: object, nbytes: int) -> None: ...
    def load_module(self, ptx: str) -> object: ...
    def unload_module(self, module: object) -> None: ...
    def function(self, module: object, name: str) -> object: ...
    def launch(self, func: object, launch: Launch, args: Sequence[tuple[str, object]]) -> None: ...
    def record_event(self) -> object: ...
    def elapsed_us(self, start: object, end: object) -> float: ...
    def synchronize(self) -> None: ...
    def dgemm(self, n: int, a: object, b: object, c: object) -> None: ...


def _check(result: object, api: str) -> object:
    """Unwrap a (status, outs...) tuple, raising HarnessError on a nonzero status."""
    if result is None:
        return None
    if isinstance(result, tuple):
        status, *outs = result
        if int(status) != CUDA_SUCCESS:
            msg = f"{api} failed with error {int(status)}"
            raise HarnessError(msg)
        return outs[0] if len(outs) == 1 else outs
    if int(result) != CUDA_SUCCESS:
        msg = f"{api} failed with error {int(result)}"
        raise HarnessError(msg)
    return None


def _import_driver() -> object:
    try:
        from cuda.bindings import driver  # noqa: PLC0415 - resolved on the rental, not locally
    except ImportError:
        try:
            from cuda import cuda as driver  # noqa: PLC0415 - cuda-python 12.5 and earlier
        except ImportError as exc:
            msg = "cuda-python is not importable on this host"
            raise HarnessError(msg) from exc
    return driver


def _import_cublas() -> ctypes.CDLL:
    # cuda-python dropped the library bindings in 12.5, so cuBLAS is reached through
    # ctypes against the nvidia-cublas-cu12 wheel's shared library instead.
    try:
        spec = importlib.util.find_spec("nvidia.cublas")
    except ModuleNotFoundError:
        spec = None
    candidates = []
    if spec is not None and spec.submodule_search_locations:
        lib_dir = Path(spec.submodule_search_locations[0]) / "lib"
        candidates.extend(sorted(lib_dir.glob("libcublas.so*")))
    found = ctypes.util.find_library("cublas")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate))
        except OSError:
            continue
    msg = "libcublas is not loadable on this host"
    raise HarnessError(msg)


class CudaGpu:
    """The Gpu surface over the CUDA driver bindings, created once per process."""

    def __init__(self) -> None:
        drv = _import_driver()
        _check(drv.cuInit(0), "cuInit")
        self._drv = drv
        self._device = _check(drv.cuDeviceGet(0), "cuDeviceGet")
        self._context = _check(
            drv.cuDevicePrimaryCtxRetain(self._device), "cuDevicePrimaryCtxRetain"
        )
        _check(drv.cuCtxSetCurrent(self._context), "cuCtxSetCurrent")
        self._stream = _check(drv.cuStreamCreate(CU_STREAM_NON_BLOCKING), "cuStreamCreate")
        self._cublas: object | None = None
        self._cublas_handle: object | None = None

    def environment(self) -> dict[str, object]:
        uuid = _check(self._drv.cuDeviceGetUuid(self._device), "cuDeviceGetUuid")
        version = int(_check(self._drv.cuDriverGetVersion(), "cuDriverGetVersion"))
        sm_khz = int(
            _check(
                self._drv.cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, self._device),
                "cuDeviceGetAttribute",
            )
        )
        mem_khz = int(
            _check(
                self._drv.cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, self._device),
                "cuDeviceGetAttribute",
            )
        )
        return {
            "gpu_uuid": bytes(getattr(uuid, "bytes", uuid)).hex(),
            "driver_version": f"{version // 1000}.{(version % 1000) // 10}",
            "sm_clock_mhz": sm_khz // 1000,
            "memory_clock_mhz": mem_khz // 1000,
        }

    def alloc(self, nbytes: int) -> object:
        return _check(self._drv.cuMemAlloc(nbytes), "cuMemAlloc")

    def free(self, ptr: object) -> None:
        _check(self._drv.cuMemFree(ptr), "cuMemFree")

    def upload(self, ptr: object, data: np.ndarray) -> None:
        _check(self._drv.cuMemcpyHtoD(ptr, data, data.nbytes), "cuMemcpyHtoD")

    def download(self, ptr: object, count: int, dtype: np.dtype) -> np.ndarray:
        data = np.empty(count, dtype=dtype)
        _check(self._drv.cuMemcpyDtoH(data, ptr, data.nbytes), "cuMemcpyDtoH")
        return data

    def copy_device(self, dst: object, src: object, nbytes: int) -> None:
        _check(self._drv.cuMemcpyDtoD(dst, src, nbytes), "cuMemcpyDtoD")

    def load_module(self, ptx: str) -> object:
        # The bindings take char* only as bytes, not str.
        return _check(self._drv.cuModuleLoadData(ptx.encode()), "cuModuleLoadData")

    def unload_module(self, module: object) -> None:
        _check(self._drv.cuModuleUnload(module), "cuModuleUnload")

    def function(self, module: object, name: str) -> object:
        return _check(self._drv.cuModuleGetFunction(module, name.encode()), "cuModuleGetFunction")

    def launch(self, func: object, launch: Launch, args: Sequence[tuple[str, object]]) -> None:
        values = tuple(value for _kind, value in args)
        types = tuple(ARG_CTYPES[kind] for kind, _value in args)
        grid_x, grid_y, grid_z = launch.grid
        block_x, block_y, block_z = launch.block
        _check(
            self._drv.cuLaunchKernel(
                func,
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                0,
                self._stream,
                (values, types),
                0,
            ),
            "cuLaunchKernel",
        )

    def record_event(self) -> object:
        event = _check(self._drv.cuEventCreate(CU_EVENT_DEFAULT), "cuEventCreate")
        _check(self._drv.cuEventRecord(event, self._stream), "cuEventRecord")
        return event

    def elapsed_us(self, start: object, end: object) -> float:
        ms = _check(self._drv.cuEventElapsedTime(start, end), "cuEventElapsedTime")
        _check(self._drv.cuEventDestroy(start), "cuEventDestroy")
        _check(self._drv.cuEventDestroy(end), "cuEventDestroy")
        return float(ms) * 1000.0

    def synchronize(self) -> None:
        _check(self._drv.cuStreamSynchronize(self._stream), "cuStreamSynchronize")

    def dgemm(self, n: int, a: object, b: object, c: object) -> None:
        if self._cublas is None:
            self._cublas = _import_cublas()
            self._cublas_handle = ctypes.c_void_p()
            status = self._cublas.cublasCreate_v2(ctypes.byref(self._cublas_handle))
            if status != 0:
                msg = f"cublasCreate_v2 failed with error {status}"
                raise HarnessError(msg)
            status = self._cublas.cublasSetStream_v2(
                self._cublas_handle, ctypes.c_void_p(int(self._stream))
            )
            if status != 0:
                msg = f"cublasSetStream_v2 failed with error {status}"
                raise HarnessError(msg)
        # cuBLAS is column-major, so a row-major C = A @ B falls out of the same three
        # buffers passed B first: C' = B' @ A'.
        alpha = ctypes.c_double(1.0)
        beta = ctypes.c_double(0.0)
        status = self._cublas.cublasDgemm_v2(
            self._cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            n,
            n,
            ctypes.byref(alpha),
            ctypes.c_void_p(int(b)),
            n,
            ctypes.c_void_p(int(a)),
            n,
            ctypes.byref(beta),
            ctypes.c_void_p(int(c)),
            n,
        )
        if status != 0:
            msg = f"cublasDgemm_v2 failed with error {status}"
            raise HarnessError(msg)


def generate_elements(spec: InputSpec, dtype: str, elements: int) -> np.ndarray:
    """One seeded buffer, bit-identical to `benchmarks.manifest.generate` on the client."""
    count = spec.length or elements
    state = spec.seed & 0xFFFFFFFF or 1
    states = []
    append = states.append
    for _ in range(count):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        append(state)
    unit = np.asarray(states, dtype=np.uint32).astype(np.float64) / UINT32_RANGE
    if spec.kind == "int_uniform":
        low = int(spec.low)
        high = int(spec.high)
        return (low + unit * (high - low)).astype(np.int64)
    values = spec.low + unit * (spec.high - spec.low)
    if dtype == "f32":
        return values.astype(np.float32)
    return values


def generate_buffer(param: ArrayParam, spec: InputSpec | None) -> np.ndarray:
    """The exact cells to upload for one array param; the tail past spec.length stays zero."""
    buffer = np.zeros(param.cells, dtype=CELL_DTYPES[param.type])
    if spec is None:
        return buffer
    values = generate_elements(spec, param.dtype, param.elements)
    if param.dtype == "f32":
        raw = values.tobytes()
        raw += bytes(-len(raw) % CELL_BYTES)
        cells = np.frombuffer(raw, dtype=np.int64)
        buffer[: cells.size] = cells
        return buffer
    buffer[: values.size] = values.astype(buffer.dtype)
    return buffer


def _elements_of(param: ArrayParam, buffer: np.ndarray) -> np.ndarray:
    if param.dtype == "f32":
        return buffer.view(np.float32)
    return buffer


def _ref_scale_i64(inputs: dict[str, np.ndarray], scalars: dict[str, float]) -> np.ndarray:
    return inputs["INPUT"] * int(scalars["SCALE"])


def _ref_axpy_f64(inputs: dict[str, np.ndarray], scalars: dict[str, float]) -> np.ndarray:
    return scalars["ALPHA"] * inputs["X"] + inputs["Y"]


def _ref_reduce_sum_f64(inputs: dict[str, np.ndarray], scalars: dict[str, float]) -> np.ndarray:
    return np.array([inputs["INPUT"][: int(scalars["N"])].sum()])


def _ref_matmul_f64(inputs: dict[str, np.ndarray], scalars: dict[str, float]) -> np.ndarray:
    n = int(scalars["N"])
    a = inputs["A"][: n * n].reshape(n, n)
    b = inputs["B"][: n * n].reshape(n, n)
    return (a @ b).ravel()


def _ref_attention_f32(inputs: dict[str, np.ndarray], scalars: dict[str, float]) -> np.ndarray:
    nrow, nkey, hdim = (int(scalars[name]) for name in ("NROW", "NKEY", "HDIM"))
    q = inputs["QMAT"][: nrow * hdim].reshape(nrow, hdim).astype(np.float64)
    k = inputs["KMAT"][: nkey * hdim].reshape(nkey, hdim).astype(np.float64)
    v = inputs["VMAT"][: nkey * hdim].reshape(nkey, hdim).astype(np.float64)
    scores = (q @ k.T) * scalars["SCALE"]
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights @ v).ravel()


REFERENCES = {
    "scale_i64": _ref_scale_i64,
    "axpy_f64": _ref_axpy_f64,
    "reduce_sum_f64": _ref_reduce_sum_f64,
    "matmul_f64": _ref_matmul_f64,
    "attention_f32": _ref_attention_f32,
}


def within_tolerance(actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float) -> bool:
    """The same symmetric isclose rule as `benchmarks.manifest.within_tolerance`."""
    if actual.shape != expected.shape:
        return False
    diff = np.abs(actual - expected)
    limit = np.maximum(rtol * np.maximum(np.abs(actual), np.abs(expected)), atol)
    return bool(np.all(diff <= limit))


def _require(mapping: dict[str, object], key: str, where: str) -> object:
    if key not in mapping:
        msg = f"{where}: missing key '{key}'"
        raise HarnessError(msg)
    return mapping[key]


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{where}: expected an object, got {value!r}"
        raise HarnessError(msg)
    return value


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{where}: expected a non-empty string, got {value!r}"
        raise HarnessError(msg)
    return value


def _positive_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = f"{where}: expected a positive integer, got {value!r}"
        raise HarnessError(msg)
    return value


def _count(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{where}: expected a non-negative integer, got {value!r}"
        raise HarnessError(msg)
    return value


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{where}: expected a number, got {value!r}"
        raise HarnessError(msg)
    return float(value)


def _scalar_value(param_type: str, value: object, where: str) -> float | int:
    number = _number(value, where)
    if param_type == "f64":
        return number
    if not number.is_integer():
        msg = f"{where}: expected an integer for i64, got {value!r}"
        raise HarnessError(msg)
    return int(number)


def _positive_number(value: object, where: str) -> float:
    number = _number(value, where)
    if number <= 0:
        msg = f"{where}: expected a positive number, got {value!r}"
        raise HarnessError(msg)
    return number


def _dims(value: object, where: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != DIMS:
        msg = f"{where}: expected {DIMS} positive integers"
        raise HarnessError(msg)
    first, second, third = value
    return (
        _positive_int(first, where),
        _positive_int(second, where),
        _positive_int(third, where),
    )


def _parse_launch(value: object, where: str) -> Launch:
    mapping = _mapping(value, where)
    return Launch(
        grid=_dims(_require(mapping, "grid", where), f"{where}.grid"),
        block=_dims(_require(mapping, "block", where), f"{where}.block"),
    )


def _array_param(params: tuple[Param, ...], name: str, where: str) -> ArrayParam:
    for param in params:
        if isinstance(param, ArrayParam) and param.name == name:
            return param
    msg = f"{where}: {name} is not an array param"
    raise HarnessError(msg)


def _parse_param(value: object, where: str) -> Param:
    mapping = _mapping(value, where)
    name = _string(_require(mapping, "name", where), f"{where}.name")
    kind = _string(_require(mapping, "kind", where), f"{where}.kind")
    cell_type = _string(_require(mapping, "type", where), f"{where}.type")
    if cell_type not in CELL_DTYPES:
        msg = f"{where}.type: unsupported type {cell_type}"
        raise HarnessError(msg)
    if kind == "scalar":
        return ScalarParam(name=name, type=cell_type)
    if kind != "array":
        msg = f"{where}.kind: expected 'scalar' or 'array', got {kind}"
        raise HarnessError(msg)
    cells = _positive_int(_require(mapping, "cells", where), f"{where}.cells")
    dtype = _string(mapping.get("dtype", cell_type), f"{where}.dtype")
    if dtype not in ELEMENT_BYTES or ELEMENT_BYTES[dtype] > ELEMENT_BYTES[cell_type]:
        msg = f"{where}.dtype: {dtype} does not fit in {cell_type} cells"
        raise HarnessError(msg)
    return ArrayParam(name=name, type=cell_type, dtype=dtype, cells=cells)


def _parse_params(value: object, where: str) -> tuple[Param, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{where}: expected a non-empty array of params"
        raise HarnessError(msg)
    params = tuple(_parse_param(item, f"{where}[{index}]") for index, item in enumerate(value))
    names = [param.name for param in params]
    if len(set(names)) != len(names):
        msg = f"{where}: duplicate param name"
        raise HarnessError(msg)
    return params


def _parse_scalars(value: object, where: str, params: tuple[Param, ...]) -> dict[str, float | int]:
    mapping = _mapping(value, where)
    scalar_types = {param.name: param.type for param in params if isinstance(param, ScalarParam)}
    scalars = {}
    for name, item in mapping.items():
        if name not in scalar_types:
            msg = f"{where}: {name} is not a scalar param of this case"
            raise HarnessError(msg)
        scalars[name] = _scalar_value(scalar_types[name], item, f"{where}.{name}")
    for name in scalar_types:
        if name not in scalars:
            msg = f"{where}: missing value for scalar param {name}"
            raise HarnessError(msg)
    return scalars


def _parse_inputs(value: object, where: str, params: tuple[Param, ...]) -> dict[str, InputSpec]:
    mapping = _mapping(value, where)
    specs = {}
    for name, entry in mapping.items():
        path = f"{where}.{name}"
        param = _array_param(params, name, path)
        table = _mapping(entry, path)
        kind = _string(_require(table, "kind", path), f"{path}.kind")
        if kind not in ("uniform", "int_uniform"):
            msg = f"{path}.kind: unsupported generator {kind}"
            raise HarnessError(msg)
        dtype = _string(_require(table, "dtype", path), f"{path}.dtype")
        if dtype != param.dtype:
            msg = f"{path}.dtype: {dtype} does not match the param's {param.dtype} elements"
            raise HarnessError(msg)
        length = _count(table.get("length", 0), f"{path}.length")
        if length > param.elements:
            msg = f"{path}.length: {length} exceeds the param's {param.elements} elements"
            raise HarnessError(msg)
        specs[name] = InputSpec(
            kind=kind,
            seed=_positive_int(_require(table, "seed", path), f"{path}.seed"),
            low=_number(table.get("low", 0.0), f"{path}.low"),
            high=_number(table.get("high", 1.0), f"{path}.high"),
            length=length,
        )
    return specs


def _parse_side(value: object, where: str) -> Side:
    mapping = _mapping(value, where)
    kind = _string(_require(mapping, "kind", where), f"{where}.kind")
    if kind == "ptx":
        return PtxSide(
            ptx=_string(_require(mapping, "ptx", where), f"{where}.ptx"),
            kernel=_string(_require(mapping, "kernel", where), f"{where}.kernel"),
            launch=_parse_launch(_require(mapping, "launch", where), f"{where}.launch"),
        )
    if kind != "host_lib":
        msg = f"{where}.kind: expected 'ptx' or 'host_lib', got {kind}"
        raise HarnessError(msg)
    library = _string(_require(mapping, "library", where), f"{where}.library")
    symbol = _string(_require(mapping, "symbol", where), f"{where}.symbol")
    if library != "cublas" or symbol != "cublasDgemm":
        msg = f"{where}: unsupported host library call {library}.{symbol}"
        raise HarnessError(msg)
    return HostLibSide(
        symbol=symbol,
        n=_positive_int(_require(mapping, "n", where), f"{where}.n"),
        a=_string(_require(mapping, "a", where), f"{where}.a"),
        b=_string(_require(mapping, "b", where), f"{where}.b"),
        c=_string(_require(mapping, "c", where), f"{where}.c"),
    )


def _check_host_lib_buffers(side: HostLibSide, params: tuple[Param, ...], where: str) -> None:
    for name in (side.a, side.b, side.c):
        param = _array_param(params, name, where)
        if param.dtype != "f64":
            msg = f"{where}: cublasDgemm needs f64 buffers, {name} holds {param.dtype}"
            raise HarnessError(msg)
        if side.n * side.n > param.elements:
            msg = f"{where}: {name} holds {param.elements} elements, dgemm needs {side.n * side.n}"
            raise HarnessError(msg)


def _parse_tolerance(value: object) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != TOLERANCE_PAIR:
        msg = "bench.tolerance: expected [rtol, atol]"
        raise HarnessError(msg)
    rtol, atol = (_number(item, "bench.tolerance") for item in value)
    if rtol < 0 or atol < 0:
        msg = "bench.tolerance: rtol and atol must be non-negative"
        raise HarnessError(msg)
    return (rtol, atol)


def _optional_string(mapping: dict[str, object], key: str, where: str) -> str:
    value = mapping.get(key, "")
    if not isinstance(value, str):
        msg = f"{where}.{key}: expected a string, got {value!r}"
        raise HarnessError(msg)
    return value


def _parse_bench(request: dict[str, object]) -> BenchRequest:
    params = _parse_params(_require(request, "params", "bench"), "bench.params")
    reference = _string(_require(request, "reference", "bench"), "bench.reference")
    if reference not in REFERENCES:
        msg = f"bench.reference: unknown reference {reference}"
        raise HarnessError(msg)
    base = _parse_side(_require(request, "base", "bench"), "bench.base")
    head = _parse_side(_require(request, "head", "bench"), "bench.head")
    for side, where in ((base, "bench.base"), (head, "bench.head")):
        if isinstance(side, HostLibSide):
            _check_host_lib_buffers(side, params, where)
    timing = _mapping(_require(request, "timing", "bench"), "bench.timing")
    environment = _mapping(request.get("environment", {}), "bench.environment")
    return BenchRequest(
        case_id=_string(_require(request, "case_id", "bench"), "bench.case_id"),
        case_version=_positive_int(
            _require(request, "case_version", "bench"), "bench.case_version"
        ),
        reference=reference,
        tolerance=_parse_tolerance(_require(request, "tolerance", "bench")),
        params=params,
        scalars=_parse_scalars(_require(request, "scalars", "bench"), "bench.scalars", params),
        inputs=_parse_inputs(request.get("inputs", {}), "bench.inputs", params),
        output_name=_array_param(
            params, _string(_require(request, "output", "bench"), "bench.output"), "bench.output"
        ).name,
        base=base,
        head=head,
        warmup=_count(_require(timing, "warmup", "bench.timing"), "bench.timing.warmup"),
        samples=_positive_int(_require(timing, "samples", "bench.timing"), "bench.timing.samples"),
        sample_ms=_positive_number(
            _require(timing, "sample_ms", "bench.timing"), "bench.timing.sample_ms"
        ),
        base_commit=_optional_string(environment, "base_commit", "bench.environment"),
        head_commit=_optional_string(environment, "head_commit", "bench.environment"),
    )


def _parse_value_param(value: object, where: str) -> ScalarValue | ArrayValue:
    mapping = _mapping(value, where)
    param_type = _string(_require(mapping, "type", where), f"{where}.type")
    if param_type in CELL_DTYPES:
        raw = _require(mapping, "value", where)
        return ScalarValue(type=param_type, value=_scalar_value(param_type, raw, f"{where}.value"))
    if param_type not in ("i64[]", "f64[]"):
        msg = f"{where}.type: unsupported param type {param_type}"
        raise HarnessError(msg)
    raw_values = _require(mapping, "values", where)
    if not isinstance(raw_values, list) or not raw_values:
        msg = f"{where}.values: expected a non-empty array"
        raise HarnessError(msg)
    base_type = param_type[:-2]
    for index, item in enumerate(raw_values):
        if base_type == "i64" and (not isinstance(item, int) or isinstance(item, bool)):
            msg = f"{where}.values[{index}]: expected an integer, got {item!r}"
            raise HarnessError(msg)
        _number(item, f"{where}.values[{index}]")
    return ArrayValue(type=base_type, values=tuple(raw_values))


def _parse_correctness(request: dict[str, object]) -> CorrectnessRequest:
    raw_params = _require(request, "params", "correctness")
    if not isinstance(raw_params, list) or not raw_params:
        msg = "correctness.params: expected a non-empty array"
        raise HarnessError(msg)
    params = tuple(
        _parse_value_param(item, f"correctness.params[{index}]")
        for index, item in enumerate(raw_params)
    )
    output_param = _count(
        _require(request, "output_param", "correctness"), "correctness.output_param"
    )
    if output_param >= len(params) or not isinstance(params[output_param], ArrayValue):
        msg = "correctness.output_param: must index an array param"
        raise HarnessError(msg)
    raw_count = request.get("output_count")
    output_count = None
    if raw_count is not None:
        output_count = _positive_int(raw_count, "correctness.output_count")
    return CorrectnessRequest(
        ptx=_string(_require(request, "ptx", "correctness"), "correctness.ptx"),
        kernel=_string(_require(request, "kernel", "correctness"), "correctness.kernel"),
        params=params,
        launch=Launch(
            grid=_dims(_require(request, "grid", "correctness"), "correctness.grid"),
            block=_dims(_require(request, "block", "correctness"), "correctness.block"),
        ),
        output_param=output_param,
        output_count=output_count,
    )


def _run_correctness(request: dict[str, object], gpu: Gpu) -> dict[str, object]:
    spec = _parse_correctness(request)
    module = gpu.load_module(spec.ptx)
    allocated: list[object] = []
    try:
        func = gpu.function(module, spec.kernel)
        args: list[tuple[str, object]] = []
        ptr_of: dict[int, object] = {}
        for index, param in enumerate(spec.params):
            if isinstance(param, ScalarValue):
                args.append((param.type, param.value))
                continue
            data = np.asarray(param.values, dtype=CELL_DTYPES[param.type])
            ptr = gpu.alloc(data.nbytes)
            allocated.append(ptr)
            gpu.upload(ptr, data)
            args.append(("ptr", ptr))
            ptr_of[index] = ptr
        # Same first-launch workaround as bench validation: launch twice, read once.
        gpu.launch(func, spec.launch, args)
        gpu.synchronize()
        gpu.launch(func, spec.launch, args)
        gpu.synchronize()
        out = spec.params[spec.output_param]
        count = spec.output_count if spec.output_count is not None else len(out.values)
        data = gpu.download(ptr_of[spec.output_param], count, CELL_DTYPES[out.type])
    finally:
        for ptr in allocated:
            gpu.free(ptr)
        gpu.unload_module(module)
    if out.type == "i64":
        return {"ok": True, "values": [int(item) for item in data]}
    return {"ok": True, "values": [float(item) for item in data]}


@dataclass
class _Prepared:
    side: Side
    func: object | None


@dataclass
class _BenchRun:
    """Everything one bench request sets up, shared by validation and sampling."""

    spec: BenchRequest
    gpu: Gpu
    prepared: dict[str, _Prepared]
    args: list[tuple[str, object]]
    buffers: dict[str, tuple[object, np.ndarray]]

    def launch_side(self, side_name: str) -> None:
        prepared = self.prepared[side_name]
        side = prepared.side
        if isinstance(side, HostLibSide):
            self.gpu.dgemm(
                side.n,
                self.buffers[side.a][0],
                self.buffers[side.b][0],
                self.buffers[side.c][0],
            )
            return
        self.gpu.launch(prepared.func, side.launch, self.args)

    def time_side(self, side_name: str, launches: int) -> float:
        start = self.gpu.record_event()
        for _ in range(launches):
            self.launch_side(side_name)
        end = self.gpu.record_event()
        self.gpu.synchronize()
        return self.gpu.elapsed_us(start, end) / launches


def _prepare_side(gpu: Gpu, side: Side, modules: list[object]) -> _Prepared:
    if isinstance(side, HostLibSide):
        return _Prepared(side=side, func=None)
    module = gpu.load_module(side.ptx)
    modules.append(module)
    return _Prepared(side=side, func=gpu.function(module, side.kernel))


def _case_mapping(
    spec: BenchRequest, environment: dict[str, object], *, correct: bool, samples: list
) -> dict[str, object]:
    return {
        "case_id": spec.case_id,
        "case_version": spec.case_version,
        "correct": correct,
        "environment": environment,
        "samples": samples,
    }


def _prepare_buffers(
    spec: BenchRequest, gpu: Gpu, allocated: list[object]
) -> tuple[dict[str, tuple[object, np.ndarray]], dict[str, np.ndarray], dict[str, str]]:
    buffers: dict[str, tuple[object, np.ndarray]] = {}
    arrays: dict[str, np.ndarray] = {}
    digests = {}
    for param in spec.params:
        if not isinstance(param, ArrayParam):
            continue
        host = generate_buffer(param, spec.inputs.get(param.name))
        ptr = gpu.alloc(host.nbytes)
        allocated.append(ptr)
        gpu.upload(ptr, host)
        buffers[param.name] = (ptr, host)
        arrays[param.name] = _elements_of(param, host)
        digests[param.name] = hashlib.sha256(host.tobytes()).hexdigest()
    return buffers, arrays, digests


def _kernel_args(
    spec: BenchRequest, buffers: dict[str, tuple[object, np.ndarray]]
) -> list[tuple[str, object]]:
    args = []
    for param in spec.params:
        if isinstance(param, ArrayParam):
            args.append(("ptr", buffers[param.name][0]))
        else:
            args.append((param.type, spec.scalars[param.name]))
    return args


def _read_output(gpu: Gpu, param: ArrayParam, ptr: object) -> np.ndarray:
    cells = gpu.download(ptr, param.cells, CELL_DTYPES[param.type])
    if param.dtype == "f32":
        return cells.view(np.float32).astype(np.float64)
    return cells.astype(np.float64)


def _validate_sides(run: _BenchRun, arrays: dict[str, np.ndarray]) -> dict[str, bool]:
    spec = run.spec
    out_param = _array_param(spec.params, spec.output_name, "bench.output")
    out_ptr, out_host = run.buffers[spec.output_name]
    expected = REFERENCES[spec.reference](arrays, spec.scalars)
    if expected.size != out_param.elements:
        msg = (
            f"bench: the {spec.reference} reference produced {expected.size} values, "
            f"but output param {spec.output_name} holds {out_param.elements}"
        )
        raise HarnessError(msg)
    snapshot = run.gpu.alloc(out_host.nbytes)
    try:
        run.gpu.copy_device(snapshot, out_ptr, out_host.nbytes)
        validation = {}
        for side_name in ("base", "head"):
            # On the 13.2-era rental drivers the first launch of a freshly loaded
            # module returns success but silently runs nothing, so each side burns
            # one launch (which also lands the lazy JIT) before the measured one.
            run.launch_side(side_name)
            run.gpu.synchronize()
            run.gpu.copy_device(out_ptr, snapshot, out_host.nbytes)
            run.launch_side(side_name)
            run.gpu.synchronize()
            actual = _read_output(run.gpu, out_param, out_ptr)
            validation[side_name] = within_tolerance(actual, expected, *spec.tolerance)
    finally:
        run.gpu.free(snapshot)
    return validation


def _collect_samples(run: _BenchRun) -> tuple[list, dict[str, int]]:
    spec = run.spec
    for _ in range(spec.warmup):
        run.launch_side("base")
        run.launch_side("head")
    run.gpu.synchronize()
    batch = {}
    for side_name in ("base", "head"):
        single = run.time_side(side_name, 1)
        launches = math.ceil(spec.sample_ms * 1000.0 / max(single, MIN_CALIBRATION_US))
        batch[side_name] = min(max(launches, 1), MAX_BATCH_LAUNCHES)
    samples = []
    for index in range(spec.samples):
        order = ("base", "head") if index % 2 == 0 else ("head", "base")
        times = {}
        for side_name in order:
            times[side_name] = run.time_side(side_name, batch[side_name])
        samples.append({"base_us": times["base"], "head_us": times["head"], "first": order[0]})
    return samples, batch


def _run_bench(request: dict[str, object], gpu: Gpu) -> dict[str, object]:
    spec = _parse_bench(request)
    environment = gpu.environment()
    environment["base_commit"] = spec.base_commit
    environment["head_commit"] = spec.head_commit
    allocated: list[object] = []
    modules: list[object] = []
    try:
        buffers, arrays, digests = _prepare_buffers(spec, gpu, allocated)
        prepared = {
            "base": _prepare_side(gpu, spec.base, modules),
            "head": _prepare_side(gpu, spec.head, modules),
        }
        run = _BenchRun(
            spec=spec,
            gpu=gpu,
            prepared=prepared,
            args=_kernel_args(spec, buffers),
            buffers=buffers,
        )
        validation = _validate_sides(run, arrays)
        if not all(validation.values()):
            result = _case_mapping(spec, environment, correct=False, samples=[])
            return {
                "ok": True,
                "result": result,
                "input_digests": digests,
                "validation": validation,
            }
        samples, batch = _collect_samples(run)
        result = _case_mapping(spec, environment, correct=True, samples=samples)
        return {
            "ok": True,
            "result": result,
            "input_digests": digests,
            "validation": validation,
            "batch_launches": batch,
        }
    finally:
        for module in modules:
            gpu.unload_module(module)
        for ptr in allocated:
            gpu.free(ptr)


_GPU_CACHE: dict[str, CudaGpu] = {}


def default_gpu() -> Gpu:
    """The process-wide device, created on first use; tests substitute a fake."""
    if "device" not in _GPU_CACHE:
        _GPU_CACHE["device"] = CudaGpu()
    return _GPU_CACHE["device"]


def handle_request(request: dict[str, object], gpu: Gpu | None = None) -> dict[str, object]:
    """Serve one request, returning the response object. HarnessError becomes ok: false."""
    try:
        mapping = _mapping(request, "request")
        device = gpu if gpu is not None else default_gpu()
        op = _string(_require(mapping, "op", "request"), "request.op")
        if op == "ping":
            return {"ok": True, "environment": device.environment()}
        if op == "correctness":
            return _run_correctness(mapping, device)
        if op == "bench":
            return _run_bench(mapping, device)
    except HarnessError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": f"request.op: unknown op {op!r}"}


def _serve_line(line: str) -> dict[str, object]:
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"JSONDecodeError: {exc}"}
    if not isinstance(request, dict):
        return {"ok": False, "error": "each request line must be a JSON object"}
    try:
        return handle_request(request)
    except Exception as exc:  # noqa: BLE001 - a failure must answer, never kill the stream
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        sys.stdout.write(json.dumps(_serve_line(line)) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
