"""Unit tests for the JSON GPU runner protocol."""

from __future__ import annotations

import base64
import ctypes
import io
import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from gpu_test import runner
from gpu_test.conftest import KernelRunner

if TYPE_CHECKING:
    import pytest


class FakeCuda:
    """Minimal successful CUDA Driver API implementation backed by host bytes."""

    class CUresult:
        CUDA_SUCCESS = 0

    def __init__(self) -> None:
        self.next_pointer = 100
        self.memory: dict[int, bytes] = {}
        self.launch_params: object = None
        self.CUresult = SimpleNamespace(CUDA_SUCCESS=0)
        methods = {
            "cuInit": self._init,
            "cuDeviceGet": self._device_get,
            "cuCtxCreate": self._context_create,
            "cuModuleLoadData": self._module_load_data,
            "cuModuleGetFunction": self._module_get_function,
            "cuMemAlloc": self._mem_alloc,
            "cuMemcpyHtoD": self._memcpy_host_to_device,
            "cuLaunchKernel": self._launch_kernel,
            "cuCtxSynchronize": self._context_synchronize,
            "cuMemcpyDtoH": self._memcpy_device_to_host,
            "cuMemFree": self._mem_free,
            "cuModuleUnload": self._module_unload,
            "cuCtxDestroy": self._context_destroy,
        }
        for name, method in methods.items():
            setattr(self, name, method)

    def _init(self, _flags: int) -> tuple[int]:
        return (0,)

    def _device_get(self, _ordinal: int) -> tuple[int, int]:
        return 0, 1

    def _context_create(self, *_args: object) -> tuple[int, int]:
        return 0, 2

    def _module_load_data(self, _ptx: object) -> tuple[int, int]:
        return 0, 3

    def _module_get_function(self, _module: int, _name: bytes) -> tuple[int, int]:
        return 0, 4

    def _mem_alloc(self, size: int) -> tuple[int, int]:
        pointer = self.next_pointer
        self.next_pointer += 1
        self.memory[pointer] = bytes(size)
        return 0, pointer

    def _memcpy_host_to_device(self, pointer: int, source: int, size: int) -> tuple[int]:
        self.memory[pointer] = ctypes.string_at(source, size)
        return (0,)

    def _launch_kernel(self, *_args: object) -> tuple[int]:
        self.launch_params = _args[-2]
        return (0,)

    def _context_synchronize(self) -> tuple[int]:
        return (0,)

    def _memcpy_device_to_host(self, destination: int, pointer: int, size: int) -> tuple[int]:
        ctypes.memmove(destination, self.memory[pointer], size)
        return (0,)

    def _mem_free(self, _pointer: int) -> tuple[int]:
        return (0,)

    def _module_unload(self, _module: int) -> tuple[int]:
        return (0,)

    def _context_destroy(self, _context: int) -> tuple[int]:
        return (0,)


def test_runner_supports_typed_params_and_multiple_outputs() -> None:
    cuda = FakeCuda()
    request = {
        "ptx_base64": base64.b64encode(b"// PTX").decode(),
        "kernel": "main",
        "grid": [2, 1, 1],
        "block": [32, 1, 1],
        "params": [
            {"type": "i64[]", "values": [1, 2, 3]},
            {"type": "f64", "value": 3.5},
            {"type": "f64[]", "values": [4.25, 5.5]},
        ],
        "outputs": [{"param": 0, "count": 2}, {"param": 2}],
    }

    assert runner.execute(request, cuda) == [
        {"param": 0, "type": "i64[]", "values": [1, 2]},
        {"param": 2, "type": "f64[]", "values": [4.25, 5.5]},
    ]
    launch_params = cast("tuple[tuple[object, ...], tuple[object, ...]]", cuda.launch_params)
    assert launch_params[0][1] == 3.5
    assert launch_params[1][1] is ctypes.c_double


def test_runner_reports_invalid_requests_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    monkeypatch.setattr(sys, "stdout", output)

    assert runner.main() == 0
    assert json.loads(output.getvalue()) == {
        "status": "error",
        "error": "ptx_base64 must be a non-empty string",
    }


class FakeCompiler:
    def compile_source(self, _source: str) -> str:
        return "// compiled PTX"


class FakeSession:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def ssh_run(self, _command: str, *, input_text: str, timeout: int) -> str:
        assert timeout == 120
        self.request = json.loads(input_text)
        return json.dumps(
            {
                "status": "ok",
                "outputs": [{"param": 0, "type": "i64[]", "values": [7, 8]}],
            }
        )


def test_kernel_runner_sends_json_request() -> None:
    session = FakeSession()
    runner = KernelRunner(session, FakeCompiler())
    source = """\\! kernel main
\\! param output i64[3]
\\! param scale f64
"""

    result = runner.run(
        source,
        params={"output": [7, 8], "scale": 2.5},
        grid=(3, 2, 1),
        block=(16, 1, 1),
        output_count=2,
    )

    assert result == [7, 8]
    assert session.request == {
        "ptx_base64": base64.b64encode(b"// compiled PTX").decode(),
        "kernel": "main",
        "grid": [3, 2, 1],
        "block": [16, 1, 1],
        "params": [
            {"type": "i64[]", "values": [7, 8, 0]},
            {"type": "f64", "value": 2.5},
        ],
        "outputs": [{"param": 0, "count": 2}],
    }
