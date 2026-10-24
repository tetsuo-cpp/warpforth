"""In-memory stand-in for the CUDA driver behind the device worker."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from warpforth_gpu.worker import CudaError

if TYPE_CHECKING:
    from warpforth_gpu.worker import Holder


@dataclass
class Launch:
    kernel: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    args: list[tuple[type, int | float]]


class FakeDriver:
    """Allocations are zeroed bytearrays; launches are recorded instead of run."""

    def __init__(self, *, load_error: str | None = None, launch_error: str | None = None) -> None:
        self.load_error = load_error
        self.launch_error = launch_error
        self.memory: dict[int, bytearray] = {}
        self.pointers: list[int] = []
        self.launches: list[Launch] = []
        self._handles = itertools.count(0x1000, 0x100)
        self._functions: dict[int, str] = {}

    def load(self, _ptx: str) -> int:
        if self.load_error is not None:
            raise CudaError(self.load_error, "ptxas fatal: parse error")
        return next(self._handles)

    def function(self, _module: int, name: str) -> int:
        handle = next(self._handles)
        self._functions[handle] = name
        return handle

    def alloc(self, size: int) -> int:
        pointer = next(self._handles)
        self.memory[pointer] = bytearray(size)
        self.pointers.append(pointer)
        return pointer

    def write(self, pointer: int, data: bytes) -> None:
        self.memory[pointer][: len(data)] = data

    def read(self, pointer: int, size: int) -> bytes:
        return bytes(self.memory[pointer][:size])

    def free(self, pointer: int) -> None:
        del self.memory[pointer]

    def launch(
        self,
        function: int,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[Holder],
    ) -> None:
        if self.launch_error is not None:
            raise CudaError(self.launch_error, "an illegal memory access was encountered")
        packed = [(type(arg), arg.value) for arg in args]
        self.launches.append(Launch(self._functions[function], grid, block, packed))
