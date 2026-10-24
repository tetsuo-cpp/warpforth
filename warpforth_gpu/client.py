"""Client for the device worker: load PTX, move buffers, and launch kernels on a GPU."""

from __future__ import annotations

import base64
import contextlib
import itertools
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from numpy.typing import DTypeLike

WORKER_SOURCE = Path(__file__).with_name("worker.py")
_CLOSE_TIMEOUT_S = 30

CLOSED = "CLOSED"
PROTOCOL = "PROTOCOL"
WORKER_EXITED = "WORKER_EXITED"


def worker_command(python: str = "python3") -> list[str]:
    """Return argv that runs the worker from its inlined source, so no file must be copied."""
    return [python, "-u", "-c", WORKER_SOURCE.read_text()]


class DeviceError(Exception):
    """The worker rejected a request, hit a CUDA error, or exited."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Device:
    """One worker process and the GPU context it owns. Requests are synchronous."""

    def __init__(self, process: subprocess.Popen[bytes], stderr: IO[bytes]) -> None:
        if process.stdin is None or process.stdout is None:
            msg = "the worker process needs piped stdin and stdout"
            raise ValueError(msg)
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._stderr = stderr
        self._stderr_text = ""
        self._ids = itertools.count(1)
        self._closed = False

    @classmethod
    def spawn(cls, argv: Sequence[str]) -> Device:
        """Start any command that speaks the worker protocol on its stdin and stdout."""
        # A file rather than a pipe, so a chatty worker cannot block on a full stderr pipe.
        stderr = tempfile.TemporaryFile()  # noqa: SIM115
        process = subprocess.Popen(  # noqa: S603
            list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr
        )
        return cls(process, stderr)

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the worker's stdin and wait for it to exit."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(BrokenPipeError):
            self._stdin.close()
        try:
            self._process.wait(timeout=_CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._stdout.close()
        self._stderr.seek(0)
        self._stderr_text = self._stderr.read().decode(errors="replace").strip()
        self._stderr.close()

    def _fail(self, code: str, message: str) -> DeviceError:
        """Close the device and build an error that carries the worker's stderr."""
        self.close()
        if self._stderr_text:
            message += f"\nworker stderr:\n{self._stderr_text}"
        return DeviceError(code, message)

    def _exited(self) -> DeviceError:
        self.close()  # the exit status is known only after the process is reaped
        return self._fail(WORKER_EXITED, f"worker exited with status {self._process.returncode}")

    def request(self, op: str, **fields: object) -> dict:
        """Send one request and return its result, raising DeviceError on any failure."""
        if self._closed:
            msg = "device is closed"
            raise DeviceError(CLOSED, msg)
        request_id = next(self._ids)
        line = json.dumps({"id": request_id, "op": op, **fields}).encode() + b"\n"
        try:
            self._stdin.write(line)
            self._stdin.flush()
        except BrokenPipeError:
            raise self._exited() from None
        reply = self._stdout.readline()
        if not reply:
            raise self._exited()
        try:
            response = json.loads(reply)
        except json.JSONDecodeError:
            raise self._fail(PROTOCOL, f"worker sent a line that is not JSON: {reply!r}") from None
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise self._fail(PROTOCOL, f"expected a reply to request {request_id}, got {reply!r}")
        if response["ok"]:
            return response["result"]
        error = response["error"]
        if response.get("fatal"):
            raise self._fail(error["code"], error["message"])
        raise DeviceError(error["code"], error["message"])

    def load(self, ptx: str) -> Module:
        result = self.request("load", ptx=ptx)
        kernels = {name: tuple(kinds) for name, kinds in result["kernels"].items()}
        return Module(self, result["module"], kernels)

    def alloc(self, size: int) -> Buffer:
        """Allocate ``size`` bytes of zeroed device memory."""
        return Buffer(self, self.request("alloc", size=size)["buffer"], size)

    def upload(self, array: np.ndarray) -> Buffer:
        """Allocate a buffer the size of ``array`` and copy it in."""
        buffer = self.alloc(array.nbytes)
        buffer.write(array)
        return buffer


@dataclass(frozen=True)
class Buffer:
    device: Device = field(repr=False)
    id: int
    size: int

    def write(self, data: bytes | np.ndarray) -> None:
        """Copy ``data`` to the start of the buffer. It may be shorter than the buffer."""
        raw = np.ascontiguousarray(data).tobytes() if isinstance(data, np.ndarray) else data
        self.device.request("write", buffer=self.id, data=base64.b64encode(raw).decode())

    def read(self, dtype: DTypeLike) -> np.ndarray:
        """Copy the whole buffer back as an array of ``dtype``."""
        data = base64.b64decode(self.device.request("read", buffer=self.id)["data"])
        return np.frombuffer(bytearray(data), dtype=dtype)

    def free(self) -> None:
        self.device.request("free", buffer=self.id)


@dataclass(frozen=True)
class Module:
    device: Device = field(repr=False)
    id: int
    kernels: dict[str, tuple[str, ...]]

    def kernel(self, name: str) -> Kernel:
        if name not in self.kernels:
            msg = f"module has no kernel {name!r}; it has {sorted(self.kernels)}"
            raise KeyError(msg)
        return Kernel(self, name, self.kernels[name])


@dataclass(frozen=True)
class Kernel:
    module: Module
    name: str
    kinds: tuple[str, ...]

    def launch(
        self,
        *,
        grid: Sequence[int] = (1, 1, 1),
        block: Sequence[int] = (1, 1, 1),
        args: Sequence[Buffer | float | np.generic] = (),
    ) -> None:
        """Launch the kernel and wait for it to finish."""
        self.module.device.request(
            "launch",
            module=self.module.id,
            kernel=self.name,
            grid=[int(n) for n in grid],
            block=[int(n) for n in block],
            args=[_encode_arg(arg) for arg in args],
        )


def _encode_arg(arg: Buffer | float | np.generic) -> dict[str, object]:
    if isinstance(arg, Buffer):
        return {"buffer": arg.id}
    if isinstance(arg, np.generic):
        return {"value": arg.item()}
    return {"value": arg}
