"""GPU test infrastructure: Compiler, KernelRunner, and fixtures."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from warpforth_gpu.client import Device, worker_command
from warpforth_gpu.vast import DEFAULT_ARCHITECTURE, VastSession

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from warpforth_gpu.client import Buffer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WARPFORTHC = PROJECT_ROOT / "build" / "bin" / "warpforthc"

SUPPORTED_ARCHITECTURES = ("sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90")


@dataclass
class ParamDecl:
    """A parsed kernel parameter declaration."""

    name: str
    is_array: bool
    size: int  # 0 for scalars
    base_type: str = "i64"  # "i64" or "f64"

    @property
    def dtype(self) -> type[np.int64 | np.float64]:
        return np.float64 if self.base_type == "f64" else np.int64


class CompileError(Exception):
    """Raised when warpforthc fails to compile Forth source."""


class Compiler:
    """Wraps the local warpforthc binary to compile Forth source to PTX."""

    def __init__(self, binary: Path = WARPFORTHC, arch: str = DEFAULT_ARCHITECTURE) -> None:
        if not binary.exists():
            msg = f"warpforthc not found at {binary}; run: cmake --build build"
            raise FileNotFoundError(msg)
        self.binary = binary
        self.arch = arch

    def compile_source(self, forth_src: str) -> str:
        """Compile Forth source code to PTX, returning PTX as a string."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".forth", delete=False) as f:
            f.write(forth_src)
            src_path = Path(f.name)

        try:
            result = subprocess.run(
                [self.binary, src_path, "--arch", self.arch],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode != 0:
                msg = f"warpforthc failed:\n{result.stderr}"
                raise CompileError(msg)
            return result.stdout
        finally:
            src_path.unlink()


def _parse_array_type(type_spec: str) -> tuple[str, int]:
    """Parse 'i64[256]' or 'f64[256]' into (base_type, size)."""
    if not type_spec.endswith("]"):
        msg = f"Invalid array type spec: {type_spec}"
        raise ValueError(msg)
    base, size_str = type_spec[:-1].split("[", 1)
    base_lower = base.lower()
    if base_lower not in ("i64", "f64"):
        msg = f"Unsupported base type: {base}"
        raise ValueError(msg)
    return base_lower, int(size_str)


def _iter_header_directives(forth_source: str) -> Generator[tuple[str, list[str]]]:
    """Yield (keyword, parts) for each \\! directive in the Forth header.

    Strips comments (-- ...) and splits on whitespace. keyword is lowercased.
    """
    for line in forth_source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("\\!"):
            continue
        directive = stripped[2:].strip()
        if "--" in directive:
            directive = directive.split("--", 1)[0].strip()
        if not directive:
            continue
        parts = directive.split()
        if parts:
            yield parts[0].lower(), parts


def _parse_kernel_name(forth_source: str) -> str:
    """Parse '\\! kernel <name>' from Forth source header."""
    for keyword, parts in _iter_header_directives(forth_source):
        if keyword == "kernel":
            try:
                return parts[1]
            except IndexError:
                msg = "Invalid header line: expected '\\! kernel <name>'"
                raise ValueError(msg) from None
    msg = "Forth source has no '\\! kernel' declaration"
    raise ValueError(msg)


def _parse_param_declarations(forth_source: str) -> list[ParamDecl]:
    """Parse '\\! param <name> <type>' declarations from Forth source.

    Returns list of ParamDecl in declaration order. Supports both array
    params (e.g. i64[256]) and scalar params (e.g. i64).
    """
    decls: list[ParamDecl] = []
    for keyword, parts in _iter_header_directives(forth_source):
        if keyword != "param":
            continue
        try:
            name, type_spec = parts[1:3]
        except ValueError:
            msg = "Invalid header line: expected '\\! param <name> <type>'"
            raise ValueError(msg) from None
        if "[" in type_spec:
            base_type, size = _parse_array_type(type_spec)
            decls.append(ParamDecl(name=name, is_array=True, size=size, base_type=base_type))
        else:
            base_type = type_spec.lower()
            if base_type not in ("i64", "f64"):
                msg = f"Unsupported scalar type: {type_spec}"
                raise ValueError(msg)
            decls.append(ParamDecl(name=name, is_array=False, size=0, base_type=base_type))
    return decls


def _initial_values(
    decls: list[ParamDecl],
    params: dict[str, list[int] | list[float] | int | float],
) -> list[np.ndarray | float]:
    """Return each param's starting value: a zero-padded array, or a scalar."""
    values: list[np.ndarray | float] = []
    for decl in decls:
        given = params.get(decl.name)
        if decl.is_array:
            if given is None:
                given = []
            if not isinstance(given, list):
                msg = f"Array param '{decl.name}' expects a list, got {type(given).__name__}"
                raise TypeError(msg)
            array = np.zeros(decl.size, dtype=decl.dtype)
            array[: len(given)] = given
            values.append(array)
        else:
            if isinstance(given, list):
                msg = f"Scalar param '{decl.name}' expects a scalar, got list"
                raise TypeError(msg)
            given = given or 0
            values.append(float(given) if decl.base_type == "f64" else given)
    return values


def open_vast_device(session: VastSession) -> Device:
    """Start a device worker on the rented instance, streaming its source over SSH."""
    return Device.spawn([*session.ssh_argv(), shlex.join(worker_command())])


class KernelRunner:
    """Compiles Forth locally and runs it on one device worker, reopened after a fatal error."""

    def __init__(self, open_device: Callable[[], Device], compiler: Compiler) -> None:
        self._open_device = open_device
        self._device: Device | None = None
        self.compiler = compiler

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None

    def _ready_device(self) -> Device:
        if self._device is None or self._device.closed:
            self._device = self._open_device()
        return self._device

    def run(
        self,
        forth_source: str,
        params: dict[str, list[int] | list[float] | int | float] | None = None,
        grid: tuple[int, int, int] = (1, 1, 1),
        block: tuple[int, int, int] = (1, 1, 1),
        output_param: int = 0,
        output_count: int | None = None,
    ) -> list[int] | list[float]:
        """Compile Forth source locally, execute on the GPU, return output values.

        Param buffer sizes are derived from the Forth source's 'param' declarations.
        The params dict maps param names to initial values:
          - Array params: list of int or float (padded with zeros to declared size)
          - Scalar params: int or float
        Params not in the dict are zero-initialized.
        """
        kernel_name = _parse_kernel_name(forth_source)
        decls = _parse_param_declarations(forth_source)
        if not decls:
            msg = "Forth source has no '\\! param' declarations"
            raise ValueError(msg)

        params = params or {}

        if output_param < 0 or output_param >= len(decls):
            msg = f"output_param {output_param} out of range (have {len(decls)} params)"
            raise ValueError(msg)
        if not decls[output_param].is_array:
            name = decls[output_param].name
            msg = f"output_param {output_param} ('{name}') is a scalar and cannot be read back"
            raise ValueError(msg)

        initial = _initial_values(decls, params)
        ptx = self.compiler.compile_source(forth_source)
        device = self._ready_device()
        kernel = device.load(ptx).kernel(kernel_name)

        buffers: dict[int, Buffer] = {}
        try:
            for index, value in enumerate(initial):
                if isinstance(value, np.ndarray):
                    buffers[index] = device.upload(value)
            args = [buffers.get(index, value) for index, value in enumerate(initial)]
            kernel.launch(grid=grid, block=block, args=args)
            output = buffers[output_param].read(decls[output_param].dtype)
            return output[:output_count].tolist()
        finally:
            if not device.closed:
                for buffer in buffers.values():
                    buffer.free()


# --- Fixtures ---


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--arch",
        choices=SUPPORTED_ARCHITECTURES,
        default=DEFAULT_ARCHITECTURE,
        help="GPU architecture used for PTX compilation and Vast.ai selection",
    )


@pytest.fixture(scope="session")
def compiler(pytestconfig: pytest.Config) -> Compiler:
    return Compiler(arch=pytestconfig.getoption("arch"))


@pytest.fixture(scope="session")
def gpu_session(pytestconfig: pytest.Config) -> Generator[VastSession]:
    api_key = os.environ.get("VASTAI_API_KEY")
    if not api_key:
        pytest.skip("VASTAI_API_KEY not set")

    with VastSession(api_key, arch=pytestconfig.getoption("arch")) as session:
        yield session


@pytest.fixture(scope="session")
def kernel_runner(gpu_session: VastSession, compiler: Compiler) -> Generator[KernelRunner]:
    runner = KernelRunner(lambda: open_vast_device(gpu_session), compiler)
    yield runner
    runner.close()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.gpu tests when VASTAI_API_KEY is not set."""
    if os.environ.get("VASTAI_API_KEY"):
        return

    skip_gpu = pytest.mark.skip(reason="VASTAI_API_KEY not set")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
