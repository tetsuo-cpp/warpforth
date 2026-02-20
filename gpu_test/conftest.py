"""GPU test infrastructure: Compiler, VastSession, KernelRunner, and fixtures."""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from vastai import VastAI

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Self

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WARPFORTHC = PROJECT_ROOT / "build" / "bin" / "warpforthc"
RUNNER_SRC = PROJECT_ROOT / "tools" / "warpforth-runner" / "warpforth-runner.cpp"

MAX_COST_PER_HOUR = 0.50
POLL_INTERVAL_S = 10
POLL_TIMEOUT_S = 300
INSTANCE_LABEL = "warpforth-test"
REMOTE_TMP = "/tmp"  # noqa: S108


@dataclass
class ParamDecl:
    """A parsed kernel parameter declaration."""

    name: str
    is_array: bool
    size: int  # 0 for scalars
    base_type: str = "i64"  # "i64" or "f64"


class CompileError(Exception):
    """Raised when warpforthc fails to compile Forth source."""


class Compiler:
    """Wraps the local warpforthc binary to compile Forth source to PTX."""

    def __init__(self, binary: Path = WARPFORTHC) -> None:
        if not binary.exists():
            msg = f"warpforthc not found at {binary}; run: cmake --build build"
            raise FileNotFoundError(msg)
        self.binary = binary

    def compile_source(self, forth_src: str) -> str:
        """Compile Forth source code to PTX, returning PTX as a string."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".forth", delete=False) as f:
            f.write(forth_src)
            src_path = Path(f.name)

        try:
            result = subprocess.run(
                [self.binary, src_path],
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


class VastSession:
    """Context manager that rents a GPU instance on Vast.ai.

    Handles searching for offers, launching an instance, waiting for SSH
    readiness, and unconditional cleanup on exit.
    """

    def __init__(self, api_key: str) -> None:
        self.sdk = VastAI(api_key)
        self.instance_id: int | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None

    def __enter__(self) -> Self:
        self._cleanup_orphans()
        self._launch()
        self._wait_for_ssh()
        atexit.register(self._atexit_cleanup)
        return self

    def __exit__(self, *_: object) -> None:
        self._destroy()

    def _cleanup_orphans(self) -> None:
        """Destroy any leftover instances with our label."""
        try:
            instances = self.sdk.show_instances()
            if not isinstance(instances, list):
                return
            for inst in instances:
                if isinstance(inst, dict) and inst.get("label") == INSTANCE_LABEL:
                    logger.warning("Destroying orphan instance %s", inst.get("id"))
                    try:
                        self.sdk.destroy_instance(id=int(inst["id"]))
                    except Exception:
                        logger.exception("Failed to destroy orphan %s", inst.get("id"))
        except Exception:
            logger.exception("Failed to check for orphans")

    def _launch(self) -> None:
        """Find the cheapest suitable offer and launch an instance."""
        query = f"num_gpus=1 rentable=True rented=False compute_cap>=700 dph<={MAX_COST_PER_HOUR}"
        offers = self.sdk.search_offers(query=query, order="dph", limit=5)

        if not offers:
            msg = "No suitable GPU offers found on Vast.ai"
            raise RuntimeError(msg)

        # Pick the cheapest offer
        offer = offers[0] if isinstance(offers, list) else offers
        offer_id = int(offer["id"]) if isinstance(offer, dict) else int(offer)

        logger.info("Launching instance from offer %s", offer_id)

        # sdk.create_instance() has a bug where it doesn't return the response,
        # so we find our instance by label via show_instances() instead.
        self.sdk.create_instance(
            id=offer_id,
            image="nvidia/cuda:12.4.0-devel-ubuntu22.04",
            disk=10.0,
            ssh=True,
            direct=True,
            label=INSTANCE_LABEL,
        )

        instances = self.sdk.show_instances()
        if isinstance(instances, list):
            for inst in instances:
                if isinstance(inst, dict) and inst.get("label") == INSTANCE_LABEL:
                    self.instance_id = int(inst["id"])
                    break

        if self.instance_id is None:
            msg = "Instance creation failed — not found in show_instances"
            raise RuntimeError(msg)

        logger.info("Instance %s created", self.instance_id)

    def _wait_for_ssh(self) -> None:
        """Poll until the instance is running and SSH is available."""
        deadline = time.monotonic() + POLL_TIMEOUT_S

        while time.monotonic() < deadline:
            instances = self.sdk.show_instances()
            if not isinstance(instances, list):
                time.sleep(POLL_INTERVAL_S)
                continue

            for inst in instances:
                if not isinstance(inst, dict):
                    continue
                if int(inst.get("id", -1)) != self.instance_id:
                    continue

                status = inst.get("actual_status", inst.get("status", ""))
                if status == "running" and inst.get("ssh_host") and inst.get("ssh_port"):
                    self.ssh_host = inst["ssh_host"]
                    self.ssh_port = int(inst["ssh_port"])
                    logger.info(
                        "Instance %s ready: %s:%s",
                        self.instance_id,
                        self.ssh_host,
                        self.ssh_port,
                    )
                    # Upload and compile the C++ runner
                    self._compile_runner()
                    return

            time.sleep(POLL_INTERVAL_S)

        msg = f"Instance {self.instance_id} did not become SSH-ready within {POLL_TIMEOUT_S}s"
        raise TimeoutError(msg)

    def _compile_runner(self) -> None:
        """Upload warpforth-runner.cpp and compile it on the remote host."""
        self.scp_upload(RUNNER_SRC, f"{REMOTE_TMP}/warpforth-runner.cpp")
        nvcc_cmd = (
            f"nvcc -o {REMOTE_TMP}/warpforth-runner"
            f" {REMOTE_TMP}/warpforth-runner.cpp -lcuda -std=c++17"
        )
        self.ssh_run(nvcc_cmd, timeout=60)

    def _destroy(self) -> None:
        """Destroy the instance, retrying up to 3 times."""
        if self.instance_id is None:
            return

        retries = 3
        for attempt in range(retries):
            try:
                self.sdk.destroy_instance(id=self.instance_id)
            except Exception:
                logger.exception(
                    "Destroy attempt %d/%d failed for instance %s",
                    attempt + 1,
                    retries,
                    self.instance_id,
                )
                if attempt < retries - 1:
                    time.sleep(5)
            else:
                logger.info("Instance %s destroyed", self.instance_id)
                self.instance_id = None
                return

        logger.error("Failed to destroy instance %s after %d attempts!", self.instance_id, retries)

    def _atexit_cleanup(self) -> None:
        """Last-resort cleanup registered with atexit."""
        if self.instance_id is not None:
            logger.warning("atexit: destroying instance %s", self.instance_id)
            self._destroy()

    def _ssh_cmd(self) -> list[str]:
        """Build base SSH command with connection options."""
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "LogLevel=ERROR",
            "-p",
            str(self.ssh_port),
            f"root@{self.ssh_host}",
        ]

    def ssh_run(self, cmd: str, *, timeout: int = 120) -> str:
        """Execute a command on the remote instance via SSH."""
        result = subprocess.run(
            [*self._ssh_cmd(), cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            msg = f"SSH command failed (rc={result.returncode}):\n{result.stderr}"
            raise RuntimeError(msg)
        return result.stdout

    def scp_upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a file to the remote instance via SCP."""
        subprocess.run(
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "LogLevel=ERROR",
                "-P",
                str(self.ssh_port),
                str(local_path),
                f"root@{self.ssh_host}:{remote_path}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )


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
            if len(parts) < 2:
                msg = "Invalid header line: expected '\\! kernel <name>'"
                raise ValueError(msg)
            return parts[1]
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
        if len(parts) < 3:
            msg = "Invalid header line: expected '\\! param <name> <type>'"
            raise ValueError(msg)
        name = parts[1]
        type_spec = parts[2]
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


class KernelRunner:
    """Orchestrates local compilation and remote GPU execution."""

    def __init__(self, session: VastSession, compiler: Compiler) -> None:
        self.session = session
        self.compiler = compiler

    def run(
        self,
        forth_source: str,
        params: dict[str, list[int] | list[float] | int | float] | None = None,
        grid: tuple[int, int, int] = (1, 1, 1),
        block: tuple[int, int, int] = (1, 1, 1),
        output_param: int = 0,
        output_count: int | None = None,
    ) -> list[int] | list[float]:
        """Compile Forth source locally, execute on remote GPU, return output values.

        Param buffer sizes are derived from the Forth source's 'param' declarations.
        The params dict maps param names to initial values:
          - Array params: list of int or float (padded with zeros to declared size)
          - Scalar params: int or float
        Params not in the dict are zero-initialized.
        """
        # Parse kernel name and param declarations
        kernel_name = _parse_kernel_name(forth_source)
        decls = _parse_param_declarations(forth_source)
        if not decls:
            msg = "Forth source has no '\\! param' declarations"
            raise ValueError(msg)

        params = params or {}

        # Validate output_param
        if output_param < 0 or output_param >= len(decls):
            msg = f"output_param {output_param} out of range (have {len(decls)} params)"
            raise ValueError(msg)
        if not decls[output_param].is_array:
            name = decls[output_param].name
            msg = f"output_param {output_param} ('{name}') is a scalar and cannot be read back"
            raise ValueError(msg)

        # Compile locally
        ptx = self.compiler.compile_source(forth_source)

        # Write PTX to temp file and upload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ptx", delete=False) as f:
            f.write(ptx)
            ptx_path = Path(f.name)

        try:
            self.session.scp_upload(ptx_path, f"{REMOTE_TMP}/kernel.ptx")
        finally:
            ptx_path.unlink()

        # Build remote command
        cmd_parts = [
            f"{REMOTE_TMP}/warpforth-runner",
            f"{REMOTE_TMP}/kernel.ptx",
            "--kernel",
            kernel_name,
        ]

        for decl in decls:
            if decl.is_array:
                values = params.get(decl.name, [])
                if not isinstance(values, list):
                    msg = f"Array param '{decl.name}' expects a list, got {type(values).__name__}"
                    raise TypeError(msg)
                zero = 0.0 if decl.base_type == "f64" else 0
                buf = [zero] * decl.size
                for i, v in enumerate(values):
                    buf[i] = v
                cmd_parts.extend(["--param", f"{decl.base_type}[]:{','.join(str(v) for v in buf)}"])
            else:
                value = params.get(decl.name, 0.0 if decl.base_type == "f64" else 0)
                if isinstance(value, list):
                    msg = f"Scalar param '{decl.name}' expects a scalar, got list"
                    raise TypeError(msg)
                cmd_parts.extend(["--param", f"{decl.base_type}:{value}"])

        cmd_parts.extend(
            [
                "--grid",
                f"{grid[0]},{grid[1]},{grid[2]}",
                "--block",
                f"{block[0]},{block[1]},{block[2]}",
                "--output-param",
                str(output_param),
            ]
        )

        if output_count is not None:
            cmd_parts.extend(["--output-count", str(output_count)])

        cmd = " ".join(cmd_parts)
        stdout = self.session.ssh_run(cmd, timeout=120)

        # Parse CSV output — type depends on the output param
        out_type = decls[output_param].base_type
        parse = float if out_type == "f64" else int
        return [parse(v) for v in stdout.strip().split(",")]


# --- Fixtures ---


@pytest.fixture(scope="session")
def compiler() -> Compiler:
    return Compiler()


@pytest.fixture(scope="session")
def gpu_session() -> Generator[VastSession]:
    api_key = os.environ.get("VASTAI_API_KEY")
    if not api_key:
        pytest.skip("VASTAI_API_KEY not set")

    with VastSession(api_key) as session:
        yield session


@pytest.fixture(scope="session")
def kernel_runner(gpu_session: VastSession, compiler: Compiler) -> KernelRunner:
    return KernelRunner(gpu_session, compiler)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.gpu tests when VASTAI_API_KEY is not set."""
    if os.environ.get("VASTAI_API_KEY"):
        return

    skip_gpu = pytest.mark.skip(reason="VASTAI_API_KEY not set")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
