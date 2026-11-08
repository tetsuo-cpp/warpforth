"""GPU test infrastructure: Compiler, VastSession, and KernelRunner."""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from requests import RequestException
from vastai import VastAI

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Self

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WARPFORTHC = PROJECT_ROOT / "build" / "bin" / "warpforthc"
RUNNER_SRC = PROJECT_ROOT / "tools" / "warpforth-runner" / "warpforth-runner.cpp"

MAX_COST_PER_HOUR = 0.50
CUDA_VERSION = "12.4"
CUDA_IMAGE = f"nvidia/cuda:{CUDA_VERSION}.0-devel-ubuntu22.04"
POLL_INTERVAL_S = 10
# Roughly half of overnight Vast launches exceeded the old 5-minute budget while pulling
# the image and booting; the retry path below only helps if the poll outlives slow boots.
POLL_TIMEOUT_S = 600
# The ssh probe options (ConnectTimeout=10 x ConnectionAttempts=3) can need ~30s, so a
# 15s subprocess cap crashed sessions that a retry would have saved.
SSH_PROBE_TIMEOUT_S = 45
INSTANCE_LABEL_PREFIX = "warpforth-test-"
REMOTE_TMP = "/tmp"  # noqa: S108
DEFAULT_ARCHITECTURE = "sm_70"
SUPPORTED_ARCHITECTURES = ("sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90")
HARNESS_SRC = PROJECT_ROOT / "gpu_test" / "remote_harness.py"
HARNESS_REMOTE = f"{REMOTE_TMP}/remote_harness.py"
# Rentals ship 12.x-era drivers (cuda_max_good>=12.4); cuda-python 13 bindings want a
# 13-era driver, so provisioning pins to the 12.x line.
CUDA_PYTHON_SPEC = "cuda-python>=12.4,<13"


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


class VastSession:
    """Context manager that rents a GPU instance on Vast.ai.

    Handles searching for offers, launching an instance, waiting for SSH
    readiness, and unconditional cleanup on exit.
    """

    def __init__(self, api_key: str, arch: str = DEFAULT_ARCHITECTURE) -> None:
        self.sdk = VastAI(api_key, retry=1)
        self.arch = arch
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        self.instance_label = f"{INSTANCE_LABEL_PREFIX}{timestamp}-{uuid4().hex}"
        self.instance_id: int | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None
        self._atexit_registered = False
        self._creation_ambiguous = False
        self._ssh_key_dir: tempfile.TemporaryDirectory[str] | None = None
        self.ssh_private_key: Path | None = None

    def __enter__(self) -> Self:
        self._log_existing_instances()
        logger.info("This GPU test session owns label=%s", self.instance_label)
        atexit.register(self._atexit_cleanup)
        self._atexit_registered = True
        try:
            self._prepare_ssh_key()
            self._launch()
            self._wait_for_ssh()
        except BaseException:
            try:
                if self._destroy():
                    self._unregister_atexit()
            finally:
                self._cleanup_ssh_key()
            raise
        else:
            return self

    def __exit__(self, *_: object) -> None:
        try:
            if self._destroy():
                self._unregister_atexit()
        finally:
            self._cleanup_ssh_key()

    def _prepare_ssh_key(self) -> None:
        """Generate an ephemeral SSH identity for this test session."""
        self._ssh_key_dir = tempfile.TemporaryDirectory(prefix="warpforth-ssh-")
        self.ssh_private_key = Path(self._ssh_key_dir.name) / "id_ed25519"
        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-f",
                str(self.ssh_private_key),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    def _attach_ssh_key(self) -> None:
        """Attach this session's ephemeral public key to its Vast instance."""
        if self.instance_id is None or self.ssh_private_key is None:
            msg = "Cannot attach SSH key before instance and identity creation"
            raise RuntimeError(msg)

        public_key = self.ssh_private_key.with_suffix(".pub").read_text().strip()
        result = self.sdk.attach_ssh(instance_id=self.instance_id, ssh_key=public_key)
        if not isinstance(result, dict) or result.get("success") is False:
            msg = f"Failed to attach ephemeral SSH key to instance {self.instance_id}"
            raise RuntimeError(msg)
        logger.info("Attached ephemeral SSH key to instance %s", self.instance_id)

    def _cleanup_ssh_key(self) -> None:
        """Remove this session's ephemeral local SSH key material."""
        if self._ssh_key_dir is not None:
            self._ssh_key_dir.cleanup()
            self._ssh_key_dir = None
            self.ssh_private_key = None

    def _show_instances(self) -> list[dict] | None:
        """Return the current account instances, or None if listing fails."""
        try:
            instances = self.sdk.show_instances()
        except Exception:
            logger.exception("Failed to list Vast.ai instances")
            return None

        if not isinstance(instances, list):
            logger.warning("Vast.ai returned an invalid instance listing")
            return None
        return [inst for inst in instances if isinstance(inst, dict)]

    def _log_existing_instances(self) -> None:
        """Log existing WarpForth instances without modifying them."""
        instances = self._show_instances()
        if instances is None:
            logger.warning("Could not check for existing WarpForth Vast.ai instances")
            return

        matching = [
            inst
            for inst in instances
            if str(inst.get("label", "")).startswith(INSTANCE_LABEL_PREFIX)
        ]
        if not matching:
            logger.info("No existing WarpForth Vast.ai instances found")
            return

        total_cost = 0.0
        for inst in matching:
            cost = float(inst.get("dph_total") or 0.0)
            total_cost += cost
            start_date = inst.get("start_date")
            age = "unknown"
            if isinstance(start_date, int | float):
                age = f"{max(0.0, time.time() - start_date) / 60:.0f}m"
            logger.warning(
                "Existing WarpForth Vast.ai instance: id=%s label=%s status=%s "
                "gpu=%s dph=$%.3f age=%s ownership=active run or orphan candidate",
                inst.get("id"),
                inst.get("label"),
                inst.get("actual_status", inst.get("status", "unknown")),
                inst.get("gpu_name", "unknown"),
                cost,
                age,
            )
        logger.warning(
            "Found %d existing WarpForth Vast.ai instance(s), total known cost=$%.3f/hr; "
            "they may belong to other active test runs and will not be destroyed",
            len(matching),
            total_cost,
        )

    def _launch(self) -> None:
        """Find the cheapest suitable offer and launch an instance."""
        minimum_compute_capability = int(self.arch.removeprefix("sm_")) * 10
        query = (
            f"num_gpus=1 rentable=True rented=False compute_cap>={minimum_compute_capability}"
            f" reliability>=0.95 inet_up>=100 dph<={MAX_COST_PER_HOUR}"
            f" cuda_max_good>={CUDA_VERSION}"
        )
        offers = self.sdk.search_offers(query=query, order="dph", limit=5)

        if not offers:
            msg = "No suitable GPU offers found on Vast.ai"
            raise RuntimeError(msg)

        offer = offers[0] if isinstance(offers, list) else offers
        details = offer if isinstance(offer, dict) else {"id": offer}
        offer_id = int(details["id"])
        logger.info(
            "Launching instance from offer %s: gpu=%s driver=%s cuda_max_good=%s",
            offer_id,
            details.get("gpu_name"),
            details.get("driver_version"),
            details.get("cuda_max_good"),
        )
        self._creation_ambiguous = True
        try:
            result = self.sdk.create_instance(
                id=offer_id,
                image=CUDA_IMAGE,
                disk=10.0,
                runtype="ssh_direc ssh_proxy",
                label=self.instance_label,
            )
        except RequestException:
            logger.warning(
                "Instance creation result is ambiguous; reconciling label=%s",
                self.instance_label,
            )
            self._reconcile_instance()
            return

        if (
            isinstance(result, dict)
            and result.get("success") is True
            and result.get("new_contract")
        ):
            self.instance_id = int(result["new_contract"])
            self._creation_ambiguous = False
            logger.info(
                "This GPU test session owns instance id=%s label=%s",
                self.instance_id,
                self.instance_label,
            )
            return

        logger.warning(
            "Instance creation response did not contain an ID; reconciling label=%s",
            self.instance_label,
        )
        self._reconcile_instance()

    def _reconcile_instance(self) -> None:
        """Find this run's instance by its exact unique label."""
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            instances = self._show_instances()
            if instances is not None:
                matches = [inst for inst in instances if inst.get("label") == self.instance_label]
                if len(matches) > 1:
                    ids = [inst.get("id") for inst in matches]
                    msg = f"Multiple instances found for unique label {self.instance_label}: {ids}"
                    raise RuntimeError(msg)
                if matches:
                    self.instance_id = int(matches[0]["id"])
                    self._creation_ambiguous = False
                    logger.info(
                        "This GPU test session owns instance id=%s label=%s",
                        self.instance_id,
                        self.instance_label,
                    )
                    return
            time.sleep(POLL_INTERVAL_S)

        msg = f"Could not reconcile instance with label {self.instance_label} after creation"
        raise TimeoutError(msg)

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
                    self._attach_ssh_key()
                    self._wait_for_sshd()
                    self._compile_runner()
                    self._provision_harness()
                    return

            time.sleep(POLL_INTERVAL_S)

        msg = f"Instance {self.instance_id} did not become SSH-ready within {POLL_TIMEOUT_S}s"
        raise TimeoutError(msg)

    def _wait_for_sshd(self) -> None:
        """Poll until sshd is actually accepting connections."""
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    [*self._ssh_cmd(), "true"],
                    capture_output=True,
                    timeout=SSH_PROBE_TIMEOUT_S,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "ssh probe to %s:%s exceeded %ds; treating it like a failed probe",
                    self.ssh_host,
                    self.ssh_port,
                    SSH_PROBE_TIMEOUT_S,
                )
                time.sleep(POLL_INTERVAL_S)
                continue
            if result.returncode == 0:
                logger.info("sshd ready on %s:%s", self.ssh_host, self.ssh_port)
                return
            time.sleep(POLL_INTERVAL_S)
        msg = (
            f"sshd on {self.ssh_host}:{self.ssh_port} did not become ready within {POLL_TIMEOUT_S}s"
        )
        raise TimeoutError(msg)

    def _compile_runner(self) -> None:
        """Upload warpforth-runner.cpp and compile it on the remote host."""
        start = time.monotonic()
        self.scp_upload(RUNNER_SRC, f"{REMOTE_TMP}/warpforth-runner.cpp")
        nvcc_cmd = (
            f"nvcc -o {REMOTE_TMP}/warpforth-runner"
            f" {REMOTE_TMP}/warpforth-runner.cpp -lcuda -std=c++17"
        )
        self.ssh_run(nvcc_cmd, timeout=60)
        logger.info("C++ runner ready in %.1fs", time.monotonic() - start)

    def _provision_harness(self) -> None:
        """Install Python dependencies and upload remote_harness.py; pip gets one retry."""
        start = time.monotonic()
        self.ssh_run(
            "command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1"
            " || (apt-get update -qq"
            " && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-pip)",
            timeout=600,
        )
        install = (
            "python3 -m pip install --quiet --disable-pip-version-check "
            f'"{CUDA_PYTHON_SPEC}" numpy nvidia-cublas-cu12'
        )
        try:
            self.ssh_run(install, timeout=600)
        except RuntimeError:
            logger.warning("cuda-python install failed; retrying once")
            self.ssh_run(install, timeout=600)
        self.scp_upload(HARNESS_SRC, HARNESS_REMOTE)
        self.ssh_run("python3 -c 'import cuda, numpy'", timeout=60)
        logger.info("remote_harness.py provisioned in %.1fs", time.monotonic() - start)

    def _record_labeled_instance_ids(
        self,
        instances: list[dict],
        instance_ids: set[int],
    ) -> None:
        """Record instances owned by this session's exact unique label."""
        labeled_ids = {
            int(inst["id"])
            for inst in instances
            if inst.get("label") == self.instance_label and inst.get("id") is not None
        }
        instance_ids.update(labeled_ids)
        if labeled_ids:
            self._creation_ambiguous = False

    def _destroy(self) -> bool:
        """Destroy only this session's instances and verify their absence."""
        instance_ids = {self.instance_id} if self.instance_id is not None else set()
        retries = 3
        for attempt in range(retries):
            instances = self._show_instances()
            if instances is not None:
                self._record_labeled_instance_ids(instances, instance_ids)
                if not self._creation_ambiguous and not instance_ids:
                    logger.info(
                        "Verified cleanup for Vast.ai label=%s instance_ids=%s",
                        self.instance_label,
                        sorted(instance_ids),
                    )
                    self.instance_id = None
                    return True

            for instance_id in instance_ids:
                try:
                    self.sdk.destroy_instance(id=instance_id)
                except Exception:
                    logger.exception(
                        "Destroy attempt %d/%d failed for instance %s",
                        attempt + 1,
                        retries,
                        instance_id,
                    )

            instances = self._show_instances()
            if instances is not None:
                self._record_labeled_instance_ids(instances, instance_ids)
                listed_ids = {int(inst["id"]) for inst in instances if inst.get("id") is not None}
                if not self._creation_ambiguous and not instance_ids & listed_ids:
                    logger.info(
                        "Verified cleanup for Vast.ai label=%s instance_ids=%s",
                        self.instance_label,
                        sorted(instance_ids),
                    )
                    self.instance_id = None
                    return True

            if attempt < retries - 1:
                time.sleep(5)

        logger.error(
            "Could not verify cleanup for Vast.ai label=%s instance_ids=%s; "
            "check the Vast.ai account and destroy these instances manually",
            self.instance_label,
            sorted(instance_ids),
        )
        return False

    def _unregister_atexit(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_cleanup)
            self._atexit_registered = False

    def _atexit_cleanup(self) -> None:
        """Last-resort cleanup registered before instance creation."""
        logger.warning(
            "atexit: checking for Vast.ai instances with label=%s instance_id=%s",
            self.instance_label,
            self.instance_id,
        )
        try:
            self._destroy()
        finally:
            self._cleanup_ssh_key()

    def _ssh_options(self) -> list[str]:
        if self.ssh_private_key is None:
            msg = "Ephemeral SSH identity is not available"
            raise RuntimeError(msg)
        return [
            "-i",
            str(self.ssh_private_key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "PreferredAuthentications=publickey",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=3",
            "-o",
            "LogLevel=ERROR",
        ]

    def _ssh_cmd(self) -> list[str]:
        """Build base SSH command with connection options."""
        return [
            "ssh",
            *self._ssh_options(),
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
                *self._ssh_options(),
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

    def ssh_popen(self, cmd: str) -> subprocess.Popen[str]:
        """Open a persistent SSH process with piped stdin/stdout for a JSON-lines service."""
        return subprocess.Popen(
            [*self._ssh_cmd(), cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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


class HarnessError(Exception):
    """The remote harness reported a failure or stopped answering."""


class HarnessClient:
    """Drives remote_harness.py on the rental over one persistent SSH JSON-lines channel.

    The protocol (one JSON request per line, one JSON response per line) is documented in
    gpu_test/remote_harness.py. A response with "ok": false raises HarnessError.
    """

    def __init__(self, session: VastSession) -> None:
        self._session = session
        self._proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> Self:
        self._proc = self._session.ssh_popen(f"python3 {HARNESS_REMOTE}")
        return self

    def __exit__(self, *_: object) -> None:
        if self._proc is not None:
            if self._proc.stdin is not None:
                with contextlib.suppress(OSError):
                    self._proc.stdin.close()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def request(self, payload: dict[str, object]) -> dict[str, object]:
        """Send one request and return its response payload."""
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            msg = "HarnessClient is not started; use it as a context manager"
            raise HarnessError(msg)
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            tail = ""
            if self._proc.stderr is not None:
                tail = self._proc.stderr.read()[-2000:].strip()
            msg = f"remote harness exited without a response: {tail}"
            raise HarnessError(msg)
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"remote harness returned a non-JSON line: {line.strip()!r}"
            raise HarnessError(msg) from exc
        if not isinstance(response, dict):
            msg = f"remote harness returned a non-object response: {response!r}"
            raise HarnessError(msg)
        if not response.get("ok"):
            raise HarnessError(str(response.get("error", "unknown harness error")))
        return response
