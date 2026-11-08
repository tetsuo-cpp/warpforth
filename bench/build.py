"""Builds one compiler revision per git ref, each in its own temporary worktree.

Every revision is configured and built from scratch rather than reused from the main checkout's
build, so an arbitrary pair of refs can be compared and a stale object file can never decide a
verdict. The cmake settings come from a working build's cache, so a fresh build uses the same MLIR
and LLVM as the checkout that owns this command.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger("bench")

TARGET = "warpforthc"
COMPILER = Path("bin") / TARGET
REQUIRED_DEFINITIONS = ("MLIR_DIR", "LLVM_DIR")
COPIED_DEFINITIONS = ("CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER")
LOG_TAIL_LINES = 20


class BuildError(Exception):
    """A ref cannot be resolved, checked out, configured, or built."""


@dataclass(frozen=True)
class Ref:
    """A revision, as the command line named it and as the commit it resolved to."""

    name: str
    sha: str

    @property
    def short(self) -> str:
        return self.sha[:12]


@dataclass(frozen=True)
class Toolchain:
    """The cmake definitions and generator a fresh build inherits from a working one."""

    definitions: tuple[tuple[str, str], ...]
    generator: str

    def configure(self, source: Path, build_dir: Path) -> list[str]:
        generator = ["-G", self.generator] if self.generator else []
        flags = [f"-D{name}={value}" for name, value in self.definitions]
        return ["cmake", "-S", str(source), "-B", str(build_dir), *generator, *flags]


def resolve(repo: Path, name: str) -> Ref:
    """The commit a ref names, so the build and the result document agree on what ran."""
    result = _git(repo, "rev-parse", "--verify", f"{name}^{{commit}}")
    if result.returncode != 0:
        msg = f"cannot resolve {name!r} to a commit: {result.stderr.strip()}"
        raise BuildError(msg)
    return Ref(name=name, sha=result.stdout.strip())


def toolchain_of(build_dir: Path) -> Toolchain:
    """The MLIR and LLVM a fresh build needs, from the environment or a configured build.

    A worktree has no build directory, so callers pass the main checkout's, found by
    `manifest.find_build`.
    """
    cached = _cache(build_dir / "CMakeCache.txt")
    definitions = []
    for name in REQUIRED_DEFINITIONS:
        value = os.environ.get(name) or cached.get(name, "")
        if not value:
            msg = f"no {name} in {build_dir / 'CMakeCache.txt'}; configure that build or set {name}"
            raise BuildError(msg)
        definitions.append((name, value))
    definitions.extend((name, cached[name]) for name in COPIED_DEFINITIONS if cached.get(name))
    return Toolchain(definitions=tuple(definitions), generator=cached.get("CMAKE_GENERATOR", ""))


@contextmanager
def worktree(repo: Path, ref: Ref) -> Iterator[Path]:
    """A detached checkout of one ref, with its build directory, removed on exit."""
    root = Path(tempfile.mkdtemp(prefix=f"warpforth-compare-{ref.short}-"))
    checkout = root / "src"
    result = _git(repo, "worktree", "add", "--detach", "--force", str(checkout), ref.sha)
    if result.returncode != 0:
        shutil.rmtree(root, ignore_errors=True)
        msg = f"cannot check out {ref.name} at {ref.short}: {result.stderr.strip()}"
        raise BuildError(msg)
    try:
        yield checkout
    finally:
        _git(repo, "worktree", "remove", "--force", str(checkout))
        _git(repo, "worktree", "prune")
        shutil.rmtree(root, ignore_errors=True)


def compiler(checkout: Path, toolchain: Toolchain, *, jobs: int) -> Path:
    """Configure and build `warpforthc` inside the checkout, returning the binary."""
    build_dir = checkout / "build"
    _run(toolchain.configure(checkout, build_dir), f"configure {checkout}")
    _run(
        ["cmake", "--build", str(build_dir), "--target", TARGET, "-j", str(jobs)],
        f"build {TARGET} at {checkout.name}",
    )
    binary = build_dir / COMPILER
    if not binary.is_file():
        msg = f"{checkout} built no {COMPILER}"
        raise BuildError(msg)
    return binary


def _cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        name, _, typed = line.partition(":")
        _kind, _, value = typed.partition("=")
        values[name] = value
    return values


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - a fixed argument list, never a shell string
        ["git", "-C", str(repo), *args],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=False,
    )


def _run(command: list[str], what: str) -> None:
    logger.info("%s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-LOG_TAIL_LINES:])
        msg = f"{what} exited {result.returncode}:\n{tail}"
        raise BuildError(msg)
