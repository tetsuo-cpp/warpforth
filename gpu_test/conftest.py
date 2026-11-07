"""GPU test fixtures and collection hooks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from gpu_test.infrastructure import (
    DEFAULT_ARCHITECTURE,
    SUPPORTED_ARCHITECTURES,
    Compiler,
    KernelRunner,
    VastSession,
)

if TYPE_CHECKING:
    from collections.abc import Generator


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
