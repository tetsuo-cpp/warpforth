"""GPU test fixtures and collection hooks."""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING

import pytest

from gpu_test.infrastructure import (
    DEFAULT_ARCHITECTURE,
    SUPPORTED_ARCHITECTURES,
    Compiler,
    HarnessClient,
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
def harness(gpu_session: VastSession) -> Generator[HarnessClient]:
    with HarnessClient(gpu_session) as client:
        yield client


@pytest.fixture(scope="session")
def kernel_runner(harness: HarnessClient, compiler: Compiler) -> KernelRunner:
    return KernelRunner(harness, compiler)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Generator[None]:
    """Fail a unit test that dials a remote host, even if the caller swallows the error.

    VastSession logs a failed instance listing and carries on, so an SDK fake missing
    `show_instances` reaches the real client and the test still passes. The attempt is recorded
    and asserted at teardown, which a swallowed exception cannot hide.
    """
    if request.node.get_closest_marker("gpu"):
        yield
        return

    attempts: list[object] = []

    def refuse(_sock: socket.socket, address: object, *_: object, **__: object) -> None:
        attempts.append(address)
        msg = f"a unit test tried to reach {address}; stub the SDK call instead"
        raise OSError(msg)

    monkeypatch.setattr(socket.socket, "connect", refuse)
    yield
    if attempts:
        pytest.fail(f"a unit test opened a socket to {attempts}")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.gpu tests when VASTAI_API_KEY is not set."""
    if os.environ.get("VASTAI_API_KEY"):
        return

    skip_gpu = pytest.mark.skip(reason="VASTAI_API_KEY not set")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
