"""Basic end-to-end GPU execution test for the WarpForth compiler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import KernelRunner

pytestmark = pytest.mark.gpu


def test_addition(kernel_runner: KernelRunner) -> None:
    """Compile 3+4, store to data[0], verify result is 7."""
    result = kernel_runner.run(
        forth_source="param data 256\n3 4 +\n0 cells data + !",
    )
    assert result[0] == 7
