"""Unit tests for local WarpForth compiler invocation."""

from pathlib import Path

from gpu_test.conftest import Compiler


def test_compiler_passes_architecture_to_warpforthc(tmp_path: Path) -> None:
    binary = tmp_path / "warpforthc"
    binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    binary.chmod(0o755)

    output = Compiler(binary=binary, arch="sm_80").compile_source("\\! kernel main\n42\n")

    assert output.splitlines()[-2:] == ["--arch", "sm_80"]
