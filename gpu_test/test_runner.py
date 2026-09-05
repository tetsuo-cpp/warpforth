"""GPU-independent tests of the JSON runner and its SSH integration."""

from __future__ import annotations

import ctypes
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import cuda.bindings
import pytest
from cuda.bindings import driver

from gpu_test import warpforth_runner
from gpu_test.conftest import KernelRunner, VastSession


@pytest.fixture
def ptx_file(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    path = tmp_path / "kernel.ptx"
    path.write_bytes(b"// PTX\n" + getattr(request, "param", b""))
    return path


@pytest.fixture
def request_data() -> dict:
    return {
        "kernel": "main",
        "grid": [2, 3, 4],
        "block": [5, 6, 7],
        "params": [
            {"type": "i64[]", "values": [-(2**63), 2**63 - 1]},
            {"type": "f64", "value": 3.14},
            {"type": "f64[]", "values": [1.25, -2.5]},
            {"type": "i64", "value": -42},
        ],
        "outputs": [{"param": 2}, {"param": 0, "count": 1}, {"param": 0, "count": 0}],
    }


@pytest.fixture
def json_files(tmp_path: Path, request_data: dict) -> tuple[Path, Path]:
    request = tmp_path / "request.json"
    request.write_text(json.dumps(request_data))
    return request, tmp_path / "result.json"


@pytest.fixture
def cuda_driver(monkeypatch: pytest.MonkeyPatch) -> Mock:
    fake = Mock(spec=driver)
    fake.CUresult = driver.CUresult
    for name in (
        "cuInit",
        "cuMemcpyHtoD",
        "cuMemcpyDtoH",
        "cuLaunchKernel",
        "cuCtxSynchronize",
        "cuMemFree",
        "cuModuleUnload",
        "cuCtxDestroy",
    ):
        getattr(fake, name).return_value = (driver.CUresult.CUDA_SUCCESS,)
    fake.cuDeviceGet.return_value = (0, 0)
    fake.cuCtxCreate.return_value = (0, 10)
    fake.cuModuleLoadData.return_value = (0, 20)
    fake.cuModuleGetFunction.return_value = (0, 30)
    fake.cuGetErrorName.return_value = (0, b"CUDA_ERROR_INVALID_PTX")
    fake.cuGetErrorString.return_value = (0, b"the provided PTX was invalid")
    allocations = []

    def allocate(size: int) -> tuple:
        storage = ctypes.create_string_buffer(size)
        allocations.append(storage)
        return (0, ctypes.addressof(storage))

    def copy(destination: int, source: int, size: int) -> tuple:
        ctypes.memmove(destination, source, size)
        return (0,)

    fake.cuMemAlloc.side_effect = allocate
    fake.cuMemcpyHtoD.side_effect = copy
    fake.cuMemcpyDtoH.side_effect = copy
    monkeypatch.setattr(cuda.bindings, "driver", fake)
    return fake


@pytest.mark.parametrize("ptx_file", [b"", b"\0"], indirect=True)
def test_launch_and_multiple_outputs(cuda_driver: Mock, request_data: dict, ptx_file: Path) -> None:
    def launch(*args: object) -> tuple:
        assert args[:9] == (30, 2, 3, 4, 5, 6, 7, 0, 0)
        pointers = (ctypes.c_void_p * 4).from_address(args[9])
        integer_address = ctypes.c_uint64.from_address(pointers[0]).value
        integers = (ctypes.c_int64 * 2).from_address(integer_address)
        assert list(integers) == [-(2**63), 2**63 - 1]
        assert ctypes.c_double.from_address(pointers[1]).value == 3.14
        float_address = ctypes.c_uint64.from_address(pointers[2]).value
        floats = (ctypes.c_double * 2).from_address(float_address)
        assert list(floats) == [1.25, -2.5]
        assert ctypes.c_int64.from_address(pointers[3]).value == -42
        integers[0] = 42
        floats[0] = 2.75
        return (0,)

    cuda_driver.cuLaunchKernel.side_effect = launch
    assert warpforth_runner.execute(ptx_file, request_data) == {
        "status": "ok",
        "outputs": [
            {"param": 2, "type": "f64[]", "values": [2.75, -2.5]},
            {"param": 0, "type": "i64[]", "values": [42]},
            {"param": 0, "type": "i64[]", "values": []},
        ],
    }
    cuda_driver.cuModuleLoadData.assert_called_once_with(b"// PTX\n\0")
    cuda_driver.cuModuleGetFunction.assert_called_once_with(20, b"main")
    cuda_driver.cuCtxSynchronize.assert_called_once()
    assert cuda_driver.cuMemcpyDtoH.call_count == 2
    assert cuda_driver.cuMemFree.call_count == 2
    cuda_driver.cuModuleUnload.assert_called_once_with(20)
    cuda_driver.cuCtxDestroy.assert_called_once_with(10)


def test_no_parameters_or_outputs(cuda_driver: Mock, ptx_file: Path) -> None:
    request = {"kernel": "main"}
    assert warpforth_runner.execute(ptx_file, request) == {"status": "ok", "outputs": []}
    cuda_driver.cuLaunchKernel.assert_called_once_with(30, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("stage", "allocations", "modules"),
    [
        ("cuModuleLoadData", 0, 0),
        ("cuMemcpyHtoD", 1, 1),
        ("cuLaunchKernel", 2, 1),
        ("cuCtxSynchronize", 2, 1),
        ("cuMemcpyDtoH", 2, 1),
    ],
)
def test_cuda_errors_cleanup(  # noqa: PLR0913
    cuda_driver: Mock,
    request_data: dict,
    ptx_file: Path,
    stage: str,
    allocations: int,
    modules: int,
) -> None:
    call = getattr(cuda_driver, stage)
    call.side_effect = None
    call.return_value = (driver.CUresult.CUDA_ERROR_INVALID_PTX,)
    with pytest.raises(RuntimeError, match="CUDA_ERROR_INVALID_PTX: the provided PTX was invalid"):
        warpforth_runner.execute(ptx_file, request_data)
    assert cuda_driver.cuMemFree.call_count == allocations
    assert cuda_driver.cuModuleUnload.call_count == modules
    cuda_driver.cuCtxDestroy.assert_called_once()


@pytest.mark.parametrize(
    "update",
    [
        {"kernel": ""},
        {"grid": [1, 2]},
        {"block": [0, 1, 1]},
        {"grid": [True, 1, 1]},
        {"params": [{"type": "i32", "value": 1}]},
        {"params": [{"type": "i64", "value": 2**63}]},
        {"params": [{"type": "i64", "value": 1.5}]},
        {"params": [{"type": "i64[]", "values": []}]},
        {"params": [{"type": "f64", "value": float("inf")}]},
        {"outputs": [{"param": -1}]},
        {"outputs": [{"param": 4}]},
        {"outputs": [{"param": 1}]},
        {"outputs": [{"param": 0, "count": -1}]},
        {"outputs": [{"param": 0, "count": 3}]},
    ],
)
def test_invalid_request(
    cuda_driver: Mock, request_data: dict, update: dict, ptx_file: Path
) -> None:
    request_data.update(update)
    with pytest.raises((ValueError, TypeError)):
        warpforth_runner.execute(ptx_file, request_data)
    cuda_driver.cuInit.assert_not_called()


@pytest.mark.parametrize("text", [None, "not json", "null", "{}"])
def test_cli_invalid_input(
    text: str | None, json_files: tuple[Path, Path], capsys: pytest.CaptureFixture, ptx_file: Path
) -> None:
    request, result = json_files
    if text is None:
        request.unlink()
    else:
        request.write_text(text)
    assert warpforth_runner.main([str(ptx_file), str(request), str(result)]) == 1
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert json.loads(result.read_text())["status"] == "error"


@pytest.mark.parametrize("contents", [None, b"", b"// PTX\0truncated"])
def test_cli_ptx_file_errors(
    contents: bytes | None,
    ptx_file: Path,
    json_files: tuple[Path, Path],
) -> None:
    if contents is None:
        ptx_file.unlink()
    else:
        ptx_file.write_bytes(contents)
    request, result = json_files
    assert warpforth_runner.main([str(ptx_file), str(request), str(result)]) == 1
    response = json.loads(result.read_text())
    assert response["status"] == "error"
    assert response["error"]


def test_cli_success(
    cuda_driver: Mock,
    ptx_file: Path,
    json_files: tuple[Path, Path],
    capsys: pytest.CaptureFixture,
) -> None:
    request, result = json_files
    result.write_text("stale result")
    assert warpforth_runner.main([str(ptx_file), str(request), str(result)]) == 0
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert json.loads(result.read_text())["status"] == "ok"
    cuda_driver.cuLaunchKernel.assert_called_once()


def test_cli_cuda_error(
    cuda_driver: Mock,
    ptx_file: Path,
    json_files: tuple[Path, Path],
    capsys: pytest.CaptureFixture,
) -> None:
    cuda_driver.cuModuleLoadData.return_value = (driver.CUresult.CUDA_ERROR_INVALID_PTX,)
    request, result = json_files
    assert warpforth_runner.main([str(ptx_file), str(request), str(result)]) == 1
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert json.loads(result.read_text()) == {
        "status": "error",
        "error": "CUDA_ERROR_INVALID_PTX: the provided PTX was invalid",
    }


FORTH = "\\! kernel main\n\\! param a i64[3]\n\\! param b f64[2]\n\\! param c f64\n"


@pytest.mark.parametrize("outputs", [None, [{"param": 1}, {"param": 0, "count": 1}]])
def test_harness_json(outputs: list | None) -> None:
    session, compiler = Mock(), Mock()
    compiler.compile_source.return_value = "// PTX\n"
    uploaded = []
    session.scp_upload.side_effect = lambda path, _remote: uploaded.append(Path(path).read_bytes())
    session.ssh_run.return_value = "ignored stdout"
    session.scp_download.side_effect = lambda _remote, path: path.write_text(
        json.dumps({"status": "ok", "outputs": [{"values": [1]}, {"values": [2]}]})
    )
    result = KernelRunner(session, compiler).run(
        FORTH, params={"a": [2**62], "c": 1.5}, outputs=outputs
    )
    assert result == ([1] if outputs is None else [[1], [2]])
    request = json.loads(uploaded[1])
    assert set(request) == {"kernel", "grid", "block", "params", "outputs"}
    assert uploaded[0] == b"// PTX\n"
    command = session.ssh_run.call_args.args[0]
    for call in session.scp_upload.call_args_list:
        local_path, remote_path = call.args
        assert not Path(local_path).exists()
        assert f" {remote_path}" in command
    remote_result, local_result = session.scp_download.call_args.args
    assert command.endswith(f" {remote_result}")
    assert not local_result.exists()
    assert "stdin" not in session.ssh_run.call_args.kwargs
    assert request["params"] == [
        {"type": "i64[]", "values": [2**62, 0, 0]},
        {"type": "f64[]", "values": [0.0, 0.0]},
        {"type": "f64", "value": 1.5},
    ]
    assert request["outputs"] == ([{"param": 0}] if outputs is None else outputs)


@pytest.mark.parametrize("returncode", [1, 255])
def test_harness_errors(returncode: int) -> None:
    session, compiler = Mock(), Mock()
    compiler.compile_source.return_value = "// PTX"
    session.ssh_run.side_effect = subprocess.CalledProcessError(returncode, "runner")
    session.scp_download.side_effect = lambda _remote, path: path.write_text(
        '{"status":"error","error":"CUDA_ERROR_INVALID_PTX"}'
    )
    expected = RuntimeError if returncode == 1 else subprocess.CalledProcessError
    with pytest.raises(expected):
        KernelRunner(session, compiler).run(FORTH)
    assert session.scp_download.call_count == (1 if returncode == 1 else 0)


def test_scp_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = object.__new__(VastSession)
    session.ssh_host = "host"
    session.ssh_port = 22
    monkeypatch.setattr(session, "_ssh_options", list)
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    destination = tmp_path / "result.json"
    session.scp_download("/tmp/result.json", destination)  # noqa: S108
    assert run.call_args.args[0] == [
        "scp",
        "-P",
        "22",
        "root@host:/tmp/result.json",
        str(destination),
    ]
    assert run.call_args.kwargs["check"] is True


def test_install_runner() -> None:
    session = Mock()
    VastSession._install_runner(session)  # noqa: SLF001
    assert session.scp_upload.call_args.args[0].name == "warpforth_runner.py"
    command = session.ssh_run.call_args.args[0]
    assert "pip install 'cuda-python>=12.8,<13'" in command
    assert "python3 -m venv" in command
    assert "nvcc" not in command
