"""Tests for the remote harness protocol, plus the gpu-marked rental smoke run.

The no-GPU tests drive `remote_harness.handle_request` with a FakeGpu, an in-process
stand-in for the CUDA driver, so the request/response protocol, input generation,
reference validation, and sample emission are checked without a key. The gpu-marked
tests rent a GPU, upload the real harness, and check that one WarpForth PTX timed
against itself reports a gap of 1.00 within noise, parsed through the bench_stats
result contract.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks import manifest
from gpu_test import bench_stats, infrastructure, remote_harness
from gpu_test.infrastructure import HarnessClient

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES = manifest.load()
SCALE_ELEMENTS = 1024
MATMUL_N = 8


def _scale_kernel(args: list, _launch: object) -> None:
    n = int(args[0])
    scale = int(args[1])
    inputs = args[2].view(np.int64)
    output = args[3].view(np.int64)
    output[:n] = inputs[:n] * scale


def _poison_kernel(args: list, _launch: object) -> None:
    n = int(args[0])
    output = args[3].view(np.int64)
    output[:n] = -1


def _copy_f64_kernel(args: list, _launch: object) -> None:
    n = int(args[0])
    args[2].view(np.float64)[:n] = args[1].view(np.float64)[:n]


def _axpy_in_place_kernel(args: list, _launch: object) -> None:
    """Y = ALPHA*X + Y, the case whose result buffer is also one of its inputs."""
    n = int(args[0])
    alpha = float(args[1])
    x = args[2].view(np.float64)[:n]
    y = args[3].view(np.float64)[:n]
    y[:] = alpha * x + y


def _matmul_kernel(args: list, _launch: object) -> None:
    n = int(args[0])
    a = args[1].view(np.float64)[: n * n].reshape(n, n)
    b = args[2].view(np.float64)[: n * n].reshape(n, n)
    args[3].view(np.float64)[: n * n] = (a @ b).ravel()


class FakeGpu:
    """In-process stand-in for CudaGpu: dict memory, Python kernels, a manual clock."""

    def __init__(self) -> None:
        self.buffers: dict[int, np.ndarray] = {}
        self.kernels: dict[str, Callable[[list, object], None]] = {}
        self.costs: dict[str, float] = {}
        self.dgemm_cost = 10.0
        self.clock_us = 0.0
        self._next_ptr = 0

    def register(self, name: str, fn: Callable[[list, object], None], cost_us: float) -> None:
        self.kernels[name] = fn
        self.costs[name] = cost_us

    def environment(self) -> dict[str, object]:
        return {
            "gpu_uuid": "fake-gpu-uuid",
            "driver_version": "12.4",
            "sm_clock_mhz": 1500,
            "memory_clock_mhz": 7000,
        }

    def alloc(self, nbytes: int) -> int:
        self._next_ptr += 1
        self.buffers[self._next_ptr] = np.zeros(nbytes, dtype=np.uint8)
        return self._next_ptr

    def free(self, ptr: int) -> None:
        del self.buffers[ptr]

    def upload(self, ptr: int, data: np.ndarray) -> None:
        self.buffers[ptr][: data.nbytes] = data.view(np.uint8)

    def download(self, ptr: int, count: int, dtype: np.dtype) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        return self.buffers[ptr][: count * itemsize].view(dtype).copy()

    def copy_device(self, dst: int, src: int, nbytes: int) -> None:
        self.buffers[dst][:nbytes] = self.buffers[src][:nbytes]

    def load_module(self, ptx: str) -> str:
        return ptx

    def unload_module(self, _module: str) -> None:
        return

    def function(self, _module: str, name: str) -> str:
        if name not in self.kernels:
            msg = f"fake has no kernel named {name}"
            raise remote_harness.HarnessError(msg)
        return name

    def launch(self, func: str, _launch: object, args: list[tuple[str, object]]) -> None:
        resolved = [self.buffers[value] if kind == "ptr" else value for kind, value in args]
        self.kernels[func](resolved, _launch)
        self.clock_us += self.costs[func]

    def record_event(self) -> float:
        return self.clock_us

    def elapsed_us(self, start: float, end: float) -> float:
        return end - start

    def synchronize(self) -> None:
        return

    def dgemm(self, n: int, a: int, b: int, c: int) -> None:
        am = self.buffers[a].view(np.float64)[: n * n].reshape(n, n)
        bm = self.buffers[b].view(np.float64)[: n * n].reshape(n, n)
        self.buffers[c].view(np.float64)[: n * n] = (am @ bm).ravel()
        self.clock_us += self.dgemm_cost


def _ptx_side(kernel: str) -> dict[str, object]:
    return {
        "kind": "ptx",
        "ptx": "fake ptx",
        "kernel": kernel,
        "launch": {"grid": [1, 1, 1], "block": [1, 1, 1]},
    }


def _scale_bench_request(base_kernel: str, head_kernel: str) -> dict[str, object]:
    return {
        "op": "bench",
        "case_id": "scale_i64",
        "case_version": 1,
        "reference": "scale_i64",
        "tolerance": [0.0, 0.0],
        "params": [
            {"name": "N", "kind": "scalar", "type": "i64"},
            {"name": "SCALE", "kind": "scalar", "type": "i64"},
            {"name": "INPUT", "kind": "array", "type": "i64", "cells": SCALE_ELEMENTS},
            {"name": "OUTPUT", "kind": "array", "type": "i64", "cells": SCALE_ELEMENTS},
        ],
        "scalars": {"N": SCALE_ELEMENTS, "SCALE": 7},
        "inputs": {
            "INPUT": {
                "kind": "int_uniform",
                "dtype": "i64",
                "seed": 11,
                "low": -1000,
                "high": 1000,
                "length": 0,
            }
        },
        "output": "OUTPUT",
        "base": _ptx_side(base_kernel),
        "head": _ptx_side(head_kernel),
        "timing": {"warmup": 2, "samples": 6, "sample_ms": 0.1},
        "environment": {"base_commit": "aaa", "head_commit": "bbb"},
    }


def _matmul_bench_request(base_kernel: str) -> dict[str, object]:
    cells = MATMUL_N * MATMUL_N
    return {
        "op": "bench",
        "case_id": "matmul_f64_naive",
        "case_version": 1,
        "reference": "matmul_f64",
        "tolerance": [1e-12, 1e-12],
        "params": [
            {"name": "N", "kind": "scalar", "type": "i64"},
            {"name": "A", "kind": "array", "type": "f64", "cells": cells},
            {"name": "B", "kind": "array", "type": "f64", "cells": cells},
            {"name": "C", "kind": "array", "type": "f64", "cells": cells},
        ],
        "scalars": {"N": MATMUL_N},
        "inputs": {
            "A": {"kind": "uniform", "dtype": "f64", "seed": 41, "low": -1.0, "high": 1.0},
            "B": {"kind": "uniform", "dtype": "f64", "seed": 42, "low": -1.0, "high": 1.0},
        },
        "output": "C",
        "base": _ptx_side(base_kernel),
        "head": {
            "kind": "host_lib",
            "library": "cublas",
            "symbol": "cublasDgemm",
            "n": MATMUL_N,
            "a": "A",
            "b": "B",
            "c": "C",
        },
        "timing": {"warmup": 1, "samples": 4, "sample_ms": 0.05},
        "environment": {"base_commit": "aaa", "head_commit": "bbb"},
    }


def _parse_single_case(response: dict[str, object]) -> bench_stats.CaseResult:
    suite_doc = {"schema_version": bench_stats.SUITE_SCHEMA_VERSION, "cases": [response["result"]]}
    return bench_stats.from_json(json.dumps(suite_doc)).cases[0]


def test_ping_reports_the_driver_environment() -> None:
    response = remote_harness.handle_request({"op": "ping"}, FakeGpu())
    assert response["ok"] is True
    environment = response["environment"]
    for key in ("gpu_uuid", "driver_version", "sm_clock_mhz", "memory_clock_mhz"):
        assert environment[key], key


def test_unknown_op_is_a_structured_error() -> None:
    response = remote_harness.handle_request({"op": "defenestrate"}, FakeGpu())
    assert response["ok"] is False
    assert "defenestrate" in str(response["error"])


def test_a_request_missing_a_required_key_is_a_structured_error() -> None:
    response = remote_harness.handle_request({"op": "bench"}, FakeGpu())
    assert response["ok"] is False
    assert "params" in str(response["error"])


def test_main_serves_one_response_per_line(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGpu()
    monkeypatch.setattr(remote_harness, "default_gpu", lambda: fake)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"op": "ping"}\nnot json\n{"op": "nope"}\n'))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    remote_harness.main()

    lines = stdout.getvalue().splitlines()
    assert len(lines) == 3
    first, second, third = (json.loads(line) for line in lines)
    assert first["ok"] is True
    assert second["ok"] is False
    assert third["ok"] is False


def test_correctness_mode_reads_back_i64_values() -> None:
    fake = FakeGpu()
    fake.register("scale", _scale_kernel, 10.0)
    response = remote_harness.handle_request(
        {
            "op": "correctness",
            "ptx": "fake ptx",
            "kernel": "scale",
            "params": [
                {"type": "i64", "value": 4},
                {"type": "i64", "value": 7},
                {"type": "i64[]", "values": [1, -2, 3, -4]},
                {"type": "i64[]", "values": [0, 0, 0, 0]},
            ],
            "grid": [1, 1, 1],
            "block": [1, 1, 1],
            "output_param": 3,
            "output_count": 4,
        },
        fake,
    )
    assert response["values"] == [7, -14, 21, -28]


def test_correctness_mode_reads_back_f64_values() -> None:
    fake = FakeGpu()
    fake.register("copy", _copy_f64_kernel, 10.0)
    response = remote_harness.handle_request(
        {
            "op": "correctness",
            "ptx": "fake ptx",
            "kernel": "copy",
            "params": [
                {"type": "i64", "value": 2},
                {"type": "f64[]", "values": [1.5, -2.25]},
                {"type": "f64[]", "values": [0.0, 0.0]},
            ],
            "grid": [1, 1, 1],
            "block": [1, 1, 1],
            "output_param": 2,
            "output_count": None,
        },
        fake,
    )
    assert response["values"] == [1.5, -2.25]


def test_correctness_launch_reads_back_one_application_of_an_in_place_kernel() -> None:
    fake = FakeGpu()
    fake.register("axpy", _axpy_in_place_kernel, 10.0)
    response = remote_harness.handle_request(
        {
            "op": "correctness",
            "ptx": "fake ptx",
            "kernel": "axpy",
            "params": [
                {"type": "i64", "value": 2},
                {"type": "f64", "value": 1.5},
                {"type": "f64[]", "values": [2.0, 4.0]},
                {"type": "f64[]", "values": [1.0, 1.0]},
            ],
            "grid": [1, 1, 1],
            "block": [1, 1, 1],
            "output_param": 3,
            "output_count": 2,
        },
        fake,
    )
    assert response["values"] == [4.0, 7.0]


class _RecordingHarness:
    """Serves one canned correctness response and keeps the request it was given."""

    def __init__(self, values: list[object]) -> None:
        self.values = values
        self.requests: list[dict[str, object]] = []

    def request(self, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append(payload)
        return {"ok": True, "values": self.values}


class _FixedCompiler:
    def __init__(self, ptx: str = "fake ptx") -> None:
        self.ptx = ptx
        self.sources: list[str] = []

    def compile_source(self, forth_src: str) -> str:
        self.sources.append(forth_src)
        return self.ptx


_SCALE_SOURCE = "\\! kernel main\n\\! param SCALE i64\n\\! param DATA i64[4]\n"


def test_kernel_runner_sends_every_header_param_in_declaration_order() -> None:
    harness = _RecordingHarness([7, 14, 21, 28])
    compiler = _FixedCompiler()
    runner = infrastructure.KernelRunner(harness, compiler)

    values = runner.run(
        _SCALE_SOURCE,
        params={"SCALE": 7, "DATA": [1, 2, 3, 4]},
        grid=(4, 1, 1),
        block=(4, 1, 1),
        output_param=1,
        output_count=4,
    )

    assert values == [7, 14, 21, 28]
    assert compiler.sources == [_SCALE_SOURCE]
    (request,) = harness.requests
    assert request["op"] == "correctness"
    assert request["ptx"] == "fake ptx"
    assert request["kernel"] == "main"
    assert request["params"] == [
        {"type": "i64", "value": 7},
        {"type": "i64[]", "values": [1, 2, 3, 4]},
    ]
    assert request["grid"] == [4, 1, 1]
    assert request["block"] == [4, 1, 1]
    assert request["output_param"] == 1
    assert request["output_count"] == 4


def test_kernel_runner_zero_fills_array_padding_and_omitted_params() -> None:
    harness = _RecordingHarness([0])
    runner = infrastructure.KernelRunner(harness, _FixedCompiler())

    runner.run(_SCALE_SOURCE, params={"DATA": [1, 2]}, output_param=1)

    assert harness.requests[0]["params"] == [
        {"type": "i64", "value": 0},
        {"type": "i64[]", "values": [1, 2, 0, 0]},
    ]
    assert harness.requests[0]["output_count"] is None


def test_kernel_runner_reads_float_output_as_floats() -> None:
    source = "\\! kernel main\n\\! param DATA f64[2]\n"
    harness = _RecordingHarness([1.5, -2.25])
    runner = infrastructure.KernelRunner(harness, _FixedCompiler())

    assert runner.run(source, params={"DATA": [1.5, 2.5]}) == [1.5, -2.25]
    assert harness.requests[0]["params"] == [{"type": "f64[]", "values": [1.5, 2.5]}]


def test_kernel_runner_rejects_a_scalar_output_param() -> None:
    source = "\\! kernel main\n\\! param SCALE i64\n\\! param DATA i64[2]\n"
    runner = infrastructure.KernelRunner(_RecordingHarness([]), _FixedCompiler())

    with pytest.raises(ValueError, match="scalar and cannot be read back"):
        runner.run(source, output_param=0)


def test_kernel_runner_rejects_an_output_param_out_of_range() -> None:
    runner = infrastructure.KernelRunner(_RecordingHarness([]), _FixedCompiler())

    with pytest.raises(ValueError, match="out of range"):
        runner.run(_SCALE_SOURCE, output_param=5)


def test_kernel_runner_surfaces_a_harness_error() -> None:
    class _FailingHarness:
        def request(self, _payload: dict[str, object]) -> dict[str, object]:
            msg = "cuModuleLoadData failed with error 218"
            raise infrastructure.HarnessError(msg)

    runner = infrastructure.KernelRunner(_FailingHarness(), _FixedCompiler())

    with pytest.raises(infrastructure.HarnessError, match="cuModuleLoadData"):
        runner.run(_SCALE_SOURCE, output_param=1)


class _BindingFakeDriver:
    """Device memory plus the binding's host-argument rule, as measured on the rental.

    cuda-python coerces its host argument to an int before it tries the buffer protocol, and
    numpy 1.x turns a one-element array into 0. A bare single-element array therefore reaches
    the driver as a null address and the copy fails with CUDA error 1, which is why every
    single-cell buffer broke while bytes and memoryview worked at the same length.
    """

    ERROR_INVALID_VALUE = 1

    def __init__(self) -> None:
        self.memory: dict[int, bytearray] = {}
        self.rejected: list[int] = []

    def surface(self) -> SimpleNamespace:
        """The object CudaGpu talks to, carrying the binding's own entry-point names."""
        return SimpleNamespace(
            cuMemAlloc=self.alloc,
            cuMemFree=self.free,
            cuMemcpyHtoD=self.to_device,
            cuMemcpyDtoH=self.to_host,
        )

    def alloc(self, nbytes: int) -> tuple[int, int]:
        ptr = len(self.memory) + 1
        self.memory[ptr] = bytearray(nbytes)
        return (0, ptr)

    def free(self, ptr: int) -> tuple[int]:
        del self.memory[ptr]
        return (0,)

    @classmethod
    def _host_bytes(cls, src: object, nbytes: int) -> bytes | None:
        if isinstance(src, np.ndarray) and src.size == 1:
            return None
        return bytes(memoryview(src))[:nbytes]

    def to_device(self, ptr: int, src: object, nbytes: int) -> tuple[int]:
        raw = self._host_bytes(src, nbytes)
        if raw is None:
            self.rejected.append(nbytes)
            return (self.ERROR_INVALID_VALUE,)
        self.memory[ptr][:nbytes] = raw
        return (0,)

    def to_host(self, dst: object, ptr: int, nbytes: int) -> tuple[int]:
        if self._host_bytes(dst, nbytes) is None:
            self.rejected.append(nbytes)
            return (self.ERROR_INVALID_VALUE,)
        memoryview(dst).cast("B")[:nbytes] = self.memory[ptr][:nbytes]
        return (0,)


def test_upload_and_download_accept_every_buffer_size() -> None:
    driver = _BindingFakeDriver()
    gpu = object.__new__(remote_harness.CudaGpu)
    gpu._drv = driver.surface()  # noqa: SLF001 - the seam the wrappers reach the bindings through

    for cells, dtype in ((1, np.float64), (1, np.int64), (2, np.float64), (1024, np.float64)):
        data = np.arange(cells, dtype=dtype)
        ptr = gpu.alloc(data.nbytes)
        gpu.upload(ptr, data)
        assert np.array_equal(gpu.download(ptr, cells, dtype), data), (cells, dtype)
        gpu.free(ptr)

    assert driver.rejected == []


def test_bench_same_kernel_reports_a_gap_of_one_through_the_contract() -> None:
    fake = FakeGpu()
    fake.register("scale", _scale_kernel, 10.0)
    response = remote_harness.handle_request(_scale_bench_request("scale", "scale"), fake)

    result = _parse_single_case(response)
    assert result.correct
    assert response["validation"] == {"base": True, "head": True}
    stats = result.stats
    assert stats is not None
    assert stats.ratio_median == 1.0
    assert bench_stats.verdict(result.samples) == bench_stats.Verdict.NO_CHANGE
    assert len(result.samples) == 6
    firsts = [sample.first for sample in result.samples]
    assert firsts == [bench_stats.Side.BASE, bench_stats.Side.HEAD] * 3
    assert result.environment.gpu_uuid == "fake-gpu-uuid"
    assert result.environment.base_commit == "aaa"


def test_bench_flags_a_slower_head() -> None:
    fake = FakeGpu()
    fake.register("fast", _scale_kernel, 10.0)
    fake.register("slow", _scale_kernel, 20.0)
    response = remote_harness.handle_request(_scale_bench_request("fast", "slow"), fake)

    result = _parse_single_case(response)
    assert result.correct
    stats = result.stats
    assert stats is not None
    assert stats.ratio_median == 2.0
    assert bench_stats.verdict(result.samples) == bench_stats.Verdict.SLOWER


def test_bench_times_a_host_library_side() -> None:
    fake = FakeGpu()
    fake.register("matmul", _matmul_kernel, 50.0)
    response = remote_harness.handle_request(_matmul_bench_request("matmul"), fake)

    result = _parse_single_case(response)
    assert result.correct
    assert response["validation"] == {"base": True, "head": True}
    stats = result.stats
    assert stats is not None
    assert stats.ratio_median == pytest.approx(0.2)
    assert bench_stats.verdict(result.samples) == bench_stats.Verdict.FASTER


def test_a_poisoned_output_fails_validation_and_produces_no_timings() -> None:
    fake = FakeGpu()
    fake.register("good", _scale_kernel, 10.0)
    fake.register("poison", _poison_kernel, 10.0)
    response = remote_harness.handle_request(_scale_bench_request("good", "poison"), fake)

    result = _parse_single_case(response)
    assert not result.correct
    assert result.samples == ()
    assert result.stats is None
    assert response["validation"] == {"base": True, "head": False}


def test_input_digests_are_deterministic_across_invocations_and_match_the_client() -> None:
    request = _scale_bench_request("scale", "scale")
    first = FakeGpu()
    first.register("scale", _scale_kernel, 10.0)
    second = FakeGpu()
    second.register("scale", _scale_kernel, 10.0)
    one = remote_harness.handle_request(request, first)
    two = remote_harness.handle_request(request, second)
    assert one["input_digests"] == two["input_digests"]

    spec = manifest.InputSpec(kind="int_uniform", dtype="i64", seed=11, low=-1000, high=1000)
    want = np.asarray(manifest.generate(spec, SCALE_ELEMENTS), dtype=np.int64).tobytes()
    digests = one["input_digests"]
    assert digests["INPUT"] == hashlib.sha256(want).hexdigest()


def test_harness_generator_matches_the_manifest_for_every_case_spec() -> None:
    for case in CASES:
        for name, spec in case.inputs.items():
            param = case.param(name)
            harness_param = remote_harness.ArrayParam(
                name=name, type=param.type, dtype=param.element_type, cells=param.elements
            )
            harness_spec = remote_harness.InputSpec(
                kind=spec.kind, seed=spec.seed, low=spec.low, high=spec.high, length=spec.length
            )
            got = remote_harness.generate_buffer(harness_param, harness_spec)
            want_values = manifest.generate(spec, param.elements)
            if param.element_type == "f32":
                want = np.asarray(manifest.pack_f32(want_values), dtype=np.int64)
            else:
                want = np.asarray(
                    want_values, dtype=np.int64 if param.type == "i64" else np.float64
                )
            assert np.array_equal(got, want), (case.id, name)


def test_harness_references_match_the_manifest_on_the_correctness_configs() -> None:
    for case in CASES:
        arrays = {}
        device_visible = {}
        for name, values in case.correctness.arrays.items():
            dtype = case.dtype_of(name)
            np_dtype = np.float32 if dtype == "f32" else np.int64 if dtype == "i64" else np.float64
            arrays[name] = np.asarray(values, dtype=np_dtype)
            # The manifest reference sees raw values; the harness sees them after
            # rounding into the param dtype, same as the kernels under test.
            device_visible[name] = arrays[name].tolist()
        scalars = {name: float(value) for name, value in case.correctness.params.items()}
        got = remote_harness.REFERENCES[case.reference](arrays, scalars)
        want = manifest.reference(case.reference, {**case.correctness.params, **device_visible})
        assert np.allclose(got.astype(np.float64), want, rtol=1e-9, atol=1e-12), case.id
        manifest_want = manifest.reference(
            case.reference, {**case.correctness.params, **case.correctness.arrays}
        )
        assert remote_harness.within_tolerance(
            got.astype(np.float64), np.asarray(manifest_want), *case.tolerance
        ), case.id


def test_within_tolerance_matches_the_manifest_rule() -> None:
    pairs = [
        ([1.0], [1.0]),
        ([1.0], [1.05]),
        ([1.0], [1.2]),
        ([-1.0], [-1.05]),
        ([0.0], [0.005]),
        ([0.0], [0.001]),
        ([1.0, 2.0], [1.0]),
    ]
    for rtol, atol in [(0.0, 0.0), (1e-9, 1e-9), (0.1, 0.0), (0.0, 0.01)]:
        for actual, expected in pairs:
            got = remote_harness.within_tolerance(
                np.asarray(actual), np.asarray(expected), rtol, atol
            )
            want = manifest.within_tolerance(actual, expected, rtol, atol)
            assert got == want, (actual, expected, rtol, atol)


class _FakeSubprocess:
    """Stands in for the subprocess module inside gpu_test.infrastructure."""

    PIPE = subprocess.PIPE
    DEVNULL = subprocess.DEVNULL
    TimeoutExpired = subprocess.TimeoutExpired
    Popen = subprocess.Popen

    def __init__(self, outcomes: list[object], tick_s: float = 0.0) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[list[str]] = []
        self._tick_s = tick_s
        self.clock = 0.0

    def run(self, cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        self.clock += self._tick_s
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


class _FakeSdk:
    """A Vast.ai SDK stand-in that answers every call locally.

    `_log_existing_instances` runs at session entry, so an incomplete fake would reach the real
    client and put a network call in a keyless run.
    """

    def show_instances(self) -> list[dict[str, object]]:
        return []


def _bare_session(monkeypatch: pytest.MonkeyPatch) -> infrastructure.VastSession:
    monkeypatch.setattr(infrastructure, "VastAI", lambda *_args, **_kwargs: _FakeSdk())
    return infrastructure.VastSession("key")


def _ssh_ready_session(monkeypatch: pytest.MonkeyPatch) -> infrastructure.VastSession:
    session = _bare_session(monkeypatch)
    session.ssh_host = "203.0.113.1"
    session.ssh_port = 12345
    session.ssh_private_key = Path("/nonexistent-key")
    monkeypatch.setattr(infrastructure.time, "sleep", lambda _seconds: None)
    return session


def test_ssh_probe_timeout_is_retried_like_a_failed_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _ssh_ready_session(monkeypatch)
    probe_ok = subprocess.CompletedProcess(args=[], returncode=0)
    fake = _FakeSubprocess([subprocess.TimeoutExpired(cmd="ssh", timeout=45), probe_ok])
    monkeypatch.setattr(infrastructure, "subprocess", fake)

    session._wait_for_sshd()  # noqa: SLF001 - drives the session's own probe seam

    assert len(fake.calls) == 2


def test_sshd_never_ready_raises_after_the_poll_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _ssh_ready_session(monkeypatch)
    fake = _FakeSubprocess(
        [subprocess.TimeoutExpired(cmd="ssh", timeout=45)] * 8,
        tick_s=infrastructure.POLL_TIMEOUT_S / 4,
    )
    monkeypatch.setattr(infrastructure, "subprocess", fake)
    monkeypatch.setattr(infrastructure.time, "monotonic", lambda: fake.clock)

    with pytest.raises(TimeoutError, match="did not become ready"):
        session._wait_for_sshd()  # noqa: SLF001 - drives the session's own probe seam

    assert len(fake.calls) >= 2


def _recording_session(monkeypatch: pytest.MonkeyPatch) -> tuple:
    session = _bare_session(monkeypatch)
    commands: list[str] = []
    uploads: list[tuple[object, str]] = []
    failures = {"install": 0}

    def fake_ssh_run(cmd: str, **_kwargs: object) -> str:
        commands.append(cmd)
        if "pip install" in cmd and failures["install"]:
            failures["install"] -= 1
            msg = "mirror unreachable"
            raise RuntimeError(msg)
        return ""

    monkeypatch.setattr(session, "ssh_run", fake_ssh_run)
    monkeypatch.setattr(session, "scp_upload", lambda src, dst: uploads.append((src, dst)))
    return session, commands, uploads, failures


def test_provision_retries_a_failed_pip_install_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session, commands, uploads, failures = _recording_session(monkeypatch)
    failures["install"] = 1
    session = _session

    session._provision_harness()  # noqa: SLF001 - drives the session's provisioning seam

    installs = [cmd for cmd in commands if "pip install" in cmd]
    assert len(installs) == 2
    assert commands[0].startswith("command -v python3")
    assert commands[-1] == "python3 -c 'import cuda, numpy'"
    assert uploads == [(infrastructure.HARNESS_SRC, infrastructure.HARNESS_REMOTE)]


def test_provision_gives_up_after_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _session, commands, uploads, failures = _recording_session(monkeypatch)
    failures["install"] = 2
    session = _session

    with pytest.raises(RuntimeError, match="mirror unreachable"):
        session._provision_harness()  # noqa: SLF001 - drives the session's provisioning seam

    installs = [cmd for cmd in commands if "pip install" in cmd]
    assert len(installs) == 2
    assert uploads == []


def _head_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def _bench_params(case: manifest.Case) -> list[dict[str, object]]:
    params = []
    for param in case.params:
        if param.is_array:
            params.append(
                {
                    "name": param.name,
                    "kind": "array",
                    "type": param.type,
                    "cells": param.elements,
                    "dtype": param.element_type,
                }
            )
        else:
            params.append({"name": param.name, "kind": "scalar", "type": param.type})
    return params


def _bench_inputs(case: manifest.Case) -> dict[str, dict[str, object]]:
    return {
        name: {
            "kind": spec.kind,
            "dtype": spec.dtype,
            "seed": spec.seed,
            "low": spec.low,
            "high": spec.high,
            "length": spec.length,
        }
        for name, spec in case.inputs.items()
    }


@pytest.mark.gpu
@pytest.mark.timeout(1500)
def test_bench_same_ptx_reports_gap_of_one_on_a_rental(
    gpu_session: infrastructure.VastSession, compiler: infrastructure.Compiler
) -> None:
    case = next(case for case in CASES if case.id == "scale_i64")
    ptx = manifest.compile_source(
        case.forth.read_text(), compiler.binary, compiler.arch, label=case.id
    )
    commit = _head_commit()
    side = {
        "kind": "ptx",
        "ptx": ptx,
        "kernel": case.kernel,
        "launch": {"grid": list(case.launch.grid), "block": list(case.launch.block)},
    }
    request = {
        "op": "bench",
        "case_id": case.id,
        "case_version": case.version,
        "reference": case.reference,
        "tolerance": list(case.tolerance),
        "params": _bench_params(case),
        "scalars": {"N": case.param("INPUT").elements, "SCALE": 7},
        "inputs": _bench_inputs(case),
        "output": case.correctness.output_param,
        "base": side,
        "head": side,
        "timing": {
            "warmup": manifest.MIN_WARMUP,
            "samples": manifest.MIN_SAMPLES,
            "sample_ms": manifest.MIN_SAMPLE_MS,
        },
        "environment": {"base_commit": commit, "head_commit": commit},
    }

    with HarnessClient(gpu_session) as harness:
        ping = harness.request({"op": "ping"})
        response = harness.request(request)

    assert ping["ok"] is True
    result = _parse_single_case(response)
    assert result.correct
    assert response["validation"] == {"base": True, "head": True}
    stats = result.stats
    assert stats is not None
    limit = bench_stats.noise_threshold(stats.ratio_mad)
    assert abs(stats.ratio_median - 1.0) <= limit
    assert bench_stats.verdict(result.samples) == bench_stats.Verdict.NO_CHANGE
    assert len(result.samples) == manifest.MIN_SAMPLES
    firsts = [sample.first for sample in result.samples]
    assert firsts[0] == bench_stats.Side.BASE
    assert firsts[1] == bench_stats.Side.HEAD
    environment = result.environment
    assert environment.gpu_uuid
    assert environment.driver_version
    assert environment.sm_clock_mhz > 0
    logger.info(
        "gap smoke: base_median=%.1fus head_median=%.1fus ratio=%.4f ratio_mad=%.4f "
        "batch_launches=%s gpu=%s driver=%s",
        stats.base_median_us,
        stats.head_median_us,
        stats.ratio_median,
        stats.ratio_mad,
        response.get("batch_launches"),
        environment.gpu_uuid,
        environment.driver_version,
    )


@pytest.mark.gpu
@pytest.mark.timeout(1500)
def test_correctness_mode_reads_back_expected_values_on_a_rental(
    gpu_session: infrastructure.VastSession, compiler: infrastructure.Compiler
) -> None:
    case = next(case for case in CASES if case.id == "scale_i64")
    ptx = manifest.compile_source(
        case.forth.read_text(), compiler.binary, compiler.arch, label=case.id
    )
    correct = case.correctness
    inputs = [int(value) for value in correct.arrays["INPUT"]]
    request = {
        "op": "correctness",
        "ptx": ptx,
        "kernel": case.kernel,
        "params": [
            {"type": "i64", "value": int(correct.params["N"])},
            {"type": "i64", "value": int(correct.params["SCALE"])},
            {"type": "i64[]", "values": inputs},
            {"type": "i64[]", "values": [0] * len(inputs)},
        ],
        "grid": list(correct.launch.grid),
        "block": list(correct.launch.block),
        "output_param": case.output_param_index(),
        "output_count": len(correct.expected),
    }

    with HarnessClient(gpu_session) as harness:
        response = harness.request(request)

    expected = [int(value) for value in correct.expected]
    assert response["values"] == expected
