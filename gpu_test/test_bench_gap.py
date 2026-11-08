"""Checks the gap command without a GPU.

The rental is faked at the harness and remote-compile seams, so these tests pin the requests the
real harness receives, both correctness gates, the geometric mean the table prints, the noise
floor two runs of the suite yield, and one broken case leaving the rest of the table reported.
The gpu-marked test drives the same flow on a rented GPU.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from bench import cli, gap
from bench.report import Report
from benchmarks import manifest
from gpu_test import bench_stats, infrastructure

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = manifest.CASE_ROOT.parent
CASES = manifest.load()
BY_ID = {case.id: case for case in CASES}
BY_KERNEL = {case.kernel: case for case in CASES}
COMPILER = manifest.find_compiler(REPO_ROOT)
requires_compiler = pytest.mark.skipif(
    not COMPILER.is_file(),
    reason=f"warpforthc not built at {COMPILER}; build it or set WARPFORTHC",
)

ENVIRONMENT = bench_stats.Environment(
    gpu_uuid="gpu-0000",
    driver_version="12.4",
    sm_clock_mhz=1500,
    memory_clock_mhz=7000,
    base_commit="aaa",
    head_commit="aaa",
)


def _readback_cells(case: manifest.Case) -> list:
    """The output cells a correct run returns, so the fake can serve a passing readback."""
    dtype = case.dtype_of(case.correctness.output_param)
    if dtype == "f32":
        return manifest.pack_f32(case.correctness.expected)
    if dtype == "i64":
        return [int(value) for value in case.correctness.expected]
    return list(case.correctness.expected)


def _bench_response(
    case: manifest.Case, *, base_us: float, head_us: float, correct: bool
) -> dict[str, object]:
    samples = [
        {
            "base_us": base_us,
            "head_us": head_us,
            "first": "base" if index % 2 == 0 else "head",
        }
        for index in range(5)
    ]
    return {
        "ok": True,
        "result": {
            "case_id": case.id,
            "case_version": case.version,
            "correct": correct,
            "environment": {
                field: getattr(ENVIRONMENT, field)
                for field in (
                    "gpu_uuid",
                    "driver_version",
                    "sm_clock_mhz",
                    "memory_clock_mhz",
                    "base_commit",
                    "head_commit",
                )
            },
            "samples": samples if correct else [],
        },
        "validation": {"base": correct, "head": correct},
    }


class FakeHarness:
    """Stands in for HarnessClient: records requests, answers by op and case id."""

    def __init__(
        self,
        gaps: dict[str, float] | None = None,
        *,
        wrong_small: Sequence[str] = (),
        wrong_timed: Sequence[str] = (),
    ) -> None:
        self.gaps = gaps or {}
        self.wrong_small = set(wrong_small)
        self.wrong_timed = set(wrong_timed)
        self.requests: list[dict[str, object]] = []

    def request(self, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append(payload)
        if payload.get("op") == "ping":
            return {
                "ok": True,
                "environment": {
                    "gpu_uuid": ENVIRONMENT.gpu_uuid,
                    "driver_version": ENVIRONMENT.driver_version,
                    "sm_clock_mhz": ENVIRONMENT.sm_clock_mhz,
                    "memory_clock_mhz": ENVIRONMENT.memory_clock_mhz,
                },
            }
        if payload["op"] == "correctness":
            case = BY_KERNEL[str(payload["kernel"])]
            if case.id in self.wrong_small:
                return {"ok": True, "values": [0] * len(_readback_cells(case))}
            return {"ok": True, "values": _readback_cells(case)}
        case = BY_ID[str(payload["case_id"])]
        return _bench_response(
            case,
            base_us=100.0,
            head_us=100.0 * self.gaps.get(case.id, 1.0),
            correct=case.id not in self.wrong_timed,
        )

    def of_op(self, op: str) -> list[dict[str, object]]:
        return [payload for payload in self.requests if payload.get("op") == op]


class FakeSession:
    """Stands in for VastSession: serves the upload-and-compile seam without a rental."""

    def __init__(self) -> None:
        self.instance_id = 4242
        self.compiles: list[tuple[str, tuple[str, ...]]] = []

    def compile_cuda_ptx(self, source: object, flags: Sequence[str], *, name: str) -> str:
        self.compiles.append((str(source), tuple(flags)))
        return f".version 8.0\n.visible .entry {name}( )\n"


def _item(case_id: str, arch: str = "sm_89") -> gap.Plan:
    item = gap.plan(BY_ID[case_id], COMPILER, arch)
    assert isinstance(item, gap.Plan), case_id
    return item


def _items(arch: str = "sm_89") -> list[gap.Item]:
    return gap.compile_baselines([_item(case.id, arch) for case in CASES], FakeSession())


def _run(harness: FakeHarness, arch: str = "sm_89") -> gap.Run:
    return gap.run_once(_items(arch), harness, {"head_commit": "aaa"})


def _report(*runs: gap.Run, arch: str = "sm_89") -> Report:
    return Report(arch=arch, environment=ENVIRONMENT, runs=runs, max_ratio_mad=0.05)


def test_element_count_counts_f32_as_two_elements_per_cell() -> None:
    attention = BY_ID["attention_f32"]
    assert attention.param("QMAT").elements == 16384
    assert gap.element_count(attention, "QMAT") == 32768
    assert gap.element_count(attention, "OMAT") == 32768
    assert gap.element_count(BY_ID["axpy_f64"], "X") == 4194304


def test_every_case_has_a_timed_scalar_rule() -> None:
    for case in CASES:
        assert gap.timed_scalars(case), case.id


def test_timed_scalars_derive_each_case_size_from_its_buffers() -> None:
    assert gap.timed_scalars(BY_ID["scale_i64"]) == {"N": 4194304, "SCALE": -7}
    assert gap.timed_scalars(BY_ID["axpy_f64"]) == {"N": 4194304, "ALPHA": 1.5}
    assert gap.timed_scalars(BY_ID["reduce_sum_f64"]) == {"N": 1048576}
    assert gap.timed_scalars(BY_ID["matmul_f64_naive"]) == {"N": 128}
    assert gap.timed_scalars(BY_ID["matmul_f64_tiled"]) == {"N": 256}
    attention = gap.timed_scalars(BY_ID["attention_f32"])
    assert (attention["NROW"], attention["NKEY"], attention["HDIM"]) == (512, 512, 64)
    assert attention["SCALE"] == pytest.approx(1.0 / math.sqrt(64))


def test_timed_scalars_move_every_buffer_the_manifest_declares() -> None:
    for case in CASES:
        scalars = gap.timed_scalars(case)
        assert set(scalars) == {param.name for param in case.params if not param.is_array}, case.id


def test_timed_scalars_agree_with_each_cases_launch_geometry() -> None:
    """The manifest's grids are sized for the timed instance, so they cross-check the derivation.

    The naive matmul launches one thread per output element, the tiled one grid-strides over 16x16
    tiles, and the attention baseline launches four warps a block with one warp per query row.
    """
    naive = BY_ID["matmul_f64_naive"]
    assert gap.timed_scalars(naive)["N"] ** 2 == naive.launch.total_threads
    tiled = BY_ID["matmul_f64_tiled"]
    tiles = (gap.timed_scalars(tiled)["N"] // 16) ** 2
    assert tiles <= tiled.launch.total_threads
    attention = BY_ID["attention_f32"]
    baseline = attention.baseline.launch
    assert baseline is not None
    assert gap.timed_scalars(attention)["NROW"] == baseline.grid[0] * 4, "four warps a block"
    assert gap.timed_scalars(attention)["NKEY"] * gap.timed_scalars(attention)["HDIM"] == (
        gap.element_count(attention, "KMAT")
    )


def test_nvcc_flags_retarget_the_arch_flag_only() -> None:
    recorded = ("-O3", "-arch=sm_89", "--use_fast_math", "-std=c++17")
    assert gap.nvcc_flags(recorded, "sm_75") == (
        "-O3",
        "-arch=sm_75",
        "--use_fast_math",
        "-std=c++17",
    )


def test_every_baseline_flag_set_carries_an_arch() -> None:
    for case in CASES:
        assert any(flag.startswith("-arch=") for flag in case.baseline.nvcc_flags), case.id


@requires_compiler
def test_plan_prepares_every_case() -> None:
    for case in CASES:
        item = _item(case.id)
        assert f".visible .entry {case.kernel}(" in item.ptx, case.id
        assert item.baseline.nvcc_flags == gap.nvcc_flags(case.baseline.nvcc_flags, "sm_89")


@requires_compiler
def test_a_baseline_carries_the_requested_arch() -> None:
    assert "-arch=sm_75" in _item("scale_i64", "sm_75").baseline.nvcc_flags


@requires_compiler
def test_correctness_request_carries_the_small_instance_in_header_order() -> None:
    request = gap.correctness_request(_item("scale_i64"))
    assert request["op"] == "correctness"
    assert request["kernel"] == "scale_i64"
    assert request["output_param"] == 3
    assert request["output_count"] == 16
    assert [param["type"] for param in request["params"]] == ["i64", "i64", "i64[]", "i64[]"]
    assert request["params"][2]["values"][:3] == [-998, -655, -857]


@requires_compiler
def test_correctness_request_packs_f32_buffers_into_i64_cells() -> None:
    case = BY_ID["attention_f32"]
    request = gap.correctness_request(_item(case.id))
    assert request["params"][4]["type"] == "i64[]"
    assert len(request["params"][4]["values"]) == case.correctness_output_cells()


@requires_compiler
def test_bench_request_uses_the_full_size_buffers_and_timed_scalars() -> None:
    case = BY_ID["matmul_f64_naive"]
    request = gap.bench_request(_item(case.id), {"kind": "ptx"}, {"head_commit": "aaa"})
    assert request["case_id"] == "matmul_f64_naive"
    assert request["scalars"] == {"N": 128}
    assert request["output"] == "C"
    assert request["timing"] == {
        "warmup": case.timing.warmup,
        "samples": case.timing.samples,
        "sample_ms": case.timing.sample_ms,
    }
    cells = {param["name"]: param["cells"] for param in request["params"] if "cells" in param}
    assert cells == {"A": 16384, "B": 16384, "C": 16384}
    assert request["head"]["kernel"] == "matmul_f64_naive"
    assert request["head"]["launch"] == {"grid": [64, 1, 1], "block": [256, 1, 1]}


@requires_compiler
def test_library_side_names_the_output_as_the_result_buffer() -> None:
    assert gap.library_side(_item("matmul_f64_tiled")) == {
        "kind": "host_lib",
        "library": "cublas",
        "symbol": "cublasDgemm",
        "n": 256,
        "a": "A",
        "b": "B",
        "c": "C",
    }


@requires_compiler
def test_a_kernel_baseline_compiles_remotely_with_the_rewritten_arch() -> None:
    session = FakeSession()
    side = gap.kernel_side(_item("scale_i64", "sm_80"), session)
    assert side["kernel"] == "scale_i64"
    assert side["launch"] == {"grid": [128, 1, 1], "block": [256, 1, 1]}
    source, flags = session.compiles[0]
    assert source.endswith("benchmarks/cases/scale_i64.cu")
    assert flags == ("-O3", "-arch=sm_80", "-std=c++17")


@requires_compiler
def test_only_a_kernel_baseline_reaches_the_remote_compiler() -> None:
    session = FakeSession()
    readied = gap.compile_baselines(_items(), session)
    assert all(isinstance(item, gap.Plan) for item in readied)
    assert [Path(source).name for source, _flags in session.compiles] == [
        Path(case.baseline.source).name for case in CASES if case.baseline.kind != "host_lib"
    ]


def test_environment_needs_every_device_field() -> None:
    ping = {
        "environment": {
            "gpu_uuid": "gpu-0000",
            "driver_version": "12.4",
            "sm_clock_mhz": 1500,
            "memory_clock_mhz": 7000,
        }
    }
    environment = gap.environment_of(ping, "abc123")
    assert environment.gpu_uuid == "gpu-0000"
    assert environment.base_commit == "abc123"
    assert environment.head_commit == "abc123"
    with pytest.raises(gap.GapError, match="memory_clock_mhz"):
        gap.environment_of({"environment": {"gpu_uuid": "x"}}, "abc123")


def test_select_cases_keeps_manifest_order_and_rejects_an_unknown_id() -> None:
    assert [case.id for case in cli.select_cases(CASES, ["axpy_f64", "scale_i64"])] == [
        "scale_i64",
        "axpy_f64",
    ]
    with pytest.raises(SystemExit, match="no case named nope"):
        cli.select_cases(CASES, ["nope"])


def test_the_parser_takes_the_gap_options() -> None:
    args = cli.build_parser().parse_args(
        ["gap", "--arch", "sm_80", "--case", "axpy_f64", "--repeats", "2", "--json"]
    )
    assert args.command == "gap"
    assert args.arch == "sm_80"
    assert args.case == ["axpy_f64"]
    assert args.repeats == 2
    assert args.json is True
    assert args.max_ratio_mad == bench_stats.MAX_STABLE_RATIO_MAD


@requires_compiler
def test_one_run_times_every_case_against_its_baseline() -> None:
    run = _run(FakeHarness({"scale_i64": 2.0, "axpy_f64": 1.5}))
    assert [row.case_id for row in run.rows] == [case.id for case in CASES]
    assert all(row.attempt.correct for row in run.rows)
    assert run.gap_of("scale_i64") == pytest.approx(2.0)
    assert run.gap_of("axpy_f64") == pytest.approx(1.5)
    assert run.gap_of("attention_f32") == pytest.approx(1.0)


@requires_compiler
def test_every_case_gets_one_correctness_check_and_one_timed_run() -> None:
    harness = FakeHarness()
    _run(harness)
    assert len(harness.of_op("correctness")) == len(CASES)
    assert len(harness.of_op("bench")) == len(CASES)


@requires_compiler
def test_a_failed_small_config_costs_only_its_own_row() -> None:
    harness = FakeHarness(wrong_small=["reduce_sum_f64"])
    run = _run(harness)
    assert {row.case_id for row in run.rows if not row.attempt.correct} == {"reduce_sum_f64"}
    assert run.gap_of("reduce_sum_f64") is None
    assert run.gap_of("scale_i64") is not None
    assert len(harness.of_op("bench")) == len(CASES) - 1
    row = next(row for row in run.rows if row.case_id == "reduce_sum_f64")
    assert "manifest" in row.note


@requires_compiler
def test_a_failed_timed_validation_produces_no_gap() -> None:
    run = _run(FakeHarness(wrong_timed=["axpy_f64"]))
    row = next(row for row in run.rows if row.case_id == "axpy_f64")
    assert not row.attempt.correct
    assert row.attempt.gap is None
    assert "validation" in row.note


@requires_compiler
def test_a_case_that_will_not_compile_keeps_its_row_and_the_rest_still_run() -> None:
    case = BY_ID["scale_i64"]
    missing = dataclasses.replace(case, forth=case.forth.with_name("missing.forth"))
    broken = gap.plan(missing, COMPILER, "sm_89")
    assert isinstance(broken, gap.Broken)
    items = [broken, *[_item(case.id) for case in CASES if case.id != "scale_i64"]]
    run = gap.run_once(
        gap.compile_baselines(items, FakeSession()), FakeHarness(), {"head_commit": "aaa"}
    )
    assert [row.case_id for row in run.rows] == [case.id for case in CASES]
    assert not run.rows[0].attempt.correct
    assert all(row.attempt.correct for row in run.rows[1:])


@requires_compiler
def test_the_table_prints_one_row_per_case_with_the_gap_and_both_medians() -> None:
    text = "\n".join(_report(_run(FakeHarness({"scale_i64": 2.0}))).lines())
    for case in CASES:
        assert case.id in text
    assert "2.00x" in text
    assert "100.0us" in text
    assert "200.0us" in text


@requires_compiler
def test_the_printed_geometric_mean_agrees_with_the_stats_module() -> None:
    run = _run(FakeHarness({"scale_i64": 2.0, "axpy_f64": 1.5}))
    report = _report(run)
    expected = bench_stats.geometric_mean_ratio(run.suite(ENVIRONMENT).cases)
    assert report.geometric_mean == pytest.approx(expected)
    line = next(line for line in report.lines() if line.startswith("geometric mean gap"))
    assert f"{expected:.3f}x" in line
    assert f"6 of {len(CASES)} stable cases" in line


@requires_compiler
def test_a_case_that_fails_is_left_out_of_the_geometric_mean() -> None:
    report = _report(_run(FakeHarness(wrong_timed=["matmul_f64_naive"])))
    assert report.complete is False
    assert f"5 of {len(CASES)} stable cases" in "\n".join(report.lines())


@requires_compiler
def test_two_runs_report_the_noise_floor_between_their_gaps() -> None:
    report = _report(
        _run(FakeHarness({"scale_i64": 2.0})),
        _run(FakeHarness({"scale_i64": 2.1})),
    )
    assert report.noise_floor == pytest.approx(0.05)
    text = "\n".join(report.lines())
    assert "same-rental repeat" in text
    assert "5.00%" in text


@requires_compiler
def test_the_result_document_records_each_baseline_and_its_flags() -> None:
    document = json.loads(_report(_run(FakeHarness(), "sm_80"), arch="sm_80").to_json())
    assert document["arch"] == "sm_80"
    assert document["geometric_mean"] == pytest.approx(1.0)
    assert len(document["cases"]) == len(CASES)
    scale = next(item for item in document["cases"] if item["case_id"] == "scale_i64")
    assert scale["baseline"]["kind"] == "ptx_kernel"
    assert scale["baseline"]["nvcc_flags"] == ["-O3", "-arch=sm_80", "-std=c++17"]
    matmul = next(item for item in document["cases"] if item["case_id"] == "matmul_f64_naive")
    assert matmul["baseline"]["kind"] == "host_lib"
    assert matmul["baseline"]["symbol"] == "cublasDgemm"
    assert len(scale["runs"][0]["samples"]) == 5


@requires_compiler
def test_the_result_document_marks_a_failed_case_and_keeps_its_reason() -> None:
    document = json.loads(_report(_run(FakeHarness(wrong_small=["reduce_sum_f64"]))).to_json())
    case = next(item for item in document["cases"] if item["case_id"] == "reduce_sum_f64")
    measurement = case["runs"][0]
    assert measurement["correct"] is False
    assert measurement["gap"] is None
    assert "manifest" in measurement["error"]


@requires_compiler
def test_a_repeat_of_the_same_rental_prints_both_runs() -> None:
    first = _run(FakeHarness({"scale_i64": 2.0}))
    text = "\n".join(_report(first, _run(FakeHarness({"scale_i64": 2.0}))).lines())
    assert "run 1" in text
    assert "run 2" in text
    assert "0.00%" in text


@pytest.mark.gpu
@pytest.mark.timeout(3600)
def test_gap_prints_a_six_case_table_on_one_rental(
    gpu_session: infrastructure.VastSession,
) -> None:
    """The plan's S2 lane 1: one rental, every timed case correctness-gated."""
    items = [_item(case.id, gpu_session.arch) for case in CASES]
    runs, environment = cli.measure(items, gpu_session, cli.head_commit(), 1)
    report = Report(
        arch=gpu_session.arch,
        environment=environment,
        runs=tuple(runs),
        max_ratio_mad=0.05,
    )
    text = "\n".join(report.lines())
    print(text)  # noqa: T201 - the lane artifact
    assert len(report.rows) == len(CASES)
    assert report.complete, text
