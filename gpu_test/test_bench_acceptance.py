"""Checks the acceptance fixture pair without a GPU.

The pair's two sources compile locally, and these tests pin the three properties the fixture relies
on: both sides compile to the same entry point with the same parameter order, the slow arm does
enough dependent work per element to be at least 2x apart, and each arm states the verdict it has to
produce. The gpu-marked test runs both arms on one rental.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from bench import acceptance, compare, gap
from benchmarks import manifest
from gpu_test import bench_stats

if TYPE_CHECKING:
    from pathlib import Path

REPO_ROOT = manifest.CASE_ROOT.parent
CASES = manifest.load()
COMPILER = manifest.find_compiler(REPO_ROOT)
TEMPLATE = acceptance.template_case(CASES)
requires_compiler = pytest.mark.skipif(
    not COMPILER.is_file(),
    reason=f"warpforthc not built at {COMPILER}; build it or set WARPFORTHC",
)


def _bench_response(case: manifest.Case, *, base_us: float, head_us: float) -> dict[str, object]:
    samples = [
        {"base_us": base_us, "head_us": head_us, "first": "base" if index % 2 == 0 else "head"}
        for index in range(5)
    ]
    return {
        "ok": True,
        "result": {
            "case_id": case.id,
            "case_version": case.version,
            "correct": True,
            "environment": {
                "gpu_uuid": "gpu-0000",
                "driver_version": "12.4",
                "sm_clock_mhz": 1500,
                "memory_clock_mhz": 7000,
                "base_commit": "fast",
                "head_commit": "slow",
            },
            "samples": samples,
        },
    }


class FixedHarness:
    """Answers every bench call with one fixed ratio."""

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio
        self.requests: list[dict[str, object]] = []

    def request(self, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append(payload)
        case = next(item for item in CASES if item.id == payload["case_id"])
        return _bench_response(case, base_us=100.0, head_us=100.0 * self.ratio)


def test_both_arms_name_files_that_exist() -> None:
    for arm in acceptance.ARMS:
        assert arm.base.is_file(), arm.base
        assert arm.head.is_file(), arm.head


def test_the_pair_is_one_fast_arm_and_one_deliberately_slow_twin() -> None:
    slow = acceptance.ARMS[0]
    same = acceptance.ARMS[1]
    assert slow.base == acceptance.FAST
    assert slow.head == acceptance.SLOW
    assert slow.expect is bench_stats.Verdict.SLOWER
    assert slow.ratio_floor >= 2.0
    assert same.base == same.head == acceptance.FAST
    assert same.expect is bench_stats.Verdict.NO_CHANGE
    assert same.ratio_floor == 0.0, "a no-change arm has no ratio to clear"


def test_the_fixture_header_documents_the_slowdown() -> None:
    text = acceptance.SLOW.read_text()
    assert "512-trip" in text
    assert "2x floor" in text


def test_the_fixture_is_not_a_case_in_the_manifest() -> None:
    assert {case.forth for case in CASES}.isdisjoint({acceptance.FAST, acceptance.SLOW})


def test_the_fixture_bodies_differ_only_in_the_per_element_work() -> None:
    def body(path: Path) -> list[str]:
        return [
            line
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("\\")
        ]

    fast = body(acceptance.FAST)
    slow = body(acceptance.SLOW)
    assert fast[0] == slow[0], "same kernel header"
    assert [line for line in slow if line.strip() != "512 0 DO S>F F>S LOOP"] == fast


@requires_compiler
def test_both_arms_compile_to_the_same_entry_point() -> None:
    for path in (acceptance.FAST, acceptance.SLOW):
        ptx = acceptance.compile_side(path, COMPILER, "sm_89")
        assert f".visible .entry {acceptance.KERNEL}(" in ptx, path


@requires_compiler
def test_both_arms_declare_the_anchor_cases_parameter_order() -> None:
    def declared(path: Path) -> list[str]:
        return re.findall(r"\\! param (\w+) ", path.read_text())

    template_order = [param.name for param in TEMPLATE.params]
    assert declared(acceptance.FAST) == template_order
    assert declared(acceptance.SLOW) == template_order


@requires_compiler
def test_the_slow_arm_executes_far_more_work_per_element() -> None:
    fast = acceptance.compile_side(acceptance.FAST, COMPILER, "sm_89")
    slow = acceptance.compile_side(acceptance.SLOW, COMPILER, "sm_89")
    loop_of = re.compile(r"\$L__BB0_3:(.*?)@%p\d+ bra \s+\$L__BB0_3", re.DOTALL)
    inner = loop_of.search(slow)
    assert inner is not None, "the twin has no inner chain loop"
    assert "cvt.rn.f64.s64" in inner.group(1)
    assert "cvt.rzi.s64.f64" in inner.group(1)
    assert "cvt.rn.f64.s64" not in fast


@requires_compiler
def test_an_identical_arm_reads_as_no_change() -> None:
    reading = acceptance.read_arm(
        acceptance.ARMS[1], TEMPLATE, COMPILER, "sm_89", FixedHarness(1.0)
    )
    assert reading.outcome.verdict is bench_stats.Verdict.NO_CHANGE
    assert reading.passed, reading.line()


@requires_compiler
def test_a_slowed_arm_reads_as_a_regression() -> None:
    reading = acceptance.read_arm(
        acceptance.ARMS[0], TEMPLATE, COMPILER, "sm_89", FixedHarness(3.0)
    )
    assert reading.outcome.verdict is bench_stats.Verdict.SLOWER
    assert reading.passed, reading.line()


@requires_compiler
def test_a_slow_twin_inside_the_noise_threshold_does_not_pass_the_arm() -> None:
    reading = acceptance.read_arm(
        acceptance.ARMS[0], TEMPLATE, COMPILER, "sm_89", FixedHarness(1.02)
    )
    assert reading.outcome.verdict is bench_stats.Verdict.NO_CHANGE
    assert not reading.passed, reading.line()


@requires_compiler
def test_a_near_threshold_slowdown_reads_as_no_change_not_a_regression() -> None:
    reading = acceptance.read_arm(
        acceptance.ARMS[1], TEMPLATE, COMPILER, "sm_89", FixedHarness(1.019)
    )
    assert reading.outcome.verdict is bench_stats.Verdict.NO_CHANGE
    assert not reading.outcome.regressed


@requires_compiler
def test_an_arm_measures_both_sides_and_no_cuda_baseline() -> None:
    harness = FixedHarness(2.0)
    reading = acceptance.read_arm(acceptance.ARMS[0], TEMPLATE, COMPILER, "sm_89", harness)
    assert len(harness.requests) == 1, "one paired call, no CUDA-side call"
    request = harness.requests[0]
    assert request["base"]["ptx"] != request["head"]["ptx"]
    assert reading.outcome.head_gap is None


@requires_compiler
def test_an_arm_launches_the_fixtures_entry_point_on_both_sides() -> None:
    harness = FixedHarness(2.0)
    acceptance.read_arm(acceptance.ARMS[0], TEMPLATE, COMPILER, "sm_89", harness)
    request = harness.requests[0]
    assert request["base"]["kernel"] == acceptance.KERNEL
    assert request["head"]["kernel"] == acceptance.KERNEL
    assert f".visible .entry {acceptance.KERNEL}(" in request["base"]["ptx"]
    assert f".visible .entry {acceptance.KERNEL}(" in request["head"]["ptx"]


@requires_compiler
def test_the_printable_lines_state_each_arms_verdict_and_result() -> None:
    readings = [
        acceptance.read_arm(acceptance.ARMS[1], TEMPLATE, COMPILER, "sm_89", FixedHarness(1.0)),
        acceptance.read_arm(acceptance.ARMS[0], TEMPLATE, COMPILER, "sm_89", FixedHarness(3.0)),
    ]
    text = "\n".join(acceptance.lines(readings))
    assert "fast.forth -> fast.forth" in text
    assert "fast.forth -> slow.forth" in text
    assert "no change" in text
    assert "slower" in text
    assert text.count("pass") == 2


def test_the_template_case_supplies_the_fixtures_buffers() -> None:
    assert TEMPLATE.id == "scale_i64"
    assert gap.timed_scalars(TEMPLATE)["N"] == 4194304
    with pytest.raises(acceptance.FixtureError, match="no scale_i64"):
        acceptance.template_case([case for case in CASES if case.id != "scale_i64"])


def test_a_pairing_states_the_verdict_it_expects() -> None:
    assert [arm.expect for arm in acceptance.ARMS] == [
        bench_stats.Verdict.SLOWER,
        bench_stats.Verdict.NO_CHANGE,
    ]
    assert isinstance(compare.Outcome(), compare.Outcome)
