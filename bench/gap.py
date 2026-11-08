"""The gap measurement: WarpForth PTX against each case's single SOTA CUDA baseline.

A gap is WarpForth time over CUDA time, so a value above one means WarpForth is slower. A case
reports a gap only when it passed both gates: the small correctness instance, and the timed
instance's own validation against the case reference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from benchmarks import manifest
from gpu_test import bench_stats, infrastructure

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

CELL_BYTES = 8
ELEMENT_BYTES = {"i64": 8, "f64": 8, "f32": 4}
LIBRARY_OPERANDS = 2
# The manifest's attention buffers hold NROW*HDIM f32 elements, and the CUDA baseline keeps
# HDIM/32 of a row in registers per lane, so 64 is the head width the case is built for.
ATTENTION_HEAD_DIM = 64
ENVIRONMENT_FIELDS = ("gpu_uuid", "driver_version", "sm_clock_mhz", "memory_clock_mhz")


class GapError(Exception):
    """A case cannot be set up for measurement."""


CASE_ERRORS = (
    infrastructure.HarnessError,
    bench_stats.StatsError,
    GapError,
    OSError,
    RuntimeError,
)


class Harness(Protocol):
    """The harness surface the gap flow uses, which `HarnessClient` implements over a rental."""

    def request(self, payload: dict[str, object]) -> dict[str, object]: ...


class CudaCompiler(Protocol):
    """The remote nvcc surface, which `VastSession` implements over ssh."""

    def compile_cuda_ptx(self, source: Path, flags: Sequence[str], *, name: str) -> str: ...


@dataclass(frozen=True)
class Baseline:
    """The CUDA side of one case, as the result document records it."""

    kind: str
    source: str
    kernel: str
    library: str
    symbol: str
    nvcc_flags: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    """One case compiled locally, ready to time: the PTX both passes use, and its timed scalars."""

    case: manifest.Case
    baseline: Baseline
    ptx: str
    scalars: dict[str, int | float]
    cuda_side: dict[str, object] | None = None


@dataclass(frozen=True)
class Broken:
    """One case that never reached the rental, with the reason it did not."""

    case: manifest.Case
    baseline: Baseline
    error: str


Item = Plan | Broken


def element_count(case: manifest.Case, name: str) -> int:
    """A buffer's element count, which is twice its cell count when it stores f32."""
    param = case.param(name)
    return param.elements * CELL_BYTES // ELEMENT_BYTES[param.element_type]


def _scale_scalars(case: manifest.Case) -> dict[str, int | float]:
    return {"N": element_count(case, "INPUT"), "SCALE": case.correctness_scalars()["SCALE"]}


def _axpy_scalars(case: manifest.Case) -> dict[str, int | float]:
    return {"N": element_count(case, "X"), "ALPHA": case.correctness_scalars()["ALPHA"]}


def _reduce_scalars(case: manifest.Case) -> dict[str, int | float]:
    return {"N": element_count(case, "INPUT")}


def _matmul_scalars(case: manifest.Case) -> dict[str, int | float]:
    return {"N": math.isqrt(element_count(case, "A"))}


def _attention_scalars(case: manifest.Case) -> dict[str, int | float]:
    head_dim = ATTENTION_HEAD_DIM
    return {
        "NROW": element_count(case, "QMAT") // head_dim,
        "NKEY": element_count(case, "KMAT") // head_dim,
        "HDIM": head_dim,
        "SCALE": 1.0 / math.sqrt(head_dim),
    }


# The manifest records buffer sizes and only the correctness instance's scalar values, so the
# timed instance derives from the reference's scalar contract. Both matmul variants share one.
_TIMED_SCALARS: dict[str, Callable[[manifest.Case], dict[str, int | float]]] = {
    "scale_i64": _scale_scalars,
    "axpy_f64": _axpy_scalars,
    "reduce_sum_f64": _reduce_scalars,
    "matmul_f64": _matmul_scalars,
    "attention_f32": _attention_scalars,
}


def timed_scalars(case: manifest.Case) -> dict[str, int | float]:
    try:
        derive = _TIMED_SCALARS[case.reference]
    except KeyError:
        msg = f"{case.id}: no timed scalar rule for reference {case.reference}"
        raise GapError(msg) from None
    return derive(case)


def nvcc_flags(recorded: Sequence[str], arch: str) -> tuple[str, ...]:
    """The manifest's flags with the arch rewritten to the rental's, so its driver JITs the PTX."""
    return tuple(f"-arch={arch}" if flag.startswith("-arch=") else flag for flag in recorded)


def _baseline_of(case: manifest.Case, arch: str) -> Baseline:
    baseline = case.baseline
    return Baseline(
        kind=baseline.kind,
        source=baseline.source,
        kernel=baseline.kernel,
        library=baseline.library,
        symbol=baseline.symbol,
        nvcc_flags=nvcc_flags(baseline.nvcc_flags, arch),
    )


def plan(case: manifest.Case, compiler: Path, arch: str) -> Item:
    """Compile one case locally, so a source that will not build costs no rental time."""
    baseline = _baseline_of(case, arch)
    try:
        ptx = manifest.compile_case(case, compiler, arch)
        scalars = timed_scalars(case)
    except (manifest.ManifestError, GapError, OSError) as exc:
        return Broken(case=case, baseline=baseline, error=str(exc))
    return Plan(case=case, baseline=baseline, ptx=ptx, scalars=scalars)


def correctness_request(item: Plan) -> dict[str, object]:
    case = item.case
    values = case.correctness_params()
    params: list[dict[str, object]] = []
    for param in case.params:
        if param.is_array:
            params.append({"type": f"{param.type}[]", "values": values[param.name]})
        else:
            params.append({"type": param.type, "value": values[param.name]})
    return {
        "op": "correctness",
        "ptx": item.ptx,
        "kernel": case.kernel,
        "params": params,
        "grid": list(case.correctness.launch.grid),
        "block": list(case.correctness.launch.block),
        "output_param": case.output_param_index(),
        "output_count": case.correctness_output_cells(),
    }


def library_side(item: Plan) -> dict[str, object]:
    """A library baseline: the case's output array holds the result, two others are operands."""
    case = item.case
    arrays = [param.name for param in case.params if param.is_array]
    output = case.correctness.output_param
    operands = [name for name in arrays if name != output]
    if len(operands) < LIBRARY_OPERANDS:
        msg = f"{case.id}: {case.baseline.symbol} needs two operand buffers, found {operands}"
        raise GapError(msg)
    return {
        "kind": "host_lib",
        "library": item.baseline.library,
        "symbol": item.baseline.symbol,
        "n": int(item.scalars["N"]),
        "a": operands[0],
        "b": operands[1],
        "c": output,
    }


def kernel_side(item: Plan, session: CudaCompiler) -> dict[str, object]:
    """Compile the case's CUDA source to PTX on the rental and describe it as a side."""
    launch = item.case.baseline.launch
    if launch is None:
        msg = f"{item.case.id}: a kernel baseline needs launch geometry"
        raise GapError(msg)
    ptx = session.compile_cuda_ptx(
        manifest.CASE_ROOT / item.baseline.source,
        item.baseline.nvcc_flags,
        name=f"baseline-{item.case.id}",
    )
    return {
        "kind": "ptx",
        "ptx": ptx,
        "kernel": item.baseline.kernel,
        "launch": {"grid": list(launch.grid), "block": list(launch.block)},
    }


def compile_baselines(items: Sequence[Item], session: CudaCompiler) -> list[Item]:
    """Build each case's CUDA side once, before any timing.

    A baseline that will not compile on the rental costs its own row and nothing else, and a
    repeat run reuses the PTX instead of paying nvcc again.
    """
    readied: list[Item] = []
    for item in items:
        if isinstance(item, Broken):
            readied.append(item)
            continue
        try:
            side = (
                library_side(item)
                if item.baseline.kind == "host_lib"
                else kernel_side(item, session)
            )
        except CASE_ERRORS as exc:
            readied.append(Broken(case=item.case, baseline=item.baseline, error=str(exc)))
            continue
        readied.append(replace(item, cuda_side=side))
    return readied


def bench_request(
    item: Plan,
    base: Mapping[str, object],
    commits: Mapping[str, str],
) -> dict[str, object]:
    case = item.case
    params: list[dict[str, object]] = []
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
    inputs = {
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
    return {
        "op": "bench",
        "case_id": case.id,
        "case_version": case.version,
        "reference": case.reference,
        "tolerance": list(case.tolerance),
        "params": params,
        "scalars": dict(item.scalars),
        "inputs": inputs,
        "output": case.correctness.output_param,
        "base": dict(base),
        "head": {
            "kind": "ptx",
            "ptx": item.ptx,
            "kernel": case.kernel,
            "launch": {"grid": list(case.launch.grid), "block": list(case.launch.block)},
        },
        "timing": {
            "warmup": case.timing.warmup,
            "samples": case.timing.samples,
            "sample_ms": case.timing.sample_ms,
        },
        "environment": dict(commits),
    }


def parse_case_result(response: Mapping[str, object]) -> bench_stats.CaseResult:
    """Read one harness bench response through the shared result contract."""
    document = {
        "schema_version": bench_stats.SUITE_SCHEMA_VERSION,
        "cases": [response["result"]],
    }
    return bench_stats.from_json(json.dumps(document)).cases[0]


def environment_of(ping: Mapping[str, object], commit: str) -> bench_stats.Environment:
    device = ping.get("environment")
    fields = dict(device) if isinstance(device, Mapping) else {}
    missing = [name for name in ENVIRONMENT_FIELDS if not fields.get(name)]
    if missing:
        msg = f"the rental reported no {', '.join(missing)}"
        raise GapError(msg)
    return bench_stats.Environment(
        gpu_uuid=str(fields["gpu_uuid"]),
        driver_version=str(fields["driver_version"]),
        sm_clock_mhz=int(fields["sm_clock_mhz"]),
        memory_clock_mhz=int(fields["memory_clock_mhz"]),
        base_commit=commit,
        head_commit=commit,
    )


@dataclass(frozen=True)
class Attempt:
    """One case in one run: samples that passed both gates, or the reason it produced none."""

    error: str = ""
    result: bench_stats.CaseResult | None = None

    @property
    def correct(self) -> bool:
        return bool(self.result is not None and self.result.correct)

    @property
    def stats(self) -> bench_stats.CaseStats | None:
        return self.result.stats if self.result is not None else None

    @property
    def gap(self) -> float | None:
        stats = self.stats
        return stats.ratio_median if stats is not None else None

    def as_result(
        self, case: manifest.Case, environment: bench_stats.Environment
    ) -> bench_stats.CaseResult:
        if self.result is not None:
            return self.result
        return bench_stats.CaseResult(
            case_id=case.id,
            case_version=case.version,
            samples=(),
            correct=False,
            environment=environment,
        )


@dataclass(frozen=True)
class Row:
    """One case across one run: its attempt, or the local failure that stopped it."""

    case: manifest.Case
    baseline: Baseline
    attempt: Attempt

    @property
    def case_id(self) -> str:
        return self.case.id

    @property
    def note(self) -> str:
        return self.attempt.error


@dataclass(frozen=True)
class Run:
    """One pass of every case in a single rental."""

    rows: tuple[Row, ...]

    def gap_of(self, case_id: str) -> float | None:
        for row in self.rows:
            if row.case_id == case_id:
                return row.attempt.gap
        return None

    def suite(self, environment: bench_stats.Environment) -> bench_stats.SuiteResult:
        return bench_stats.SuiteResult(
            cases=tuple(row.attempt.as_result(row.case, environment) for row in self.rows)
        )


def _mismatch(
    got: Sequence[float], expected: Sequence[float], tolerance: tuple[float, float]
) -> str:
    """Name the first wrong value, so a failed gate points at a number instead of a verdict."""
    for index, (actual, want) in enumerate(zip(got, expected, strict=False)):
        if not manifest.within_tolerance([actual], [want], *tolerance):
            return f"small config output {index} is {actual!r}, the manifest expects {want!r}"
    return "small config returned the wrong number of values"


def measure(
    item: Plan,
    harness: Harness,
    commits: Mapping[str, str],
) -> Attempt:
    """Gate a case at both sizes, then time it against its baseline.

    Every failure is contained to this case, so one broken kernel still leaves the other rows of
    the table reported.
    """
    case = item.case
    try:
        check = harness.request(correctness_request(item))
        got = case.correctness_readback(check["values"])
        expected = case.correctness.expected
        if not manifest.within_tolerance(got[: len(expected)], expected, *case.tolerance):
            return Attempt(error=_mismatch(got, expected, case.tolerance))
        if item.cuda_side is None:
            msg = f"{item.case.id}: the CUDA baseline was never built"
            raise GapError(msg)
        response = harness.request(bench_request(item, item.cuda_side, commits))
        result = parse_case_result(response)
    except CASE_ERRORS as exc:
        return Attempt(error=str(exc))
    if not result.correct:
        return Attempt(error="timed config failed validation")
    return Attempt(result=result)


def run_once(
    items: Sequence[Item],
    harness: Harness,
    commits: Mapping[str, str],
) -> Run:
    rows: list[Row] = []
    for item in items:
        if isinstance(item, Broken):
            rows.append(Row(item.case, item.baseline, Attempt(error=item.error)))
            continue
        rows.append(Row(item.case, item.baseline, measure(item, harness, commits)))
    return Run(tuple(rows))
