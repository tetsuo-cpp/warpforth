"""Renders the gap table and its result document.

Every case gets a row, including one that failed, so a broken case cannot hide the rest of the
suite. The geometric mean goes through the stats module, which also decides which cases are stable
enough to average.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpu_test import bench_stats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bench.gap import Baseline, Row, Run

GAP_SCHEMA_VERSION = 1
CASE_WIDTH = 20
ENVIRONMENT_KEYS = (
    "gpu_uuid",
    "driver_version",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "base_commit",
    "head_commit",
)


@dataclass(frozen=True)
class Report:
    """One `bench gap` invocation, on both its text and JSON surfaces."""

    arch: str
    environment: bench_stats.Environment
    runs: tuple[Run, ...]
    max_ratio_mad: float

    @property
    def rows(self) -> tuple[Row, ...]:
        return self.runs[0].rows

    @property
    def complete(self) -> bool:
        return all(row.attempt.correct for row in self.rows)

    @property
    def geometric_mean(self) -> float | None:
        return self._mean(self.runs[0])

    @property
    def noise_floor(self) -> float | None:
        """The largest per-case gap change between repeats, as a fraction of the earlier gap."""
        changes = []
        for earlier, later in zip(self.runs, self.runs[1:], strict=False):
            for row in later.rows:
                before = earlier.gap_of(row.case_id)
                gap = row.attempt.gap
                if gap is not None and before:
                    changes.append(abs(gap / before - 1.0))
        return max(changes) if changes else None

    def _mean(self, run: Run) -> float | None:
        try:
            return bench_stats.geometric_mean_ratio(
                run.suite(self.environment).cases, max_ratio_mad=self.max_ratio_mad
            )
        except bench_stats.SampleError:
            return None

    def _stable(self, run: Run) -> int:
        return sum(
            1
            for result in run.suite(self.environment).cases
            if result.stats is not None and result.stats.ratio_mad <= self.max_ratio_mad
        )

    def lines(self) -> list[str]:
        lines = [
            f"gap against SOTA CUDA at {self.arch}, WarpForth time over CUDA time, one rental",
            self._environment_line(),
            "",
            f"{'case':<{CASE_WIDTH}} {'gap':>7} {'warpforth':>12} {'cuda':>12} {'mad':>6}  ok",
            *(_row_line(row) for row in self.rows),
            "",
            self._mean_line(),
        ]
        if len(self.runs) > 1:
            lines.extend(["", *self._repeat_lines()])
        return lines

    def _environment_line(self) -> str:
        return " ".join(f"{key}={getattr(self.environment, key)}" for key in ENVIRONMENT_KEYS)

    def _mean_line(self) -> str:
        run = self.runs[0]
        mean = self._mean(run)
        total = len(run.rows)
        if mean is None:
            return (
                "geometric mean: none, no case passed both gates with a stable ratio"
                f" ({total} cases)"
            )
        return (
            f"geometric mean gap {mean:.3f}x over {self._stable(run)} of {total} stable cases"
            f" (cutoff ratio MAD {self.max_ratio_mad:.3f})"
        )

    def _repeat_lines(self) -> list[str]:
        headers = "".join(f"{f'run {index + 1}':>9}" for index in range(len(self.runs)))
        lines = [
            f"same-rental repeat, {len(self.runs)} runs",
            f"{'case':<{CASE_WIDTH}}{headers}",
        ]
        for row in self.rows:
            cells = "".join(_cell(run.gap_of(row.case_id)) for run in self.runs)
            lines.append(f"{row.case_id:<{CASE_WIDTH}}{cells}")
        means = "".join(_cell(self._mean(run)) for run in self.runs)
        lines.append(f"{'geometric mean':<{CASE_WIDTH}}{means}")
        floor = self.noise_floor
        if floor is not None:
            lines.append(f"measured gap noise floor {floor * 100:.2f}% (largest per-case change)")
        return lines

    def document(self) -> dict[str, object]:
        return {
            "schema_version": GAP_SCHEMA_VERSION,
            "arch": self.arch,
            "environment": _environment_document(self.environment),
            "max_ratio_mad": self.max_ratio_mad,
            "noise_floor": self.noise_floor,
            "geometric_mean": self.geometric_mean,
            "cases": [_case_document(row, self.runs) for row in self.rows],
        }

    def to_json(self) -> str:
        return json.dumps(self.document(), indent=2, sort_keys=True)


def _environment_document(environment: bench_stats.Environment) -> dict[str, object]:
    return {key: getattr(environment, key) for key in ENVIRONMENT_KEYS}


def _case_document(row: Row, runs: Sequence[Run]) -> dict[str, object]:
    return {
        "case_id": row.case_id,
        "baseline": _baseline_document(row.baseline),
        "runs": [_measurement_document(run, row.case_id) for run in runs],
    }


def _baseline_document(baseline: Baseline) -> dict[str, object]:
    return {
        "kind": baseline.kind,
        "source": baseline.source,
        "kernel": baseline.kernel,
        "library": baseline.library,
        "symbol": baseline.symbol,
        "nvcc_flags": list(baseline.nvcc_flags),
    }


def _measurement_document(run: Run, case_id: str) -> dict[str, object]:
    row = next(row for row in run.rows if row.case_id == case_id)
    stats = row.attempt.stats
    result = row.attempt.result
    samples = []
    if result is not None:
        samples = [
            {"base_us": sample.base_us, "head_us": sample.head_us, "first": sample.first.value}
            for sample in result.samples
        ]
    return {
        "correct": row.attempt.correct,
        "error": row.note,
        "gap": row.attempt.gap,
        "warpforth_us": stats.head_median_us if stats else None,
        "cuda_us": stats.base_median_us if stats else None,
        "ratio_mad": stats.ratio_mad if stats else None,
        "samples": samples,
    }


def _cell(value: float | None) -> str:
    return f"{value:>9.3f}" if value is not None else f"{'-':>9}"


def _row_line(row: Row) -> str:
    stats = row.attempt.stats
    gap = f"{row.attempt.gap:.2f}x" if row.attempt.gap is not None else "-"
    warpforth = f"{stats.head_median_us:.1f}us" if stats else "-"
    cuda = f"{stats.base_median_us:.1f}us" if stats else "-"
    mad = f"{stats.ratio_mad:.3f}" if stats else "-"
    ok = "yes" if row.attempt.correct else "no"
    line = f"{row.case_id:<{CASE_WIDTH}} {gap:>7} {warpforth:>12} {cuda:>12} {mad:>6}  {ok}"
    return f"{line}  {row.note}" if row.note else line
