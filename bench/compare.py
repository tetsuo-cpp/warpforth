"""Compares two compiler revisions on the same case suite, in one rental.

Each case is compiled twice, once by each revision's `warpforthc`, and both modules go into one
harness bench call, so the base and head times are paired samples from the same session. A verdict
comes from the stats module's own threshold, so a difference inside the noise reads as no change.

The case suite is the working tree's, unchanged across both refs, because the comparison is about
the compiler and not about the cases. The head side is also timed against its case's SOTA CUDA
baseline, so a verdict says which compiler is faster and the gap says how far it still sits from
CUDA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from bench import gap
from benchmarks import manifest
from gpu_test import bench_stats

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

CASE_WIDTH = 18
VERDICT_WIDTH = 10
ENVIRONMENT_FIELDS = (
    "gpu_uuid",
    "driver_version",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "base_commit",
    "head_commit",
)


@dataclass(frozen=True)
class BaseSide:
    """One case as the base revision compiles it, or the reason it could not."""

    ptx: str = ""
    error: str = ""

    def side(self, case: manifest.Case) -> dict[str, object]:
        return gap.ptx_side(self.ptx, case.kernel, case.launch)


@dataclass(frozen=True)
class Outcome:
    """One case's paired verdict, and the head side's gap against CUDA from the same session."""

    verdict: bench_stats.Verdict | None = None
    stats: bench_stats.CaseStats | None = None
    head_gap: float | None = None
    error: str = ""
    gap_error: str = ""

    @property
    def regressed(self) -> bool:
        return self.verdict is bench_stats.Verdict.SLOWER

    @property
    def note(self) -> str:
        return " ".join(part for part in (self.error, self.gap_error) if part)


@dataclass(frozen=True)
class Row:
    case: manifest.Case
    outcome: Outcome


def compile_base(case: manifest.Case, compiler: Path, arch: str) -> BaseSide:
    """Compile one case with the base revision, so a base that cannot build it costs no rental."""
    try:
        return BaseSide(ptx=manifest.compile_case(case, compiler, arch))
    except (manifest.ManifestError, OSError) as exc:
        return BaseSide(error=f"base: {exc}")


def compare_case(
    head: gap.Plan,
    base_side: Mapping[str, object],
    harness: gap.Harness,
    commits: Mapping[str, str],
) -> Outcome:
    """Time one base side against this case's head PTX, then the head against its CUDA baseline."""
    try:
        request = gap.bench_request(head, base_side, commits)
        result = gap.parse_case_result(harness.request(request))
    except gap.CASE_ERRORS as exc:
        return Outcome(error=str(exc))
    if not result.correct:
        return Outcome(error="the paired run failed validation")
    stats = result.stats
    if stats is None:
        return Outcome(error="the paired run produced no samples")
    cuda = gap.bench_against_cuda(head, harness, commits) if head.cuda_side else None
    return Outcome(
        verdict=bench_stats.verdict(result.samples),
        stats=stats,
        head_gap=cuda.gap if cuda else None,
        gap_error="" if cuda is None or cuda.result is not None else f"CUDA baseline: {cuda.error}",
    )


@dataclass(frozen=True)
class Report:
    """One `bench compare` invocation, on its text and JSON surfaces."""

    arch: str
    environment: bench_stats.Environment
    rows: tuple[Row, ...]

    @property
    def regressions(self) -> tuple[str, ...]:
        return tuple(row.case.id for row in self.rows if row.outcome.regressed)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(row.case.id for row in self.rows if row.outcome.error)

    def lines(self) -> list[str]:
        header = (
            f"{'case':<{CASE_WIDTH}} {'verdict':<{VERDICT_WIDTH}} {'ratio':>7} {'mad':>6}"
            f" {'base':>10} {'head':>10} {'gap':>7}  ok"
        )
        return [
            f"compare base={self.environment.base_commit[:12]}"
            f" head={self.environment.head_commit[:12]} at {self.arch},"
            " head time over base time, one rental",
            self._environment_line(),
            "",
            header,
            *(_row_line(row) for row in self.rows),
            "",
            self._sum_line(),
        ]

    def _environment_line(self) -> str:
        return " ".join(f"{name}={getattr(self.environment, name)}" for name in ENVIRONMENT_FIELDS)

    def _sum_line(self) -> str:
        total = len(self.rows)
        if self.regressions:
            return f"{len(self.regressions)} of {total} cases slower: {', '.join(self.regressions)}"
        if self.failures:
            return f"{len(self.failures)} of {total} cases could not be compared"
        return f"no case slower across {total} cases"

    def document(self) -> dict[str, object]:
        return {
            "arch": self.arch,
            "environment": {name: getattr(self.environment, name) for name in ENVIRONMENT_FIELDS},
            "cases": [
                {
                    "case_id": row.case.id,
                    "verdict": row.outcome.verdict.value if row.outcome.verdict else None,
                    "ratio_median": row.outcome.stats.ratio_median if row.outcome.stats else None,
                    "ratio_mad": row.outcome.stats.ratio_mad if row.outcome.stats else None,
                    "head_gap": row.outcome.head_gap,
                    "error": row.outcome.note,
                }
                for row in self.rows
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.document(), indent=2, sort_keys=True)


def environment_of(ping: Mapping[str, object], base: str, head: str) -> bench_stats.Environment:
    """The rental's device metadata, stamped with both revisions this run compared."""
    return replace(gap.environment_of(ping, base), head_commit=head)


def rows_of(
    items: Sequence[gap.Item],
    bases: Mapping[str, BaseSide],
    harness: gap.Harness,
    commits: Mapping[str, str],
) -> list[Row]:
    """Every case in suite order, with a paired outcome or the reason it has none."""
    rows = []
    for item in items:
        if isinstance(item, gap.Broken):
            rows.append(Row(item.case, Outcome(error=item.error)))
            continue
        base = bases[item.case.id]
        if base.error:
            rows.append(Row(item.case, Outcome(error=base.error)))
            continue
        side = base.side(item.case)
        rows.append(Row(item.case, compare_case(item, side, harness, commits)))
    return rows


def _row_line(row: Row) -> str:
    outcome = row.outcome
    stats = outcome.stats
    verdict = outcome.verdict.value if outcome.verdict else "-"
    ratio = f"{stats.ratio_median:.3f}" if stats else "-"
    mad = f"{stats.ratio_mad:.3f}" if stats else "-"
    before = f"{stats.base_median_us:.1f}us" if stats else "-"
    after = f"{stats.head_median_us:.1f}us" if stats else "-"
    gap_text = f"{outcome.head_gap:.2f}x" if outcome.head_gap is not None else "-"
    ok = "no" if outcome.error else "yes"
    line = (
        f"{row.case.id:<{CASE_WIDTH}} {verdict:<{VERDICT_WIDTH}} {ratio:>7} {mad:>6}"
        f" {before:>10} {after:>10} {gap_text:>7}  {ok}"
    )
    return f"{line}  {outcome.note}" if outcome.note else line
