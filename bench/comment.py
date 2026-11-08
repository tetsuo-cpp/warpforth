from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bench import build, compare

MARKER = "<!-- warpforth-bench-compare -->"
_MERGE_PARENTS = 2
_MIN_FENCE = 3
_RATIO_DIGITS = 3
_GAP_DIGITS = 2
_TIME_DIGITS = 1
_TEXT_COLUMNS = 2
_NUMERIC_COLUMNS = 5
_HEADERS = ("case", "verdict", "ratio", "mad", "base", "head", "gap")


@dataclass(frozen=True, slots=True)
class Revisions:
    base: build.Ref
    head: build.Ref
    pull_head: str | None

    @classmethod
    def of(cls, base: build.Ref, head: build.Ref, head_parents: Sequence[str]) -> Revisions:
        pull_head = None
        if len(head_parents) == _MERGE_PARENTS and head_parents[0] == base.sha:
            pull_head = head_parents[1]
        return cls(base, head, pull_head)


@dataclass(frozen=True, slots=True)
class Unfinished:
    revisions: Revisions


@dataclass(frozen=True, slots=True)
class Failed:
    revisions: Revisions
    reason: str

    @classmethod
    def from_error(cls, revisions: Revisions, error: BaseException) -> Failed:
        if isinstance(error, KeyboardInterrupt):
            return cls(revisions, "interrupted")
        return cls(revisions, f"{type(error).__name__}: {error}")


@dataclass(frozen=True, slots=True)
class Measured:
    revisions: Revisions
    report: compare.Report


State = Unfinished | Failed | Measured


def render(state: State) -> str:
    lines = [MARKER, _headline(state), ""]
    result = _result(state)
    if result is not None:
        lines.extend([result, ""])
    lines.append(_details(state))
    return "\n".join(lines) + "\n"


def _headline(state: State) -> str:
    return f"### Bench compare against `{state.revisions.base.name}`: {_status(state)}"


def _status(state: State) -> str:
    if isinstance(state, Measured):
        return state.report.summary
    if isinstance(state, Failed):
        return "no result"
    return "did not finish"


def _result(state: State) -> str | None:
    if isinstance(state, Measured):
        return _table(state.report)
    if isinstance(state, Failed):
        return _fence(state.reason)
    return None


def _table(report: compare.Report) -> str:
    notes = any(row.outcome.note for row in report.rows)
    headers = [*_HEADERS, "note"] if notes else list(_HEADERS)
    rows = [_markdown_row(headers), _separator(len(headers))]
    for row in report.rows:
        cells = _cells(row)
        if notes:
            cells.append(_cell(row.outcome.note))
        rows.append(_markdown_row(cells))
    return "\n".join(rows)


def _cells(row: compare.Row) -> list[str]:
    outcome = row.outcome
    stats = outcome.stats
    verdict = outcome.verdict.value if outcome.verdict else "-"
    ratio = f"{stats.ratio_median:.{_RATIO_DIGITS}f}" if stats else "-"
    mad = f"{stats.ratio_mad:.{_RATIO_DIGITS}f}" if stats else "-"
    base = f"{stats.base_median_us:.{_TIME_DIGITS}f} us" if stats else "-"
    head = f"{stats.head_median_us:.{_TIME_DIGITS}f} us" if stats else "-"
    gap = f"{outcome.head_gap:.{_GAP_DIGITS}f}x" if outcome.head_gap is not None else "-"
    return [_cell(value) for value in (row.case.id, verdict, ratio, mad, base, head, gap)]


def _markdown_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _separator(columns: int) -> str:
    cells = []
    for index in range(columns):
        numeric = _TEXT_COLUMNS <= index < _TEXT_COLUMNS + _NUMERIC_COLUMNS
        cells.append("---:" if numeric else "---")
    return _markdown_row(cells)


def _cell(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())


def _details(state: State) -> str:
    revisions = state.revisions
    bullets = [f"Base: `{revisions.base.sha}`", f"Head: `{revisions.head.sha}`"]
    if isinstance(state, Measured):
        bullets.append(_rental(state.report))
    return "\n".join(f"- {bullet}" for bullet in bullets)


def _rental(report: compare.Report) -> str:
    environment = report.environment
    return (
        f"Rental: `{environment.gpu_uuid}`, driver {environment.driver_version}, "
        f"SM clock {environment.sm_clock_mhz} MHz, "
        f"memory clock {environment.memory_clock_mhz} MHz, `{report.arch}`"
    )


def _fence(text: str) -> str:
    ticks = "`" * max(_backtick_run(text) + 1, _MIN_FENCE)
    return f"{ticks}\n{text}\n{ticks}"


def _backtick_run(text: str) -> int:
    longest = 0
    run = 0
    for char in text:
        if char == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest
