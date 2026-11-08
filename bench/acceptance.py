"""The acceptance fixture pair: a kernel and a deliberately slowed twin of the same shape.

The pair proves the compare instrument both ways on one rental. The slowed twin must come back a
regression, and the same source in both slots must come back no change, each decided by the stats
module's own threshold. Both arms borrow one case's buffers, seeds, launch geometry, reference, and
tolerance, so the only thing that differs between them is the kernel body.

The arms run through `compare.compare_case`, the function `bench compare` times its cases with, so
the fixture proves the real verdict path rather than a copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bench import compare, gap
from benchmarks import manifest
from gpu_test import bench_stats, infrastructure

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

FIXTURE_ROOT = manifest.CASE_ROOT / "acceptance"
FAST = FIXTURE_ROOT / "fast.forth"
SLOW = FIXTURE_ROOT / "slow.forth"
KERNEL = "acceptance_scale"
TEMPLATE_CASE = "scale_i64"
REGRESSION_FLOOR = 2.0


class FixtureError(Exception):
    """The fixture pair cannot be read or compiled."""


@dataclass(frozen=True)
class Arm:
    """One pairing of the fixture's two files, with the verdict it has to produce."""

    base: Path
    head: Path
    expect: bench_stats.Verdict
    ratio_floor: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.base.name} -> {self.head.name}"


ARMS = (
    Arm(base=FAST, head=SLOW, expect=bench_stats.Verdict.SLOWER, ratio_floor=REGRESSION_FLOOR),
    Arm(base=FAST, head=FAST, expect=bench_stats.Verdict.NO_CHANGE),
)


@dataclass(frozen=True)
class Reading:
    """What one arm measured."""

    arm: Arm
    outcome: compare.Outcome

    @property
    def passed(self) -> bool:
        ratio = self.outcome.stats.ratio_median if self.outcome.stats else 0.0
        return (
            not self.outcome.note
            and self.outcome.verdict is self.arm.expect
            and ratio >= self.arm.ratio_floor
        )

    def line(self) -> str:
        stats = self.outcome.stats
        ratio = f"{stats.ratio_median:.3f}" if stats else "-"
        verdict = self.outcome.verdict.value if self.outcome.verdict else "-"
        status = "pass" if self.passed else "fail"
        note = f"  {self.outcome.note}" if self.outcome.note else ""
        return f"{self.arm.label:<24} {verdict:<10} {ratio:>7}  {status}{note}"


def template_case(cases: Sequence[manifest.Case]) -> manifest.Case:
    """The case the fixture borrows its buffers, seeds, launch, and tolerance from."""
    for case in cases:
        if case.id == TEMPLATE_CASE:
            return case
    msg = f"the case suite has no {TEMPLATE_CASE} case to supply the fixture's buffers"
    raise FixtureError(msg)


def compile_side(source: Path, compiler: Path, arch: str) -> str:
    try:
        return manifest.compile_text(
            source.read_text(), KERNEL, compiler, arch, label=f"acceptance/{source.name}"
        )
    except (manifest.ManifestError, OSError) as exc:
        raise FixtureError(str(exc)) from exc


def read_arm(
    arm: Arm, case: manifest.Case, compiler: Path, arch: str, harness: gap.Harness
) -> Reading:
    """Compile both sides and time them against each other through the compare path."""
    commits = {"base_commit": arm.base.stem, "head_commit": arm.head.stem}
    try:
        base_ptx = compile_side(arm.base, compiler, arch)
        head_ptx = compile_side(arm.head, compiler, arch)
    except FixtureError as exc:
        return Reading(arm=arm, outcome=compare.Outcome(error=str(exc)))
    head = gap.planned(case, head_ptx, KERNEL, case.launch, arch)
    base_side = gap.ptx_side(base_ptx, KERNEL, case.launch)
    return Reading(arm=arm, outcome=compare.compare_case(head, base_side, harness, commits))


def run(cases: Sequence[manifest.Case], compiler: Path, arch: str, api_key: str) -> list[Reading]:
    """Rent once and run both arms, so the whole fixture costs one session."""
    case = template_case(cases)
    with (
        infrastructure.VastSession(api_key, arch=arch) as session,
        infrastructure.HarnessClient(session) as harness,
    ):
        return [read_arm(arm, case, compiler, arch, harness) for arm in ARMS]


def lines(readings: Sequence[Reading]) -> list[str]:
    header = f"{'pair':<24} {'verdict':<10} {'ratio':>7}  result"
    return [header, *(reading.line() for reading in readings)]
