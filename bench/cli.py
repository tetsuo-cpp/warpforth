"""The `bench` command line: the CUDA gap table and the base-versus-head comparison."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bench import acceptance, build, comment, compare, gap
from bench.report import Report
from benchmarks import manifest
from gpu_test import bench_stats, infrastructure

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger("bench")
REPO_ROOT = manifest.CASE_ROOT.parent
EXIT_INCOMPLETE = 1
EXIT_REGRESSED = 2
BUILD_JOBS = max(1, (os.cpu_count() or 2))


def head_commit() -> str:
    result = subprocess.run(  # noqa: S603 - a fixed argument list, never a shell string
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def api_key() -> str:
    key = os.environ.get("VASTAI_API_KEY", "")
    if not key:
        msg = "VASTAI_API_KEY is not set; load it with: set -a; source .env; set +a"
        raise SystemExit(msg)
    return key


def select_cases(cases: Sequence[manifest.Case], wanted: Sequence[str]) -> list[manifest.Case]:
    if not wanted:
        return list(cases)
    missing = sorted(set(wanted) - {case.id for case in cases})
    if missing:
        msg = f"no case named {', '.join(missing)}"
        raise SystemExit(msg)
    keep = set(wanted)
    return [case for case in cases if case.id in keep]


def measure(
    items: Sequence[gap.Item],
    session: infrastructure.VastSession,
    commit: str,
    repeats: int,
) -> tuple[list[gap.Run], bench_stats.Environment]:
    """Build every CUDA baseline once, then run the suite `repeats` times over one channel."""
    commits = {"base_commit": commit, "head_commit": commit}
    readied = gap.compile_baselines(items, session)
    for item in readied:
        if isinstance(item, gap.Broken):
            logger.error("%s: %s", item.case.id, item.error)
    runs: list[gap.Run] = []
    with infrastructure.HarnessClient(session) as harness:
        environment = gap.environment_of(harness.request({"op": "ping"}), commit)
        for index in range(repeats):
            logger.info(
                "suite run %d of %d on instance %s", index + 1, repeats, session.instance_id
            )
            runs.append(gap.run_once(readied, harness, commits))
    return runs, environment


def run(args: argparse.Namespace) -> int:
    cases = select_cases(manifest.load(), args.case)
    key = api_key()
    compiler = manifest.find_compiler(REPO_ROOT)
    if not compiler.is_file():
        logger.error("warpforthc not found at %s; run: cmake --build build", compiler)
        return EXIT_INCOMPLETE

    logger.info("compiling %d cases locally with %s at %s", len(cases), compiler, args.arch)
    items = [gap.plan(case, compiler, args.arch) for case in cases]
    for item in items:
        if isinstance(item, gap.Broken):
            logger.error("%s: %s", item.case.id, item.error)

    with infrastructure.VastSession(key, arch=args.arch) as session:
        runs, environment = measure(items, session, head_commit(), args.repeats)

    report = Report(
        arch=args.arch,
        environment=environment,
        runs=tuple(runs),
        max_ratio_mad=args.max_ratio_mad,
    )
    for line in report.lines():
        print(line)  # noqa: T201 - the command's own output
    if args.json:
        print(report.to_json())  # noqa: T201
    return 0 if report.complete else EXIT_INCOMPLETE


def compile_both(
    cases: Sequence[manifest.Case],
    base: build.Ref,
    head: build.Ref,
    args: argparse.Namespace,
) -> tuple[dict[str, compare.BaseSide], list[gap.Item]]:
    """A worktree is torn down as soon as it has produced its PTX, so the two compilers never
    share a build directory and nothing depends on where the temporary checkouts live.
    """
    logger.info("base %s at %s, head %s at %s", base.name, base.short, head.name, head.short)
    toolchain = build.toolchain_of(manifest.find_build(REPO_ROOT))
    logger.info("compiling %d cases with both revisions at %s", len(cases), args.arch)
    with build.worktree(REPO_ROOT, base) as checkout:
        compiler = build.compiler(checkout, toolchain, jobs=args.jobs)
        bases = {case.id: compare.compile_base(case, compiler, args.arch) for case in cases}
    with build.worktree(REPO_ROOT, head) as checkout:
        compiler = build.compiler(checkout, toolchain, jobs=args.jobs)
        items = [gap.plan(case, compiler, args.arch) for case in cases]
    for item, base_error in zip(items, (bases[case.id].error for case in cases), strict=True):
        error = item.error if isinstance(item, gap.Broken) else base_error
        if error:
            logger.error("%s: %s", item.case.id, error)
    return bases, items


@dataclass(frozen=True, slots=True)
class UnbuiltHead:
    pass


def compare_refs(
    cases: Sequence[manifest.Case],
    base: build.Ref,
    head: build.Ref,
    key: str,
    args: argparse.Namespace,
) -> compare.Report | UnbuiltHead:
    bases, items = compile_both(cases, base, head, args)
    if all(isinstance(item, gap.Broken) for item in items):
        return UnbuiltHead()
    commits = {"base_commit": base.sha, "head_commit": head.sha}
    logger.info("building the CUDA baselines and timing %d cases on one rental", len(items))
    with infrastructure.VastSession(key, arch=args.arch) as session:
        readied = gap.compile_baselines(items, session)
        with infrastructure.HarnessClient(session) as harness:
            environment = compare.environment_of(
                harness.request({"op": "ping"}), base.sha, head.sha
            )
            rows = compare.rows_of(readied, bases, harness, commits)
    return compare.Report(arch=args.arch, environment=environment, rows=tuple(rows))


def run_compare(args: argparse.Namespace) -> int:
    cases = select_cases(manifest.load(), args.case)
    key = api_key()
    base = build.resolve(REPO_ROOT, args.base)
    head = build.resolve(REPO_ROOT, args.head)
    revisions = comment.Revisions.of(base, head, build.parents(REPO_ROOT, head))
    publish = comment_writer(args.comment)
    publish(comment.Unfinished(revisions))
    try:
        outcome = compare_refs(cases, base, head, key, args)
    except BaseException as exc:
        publish(comment.Failed.from_error(revisions, exc))
        raise
    if isinstance(outcome, UnbuiltHead):
        publish(comment.Failed(revisions, "no case compiled with the head revision"))
        return EXIT_INCOMPLETE
    report = outcome
    publish(comment.Measured(revisions, report))
    for line in report.lines():
        print(line)  # noqa: T201 - the command's own output
    if args.json:
        print(report.to_json())  # noqa: T201
    if report.regressions:
        return EXIT_REGRESSED
    return 0 if not report.failures else EXIT_INCOMPLETE


def comment_writer(path: Path | None) -> Callable[[comment.State], None]:
    def write(state: comment.State) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(comment.render(state), encoding="utf-8", newline="\n")
        temporary.replace(path)

    return write


def run_acceptance(args: argparse.Namespace) -> int:
    cases = manifest.load()
    key = api_key()
    compiler = manifest.find_compiler(REPO_ROOT)
    if not compiler.is_file():
        logger.error("warpforthc not found at %s; run: cmake --build build", compiler)
        return EXIT_INCOMPLETE
    readings = acceptance.run(cases, compiler, args.arch, key)
    for line in acceptance.lines(readings):
        print(line)  # noqa: T201 - the command's own output
    return 0 if all(reading.passed for reading in readings) else EXIT_INCOMPLETE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="Measure WarpForth against each case's SOTA CUDA baseline on one rental.",
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-v", "--verbose", action="store_true", help="log at debug level")
    subcommands = parser.add_subparsers(dest="command", required=True)
    gap_command = subcommands.add_parser("gap", parents=[shared], help="print the per-case gap")
    add_case_options(gap_command)
    gap_command.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run the whole suite this many times on one rental, to measure the noise floor",
    )
    gap_command.add_argument(
        "--max-ratio-mad",
        type=float,
        default=bench_stats.MAX_STABLE_RATIO_MAD,
        help="leave cases that scatter more than this out of the geometric mean",
    )
    compare_command = subcommands.add_parser(
        "compare", parents=[shared], help="print a verdict per case for a base and head ref"
    )
    add_case_options(compare_command)
    compare_command.add_argument("base", help="the ref every change is measured against")
    compare_command.add_argument("head", help="the ref that may be faster or slower")
    compare_command.add_argument(
        "--jobs",
        type=int,
        default=BUILD_JOBS,
        help="parallel compile jobs for each of the two fresh builds",
    )
    compare_command.add_argument(
        "--comment",
        type=Path,
        help="write the review comment to this path as the run goes, without posting it",
    )
    acceptance_command = subcommands.add_parser(
        "acceptance",
        parents=[shared],
        help="check the fixture pair still flags a slower twin and clears an identical pair",
    )
    add_case_options(acceptance_command)
    parser.add_argument("-v", "--verbose", action="store_true", help="log at debug level")
    return parser


def add_case_options(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--arch",
        default=manifest.DEFAULT_ARCH,
        choices=infrastructure.SUPPORTED_ARCHITECTURES,
        help="compute capability both sides compile for, and the one the rental must meet",
    )
    command.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="ID",
        help="measure only this case; repeatable",
    )
    command.add_argument("--json", action="store_true", help="also print the result document")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.command == "compare":
        return run_compare(args)
    if args.command == "acceptance":
        return run_acceptance(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
