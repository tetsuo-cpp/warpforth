"""The `bench gap` command and its argument surface."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING

from bench import gap
from bench.report import Report
from benchmarks import manifest
from gpu_test import bench_stats, infrastructure

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("bench")
REPO_ROOT = manifest.CASE_ROOT.parent
EXIT_INCOMPLETE = 1


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="Measure WarpForth against each case's SOTA CUDA baseline on one rental.",
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-v", "--verbose", action="store_true", help="log at debug level")
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("gap", parents=[shared], help="print the per-case gap table")
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
    command.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run the whole suite this many times on one rental, to measure the noise floor",
    )
    command.add_argument(
        "--max-ratio-mad",
        type=float,
        default=bench_stats.MAX_STABLE_RATIO_MAD,
        help="leave cases that scatter more than this out of the geometric mean",
    )
    command.add_argument("--json", action="store_true", help="also print the result document")
    parser.add_argument("-v", "--verbose", action="store_true", help="log at debug level")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
