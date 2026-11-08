"""Checks the compare command without a GPU or a second build.

The two compiler revisions are faked at the compile seam, and the rental at the harness seam, so
these tests pin what the command resolves, what it compiles twice, the request the harness receives
for a paired run, the verdict each pairing produces, and the exit codes. The git worktree and cmake
calls are pinned against a temporary repository, which is cheap and needs no compiler.
"""

from __future__ import annotations

import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from bench import build, cli, compare, gap
from benchmarks import manifest
from gpu_test import bench_stats

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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
    base_commit="aaa111",
    head_commit="bbb222",
)


def _readback_cells(case: manifest.Case) -> list:
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
        {"base_us": base_us, "head_us": head_us, "first": "base" if index % 2 == 0 else "head"}
        for index in range(5)
    ]
    return {
        "ok": True,
        "result": {
            "case_id": case.id,
            "case_version": case.version,
            "correct": correct,
            "environment": {
                "gpu_uuid": ENVIRONMENT.gpu_uuid,
                "driver_version": ENVIRONMENT.driver_version,
                "sm_clock_mhz": ENVIRONMENT.sm_clock_mhz,
                "memory_clock_mhz": ENVIRONMENT.memory_clock_mhz,
                "base_commit": ENVIRONMENT.base_commit,
                "head_commit": ENVIRONMENT.head_commit,
            },
            "samples": samples if correct else [],
        },
        "validation": {"base": correct, "head": correct},
    }


class FakeHarness:
    """Stands in for HarnessClient: records requests, answers by op, kernel, and case id.

    Each case gets one paired call followed by one call against its CUDA baseline, so the fake
    serves the second call a gap from `gaps` and the first from `ratios`.
    """

    def __init__(
        self,
        ratios: dict[str, float] | None = None,
        *,
        gaps: dict[str, float] | None = None,
        bad: Sequence[str] = (),
    ) -> None:
        self.ratios = ratios or {}
        self.gaps = gaps or {}
        self.bad = set(bad)
        self.requests: list[dict[str, object]] = []
        self.seen: dict[str, int] = {}

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
            return {"ok": True, "values": _readback_cells(case)}
        case = BY_ID[str(payload["case_id"])]
        index = self.seen.get(case.id, 0)
        self.seen[case.id] = index + 1
        ratio = self.ratios.get(case.id, 1.0) if index == 0 else self.gaps.get(case.id, 1.0)
        return _bench_response(
            case,
            base_us=100.0,
            head_us=100.0 * ratio,
            correct=case.id not in self.bad,
        )

    def of_op(self, op: str) -> list[dict[str, object]]:
        return [payload for payload in self.requests if payload.get("op") == op]


class FakeSession:
    """Serves the remote CUDA compile seam without a rental, recording each compile."""

    instance_id = 4242

    def __init__(self) -> None:
        self.compiles: list[tuple[Path, tuple[str, ...]]] = []

    def compile_cuda_ptx(self, source: Path, flags: Sequence[str], *, name: str) -> str:
        self.compiles.append((source, tuple(flags)))
        return f".version 8.0\n.visible .entry {name}( )\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repository with one tag per ref name, so ref resolution is exercised."""
    root = tmp_path / "repo"
    root.mkdir()

    def run(*args: str) -> None:
        subprocess.run(  # noqa: S603
            ["git", "-C", str(root), *args],  # noqa: S607 - git from PATH
            check=True,
            capture_output=True,
        )

    run("init", "--template=")
    run("config", "user.email", "bench@example.invalid")
    run("config", "user.name", "bench")
    run("config", "commit.gpgsign", "false")
    (root / "file.txt").write_text("first\n")
    run("add", ".")
    run("commit", "-m", "first")
    run("tag", "base")
    (root / "file.txt").write_text("second\n")
    run("commit", "-am", "second")
    run("tag", "head")
    return root


def _resolved(repo: Path, name: str) -> build.Ref:
    return build.resolve(repo, name)


def test_resolve_names_the_commit_a_ref_points_at(repo: Path) -> None:
    base = _resolved(repo, "base")
    head = _resolved(repo, "head")
    assert base.sha != head.sha
    assert base.name == "base"
    assert len(base.short) == 12


def test_resolve_rejects_a_ref_that_does_not_exist(repo: Path) -> None:
    with pytest.raises(build.BuildError, match="cannot resolve"):
        build.resolve(repo, "no-such-ref")


def test_toolchain_requires_the_mlir_and_llvm_directories(tmp_path: Path) -> None:
    with pytest.raises(build.BuildError, match="MLIR_DIR"):
        build.toolchain_of(tmp_path)


def test_toolchain_reads_the_definitions_a_working_build_kept(tmp_path: Path) -> None:
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        "CMAKE_GENERATOR:INTERNAL=Ninja\n"
        "MLIR_DIR:PATH=/opt/mlir\n"
        "LLVM_DIR:PATH=/opt/llvm\n"
        "CMAKE_BUILD_TYPE:STRING=Release\n"
        "SOMETHING_ELSE:STRING=ignored\n"
    )
    toolchain = build.toolchain_of(tmp_path)
    assert toolchain.generator == "Ninja"
    assert toolchain.definitions == (
        ("MLIR_DIR", "/opt/mlir"),
        ("LLVM_DIR", "/opt/llvm"),
        ("CMAKE_BUILD_TYPE", "Release"),
    )
    assert toolchain.configure(Path("/src"), Path("/out"))[:3] == [
        "cmake",
        "-S",
        "/src",
    ]
    assert "-G" in toolchain.configure(Path("/src"), Path("/out"))


def test_environment_in_the_env_var_beats_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "CMakeCache.txt").write_text(
        "MLIR_DIR:PATH=/cached/mlir\nLLVM_DIR:PATH=/cached/llvm\n"
    )
    monkeypatch.setenv("MLIR_DIR", "/env/mlir")
    toolchain = build.toolchain_of(tmp_path)
    assert ("MLIR_DIR", "/env/mlir") in toolchain.definitions
    assert ("LLVM_DIR", "/cached/llvm") in toolchain.definitions


def test_worktree_checks_out_the_ref_and_cleans_up_after(repo: Path) -> None:
    ref = _resolved(repo, "base")
    with build.worktree(repo, ref) as checkout:
        assert checkout.is_dir()
        assert (checkout / "file.txt").read_text() == "first\n"
        held = checkout
    assert not held.exists()
    listed = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), "worktree", "list"],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
    )
    assert "compare-" not in listed.stdout


@requires_compiler
def _plan(name: str) -> gap.Plan:
    item = gap.plan(BY_ID[name], COMPILER, "sm_89")
    assert isinstance(item, gap.Plan), item
    return item


def _items() -> list[gap.Item]:
    return gap.compile_baselines([_plan(case.id) for case in CASES], FakeSession())


def _bases() -> dict[str, compare.BaseSide]:
    return {case.id: compare.BaseSide(ptx=".version 8.0\n") for case in CASES}


@requires_compiler
def test_compare_builds_each_cuda_baseline_once_for_both_verdicts() -> None:
    session = FakeSession()
    items = gap.compile_baselines(_items(), session)
    compare.rows_of(items, _bases(), FakeHarness(), {})
    compiled = [path.name for path, _flags in session.compiles]
    assert compiled == [
        Path(case.baseline.source).name for case in CASES if case.baseline.kind != "host_lib"
    ]


@requires_compiler
def test_a_paired_request_carries_both_revisions_in_the_harness_slots() -> None:
    head = _plan("scale_i64")
    base = compare.BaseSide(ptx=".version 8.0 // base\n")
    request = gap.bench_request(head, base.side(head.case), {"head_commit": "b"})
    assert request["base"]["ptx"] == base.ptx
    assert request["head"]["ptx"] == head.ptx
    assert request["base"]["kernel"] == request["head"]["kernel"] == "scale_i64"
    assert request["base"]["launch"] == request["head"]["launch"]


@requires_compiler
def test_compare_times_the_head_against_its_cuda_baseline_in_the_same_session() -> None:
    harness = FakeHarness({"scale_i64": 1.5}, gaps={"scale_i64": 2.25})
    items = gap.compile_baselines([_plan("scale_i64")], FakeSession())
    rows = compare.rows_of(items, _bases(), harness, {"head_commit": "b"})
    assert len(harness.of_op("bench")) == 2, "one paired run and one against CUDA"
    assert rows[0].outcome.verdict is bench_stats.Verdict.SLOWER
    assert rows[0].outcome.stats is not None
    assert rows[0].outcome.stats.ratio_median == pytest.approx(1.5)
    assert rows[0].outcome.head_gap == pytest.approx(2.25)
    assert rows[0].outcome.error == ""


@requires_compiler
def test_compare_takes_the_base_ptx_before_reaching_the_rental() -> None:
    harness = FakeHarness()
    items = gap.compile_baselines(_items(), FakeSession())
    compare.rows_of(items, _bases(), harness, {"head_commit": "b"})
    assert len(harness.of_op("bench")) == 2 * len(CASES)


@requires_compiler
def test_no_change_comes_from_the_stats_threshold_not_from_an_equal_median() -> None:
    harness = FakeHarness(dict.fromkeys([case.id for case in CASES], 1.01))
    rows = compare.rows_of(gap.compile_baselines(_items(), FakeSession()), _bases(), harness, {})
    assert {row.outcome.verdict for row in rows} == {bench_stats.Verdict.NO_CHANGE}
    assert rows[0].outcome.stats is not None
    assert rows[0].outcome.stats.ratio_median == pytest.approx(1.01)


@requires_compiler
def test_a_base_that_will_not_compile_costs_only_its_row() -> None:
    bases = _bases() | {"axpy_f64": compare.BaseSide(error="base: nope")}
    rows = compare.rows_of(gap.compile_baselines(_items(), FakeSession()), bases, FakeHarness(), {})
    assert [row.case.id for row in rows] == [case.id for case in CASES]
    failed = next(row for row in rows if row.case.id == "axpy_f64")
    assert failed.outcome.error == "base: nope"
    assert all(row.outcome.error == "" for row in rows if row.case.id != "axpy_f64")


@requires_compiler
def test_a_case_that_fails_validation_reports_its_reason_and_no_verdict() -> None:
    rows = compare.rows_of(
        gap.compile_baselines(_items(), FakeSession()),
        _bases(),
        FakeHarness({"scale_i64": 1.0}, bad=["scale_i64"]),
        {},
    )
    failed = rows[0]
    assert failed.outcome.verdict is None
    assert "validation" in failed.outcome.error


@requires_compiler
def test_the_report_marks_each_regression_and_names_it_in_the_summary() -> None:
    rows = compare.rows_of(
        gap.compile_baselines(_items(), FakeSession()),
        _bases(),
        FakeHarness({"scale_i64": 3.0, "axpy_f64": 1.01}),
        {},
    )
    report = compare.Report(arch="sm_89", environment=ENVIRONMENT, rows=tuple(rows))
    assert report.regressions == ("scale_i64",)
    text = "\n".join(report.lines())
    assert "slower" in text
    assert "3.000" in text
    assert "1 of 6 cases slower: scale_i64" in text


@requires_compiler
def test_a_clean_report_says_no_case_is_slower() -> None:
    rows = compare.rows_of(
        gap.compile_baselines(_items(), FakeSession()), _bases(), FakeHarness(), {}
    )
    report = compare.Report(arch="sm_89", environment=ENVIRONMENT, rows=tuple(rows))
    assert report.regressions == ()
    assert "no case slower across 6 cases" in "\n".join(report.lines())


@requires_compiler
def test_the_report_states_both_revisions_and_the_rental() -> None:
    rows = compare.rows_of(
        gap.compile_baselines(_items(), FakeSession()), _bases(), FakeHarness(), {}
    )
    report = compare.Report(arch="sm_89", environment=ENVIRONMENT, rows=tuple(rows))
    text = "\n".join(report.lines())
    assert "base=aaa111" in text
    assert "head=bbb222" in text
    assert f"gpu_uuid={ENVIRONMENT.gpu_uuid}" in text
    document = report.document()
    assert document["environment"]["base_commit"] == "aaa111"
    assert document["environment"]["head_commit"] == "bbb222"
    assert len(document["cases"]) == len(CASES)


def test_environment_of_stamps_both_revisions() -> None:
    ping = {
        "environment": {
            "gpu_uuid": "gpu-0001",
            "driver_version": "13.3",
            "sm_clock_mhz": 2490,
            "memory_clock_mhz": 8501,
        }
    }
    environment = compare.environment_of(ping, "base-sha", "head-sha")
    assert environment.base_commit == "base-sha"
    assert environment.head_commit == "head-sha"
    assert environment.sm_clock_mhz == 2490


@requires_compiler
def test_each_case_compiles_inside_its_refs_own_worktree(tmp_path: Path) -> None:
    """A case compiled after its worktree is torn down fails on a compiler path that is gone."""
    built_in: list[Path] = []
    counter = iter(range(2))
    real_plan = gap.plan

    @contextmanager
    def checkout(_repo: Path, _ref: build.Ref) -> Iterator[Path]:
        root = tmp_path / f"checkout-{next(counter)}"
        root.mkdir()
        try:
            yield root
        finally:
            shutil.rmtree(root)

    def plan(case: manifest.Case, compiler: Path, arch: str) -> gap.Item:
        assert compiler.is_file(), f"{compiler} was removed before the cases compiled"
        built_in.append(compiler)
        return real_plan(case, compiler, arch)

    def compile_base(case: manifest.Case, compiler: Path, _arch: str) -> compare.BaseSide:
        assert compiler.is_file(), f"{compiler} was removed before the base compiled"
        built_in.append(compiler)
        return compare.BaseSide(ptx=f".visible .entry {case.kernel}( )\n")

    args = cli.build_parser().parse_args(["compare", "base", "head", "--jobs", "2"])
    with (
        mock.patch.object(build, "resolve", return_value=build.Ref("ref", "a" * 40)),
        mock.patch.object(build, "worktree", side_effect=checkout),
        mock.patch.object(build, "toolchain_of", return_value=build.Toolchain((), "")),
        mock.patch.object(build, "compiler", side_effect=_fake_compiler),
        mock.patch.object(compare, "compile_base", side_effect=compile_base),
        mock.patch.object(gap, "plan", side_effect=plan),
    ):
        cli.compile_both(CASES[:2], args)
    assert len(built_in) == 2 * len(CASES[:2])
    assert len({path.parent.parent.parent for path in built_in}) == 2, "one build per ref"


def test_a_compiler_that_fails_to_build_is_a_clean_error() -> None:
    failure = subprocess.CompletedProcess([], 1, "", "ninja: build stopped")
    toolchain = build.Toolchain(definitions=(), generator="")
    with (
        mock.patch.object(build.subprocess, "run", return_value=failure),
        pytest.raises(build.BuildError, match="exited 1"),
    ):
        build.compiler(Path("/src"), toolchain, jobs=1)


def _ref(repo: Path, name: str) -> build.Ref:
    sha = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{name}^{{commit}}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return build.Ref(name=name, sha=sha)


def _fake_compiler(checkout: Path, _toolchain: build.Toolchain, *, jobs: int) -> Path:
    """A stand-in binary that writes valid PTX to the `-o` path for every case kernel."""
    binary = checkout / "build" / "bin" / "warpforthc"
    binary.parent.mkdir(parents=True)
    entries = "\\n".join(f".visible .entry {case.kernel}( )" for case in CASES)
    binary.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then shift; out="$1"; fi\n'
        "  shift\n"
        "done\n"
        f'printf "%s\\n" "{entries}" > "$out"\n'
        f"# jobs={jobs}\n"
    )
    binary.chmod(0o755)
    return binary


def test_the_parser_takes_the_compare_refs_and_the_build_jobs() -> None:
    args = cli.build_parser().parse_args(["compare", "canon", "bench/x", "--jobs", "4"])
    assert args.command == "compare"
    assert (args.base, args.head) == ("canon", "bench/x")
    assert args.jobs == 4
    assert args.arch == manifest.DEFAULT_ARCH


def test_the_parser_takes_the_acceptance_options() -> None:
    args = cli.build_parser().parse_args(["acceptance", "--arch", "sm_80"])
    assert args.command == "acceptance"
    assert args.arch == "sm_80"


def test_the_gap_parser_is_unchanged_by_the_shared_options() -> None:
    args = cli.build_parser().parse_args(["gap", "--repeats", "2", "--max-ratio-mad", "0.1"])
    assert args.command == "gap"
    assert args.repeats == 2
    assert args.max_ratio_mad == 0.1
    assert args.arch == manifest.DEFAULT_ARCH
