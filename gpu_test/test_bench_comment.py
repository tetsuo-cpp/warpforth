from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from bench import build, cli, comment, compare
from benchmarks import manifest
from gpu_test import bench_stats

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
PULL_SHA = "c" * 40
OTHER_SHA = "d" * 40
TABLE_HEADER = "| case | verdict | ratio | mad | base | head | gap |"
RENTAL = "- Rental: `gpu-0000`, driver 12.4, SM clock 1500 MHz, memory clock 7000 MHz, `sm_89`"
BY_ID = {case.id: case for case in manifest.load()}


def _ref(name: str, sha: str) -> build.Ref:
    return build.Ref(name=name, sha=sha)


def _revisions(*, pull_head: str | None) -> comment.Revisions:
    parents = (BASE_SHA, pull_head) if pull_head is not None else (OTHER_SHA, PULL_SHA)
    return comment.Revisions.of(_ref("canon", BASE_SHA), _ref("HEAD", HEAD_SHA), parents)


def _samples(head_us: float) -> tuple[bench_stats.BenchSample, ...]:
    return tuple(
        bench_stats.BenchSample(
            base_us=100.0,
            head_us=head_us,
            first=bench_stats.Side.BASE if index % 2 == 0 else bench_stats.Side.HEAD,
        )
        for index in range(5)
    )


def _row(case_id: str, *, head_us: float) -> compare.Row:
    samples = _samples(head_us)
    return compare.Row(
        BY_ID[case_id],
        compare.Outcome(
            verdict=bench_stats.verdict(samples),
            stats=bench_stats.CaseStats.from_samples(samples),
            head_gap=1.25,
        ),
    )


def _report() -> compare.Report:
    return compare.Report(
        arch="sm_89",
        environment=bench_stats.Environment(
            gpu_uuid="gpu-0000",
            driver_version="12.4",
            sm_clock_mhz=1500,
            memory_clock_mhz=7000,
            base_commit=BASE_SHA,
            head_commit=HEAD_SHA,
        ),
        rows=(_row("scale_i64", head_us=300.0), _row("axpy_f64", head_us=100.0)),
    )


def _fences(body: str) -> list[str]:
    lines = body.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if len(line) >= 3 and set(line) == {"`"}:
            close = next(
                (cursor for cursor in range(index + 1, len(lines)) if lines[cursor] == line),
                None,
            )
            assert close is not None, f"unclosed fence of {len(line)} backticks"
            blocks.append("\n".join(lines[index + 1 : close]))
            index = close + 1
            continue
        index += 1
    return blocks


def _longest_backtick_run(text: str) -> int:
    longest = 0
    run = 0
    for char in text:
        if char == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def test_each_state_opens_with_the_marker_and_headline() -> None:
    report = _report()
    revisions = _revisions(pull_head=PULL_SHA)
    states = (
        (comment.Unfinished(revisions), "did not finish"),
        (comment.Failed(revisions, "the rental died"), "no result"),
        (comment.Measured(revisions, report), report.summary),
    )
    for state, status in states:
        lines = comment.render(state).splitlines()
        assert lines[0] == comment.MARKER
        assert lines[1] == f"### Bench compare against `canon`: {status}"
        assert lines[2] == ""
        rendered = comment.render(state)
        assert f"- Base: `{BASE_SHA}`" in rendered
        assert f"- Head: `{HEAD_SHA}`" in rendered
        assert "Case sources" not in rendered
        assert "head time over base time" not in rendered
        if isinstance(state, comment.Measured):
            assert RENTAL in rendered
        else:
            assert "Rental:" not in rendered


def test_a_measured_merge_uses_a_markdown_table_then_bullets() -> None:
    report = _report()
    revisions = _revisions(pull_head=PULL_SHA)
    body = comment.render(comment.Measured(revisions, report))
    table, details = body.split("\n\n", maxsplit=2)[1:]
    assert table.splitlines()[0] == TABLE_HEADER
    assert "| scale_i64 | slower | 3.000 | 0.000 | 100.0 us | 300.0 us | 1.25x |" in table
    assert "| axpy_f64 | no change | 1.000 | 0.000 | 100.0 us | 100.0 us | 1.25x |" in table
    assert "compare base=" not in body
    assert _fences(body) == []
    assert PULL_SHA not in body
    assert details.splitlines() == [
        f"- Base: `{BASE_SHA}`",
        f"- Head: `{HEAD_SHA}`",
        RENTAL,
    ]


def test_a_plain_compare_names_both_full_shas() -> None:
    revisions = comment.Revisions.of(_ref("canon", BASE_SHA), _ref("topic", HEAD_SHA), (BASE_SHA,))
    body = comment.render(comment.Measured(revisions, _report()))
    assert f"- Base: `{BASE_SHA}`" in body
    assert f"- Head: `{HEAD_SHA}`" in body
    assert "topic" not in body
    assert "Pull request head" not in body


def test_a_failure_note_with_a_pipe_stays_inside_one_table_cell() -> None:
    broken = compare.Row(BY_ID["scale_i64"], compare.Outcome(error="base: a|b\nsecond"))
    report = compare.Report(arch="sm_89", environment=_report().environment, rows=(broken,))
    body = comment.render(comment.Measured(_revisions(pull_head=None), report))
    assert "| case | verdict | ratio | mad | base | head | gap | note |" in body
    assert "| scale_i64 | - | - | - | - | - | - | base: a\\|b second |" in body


def test_revisions_of_sets_pull_head_only_when_the_first_parent_is_the_base() -> None:
    base = _ref("canon", BASE_SHA)
    head = _ref("HEAD", HEAD_SHA)
    assert comment.Revisions.of(base, head, (BASE_SHA, PULL_SHA)).pull_head == PULL_SHA
    assert comment.Revisions.of(base, head, (OTHER_SHA, PULL_SHA)).pull_head is None
    assert comment.Revisions.of(base, head, (BASE_SHA,)).pull_head is None
    assert comment.Revisions.of(base, head, ()).pull_head is None
    assert comment.Revisions.of(base, head, (BASE_SHA, PULL_SHA, OTHER_SHA)).pull_head is None


def test_a_triple_backtick_in_the_reason_stays_inside_one_longer_fence() -> None:
    reason = "nvcc said\n```\nstill the reason\n```\n"
    body = comment.render(comment.Failed(_revisions(pull_head=PULL_SHA), reason))
    blocks = _fences(body)
    assert blocks == [reason]
    opening = next(line for line in body.splitlines() if set(line) == {"`"})
    assert len(opening) == max(3, _longest_backtick_run(reason) + 1)
    assert len(opening) > 3


def test_from_error_names_the_type_and_calls_an_interrupt_interrupted() -> None:
    revisions = _revisions(pull_head=None)
    failed = comment.Failed.from_error(revisions, build.BuildError("missing llvm"))
    assert failed.reason == "BuildError: missing llvm"
    assert comment.Failed.from_error(revisions, KeyboardInterrupt()).reason == "interrupted"


def test_a_slower_compare_exits_2_and_the_headline_says_slower(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VASTAI_API_KEY", "dummy")
    path = tmp_path / "out" / "bench-comment.md"
    report = _report()
    with (
        mock.patch.object(
            build, "resolve", side_effect=[_ref("canon", BASE_SHA), _ref("HEAD", HEAD_SHA)]
        ),
        mock.patch.object(build, "parents", return_value=(BASE_SHA, PULL_SHA)),
        mock.patch.object(cli, "compare_refs", return_value=report),
    ):
        code = cli.main(["compare", "canon", "HEAD", "--comment", str(path)])
    assert code == 2
    assert "slower" in path.read_text(encoding="utf-8").splitlines()[1]


def test_a_build_error_propagates_and_the_comment_says_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VASTAI_API_KEY", "dummy")
    path = tmp_path / "bench-comment.md"

    def explode(*_args: object, **_kwargs: object) -> None:
        msg = "toolchain missing"
        raise build.BuildError(msg)

    with (
        mock.patch.object(
            build, "resolve", side_effect=[_ref("canon", BASE_SHA), _ref("HEAD", HEAD_SHA)]
        ),
        mock.patch.object(build, "parents", return_value=(BASE_SHA, PULL_SHA)),
        mock.patch.object(cli, "compare_refs", side_effect=explode),
        pytest.raises(build.BuildError, match="toolchain missing"),
    ):
        cli.main(["compare", "canon", "HEAD", "--comment", str(path)])
    text = path.read_text(encoding="utf-8")
    headline = text.splitlines()[1]
    assert "no result" in headline
    assert "BuildError" in text
    assert _fences(text) == ["BuildError: toolchain missing"]


def test_the_comment_says_did_not_finish_when_compare_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VASTAI_API_KEY", "dummy")
    path = tmp_path / "bench-comment.md"
    seen: list[str] = []

    def entered(*_args: object, **_kwargs: object) -> compare.Report:
        seen.append(path.read_text(encoding="utf-8").splitlines()[1])
        return compare.Report(
            arch="sm_89",
            environment=_report().environment,
            rows=(_row("axpy_f64", head_us=100.0),),
        )

    with (
        mock.patch.object(
            build, "resolve", side_effect=[_ref("canon", BASE_SHA), _ref("HEAD", HEAD_SHA)]
        ),
        mock.patch.object(build, "parents", return_value=(BASE_SHA,)),
        mock.patch.object(cli, "compare_refs", side_effect=entered),
    ):
        cli.main(["compare", "canon", "HEAD", "--comment", str(path)])
    assert seen == ["### Bench compare against `canon`: did not finish"]


def test_a_missing_ref_writes_no_comment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VASTAI_API_KEY", "dummy")
    path = tmp_path / "bench-comment.md"
    with pytest.raises(build.BuildError, match="cannot resolve"):
        cli.main(["compare", "no-such-bench-ref", "HEAD", "--comment", str(path)])
    assert not path.exists()


def test_parents_of_a_no_ff_merge_are_first_then_second(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(  # noqa: S603 - a fixed argument list, never a shell string
            ["git", "-C", str(repo), *args],  # noqa: S607 - git from PATH
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "--template=", "-b", "master")
    git("config", "user.email", "bench@example.invalid")
    git("config", "user.name", "bench")
    git("config", "commit.gpgsign", "false")
    (repo / "file.txt").write_text("base\n")
    git("add", ".")
    git("commit", "-m", "base")
    first = git("rev-parse", "HEAD")
    git("checkout", "-b", "feature")
    (repo / "file.txt").write_text("feature\n")
    git("commit", "-am", "feature")
    second = git("rev-parse", "HEAD")
    git("checkout", "master")
    git("merge", "--no-ff", "feature", "-m", "merge")
    assert build.parents(repo, build.resolve(repo, "HEAD")) == (first, second)


def test_only_compare_takes_a_comment_path() -> None:
    args = cli.build_parser().parse_args(["compare", "canon", "HEAD", "--comment", "out.md"])
    assert args.comment == Path("out.md")
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["gap", "--comment", "out.md"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["acceptance", "--comment", "out.md"])
