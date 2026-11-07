"""Hand-computed fixtures for the paired-sample statistics and result contract."""

from __future__ import annotations

import json
import math

import pytest

from gpu_test.bench_stats import (
    BenchSample,
    CaseResult,
    ContractError,
    Environment,
    SampleError,
    Side,
    SuiteResult,
    Verdict,
    from_json,
    geometric_mean_ratio,
    median,
    median_absolute_deviation,
    noise_threshold,
    paired_ratios,
    to_json,
    verdict,
)

ENVIRONMENT = Environment(
    gpu_uuid="GPU-3f2a1c44-9d7e-4b21-8a55-0c1de6f7b812",
    driver_version="550.90.07",
    sm_clock_mhz=2100,
    memory_clock_mhz=10501,
    base_commit="76ebea6",
    head_commit="a1b2c3d",
)


def batches(*pairs: tuple[float, float]) -> tuple[BenchSample, ...]:
    first = Side.BASE
    samples = []
    for base_us, head_us in pairs:
        samples.append(BenchSample(base_us=base_us, head_us=head_us, first=first))
        first = Side.HEAD if first is Side.BASE else Side.BASE
    return tuple(samples)


def case(
    case_id: str,
    pairs: tuple[tuple[float, float], ...],
    *,
    case_version: int = 1,
    correct: bool = True,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        case_version=case_version,
        samples=batches(*pairs),
        correct=correct,
        environment=ENVIRONMENT,
    )


def test_even_count_median_averages_the_middle_pair() -> None:
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert median([1.0, 2.0, 3.0]) == 2.0


def test_even_count_mad_uses_the_even_count_median_again() -> None:
    assert median_absolute_deviation([1.0, 2.0, 3.0, 4.0]) == 1.0


def test_paired_ratios_are_head_over_base_per_batch() -> None:
    samples = batches((100.0, 90.0), (200.0, 220.0))
    assert paired_ratios(samples) == (0.9, 1.1)
    assert median(paired_ratios(samples)) == 1.0
    assert median_absolute_deviation(paired_ratios(samples)) == pytest.approx(0.1)


def test_case_stats_read_off_the_samples() -> None:
    result = case("axpy", ((100.0, 90.0), (200.0, 220.0), (300.0, 330.0), (400.0, 400.0)))
    assert result.stats.base_median_us == 250.0
    assert result.stats.head_median_us == 275.0
    assert result.stats.base_mad_us == 100.0
    assert result.stats.head_mad_us == 90.0
    assert result.stats.ratios == (0.9, 1.1, 1.1, 1.0)
    assert result.stats.ratio_median == 1.05
    assert result.stats.ratio_mad == pytest.approx(0.05)


def test_verdict_flags_a_clean_slowdown_and_a_clean_speedup() -> None:
    slower = case("matmul", ((100.0, 130.0), (200.0, 260.0), (300.0, 390.0)))
    faster = case("matmul", ((100.0, 70.0), (200.0, 140.0), (300.0, 210.0)))
    assert verdict(slower.samples) is Verdict.SLOWER
    assert verdict(faster.samples) is Verdict.FASTER


def test_verdict_is_no_change_when_raw_medians_differ_but_noise_swamps_them() -> None:
    result = case("reduce", ((100.0, 110.0), (200.0, 210.0), (300.0, 300.0)))
    assert result.stats.base_median_us == 200.0
    assert result.stats.head_median_us == 210.0
    assert result.stats.ratio_median == 1.05
    assert result.stats.ratio_mad == pytest.approx(0.05)
    assert noise_threshold(result.stats.ratio_mad) == pytest.approx(0.15)
    assert verdict(result.samples) is Verdict.NO_CHANGE


def test_verdict_is_no_change_exactly_at_the_threshold() -> None:
    samples = ((100.0, 102.0), (200.0, 204.0), (300.0, 306.0))
    at_threshold = case("scale", samples)
    over_threshold = case("scale", ((100.0, 103.0), (200.0, 206.0), (300.0, 309.0)))
    assert at_threshold.stats.ratio_median == pytest.approx(1.02)
    assert at_threshold.stats.ratio_mad == 0.0
    assert noise_threshold(0.0) == 0.02
    assert verdict(at_threshold.samples) is Verdict.NO_CHANGE
    assert verdict(over_threshold.samples) is Verdict.SLOWER


def test_noise_threshold_takes_the_larger_of_the_floor_and_three_mads() -> None:
    assert noise_threshold(0.001) == 0.02
    assert noise_threshold(0.01) == pytest.approx(0.03)


def test_verdict_threshold_is_parameterized() -> None:
    result = case("attention", ((100.0, 105.0), (200.0, 210.0), (300.0, 315.0)))
    assert verdict(result.samples) is Verdict.SLOWER
    assert verdict(result.samples, minimum_relative=0.06) is Verdict.NO_CHANGE
    assert verdict(result.samples, minimum_relative=0.01, mad_multiplier=0.0) is Verdict.SLOWER


def test_geometric_mean_excludes_unstable_cases() -> None:
    stable_slowdown = case("matmul_tiled", ((100.0, 200.0), (200.0, 400.0), (300.0, 600.0)))
    stable_speedup = case("axpy", ((100.0, 50.0), (200.0, 100.0), (300.0, 150.0)))
    unstable = case("reduce", ((100.0, 80.0), (100.0, 120.0), (100.0, 80.0), (100.0, 120.0)))
    failed = CaseResult(
        case_id="attention",
        case_version=1,
        samples=(),
        correct=False,
        environment=ENVIRONMENT,
    )
    assert stable_slowdown.stats.ratio_median == 2.0
    assert stable_speedup.stats.ratio_median == 0.5
    assert unstable.stats.ratio_mad == pytest.approx(0.2)
    expected = math.sqrt(2.0 * 0.5)
    suite = SuiteResult(cases=(stable_slowdown, stable_speedup, unstable, failed))
    assert geometric_mean_ratio(suite.cases) == pytest.approx(expected)


def test_geometric_mean_rejects_a_suite_with_no_stable_case() -> None:
    unstable = case("reduce", ((100.0, 80.0), (100.0, 120.0)))
    with pytest.raises(SampleError, match="no stable cases"):
        geometric_mean_ratio((unstable,))


def test_order_flag_records_which_side_ran_first() -> None:
    result = case("scale", ((100.0, 100.0), (100.0, 100.0), (100.0, 100.0), (100.0, 100.0)))
    assert [sample.first for sample in result.samples] == [
        Side.BASE,
        Side.HEAD,
        Side.BASE,
        Side.HEAD,
    ]


def test_json_round_trip_is_byte_identical() -> None:
    suite = SuiteResult(
        cases=(
            case("scale_i64", ((100.0, 98.5), (101.0, 99.0), (99.0, 97.5))),
            case("axpy_f64", ((250.0, 250.5), (249.0, 250.0), (251.0, 249.5)), case_version=2),
        )
    )
    first = to_json(suite)
    parsed = from_json(first)
    assert parsed == suite
    assert to_json(parsed) == first


def test_metadata_survives_the_round_trip() -> None:
    suite = SuiteResult(cases=(case("matmul", ((100.0, 99.0), (100.0, 99.0))),))
    parsed = from_json(to_json(suite))
    assert parsed.cases[0].environment == ENVIRONMENT
    assert parsed.cases[0].case_version == 1
    assert parsed.cases[0].correct is True


def test_json_carries_the_derived_stats_for_readers() -> None:
    suite = SuiteResult(cases=(case("axpy", ((100.0, 90.0), (200.0, 220.0), (300.0, 330.0))),))
    payload = json.loads(to_json(suite))
    stats = payload["cases"][0]["stats"]
    assert stats["ratio_median"] == pytest.approx(1.1)
    assert stats["base_median_us"] == 200.0
    assert payload["schema_version"] == 1


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ('{"cases": [', "not valid JSON"),
        ("[]", "suite must be a JSON object"),
        ('{"cases": []}', "schema_version is missing"),
        ('{"schema_version": 2, "cases": []}', "schema_version must be 1"),
        ('{"schema_version": 1}', "cases is missing"),
        ('{"schema_version": 1, "cases": {}}', "cases must be a JSON array"),
        ('{"schema_version": 1, "cases": [{}]}', "case_id is missing"),
    ],
)
def test_malformed_json_raises_a_contract_error(payload: str, match: str) -> None:
    with pytest.raises(ContractError, match=match):
        from_json(payload)


def test_malformed_sample_raises_a_contract_error_with_its_path() -> None:
    good = json.loads(to_json(SuiteResult(cases=(case("axpy", ((100.0, 100.0),)),))))
    good["cases"][0]["samples"][0]["first"] = "middle"
    with pytest.raises(ContractError, match=r"cases\[0\]\.samples\[0\]"):
        from_json(json.dumps(good))

    good["cases"][0]["samples"][0]["first"] = "base"
    good["cases"][0]["samples"][0]["base_us"] = 0.0
    with pytest.raises(ContractError, match="must be finite and positive"):
        from_json(json.dumps(good))


def test_empty_sample_sets_raise_sample_errors() -> None:
    with pytest.raises(SampleError, match="median of an empty sample set"):
        median([])
    with pytest.raises(SampleError, match="median of an empty sample set"):
        median_absolute_deviation([])
    with pytest.raises(SampleError, match="median of an empty sample set"):
        verdict(())


def test_case_without_samples_has_no_stats() -> None:
    failed = CaseResult(
        case_id="broken",
        case_version=1,
        samples=(),
        correct=False,
        environment=ENVIRONMENT,
    )
    assert failed.stats is None
    parsed = from_json(to_json(SuiteResult(cases=(failed,))))
    assert parsed.cases[0].stats is None
    assert parsed.cases[0].correct is False


def test_non_positive_batch_time_is_rejected() -> None:
    with pytest.raises(SampleError, match="finite and positive"):
        BenchSample(base_us=100.0, head_us=0.0, first=Side.BASE)
