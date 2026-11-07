"""Paired-sample statistics and the JSON result contract for the bench program.

Times are microseconds. A ratio is head over base, so a ratio below one means head is faster.
A batch times both sides back to back and records which side ran first, so drift across a session
lands on both sides instead of on one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

MINIMUM_RELATIVE_CHANGE = 0.02
MAD_MULTIPLIER = 3.0
MAX_STABLE_RATIO_MAD = 0.05
SUITE_SCHEMA_VERSION = 1


class StatsError(Exception):
    """Base for every error this module raises."""


class ContractError(StatsError, ValueError):
    """The result JSON does not match the contract."""


class SampleError(StatsError, ValueError):
    """The samples cannot support the requested statistic."""


class Side(StrEnum):
    BASE = "base"
    HEAD = "head"


class Verdict(StrEnum):
    FASTER = "faster"
    SLOWER = "slower"
    NO_CHANGE = "no change"


def median(values: Sequence[float]) -> float:
    if not values:
        msg = "median of an empty sample set"
        raise SampleError(msg)
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def median_absolute_deviation(values: Sequence[float]) -> float:
    center = median(values)
    return median([abs(value - center) for value in values])


def paired_ratios(samples: Sequence[BenchSample]) -> tuple[float, ...]:
    return tuple(sample.head_us / sample.base_us for sample in samples)


def noise_threshold(
    ratio_mad: float,
    *,
    minimum_relative: float = MINIMUM_RELATIVE_CHANGE,
    mad_multiplier: float = MAD_MULTIPLIER,
) -> float:
    return max(minimum_relative, mad_multiplier * ratio_mad)


def verdict(
    samples: Sequence[BenchSample],
    *,
    minimum_relative: float = MINIMUM_RELATIVE_CHANGE,
    mad_multiplier: float = MAD_MULTIPLIER,
) -> Verdict:
    """A change exactly at the threshold counts as no change.

    The ratio is compared against `1 + limit` rather than checking the delta against `limit`, so a
    delta that lands exactly on the threshold cannot flip a verdict on a rounding step.
    """
    ratios = paired_ratios(samples)
    center = median(ratios)
    limit = noise_threshold(
        median_absolute_deviation(ratios),
        minimum_relative=minimum_relative,
        mad_multiplier=mad_multiplier,
    )
    if center < 1.0 - limit:
        return Verdict.FASTER
    if center > 1.0 + limit:
        return Verdict.SLOWER
    return Verdict.NO_CHANGE


def geometric_mean_ratio(
    cases: Sequence[CaseResult],
    *,
    max_ratio_mad: float = MAX_STABLE_RATIO_MAD,
) -> float:
    """Average the case ratios in log space, skipping cases that failed or scattered."""
    stable = []
    for case in cases:
        stats = case.stats
        if stats is not None and stats.ratio_mad <= max_ratio_mad:
            stable.append(stats.ratio_median)
    if not stable:
        msg = f"suite has no stable cases below ratio MAD {max_ratio_mad}, cannot average"
        raise SampleError(msg)
    return math.exp(math.fsum(math.log(ratio) for ratio in stable) / len(stable))


@dataclass(frozen=True, slots=True)
class BenchSample:
    base_us: float
    head_us: float
    first: Side

    def __post_init__(self) -> None:
        for side, value in (("base_us", self.base_us), ("head_us", self.head_us)):
            if not (math.isfinite(value) and value > 0):
                msg = f"{side} must be finite and positive, got {value}"
                raise SampleError(msg)


@dataclass(frozen=True, slots=True)
class Environment:
    gpu_uuid: str
    driver_version: str
    sm_clock_mhz: int
    memory_clock_mhz: int
    base_commit: str
    head_commit: str


@dataclass(frozen=True, slots=True)
class CaseStats:
    """Everything derivable from one case's samples."""

    base_median_us: float
    head_median_us: float
    base_mad_us: float
    head_mad_us: float
    ratios: tuple[float, ...]
    ratio_median: float
    ratio_mad: float

    @classmethod
    def from_samples(cls, samples: Sequence[BenchSample]) -> Self:
        ratios = paired_ratios(samples)
        return cls(
            base_median_us=median([sample.base_us for sample in samples]),
            head_median_us=median([sample.head_us for sample in samples]),
            base_mad_us=median_absolute_deviation([sample.base_us for sample in samples]),
            head_mad_us=median_absolute_deviation([sample.head_us for sample in samples]),
            ratios=ratios,
            ratio_median=median(ratios),
            ratio_mad=median_absolute_deviation(ratios),
        )


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One case. Samples are empty when correctness failed before the harness could time it."""

    case_id: str
    case_version: int
    samples: tuple[BenchSample, ...]
    correct: bool
    environment: Environment

    @property
    def stats(self) -> CaseStats | None:
        return CaseStats.from_samples(self.samples) if self.samples else None


@dataclass(frozen=True, slots=True)
class SuiteResult:
    cases: tuple[CaseResult, ...]


def to_json(suite: SuiteResult) -> str:
    root = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "cases": [_case_to_mapping(case) for case in suite.cases],
    }
    return json.dumps(root, indent=2, sort_keys=True)


def from_json(text: str) -> SuiteResult:
    try:
        root = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"result JSON is not valid JSON: {exc}"
        raise ContractError(msg) from exc
    return _suite_from_mapping(root)


def _case_to_mapping(case: CaseResult) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "case_version": case.case_version,
        "correct": case.correct,
        "environment": {
            "gpu_uuid": case.environment.gpu_uuid,
            "driver_version": case.environment.driver_version,
            "sm_clock_mhz": case.environment.sm_clock_mhz,
            "memory_clock_mhz": case.environment.memory_clock_mhz,
            "base_commit": case.environment.base_commit,
            "head_commit": case.environment.head_commit,
        },
        "samples": [
            {"base_us": sample.base_us, "head_us": sample.head_us, "first": sample.first.value}
            for sample in case.samples
        ],
        "stats": _stats_to_mapping(case.stats),
    }


# The `stats` block is a readable view so a reader gets medians without recomputing. Parsing drops
# it and derives again from `samples`, and extra keys are tolerated so a harness can add
# diagnostics without breaking the contract.
def _stats_to_mapping(stats: CaseStats | None) -> dict[str, object] | None:
    if stats is None:
        return None
    return {
        "base_median_us": stats.base_median_us,
        "head_median_us": stats.head_median_us,
        "base_mad_us": stats.base_mad_us,
        "head_mad_us": stats.head_mad_us,
        "ratios": list(stats.ratios),
        "ratio_median": stats.ratio_median,
        "ratio_mad": stats.ratio_mad,
    }


def _suite_from_mapping(root: object) -> SuiteResult:
    mapping = _as_mapping(root, "suite")
    version = _as_int(mapping, "schema_version", "suite")
    if version != SUITE_SCHEMA_VERSION:
        msg = f"suite.schema_version must be {SUITE_SCHEMA_VERSION}, got {version}"
        raise ContractError(msg)
    cases = _as_list(mapping, "cases", "suite")
    parsed = tuple(_case_from_mapping(item, f"cases[{index}]") for index, item in enumerate(cases))
    return SuiteResult(cases=parsed)


def _case_from_mapping(value: object, path: str) -> CaseResult:
    mapping = _as_mapping(value, path)
    case_id = _as_str(mapping, "case_id", path)
    case_version = _as_int(mapping, "case_version", path)
    correct = _as_bool(mapping, "correct", path)
    environment = _environment_from_mapping(
        _require(mapping, "environment", path), f"{path}.environment"
    )
    samples = tuple(
        _sample_from_mapping(item, f"{path}.samples[{index}]")
        for index, item in enumerate(_as_list(mapping, "samples", path))
    )
    return CaseResult(
        case_id=case_id,
        case_version=case_version,
        samples=samples,
        correct=correct,
        environment=environment,
    )


def _environment_from_mapping(value: object, path: str) -> Environment:
    mapping = _as_mapping(value, path)
    return Environment(
        gpu_uuid=_as_str(mapping, "gpu_uuid", path),
        driver_version=_as_str(mapping, "driver_version", path),
        sm_clock_mhz=_as_int(mapping, "sm_clock_mhz", path),
        memory_clock_mhz=_as_int(mapping, "memory_clock_mhz", path),
        base_commit=_as_str(mapping, "base_commit", path),
        head_commit=_as_str(mapping, "head_commit", path),
    )


def _sample_from_mapping(value: object, path: str) -> BenchSample:
    mapping = _as_mapping(value, path)
    first = _as_str(mapping, "first", path)
    try:
        return BenchSample(
            base_us=_as_float(mapping, "base_us", path),
            head_us=_as_float(mapping, "head_us", path),
            first=Side(first),
        )
    except (SampleError, ValueError) as exc:
        msg = f"{path}: {exc}"
        raise ContractError(msg) from exc


def _require(mapping: Mapping[str, object], key: str, path: str) -> object:
    if key not in mapping:
        msg = f"{path}.{key} is missing"
        raise ContractError(msg)
    return mapping[key]


def _as_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{path} must be a JSON object"
        raise ContractError(msg)
    return value


def _as_list(mapping: Mapping[str, object], key: str, path: str) -> list[object]:
    value = _require(mapping, key, path)
    if not isinstance(value, list):
        msg = f"{path}.{key} must be a JSON array"
        raise ContractError(msg)
    return value


def _as_str(mapping: Mapping[str, object], key: str, path: str) -> str:
    value = _require(mapping, key, path)
    if not isinstance(value, str):
        msg = f"{path}.{key} must be a string"
        raise ContractError(msg)
    return value


def _as_int(mapping: Mapping[str, object], key: str, path: str) -> int:
    value = _require(mapping, key, path)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{path}.{key} must be an integer"
        raise ContractError(msg)
    return value


def _as_float(mapping: Mapping[str, object], key: str, path: str) -> float:
    value = _require(mapping, key, path)
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"{path}.{key} must be a number"
        raise ContractError(msg)
    return float(value)


def _as_bool(mapping: Mapping[str, object], key: str, path: str) -> bool:
    value = _require(mapping, key, path)
    if not isinstance(value, bool):
        msg = f"{path}.{key} must be a boolean"
        raise ContractError(msg)
    return value
