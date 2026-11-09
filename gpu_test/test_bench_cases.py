"""Checks the benchmark case suite.

Every `.forth` source must compile and every entry in `cases.toml` must agree with its sources
before a case can be timed. The no-GPU tests run anywhere. The gpu-marked ones rent a GPU and run
each case's correctness instance through `KernelRunner`, so a kernel that computes wrong numbers
fails against the real runtime rather than a model of it.

A case has exactly one baseline, the strongest available CUDA implementation. These tests enforce
the parts of that policy a machine can check: one baseline entry, a kind that matches the files
declared for it, and the suite covering every hardware feature the program expects it to.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks import manifest
from benchmarks.manifest import MAX_TOTAL_THREADS, ManifestError

if TYPE_CHECKING:
    from gpu_test.infrastructure import KernelRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = REPO_ROOT / "benchmarks"
MANIFEST = BENCHMARKS / "cases.toml"
CASES = manifest.load(MANIFEST)
COMPILER = manifest.find_compiler(REPO_ROOT)
CASE_IDS = [case.id for case in CASES]

# Features the Forth kernels do not match. The suite must exercise all of them at least once, so a
# gap it reports is a feature gap and not just a tiling difference.
REQUIRED_FEATURES = (
    "128-bit vector",
    "warp shuffle",
    "fused online softmax",
    "register",
)

requires_compiler = pytest.mark.skipif(
    not COMPILER.is_file(),
    reason=f"warpforthc not built at {COMPILER}; build it or set WARPFORTHC",
)


def load_mutated(replace_from: str, replace_to: str) -> tuple[manifest.Case, ...]:
    """Load the real manifest with one substring replaced, to prove a bad entry is rejected."""
    text = MANIFEST.read_text().replace(replace_from, replace_to, 1)
    if text == MANIFEST.read_text():
        msg = f"the mutation target {replace_from!r} is not in the manifest"
        raise AssertionError(msg)
    path = MANIFEST.parent / "cases_mutated.toml"
    path.write_text(text)
    try:
        return manifest.load(path)
    finally:
        path.unlink()


def numpy_reference(case: manifest.Case) -> list[float]:
    """The case's expected outputs recomputed with numpy from its own correctness inputs."""
    correct = case.correctness
    arrays = {name: np.array(values, dtype=np.float64) for name, values in correct.arrays.items()}
    scalars = correct.params

    if case.reference == "scale_i64":
        return (arrays["INPUT"] * scalars["SCALE"]).tolist()
    if case.reference == "axpy_f64":
        return (scalars["ALPHA"] * arrays["X"] + arrays["Y"]).tolist()
    if case.reference == "reduce_sum_f64":
        return [float(arrays["INPUT"][: int(scalars["N"])].sum())]
    if case.reference == "matmul_f64":
        n = int(scalars["N"])
        return (arrays["A"].reshape(n, n) @ arrays["B"].reshape(n, n)).ravel().tolist()
    nrow, nkey, hdim = int(scalars["NROW"]), int(scalars["NKEY"]), int(scalars["HDIM"])
    scores = (arrays["QMAT"].reshape(nrow, hdim) @ arrays["KMAT"].reshape(nkey, hdim).T) * scalars[
        "SCALE"
    ]
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights @ arrays["VMAT"].reshape(nkey, hdim)).ravel().tolist()


@requires_compiler
def test_every_case_source_compiles() -> None:
    for case in CASES:
        ptx = manifest.compile_case(case, COMPILER)
        assert f".visible .entry {case.kernel}(" in ptx, case.id


@requires_compiler
def test_every_case_source_compiles_for_older_and_newer_architectures() -> None:
    for case in CASES:
        for arch in ("sm_70", "sm_90"):
            ptx = manifest.compile_case(case, COMPILER, arch)
            assert f".target {arch}" in ptx, (case.id, arch)


@requires_compiler
def test_correctness_source_is_the_benchmarked_kernel_at_a_smaller_size() -> None:
    """The correctness instance must time the same kernel, not a variant built for small inputs.

    A header's array length only bounds the buffer the runtime allocates, so shrinking it for the
    correctness instance has to leave the emitted PTX untouched.
    """
    for case in CASES:
        shrunk = manifest.compile_source(case.correctness_source(), COMPILER, label=case.id)
        assert shrunk == manifest.compile_case(case, COMPILER), case.id


def test_kernel_name_matches_the_source_header() -> None:
    for case in CASES:
        assert f"\\! kernel {case.kernel}\n" in case.forth.read_text(), case.id


def test_forth_params_match_the_manifest_in_order() -> None:
    for case in CASES:
        declared = [
            (parts[2], parts[3])
            for line in case.forth.read_text().splitlines()
            if line.startswith("\\! param ")
            for parts in [line.split()]
        ]
        expected = [
            (param.name, f"{param.type}[{param.elements}]" if param.is_array else param.type)
            for param in case.params
        ]
        assert declared == expected, case.id


def test_f32_buffers_declare_half_as_many_cells_as_they_hold_elements() -> None:
    for case in CASES:
        for name, spec in case.inputs.items():
            if spec.dtype != "f32":
                continue
            cells = case.param(name).elements
            assert spec.length == cells * 2, (case.id, name)


def test_baseline_kind_agrees_with_its_files() -> None:
    for case in CASES:
        baseline = case.baseline
        if baseline.kind == "host_lib":
            assert baseline.library, case.id
            assert baseline.symbol, case.id
            assert baseline.launch is None, case.id
            continue
        source = BENCHMARKS / baseline.source
        assert source.is_file(), case.id
        assert f"__global__ void {baseline.kernel}(" in source.read_text(), case.id
        assert baseline.launch is not None, case.id


def test_every_baseline_records_flags_and_features() -> None:
    for case in CASES:
        assert "-arch=sm_89" in case.baseline.nvcc_flags, case.id
        assert case.baseline.features, case.id


def test_suite_covers_the_hardware_features_warpforth_lacks() -> None:
    covered = " ".join(" ".join(case.baseline.features) for case in CASES).lower()
    for feature in REQUIRED_FEATURES:
        assert feature in covered, feature


def test_no_case_exceeds_the_thread_budget() -> None:
    for case in CASES:
        assert case.launch.total_threads <= MAX_TOTAL_THREADS, case.id
        if case.baseline.launch is not None:
            assert case.baseline.launch.total_threads <= MAX_TOTAL_THREADS, case.id


def test_timing_config_meets_the_plan_minimums() -> None:
    for case in CASES:
        assert case.timing.warmup >= manifest.MIN_WARMUP, case.id
        assert case.timing.samples >= manifest.MIN_SAMPLES, case.id
        assert case.timing.sample_ms >= manifest.MIN_SAMPLE_MS, case.id


def test_correctness_instance_matches_its_reference() -> None:
    for case in CASES:
        args = {**case.correctness_scalars(), **case.correctness.arrays}
        computed = manifest.reference(case.reference, args)
        assert manifest.within_tolerance(
            computed[: len(case.correctness.expected)], case.correctness.expected, *case.tolerance
        ), case.id


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_expected_outputs_match_a_numpy_recomputation(case: manifest.Case) -> None:
    """A second, independent implementation, so a manifest value cannot drift with the loader's."""
    computed = numpy_reference(case)
    expected = case.correctness.expected
    rtol, atol = case.tolerance
    if case.dtype_of(case.correctness.output_param) == "i64":
        rtol, atol = 0.0, 0.0
    assert manifest.within_tolerance(computed[: len(expected)], expected, rtol, atol), case.id


def test_correctness_instance_fits_its_declared_buffers() -> None:
    for case in CASES:
        for name, values in case.correctness.arrays.items():
            assert values, (case.id, name)
            assert len(values) <= case.param(name).elements, (case.id, name)
        assert case.correctness.output_param in case.correctness.arrays, case.id


def test_seeded_inputs_are_deterministic() -> None:
    for case in CASES:
        for name, spec in case.inputs.items():
            first = manifest.generate(spec, case.param(name).elements)
            second = manifest.generate(spec, case.param(name).elements)
            assert manifest.digest(first) == manifest.digest(second), (case.id, name)


def test_seeded_inputs_stay_inside_their_declared_range() -> None:
    for case in CASES:
        for name, spec in case.inputs.items():
            values = manifest.generate(spec, case.param(name).elements)
            assert min(values) >= spec.low, (case.id, name)
            assert max(values) < spec.high, (case.id, name)


def test_an_unknown_key_is_rejected() -> None:
    with pytest.raises(ManifestError, match="unknown key"):
        load_mutated('id = "scale_i64"', 'id = "scale_i64"\nsurprise = 1')


def test_a_missing_required_key_is_rejected() -> None:
    with pytest.raises(ManifestError, match="missing key"):
        load_mutated("warmup = 20\n", "")


def test_a_second_baseline_is_rejected() -> None:
    second = '[cases.ceiling]\nkind = "host_lib"\nlibrary = "cublas"\nsymbol = "cublasDgemm"\n\n'
    with pytest.raises(ManifestError, match="unknown key"):
        load_mutated("[cases.timing]", second + "[cases.timing]")


def test_an_oversized_launch_is_rejected() -> None:
    with pytest.raises(ManifestError, match="thread budget"):
        load_mutated("grid = [128, 1, 1]", "grid = [4096, 1, 1]")


def test_a_baseline_kind_without_its_files_is_rejected() -> None:
    with pytest.raises(ManifestError, match="no file"):
        load_mutated('source = "cases/scale_i64.cu"', 'source = "cases/missing.cu"')


def test_expected_outputs_that_disagree_with_the_reference_are_rejected() -> None:
    with pytest.raises(ManifestError, match="disagrees with the scale_i64 reference"):
        load_mutated("  6986, 4585", "  6987, 4585")


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_forth_kernel_matches_the_expected_outputs(
    case: manifest.Case, kernel_runner: KernelRunner
) -> None:
    correct = case.correctness
    expected = correct.expected
    cells = kernel_runner.run(
        forth_source=case.correctness_source(),
        params=case.correctness_params(),
        grid=correct.launch.grid,
        block=correct.launch.block,
        output_param=case.output_param_index(),
        output_count=case.correctness_output_cells(),
    )
    got = case.correctness_readback(cells)[: len(expected)]
    assert manifest.within_tolerance(got, expected, *case.tolerance), (case.id, got, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_forth_kernel_strides_over_multiple_turns(
    case: manifest.Case, kernel_runner: KernelRunner
) -> None:
    """One block must walk every turn of the grid-stride loop and still reach the same answer.

    A correctness launch gives each thread one turn at most, which cannot catch a wrong stride step
    or a stack slot left behind per turn. Squeezing the grid to one block makes each thread loop.
    """
    correct = case.correctness
    expected = correct.expected
    cells = kernel_runner.run(
        forth_source=case.correctness_source(),
        params=case.correctness_params(),
        grid=(1, 1, 1),
        block=correct.launch.block,
        output_param=case.output_param_index(),
        output_count=case.correctness_output_cells(),
    )
    got = case.correctness_readback(cells)[: len(expected)]
    assert manifest.within_tolerance(got, expected, *case.tolerance), (case.id, got, expected)
