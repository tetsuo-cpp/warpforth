"""Strict loader for `cases.toml`, plus the references its correctness configs check against.

The manifest is benchmark data, so a typo in a key must fail loudly instead of silently taking a
default. Every path and cross-reference resolves at load time, so a broken case fails at the
manifest rather than inside a rented session.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = CASE_ROOT / "cases.toml"

BASELINE_KINDS = frozenset({"ptx_kernel", "device_lib_ptx", "host_lib"})

# The runtime allocates a 2 KiB stack buffer per thread, so a case that launches far more threads
# than this exhausts memory before it computes anything.
MAX_TOTAL_THREADS = 65_536

MIN_WARMUP = 10
MIN_SAMPLES = 30
MIN_SAMPLE_MS = 50.0

_PAIR = 2
_TRIPLE = 3

_ELEMENT_BYTES = {"i64": 8, "f64": 8, "f32": 4}

_ARRAY_PARAM_LINE = re.compile(r"^\\! param (\w+) (\w+)\[(\d+)\]$")


def _cells_for(values: list[float], dtype: str) -> int:
    """The i64 cells a buffer of these values occupies in an `i64[N]` header."""
    return -(-len(values) * _ELEMENT_BYTES[dtype] // 8)


class ManifestError(Exception):
    """`cases.toml` does not match the schema."""


@dataclass(frozen=True)
class Param:
    """A kernel argument.

    `type` is the cell type the header declares, which is i64 or f64. `dtype` is what the buffer
    really holds. They differ when a case stores f32 in i64 cells, which is the only way to hand
    narrow floats to the current word set: the header counts cells, so such a buffer declares half
    as many cells as it has elements.
    """

    name: str
    type: str
    elements: int
    dtype: str = ""

    @property
    def is_array(self) -> bool:
        return self.elements > 0

    @property
    def element_type(self) -> str:
        return self.dtype or self.type


@dataclass(frozen=True)
class InputSpec:
    """A seeded generator for one array parameter."""

    kind: str
    dtype: str
    seed: int
    low: float = 0.0
    high: float = 0.0
    length: int = 0


@dataclass(frozen=True)
class Launch:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]

    @property
    def total_threads(self) -> int:
        grid_threads = self.grid[0] * self.grid[1] * self.grid[2]
        block_threads = self.block[0] * self.block[1] * self.block[2]
        return grid_threads * block_threads


@dataclass(frozen=True)
class Baseline:
    """The single strongest CUDA implementation for a case."""

    kind: str
    source: str
    kernel: str
    nvcc_flags: tuple[str, ...]
    features: tuple[str, ...]
    launch: Launch | None
    library: str = ""
    symbol: str = ""


@dataclass(frozen=True)
class Timing:
    warmup: int
    samples: int
    sample_ms: float


@dataclass(frozen=True)
class Correctness:
    """A small instance of the case, with its inputs and its reference outputs written out.

    The values are literal rather than seeded, so a reviewer can check the expected outputs by
    hand and the test can recompute them from the inputs without touching a GPU.
    """

    params: dict[str, float]
    arrays: dict[str, list[float]]
    expected: list[float]
    output_param: str
    launch: Launch


@dataclass(frozen=True)
class Case:
    id: str
    version: int
    forth: Path
    kernel: str
    params: tuple[Param, ...]
    inputs: dict[str, InputSpec]
    launch: Launch
    correctness: Correctness
    baseline: Baseline
    timing: Timing
    reference: str
    tolerance: tuple[float, float]

    def param(self, name: str) -> Param:
        for decl in self.params:
            if decl.name == name:
                return decl
        msg = f"{self.id}: no parameter named {name}"
        raise ManifestError(msg)

    def dtype_of(self, name: str) -> str:
        """The element type the buffer really holds, which is finer than the header's cell type."""
        return self.param(name).element_type

    def correctness_scalars(self) -> dict[str, int | float]:
        """The correctness instance's scalar arguments, typed as their parameters declare."""
        typed: dict[str, int | float] = {}
        for name, value in self.correctness.params.items():
            typed[name] = int(value) if self.param(name).type == "i64" else value
        return typed

    def output_param_index(self) -> int:
        """The position of the case's output argument, which is the order the header declares."""
        return [decl.name for decl in self.params].index(self.correctness.output_param)

    def correctness_source(self) -> str:
        """The case source with every array header cut down to the correctness instance's size.

        A header's array length only bounds the buffer the runtime allocates, so a smaller length
        emits byte-identical PTX and the correctness run needs to ship 64 values instead of 4M.
        """
        cells = {
            name: _cells_for(values, self.dtype_of(name))
            for name, values in self.correctness.arrays.items()
        }
        lines = []
        for line in self.forth.read_text().splitlines():
            match = _ARRAY_PARAM_LINE.match(line)
            if match and match.group(1) in cells:
                name, cell_type = match.group(1), match.group(2)
                lines.append(f"\\! param {name} {cell_type}[{cells[name]}]")
            else:
                lines.append(line)
        return "\n".join(lines) + "\n"

    def correctness_params(self) -> dict[str, int | float | list[int] | list[float]]:
        """The correctness instance as `KernelRunner` arguments, in the header's declared types."""
        params: dict[str, int | float | list[int] | list[float]] = dict(self.correctness_scalars())
        for name, values in self.correctness.arrays.items():
            dtype = self.dtype_of(name)
            if dtype == "f32":
                params[name] = pack_f32(values)
            elif dtype == "i64":
                params[name] = [int(value) for value in values]
            else:
                params[name] = list(values)
        return params

    def correctness_readback(self, cells: list[int] | list[float]) -> list[float]:
        """The output cells the runtime returned, as the values the manifest states."""
        dtype = self.dtype_of(self.correctness.output_param)
        if dtype == "f32":
            return unpack_f32([int(cell) for cell in cells], len(self.correctness.expected))
        return [float(cell) for cell in cells]

    def correctness_output_cells(self) -> int:
        """How many cells to read back, which is half the element count for an f32 buffer."""
        name = self.correctness.output_param
        return _cells_for(self.correctness.arrays[name], self.dtype_of(name))


_CASE_KEYS = frozenset(
    {
        "id",
        "version",
        "forth",
        "kernel",
        "params",
        "inputs",
        "launch",
        "correctness",
        "baseline",
        "timing",
        "reference",
        "tolerance",
    }
)


def _require(mapping: dict[str, object], key: str, where: str) -> object:
    if key not in mapping:
        msg = f"{where}: missing key '{key}'"
        raise ManifestError(msg)
    return mapping[key]


def _reject_unknown(mapping: dict[str, object], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        msg = f"{where}: unknown key(s) {', '.join(unknown)}"
        raise ManifestError(msg)


def _as_table(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{where}: expected a table"
        raise ManifestError(msg)
    return value


def _as_positive_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = f"{where}: expected a positive integer, got {value!r}"
        raise ManifestError(msg)
    return value


def _as_str(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{where}: expected a non-empty string, got {value!r}"
        raise ManifestError(msg)
    return value


def _as_float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{where}: expected a number, got {value!r}"
        raise ManifestError(msg)
    return float(value)


def _as_str_list(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"{where}: expected a list of strings"
        raise ManifestError(msg)
    return tuple(value)


def _as_launch(value: object, where: str) -> Launch:
    mapping = _as_table(value, where)
    _reject_unknown(mapping, frozenset({"grid", "block"}), where)
    return Launch(
        grid=_triple(_require(mapping, "grid", where), f"{where}.grid"),
        block=_triple(_require(mapping, "block", where), f"{where}.block"),
    )


def _triple(value: object, where: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != _TRIPLE:
        msg = f"{where}: expected {_TRIPLE} integers"
        raise ManifestError(msg)
    first, second, third = value
    return (
        _as_positive_int(first, where),
        _as_positive_int(second, where),
        _as_positive_int(third, where),
    )


def _parse_param(entry: object, where: str) -> Param:
    mapping = _as_table(entry, where)
    _reject_unknown(mapping, frozenset({"name", "type", "elements", "dtype"}), where)
    param_type = _as_str(_require(mapping, "type", where), f"{where}.type")
    if param_type not in ("i64", "f64"):
        msg = f"{where}.type: unsupported type {param_type}"
        raise ManifestError(msg)
    dtype = mapping.get("dtype", "")
    if dtype:
        dtype = _as_str(dtype, f"{where}.dtype")
        if dtype not in ("i64", "f64", "f32"):
            msg = f"{where}.dtype: unsupported dtype {dtype}"
            raise ManifestError(msg)
        if _ELEMENT_BYTES[dtype] > _ELEMENT_BYTES[param_type]:
            msg = f"{where}.dtype: {dtype} does not fit in {param_type} cells"
            raise ManifestError(msg)
    elements = mapping.get("elements", 0)
    if not isinstance(elements, int) or isinstance(elements, bool) or elements < 0:
        msg = f"{where}.elements: expected a non-negative integer, got {elements!r}"
        raise ManifestError(msg)
    return Param(
        name=_as_str(_require(mapping, "name", where), f"{where}.name"),
        type=param_type,
        elements=elements,
        dtype=dtype,
    )


def _parse_input(entry: object, where: str) -> InputSpec:
    mapping = _as_table(entry, where)
    _reject_unknown(
        mapping,
        frozenset({"name", "kind", "dtype", "seed", "low", "high", "length"}),
        where,
    )
    kind = _as_str(_require(mapping, "kind", where), f"{where}.kind")
    if kind not in ("uniform", "int_uniform"):
        msg = f"{where}.kind: unsupported generator {kind}"
        raise ManifestError(msg)
    dtype = _as_str(_require(mapping, "dtype", where), f"{where}.dtype")
    if dtype not in ("i64", "f64", "f32"):
        msg = f"{where}.dtype: unsupported dtype {dtype}"
        raise ManifestError(msg)
    return InputSpec(
        kind=kind,
        dtype=dtype,
        seed=_as_positive_int(_require(mapping, "seed", where), f"{where}.seed"),
        low=float(mapping.get("low", 0.0)),
        high=float(mapping.get("high", 1.0)),
        length=int(mapping.get("length", 0)),
    )


def _parse_baseline(entry: object, where: str, case_root: Path) -> Baseline:
    mapping = _as_table(entry, where)
    _reject_unknown(
        mapping,
        frozenset(
            {"kind", "source", "kernel", "nvcc_flags", "features", "launch", "library", "symbol"}
        ),
        where,
    )
    kind = _as_str(_require(mapping, "kind", where), f"{where}.kind")
    if kind not in BASELINE_KINDS:
        msg = f"{where}.kind: expected one of {sorted(BASELINE_KINDS)}, got {kind}"
        raise ManifestError(msg)

    flags = _as_str_list(_require(mapping, "nvcc_flags", where), f"{where}.nvcc_flags")
    features = _as_str_list(_require(mapping, "features", where), f"{where}.features")
    if not features:
        msg = f"{where}.features: a baseline must name the hardware features it exercises"
        raise ManifestError(msg)

    launch = _as_launch(mapping["launch"], f"{where}.launch") if "launch" in mapping else None
    if kind == "host_lib":
        if launch is not None:
            msg = f"{where}.launch: a host library has no launch geometry"
            raise ManifestError(msg)
        return Baseline(
            kind=kind,
            source="",
            kernel="",
            nvcc_flags=flags,
            features=features,
            launch=None,
            library=_as_str(_require(mapping, "library", where), f"{where}.library"),
            symbol=_as_str(_require(mapping, "symbol", where), f"{where}.symbol"),
        )

    if launch is None:
        msg = f"{where}.launch: a kernel baseline needs launch geometry"
        raise ManifestError(msg)
    source = _as_str(_require(mapping, "source", where), f"{where}.source")
    source_path = case_root / source
    if not source_path.is_file():
        msg = f"{where}.source: no file at {source_path}"
        raise ManifestError(msg)
    if "library" in mapping or "symbol" in mapping:
        msg = f"{where}: library and symbol belong to a host_lib baseline only"
        raise ManifestError(msg)
    return Baseline(
        kind=kind,
        source=source,
        kernel=_as_str(_require(mapping, "kernel", where), f"{where}.kernel"),
        nvcc_flags=flags,
        features=features,
        launch=launch,
    )


def _parse_timing(entry: object, where: str) -> Timing:
    mapping = _as_table(entry, where)
    _reject_unknown(mapping, frozenset({"warmup", "samples", "sample_ms"}), where)
    warmup = _as_positive_int(_require(mapping, "warmup", where), f"{where}.warmup")
    samples = _as_positive_int(_require(mapping, "samples", where), f"{where}.samples")
    sample_ms = float(_require(mapping, "sample_ms", where))
    if warmup < MIN_WARMUP:
        msg = f"{where}.warmup: needs at least {MIN_WARMUP}, got {warmup}"
        raise ManifestError(msg)
    if samples < MIN_SAMPLES:
        msg = f"{where}.samples: needs at least {MIN_SAMPLES}, got {samples}"
        raise ManifestError(msg)
    if sample_ms < MIN_SAMPLE_MS:
        msg = f"{where}.sample_ms: needs at least {MIN_SAMPLE_MS}, got {sample_ms}"
        raise ManifestError(msg)
    return Timing(warmup=warmup, samples=samples, sample_ms=sample_ms)


def _parse_correctness(entry: object, where: str) -> Correctness:
    mapping = _as_table(entry, where)
    _reject_unknown(
        mapping,
        frozenset({"params", "arrays", "expected", "output_param", "launch"}),
        where,
    )
    params = _as_table(_require(mapping, "params", where), f"{where}.params")
    scalars = {}
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            msg = f"{where}.params.{name}: expected a number, got {value!r}"
            raise ManifestError(msg)
        scalars[name] = float(value)

    arrays = _as_table(_require(mapping, "arrays", where), f"{where}.arrays")
    buffers: dict[str, list[float]] = {}
    for name, value in arrays.items():
        if not isinstance(value, list) or not all(
            isinstance(item, int | float) and not isinstance(item, bool) for item in value
        ):
            msg = f"{where}.arrays.{name}: expected a list of numbers"
            raise ManifestError(msg)
        buffers[name] = [float(item) for item in value]

    expected = _require(mapping, "expected", where)
    if not isinstance(expected, list) or not all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in expected
    ):
        msg = f"{where}.expected: expected a list of numbers"
        raise ManifestError(msg)
    return Correctness(
        params=scalars,
        arrays=buffers,
        expected=[float(item) for item in expected],
        output_param=_as_str(_require(mapping, "output_param", where), f"{where}.output_param"),
        launch=_as_launch(_require(mapping, "launch", where), f"{where}.launch"),
    )


def _as_table_array(value: object, where: str) -> list[object]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        msg = f"{where}: expected an array of tables"
        raise ManifestError(msg)
    return value


def _parse_params(mapping: dict[str, object], where: str) -> tuple[Param, ...]:
    entries = _as_table_array(_require(mapping, "params", where), f"{where}.params")
    params = tuple(
        _parse_param(item, f"{where}.params[{index}]") for index, item in enumerate(entries)
    )
    names = [decl.name for decl in params]
    if len(set(names)) != len(names):
        msg = f"{where}.params: duplicate parameter name"
        raise ManifestError(msg)
    return params


def _parse_inputs(
    mapping: dict[str, object], where: str, params: tuple[Param, ...]
) -> dict[str, InputSpec]:
    entries = _as_table_array(_require(mapping, "inputs", where), f"{where}.inputs")

    inputs: dict[str, InputSpec] = {}
    for index, raw in enumerate(entries):
        path = f"{where}.inputs[{index}]"
        table = _as_table(raw, path)
        name = _as_str(_require(table, "name", path), f"{path}.name")
        if name in inputs:
            msg = f"{path}: duplicate input for {name}"
            raise ManifestError(msg)
        inputs[name] = _parse_input(table, path)

    for name, spec in inputs.items():
        decl = next((decl for decl in params if decl.name == name), None)
        if decl is None:
            msg = f"{where}.inputs: {name} is not a parameter"
            raise ManifestError(msg)
        if not decl.is_array:
            msg = f"{where}.inputs.{name}: {name} is a scalar, it cannot have an input spec"
            raise ManifestError(msg)
        if spec.dtype != decl.element_type:
            msg = (
                f"{where}.inputs.{name}: dtype {spec.dtype} does not match the parameter's "
                f"{decl.element_type} elements"
            )
            raise ManifestError(msg)
        if spec.length and spec.length * _ELEMENT_BYTES[spec.dtype] > decl.elements * 8:
            msg = (
                f"{where}.inputs.{name}: {spec.length} {spec.dtype} values need more bytes "
                f"than the {decl.elements} cells the header declares"
            )
            raise ManifestError(msg)
    return inputs


def _parse_case(entry: object, where: str) -> Case:
    mapping = _as_table(entry, where)
    _reject_unknown(mapping, _CASE_KEYS, where)

    forth = CASE_ROOT / _as_str(_require(mapping, "forth", where), f"{where}.forth")
    if not forth.is_file():
        msg = f"{where}.forth: no file at {forth}"
        raise ManifestError(msg)

    params = _parse_params(mapping, where)
    inputs = _parse_inputs(mapping, where, params)
    tolerance = _as_number_pair(_require(mapping, "tolerance", where), where)
    reference = _as_str(_require(mapping, "reference", where), f"{where}.reference")
    if reference not in REFERENCES:
        msg = f"{where}.reference: unknown reference {reference}"
        raise ManifestError(msg)

    case = Case(
        id=_as_str(_require(mapping, "id", where), f"{where}.id"),
        version=_as_positive_int(_require(mapping, "version", where), f"{where}.version"),
        forth=forth,
        kernel=_as_str(_require(mapping, "kernel", where), f"{where}.kernel"),
        params=params,
        inputs=inputs,
        launch=_as_launch(_require(mapping, "launch", where), f"{where}.launch"),
        correctness=_parse_correctness(
            _require(mapping, "correctness", where), f"{where}.correctness"
        ),
        baseline=_parse_baseline(
            _require(mapping, "baseline", where), f"{where}.baseline", CASE_ROOT
        ),
        timing=_parse_timing(_require(mapping, "timing", where), f"{where}.timing"),
        reference=reference,
        tolerance=tolerance,
    )
    _check_case(case, where)
    return case


def _as_number_pair(value: object, where: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != _PAIR:
        msg = f"{where}.tolerance: expected [rtol, atol]"
        raise ManifestError(msg)
    rtol, atol = value
    return (_as_float(rtol, where), _as_float(atol, where))


def _check_launch_budget(launch: Launch | None, label: str, where: str) -> None:
    if launch is not None and launch.total_threads > MAX_TOTAL_THREADS:
        msg = (
            f"{where}.{label}: {launch.total_threads} threads exceeds the "
            f"{MAX_TOTAL_THREADS}-thread budget from the 2 KiB per-thread stack"
        )
        raise ManifestError(msg)


def _check_correctness_shape(case: Case, where: str) -> None:
    corr = case.correctness
    output = case.param(corr.output_param)
    if not output.is_array:
        msg = f"{where}.correctness.output_param: {output.name} is not an array"
        raise ManifestError(msg)
    if corr.output_param not in corr.arrays:
        msg = f"{where}.correctness.output_param: {corr.output_param} is not in arrays"
        raise ManifestError(msg)

    for name, values in corr.arrays.items():
        param = case.param(name)
        if not param.is_array:
            msg = f"{where}.correctness.arrays.{name}: {name} is a scalar"
            raise ManifestError(msg)
        if len(values) > param.elements:
            msg = (
                f"{where}.correctness.arrays.{name}: {len(values)} values exceed the "
                f"{param.elements} elements the header declares"
            )
            raise ManifestError(msg)

    for name in corr.params:
        if case.param(name).is_array:
            msg = f"{where}.correctness.params.{name}: {name} is an array, list it in arrays"
            raise ManifestError(msg)

    if not corr.expected:
        msg = f"{where}.correctness.expected: needs at least one value"
        raise ManifestError(msg)
    written = len(corr.arrays[corr.output_param])
    if len(corr.expected) > written:
        msg = (
            f"{where}.correctness.expected: {len(corr.expected)} values exceed the "
            f"{written} {corr.output_param} values"
        )
        raise ManifestError(msg)


def _check_case(case: Case, where: str) -> None:
    _check_launch_budget(case.launch, "launch", where)
    _check_launch_budget(case.baseline.launch, "baseline.launch", where)
    _check_correctness_shape(case, where)

    head = case.correctness.expected
    computed = reference(case.reference, {**case.correctness.params, **case.correctness.arrays})
    if not within_tolerance(computed[: len(head)], head, *case.tolerance):
        msg = (
            f"{where}.correctness.expected: disagrees with the {case.reference} reference; "
            f"expected {head[:4]}, computed {computed[:4]}"
        )
        raise ManifestError(msg)


def load(path: Path | str = DEFAULT_MANIFEST) -> tuple[Case, ...]:
    """Parse and validate a manifest, returning its cases in file order."""
    manifest_path = Path(path)
    try:
        text = manifest_path.read_bytes()
    except OSError as exc:
        msg = f"cannot read {manifest_path}: {exc}"
        raise ManifestError(msg) from exc
    try:
        root = tomllib.loads(text.decode())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        msg = f"{manifest_path} is not valid TOML: {exc}"
        raise ManifestError(msg) from exc

    _reject_unknown(root, frozenset({"cases"}), str(manifest_path))
    entries = _require(root, "cases", str(manifest_path))
    if not isinstance(entries, list) or not entries:
        msg = f"{manifest_path}: 'cases' must be a non-empty array of tables"
        raise ManifestError(msg)

    cases = tuple(_parse_case(entry, f"cases[{index}]") for index, entry in enumerate(entries))
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        msg = f"{manifest_path}: duplicate case id"
        raise ManifestError(msg)
    return cases


def generate(spec: InputSpec, elements: int) -> list:
    """A seeded, deterministic buffer for one array parameter.

    The generator is a small xorshift rather than `random`, so the same seed yields the same
    buffer on any Python build and in any process.
    """
    count = spec.length or elements
    state = spec.seed & 0xFFFFFFFF or 1
    values: list = []
    for _ in range(count):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        unit = state / 0x100000000
        if spec.kind == "int_uniform":
            low = int(spec.low)
            high = int(spec.high)
            values.append(int(low + unit * (high - low)))
        else:
            values.append(spec.low + unit * (spec.high - spec.low))
    return values


def digest(values: list) -> str:
    return hashlib.sha256(repr(values).encode()).hexdigest()


def pack_f32(values: list[float]) -> list[int]:
    """f32 values as the i64 cells an `i64[N]` header declares, two elements per cell."""
    packed = struct.pack(f"<{len(values)}f", *values)
    packed += bytes(-len(packed) % 8)
    return [item[0] for item in struct.iter_unpack("<q", packed)]


def unpack_f32(cells: list[int], count: int) -> list[float]:
    raw = b"".join(struct.pack("<q", cell) for cell in cells)
    return list(struct.unpack(f"<{count}f", raw[: count * 4]))


def reference(name: str, params: dict[str, object]) -> list[float]:
    """The independent reference for a case, computed with plain Python arithmetic."""
    return REFERENCES[name](params)


def _ref_scale(params: dict[str, object]) -> list[float]:
    scale = params["SCALE"]
    return [value * scale for value in params["INPUT"]]


def _ref_axpy(params: dict[str, object]) -> list[float]:
    alpha = params["ALPHA"]
    return [alpha * x + y for x, y in zip(params["X"], params["Y"], strict=True)]


def _ref_reduce(params: dict[str, object]) -> list[float]:
    return [math.fsum(params["INPUT"][: int(params["N"])])]


def _matmul(a: list[float], b: list[float], n: int) -> list[float]:
    out = []
    for row in range(n):
        for col in range(n):
            terms = [a[row * n + k] * b[k * n + col] for k in range(n)]
            out.append(math.fsum(terms))
    return out


def _ref_matmul(params: dict[str, object]) -> list[float]:
    return _matmul(params["A"], params["B"], int(params["N"]))


def _ref_attention(params: dict[str, object]) -> list[float]:
    nrow, nkey, hdim = int(params["NROW"]), int(params["NKEY"]), int(params["HDIM"])
    q, k, v = params["QMAT"], params["KMAT"], params["VMAT"]
    scale = params["SCALE"]
    out = []
    for row in range(nrow):
        scores = [
            math.fsum(q[row * hdim + d] * k[key * hdim + d] for d in range(hdim)) * scale
            for key in range(nkey)
        ]
        peak = max(scores)
        weights = [math.exp(score - peak) for score in scores]
        total = math.fsum(weights)
        out.extend(
            math.fsum(weights[key] * v[key * hdim + d] for key in range(nkey)) / total
            for d in range(hdim)
        )
    return out


REFERENCES = {
    "scale_i64": _ref_scale,
    "axpy_f64": _ref_axpy,
    "reduce_sum_f64": _ref_reduce,
    "matmul_f64": _ref_matmul,
    "attention_f32": _ref_attention,
}


def within_tolerance(actual: list[float], expected: list[float], rtol: float, atol: float) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        math.isclose(got, want, rel_tol=rtol, abs_tol=atol)
        for got, want in zip(actual, expected, strict=True)
    )


DEFAULT_BUILD = Path("build")
COMPILER_PATH = Path("bin") / "warpforthc"
DEFAULT_COMPILER = DEFAULT_BUILD / COMPILER_PATH
DEFAULT_ARCH = "sm_89"


def _git_common_dir(root: Path) -> Path | None:
    result = subprocess.run(  # noqa: S603 - a fixed argument list, never a shell string
        ["git", "-C", str(root), "rev-parse", "--git-common-dir"],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    common = Path(result.stdout.strip())
    return common if common.is_absolute() else root / common


def find_build(root: Path) -> Path:
    """The build directory for a checkout, falling back to the one owning the shared git directory.

    `WARPFORTH_BUILD` overrides the search. A git worktree has no build directory of its own, so the
    fallback is the main checkout's build, which is the one a worktree should use.
    """
    override = os.environ.get("WARPFORTH_BUILD")
    candidates = [Path(override)] if override else []
    candidates.append(root / DEFAULT_BUILD)
    common = _git_common_dir(root)
    if common is not None:
        candidates.append(common.parent / DEFAULT_BUILD)
    for candidate in candidates:
        if (candidate / COMPILER_PATH).is_file():
            return candidate
    return candidates[0]


def find_compiler(root: Path) -> Path:
    """warpforthc for a checkout, found the same way as its build directory."""
    override = os.environ.get("WARPFORTHC")
    return Path(override) if override else find_build(root) / COMPILER_PATH


def compile_source(source: str, compiler: Path, arch: str = DEFAULT_ARCH, *, label: str) -> str:
    """Compile one Forth source through warpforthc, returning the PTX text."""
    with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as handle:
        output = Path(handle.name)
    with tempfile.NamedTemporaryFile("w", suffix=".forth", delete=False) as forth:
        forth.write(source)
        forth_path = Path(forth.name)
    try:
        result = subprocess.run(  # noqa: S603 - the command is a fixed list, not a shell string
            [str(compiler), str(forth_path), "--arch", arch, "-o", str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            msg = f"{label}: warpforthc exited {result.returncode}\n{result.stderr}"
            raise ManifestError(msg)
        return output.read_text()
    finally:
        output.unlink(missing_ok=True)
        forth_path.unlink(missing_ok=True)


def compile_case(case: Case, compiler: Path, arch: str = DEFAULT_ARCH) -> str:
    """Compile a case's Forth source, checking the PTX exposes the entry the manifest names."""
    ptx = compile_source(case.forth.read_text(), compiler, arch, label=case.id)
    return check_entry(ptx, case.kernel, case.id)


def compile_text(
    source: str, kernel: str, compiler: Path, arch: str = DEFAULT_ARCH, *, label: str
) -> str:
    """Compile one Forth source that the manifest does not list, checking its entry point."""
    return check_entry(compile_source(source, compiler, arch, label=label), kernel, label)


def check_entry(ptx: str, kernel: str, label: str) -> str:
    if f".visible .entry {kernel}(" not in ptx:
        msg = f"{label}: the PTX has no entry named {kernel}"
        raise ManifestError(msg)
    return ptx
