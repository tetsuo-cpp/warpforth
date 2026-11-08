# WarpForth benchmark cases

The case suite measures how far WarpForth sits from the strongest CUDA implementation of each
problem. Every case has exactly one baseline, so a number is either "this much slower than the best
available CUDA" or nothing at all. There are no separate ceiling rows.

## One baseline per case

The baseline is the fastest practical CUDA implementation for that case. A vendor library wins where
one genuinely is best; a hand-written kernel wins where it is not. The manifest records which, so the
choice is visible rather than implied:

| Case | Baseline kind | What it is |
| --- | --- | --- |
| `scale_i64` | `ptx_kernel` | Hand-written 128-bit vectorized kernel. No library covers i64 scaling. |
| `axpy_f64` | `ptx_kernel` | Hand-written 128-bit vectorized kernel with fused multiply-add. Already at the bandwidth ceiling. |
| `reduce_sum_f64` | `device_lib_ptx` | CUB `BlockReduce` kernel, compiled and timed like any kernel. |
| `matmul_f64_naive` | `host_lib` | cuBLAS `Dgemm`, the practical SOTA for f64 GEMM. |
| `matmul_f64_tiled` | `host_lib` | cuBLAS `Dgemm`, same call against the tiled WarpForth variant. |
| `attention_f32` | `ptx_kernel` | Hand-written fused online-softmax kernel with warp-shuffle reductions. |

The attention baseline is hand-written because no practical FlashAttention or cuDNN SDPA binding
exists for the remote harness, which provisions `cuda-python` and `numpy` only. `flash-attn` needs
torch and compiles a large CUDA extension from source; the cuDNN frontend needs its own wheel and a
graph build. The hand-written kernel covers the same ground: online softmax fused with the output
accumulation in one pass, so a row of scores never touches memory.

Hand-written baselines deliberately use hardware WarpForth cannot express yet, so the measured gap
is a feature gap and not a tiling accident. The suite covers 128-bit vector loads, warp shuffles,
register-resident accumulators, fused multiply-add, and fused online softmax.

## Numerics

The two sides do not agree bitwise and are never compared bitwise. Each case carries:

- a small correctness instance, with its inputs and its expected outputs written out in the manifest,
- a per-case tolerance.

The Forth side is validated against the case reference within that tolerance. The CUDA side is
validated against the same reference within the same tolerance. Reduction order, fast-math flags,
and the accumulation shape all differ between the two, by design.

## Manifest

`cases.toml` holds per case: id, version, source path, kernel name, parameters with cell types and
sizes, a seeded generator per array, launch geometry for both sides, one baseline entry, the
correctness instance, tolerances, and the timing config.

Launch geometry is recorded per side because it differs. The WarpForth runtime allocates a 2 KiB
stack per thread, so every kernel uses a grid-stride loop with a bounded grid, and no case launches
more than `MAX_TOTAL_THREADS` threads in `benchmarks/manifest.py`.

Timing uses a warmup of at least 10, at least 30 samples, and at least 50 ms per sample batch.

The manifest loader is strict. An unknown key, a missing field, a baseline kind that disagrees with
its declared files, or an expected output that does not match the case reference all fail at load
time with a located error. Case sizes are initial guesses aimed at 0.1 to 10 ms per launch on a
mid-range GPU, and they are data: tune them in the manifest without touching code.

## Checked without a GPU

```bash
uv run pytest gpu_test/test_bench_cases.py
```

That compiles every `.forth` through `warpforthc` on three architectures and validates the
manifest: references resolve, kernel names and parameter order agree with the `.forth` headers, the
baseline policy holds, no launch exceeds the thread budget, the seeded generators are deterministic,
and each correctness instance matches a numpy recomputation of its expected outputs.

The tests need `build/bin/warpforthc`, which `cmake --build build` produces. From a git worktree
they find the main checkout's binary, and `WARPFORTHC` overrides the path. With no binary at all,
the compile tests skip and the rest still run.

## Checked on a rented GPU

Case correctness is decided by running the kernels, not by a model of them. With `VASTAI_API_KEY`
set, the gpu-marked tests rent one GPU through the existing `VastSession` and run each case's
correctness instance through `KernelRunner`:

```bash
uv run pytest -m gpu
```

Each case is launched at its correctness config's geometry with its scalars and its small input
buffers, and the outputs must match the manifest's expected values within the case's tolerance. The
whole suite shares one rental, and the runner path is the same one the rest of `gpu_test/` uses.

A correctness instance passes its array headers at the size it actually needs, so the run ships tens
of values instead of millions. That shrink is PTX-neutral, and
`test_correctness_source_is_the_benchmarked_kernel_at_a_smaller_size` pins it, so a case is verified
against the exact kernel the suite times.

## The gap table on one rental

`bench gap` answers how close WarpForth sits to SOTA CUDA. It rents one GPU, compiles every case's
`.forth` to PTX locally with `warpforthc`, compiles each hand-written or device-library CUDA baseline
to PTX on that rental with `nvcc` and the case's recorded flags, calls `cuBLAS` directly for the
`host_lib` cases, then runs the harness once per case for the small correctness config and once for
the timed config.

```bash
set -a; source .env; set +a
uv run python -m bench gap
```

```text
gap against SOTA CUDA at sm_89, WarpForth time over CUDA time, one rental
gpu_uuid=... driver_version=13.3 sm_clock_mhz=2490 memory_clock_mhz=8501 base_commit=... head_commit=...

case                     gap    warpforth         cuda    mad  ok
scale_i64              1.01x      272.2us      270.0us  0.001  yes
axpy_f64                   -            -            -      -  no  small config output 0 is -15.95..., the manifest expects -9.97...
reduce_sum_f64             -            -            -      -  no  cuMemcpyHtoD failed with error 1
matmul_f64_naive       1.49x       43.6us       29.2us  0.000  yes
matmul_f64_tiled       1.61x      349.9us      217.6us  0.002  yes
attention_f32          9.68x     1203.3us      124.3us  0.001  yes

geometric mean gap 2.199x over 4 of 6 stable cases (cutoff ratio MAD 0.050)
```

Those two failures are defects in the harness, not in the cases or this command, and they are open
work. The `axpy_f64` correctness op is launched twice before one readback, and this case writes into
its own input, so it applies its update twice. The `reduce_sum_f64` output buffer is a single `f64`,
and the harness cannot upload an 8-byte numpy array.

A gap is WarpForth time over CUDA time, so a larger number is a larger distance to close. A row
carries a gap only when it passed both gates: the small correctness instance against the manifest's
expected values, and the timed instance's own validation against the case reference. A case that
fails a gate keeps its row with the reason and is left out of the geometric mean, and the command
exits 1 when any case failed.

`--repeats 2` runs the whole suite twice on the same rental and prints a second table of both runs
plus the largest per-case change between them, which is the measured noise floor. `--case ID`
restricts the run, `--arch` picks the compute capability both sides compile for, and `--json` prints
the result document with every raw sample and each baseline's flags.

The reference for a case is computed on the device host, so the two sides are compared to the same
values rather than to each other. Reduction order and fast-math flags differ by design.

