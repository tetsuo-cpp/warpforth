---
name: verify-warpforth
description: Drive the real WarpForth toolchain to prove a change works. Compiles Forth with build/bin/warpforthc (and the warpforth-translate | warpforth-opt stages) locally, and executes kernels on a rented Vast.ai GPU via warpforth-runner. Use after changing the parser, dialect, conversions, pipeline, or CLI flags, when lit tests alone don't show the kernel actually computes the right numbers.
---

# Verify WarpForth

WarpForth is a CLI compiler: the user writes a `.forth` kernel, runs `warpforthc` to get PTX, and runs that PTX on an NVIDIA GPU with `warpforth-runner`. There is no server or UI. This machine (macOS) has no CUDA GPU, no `nvcc`, no `ptxas`, so:

- **Local surface** (free, always available): `warpforthc`, `warpforth-translate`, `warpforth-opt`. Proves parsing, diagnostics, lowering, and that PTX is emitted.
- **GPU surface** (costs money, needs `VASTAI_API_KEY`): rents one Vast.ai GPU (cap $0.50/hr, `nvidia/cuda:12.4.0-devel-ubuntu22.04`, host driver must support CUDA 12.4), builds `warpforth-runner` there with `nvcc`, executes PTX, reads back a buffer. This is the only proof that a kernel computes the right values. **Ask the user before each GPU run**: it spends their money.

Other surfaces, not mapped yet: the GPT-2 demo (`demo/`, `docs/demo.md`) needs a PyTorch Vast.ai image and a manual scp.

All commands below run from the repo root. Helpers live in `.claude/skills/verify-warpforth/scripts/` (written `$S` below; set `S=.claude/skills/verify-warpforth/scripts`).

## Launch

There is no long-running process. Launch means "build is current":

```bash
cmake --build build
```

Ready when it exits 0 and `$S/doctor.sh` prints no `FAIL`. Each compile is its own short-lived process, so any number of agents can compile side by side. Teardown: nothing.

GPU launch happens inside `$S/gpu_run.py` (below): it searches offers, creates an instance labeled `warpforth-test-<timestamp>-<uuid>`, waits for sshd, compiles the runner with `nvcc`. Usually 2-7 minutes (the first proof run took 6: 5 to boot, 1 for nvcc and the jobs). Concurrent GPU runs are safe: each owns its own label and destroys only that.

## Doctor

```bash
$S/doctor.sh            # local: binaries exist, `ninja -n` shows no pending build steps, smoke compile emits `.visible .entry main`
$S/doctor.sh --remote   # plus: uv/ssh/scp on PATH, Vast.ai key accepted, lists any running warpforth-test-* instances
```

Read-only (the smoke compile writes only to a mktemp dir). Exit 0 means worth driving. A stale build is the most common false result: a verification run against old binaries proves nothing, so rebuild when doctor says stale. `--remote` warnings about existing instances are informational; they may be another agent's live run.

## Drive

**Local compile, with every stage kept:**

```bash
$S/compile.sh path/to/kernel.forth [sm_70|sm_75|sm_80|sm_86|sm_89|sm_90]
```

Runs exactly `build/bin/warpforthc <file> --arch <arch> -o kernel.ptx` (the user path) and, on success, also `warpforth-translate --forth-to-mlir` and `warpforth-opt --pass-pipeline='builtin.module(warpforth-pipeline{chip=<arch>})'` for inspection. Prints `evidence: <dir> (rc=N)`; exit code is warpforthc's. To drive a raw command yourself, use the binaries in `build/bin/` directly; write Forth with a quoted heredoc (`cat > k.forth <<'FORTH'`) because `\!` headers break under shell escaping.

**GPU execution (ask the user first):**

```bash
uv run python $S/gpu_run.py <spec.json>
```

One spec = one rented GPU for all its jobs. Spec format is in the script's docstring: per job a `.forth` source, `params` by header name, `grid`, `block`, `output_param` (index into the `\! param` list, must be an array), `output_count`, and optional `expect` (a prefix of the output; floats compared with `isclose`). It reuses `gpu_test/conftest.py`'s `VastSession` and `KernelRunner`, the same path as `uv run pytest -m gpu`. Exit 0 only if every `expect` matched and instance destruction was verified. A ready-made spec covering int arithmetic, multi-thread matmul (the README quickstart), and f64 math is at `.claude/skills/verify-warpforth/fixtures/smoke/spec.json`. For a change, write a new spec next to the kernels you want to prove, in scratch space, not inside the skill.

The full pytest GPU suite (`uv run pytest -v -m gpu`, ~100+ kernels) is the regression net; `gpu_run.py` is for proving the specific kernel your change affects with captured evidence.

## Evidence

Everything lands in `build/verify-evidence/<run-id>/` (`build/` is gitignored; nothing cleans it):

- `compile.sh`: `kernel.forth`, `kernel.ptx`, `warpforthc.stderr`, `rc`, `kernel.mlir`, `lowered.mlir`.
- `gpu_run.py`: `spec.json`, `session.log` (offer, instance id, ssh, destroy verification), `owned_label.txt`, per job `kernel.forth`, `kernel.ptx`, `result.json` (output, expect, pass, or traceback), and `summary.json` (`cleanup_verified`).

Proof standards:

- Drive `warpforthc` / `warpforth-runner` as a user would. Do not call internal C++ APIs or hand-edit MLIR to make a case pass.
- A compile proof shows the source, the command's exit code, and the PTX (or the diagnostic for an error case, with its `file:line:col`). "It compiled" is not proof of correctness. Correct values need a GPU run.
- A GPU proof shows the kernel source, the PTX that ran, the inputs (the spec), and the read-back buffer matching an expectation you computed independently, not one copied from the output.
- Quote the `evidence:` path in your report.
- `--output-param` reads back one buffer. To prove a side effect in a second buffer, add a second job that outputs it.

## Cleanup

- Local: nothing to tear down. Leave `build/verify-evidence/`.
- GPU: `gpu_run.py` destroys its own instance on exit (and in an `atexit` hook). If it was killed, or `summary.json` lacks `"cleanup_verified": true`, run:

  ```bash
  uv run python $S/cleanup.py build/verify-evidence/<run-id>
  ```

  It destroys only instances whose label equals that run's `owned_label.txt`, then confirms none remain. Never destroy instances by the `warpforth-test-` prefix: concurrent runs share it. Cleanup never touches the evidence directory.

## Helpers

| Script | Invocation | Does |
| --- | --- | --- |
| `doctor.sh` | `$S/doctor.sh [--remote]` | read-only health check |
| `compile.sh` | `$S/compile.sh <kernel.forth> [arch]` | user-path compile plus stage dumps into evidence |
| `gpu_run.py` | `uv run python $S/gpu_run.py <spec.json>` | rent a GPU, run jobs, assert, destroy |
| `cleanup.py` | `uv run python $S/cleanup.py <evidence-dir>` | destroy a run's leftover instance by exact label |

`VASTAI_API_KEY` is read from the environment, then from `.env` in the repo root. Never print it.

## Feature map

See `features/README.md`. A proof that drives one entry point is incomplete when the feature file lists others.
