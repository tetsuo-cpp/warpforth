# WarpForth verification map

This directory is the maintained source for verifying the user-facing behavior of the WarpForth toolchain. Read this index before driving, then use the matching feature file as the recipe. Commands assume the repo root as cwd and `S=.claude/skills/verify-warpforth/scripts`.

## Baseline preconditions

- `cmake --build build` exits 0.
- `$S/doctor.sh` prints no `FAIL` (binaries present, build not stale, smoke compile emits a kernel).
- GPU journeys only: the user approved spending on a Vast.ai rental for this run, and `$S/doctor.sh --remote` accepts the key.

## Driving conventions

- Write kernels with a quoted heredoc (`cat > k.forth <<'FORTH'`) into scratch space, never into `test/` or the skill.
- Every kernel starts with `\! kernel <name>`; header directives (`\! param`, `\! shared`) come before any code.
- Drive the shipped binaries in `build/bin/`. Do not substitute lit/FileCheck runs for a journey: lit is the regression net, this map is the user path.
- One `gpu_run.py` spec per GPU rental. Batch every GPU job a proof needs into one spec.

## Proof and skip reporting

- Compile proof: source, command, exit code, and PTX or the diagnostic text (from `compile.sh` evidence).
- Value proof: a `gpu_run.py` evidence dir whose `result.json` has `"pass": true` against an expectation computed independently of the output, and `summary.json` has `"cleanup_verified": true`.
- Quote the evidence directory with every claim.
- If the GPU journey was not run (no key, user declined spend), say "values not verified on GPU" rather than reporting the feature verified from PTX alone.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior, then exactly four H2 sections in this order: `Sub-features`, `How to get to it (user POV)`, `Driving it with <harness>` (starting with `Preconditions:`), and `Gotchas`.

## Features

- [Compile to PTX](./compile-to-ptx.md): `warpforthc` input/output, `--arch`, default target.
- [Compiler diagnostics](./compiler-diagnostics.md): parse/header errors with source locations, bad flags, missing files.
- [Staged pipeline](./staged-pipeline.md): `warpforth-translate` and `warpforth-opt` stages and pipeline options.
- [GPU execution](./gpu-execution.md): params, grid/block, read-back values on a real GPU.

Not yet mapped: the GPT-2 attention demo (`docs/demo.md`).
