# Staged pipeline

A user debugging a kernel runs the compiler in stages: Forth to `forth`-dialect MLIR, individual lowering passes or the whole `warpforth-pipeline` with target options, then MLIR to PTX, and inspects each intermediate.

## Sub-features

- `stage-translate` `warpforth-translate --forth-to-mlir` emits `forth` dialect ops.
- `stage-passes` `warpforth-opt --convert-forth-to-memref` and `--convert-forth-to-gpu` run single lowerings.
- `stage-pipeline` `warpforth-opt --warpforth-pipeline` lowers to a `gpu.binary`.
- `stage-options` pipeline options `chip`, `features`, `opt-level` (0-3), `compilation-target` (`llvm|isa|bin|fatbin`).
- `stage-ptx` `warpforth-translate --mlir-to-ptx` turns pipeline output into PTX.

## How to get to it (user POV)

- `build/bin/warpforth-translate --forth-to-mlir k.forth | build/bin/warpforth-opt --warpforth-pipeline | build/bin/warpforth-translate --mlir-to-ptx`
- `build/bin/warpforth-opt --pass-pipeline='builtin.module(warpforth-pipeline{chip=sm_80 features=+ptx70 opt-level=3 compilation-target=isa})' k.mlir`

## Driving it with build/bin tools

Preconditions:

- Baseline preconditions hold. `K=.claude/skills/verify-warpforth/fixtures/smoke/matmul.forth`.

- **Translate.** Run `build/bin/warpforth-translate --forth-to-mlir $K`. Output contains `forth.` ops and a `func.func` for `main`; exit 0.
- **Single pass.** Pipe that into `build/bin/warpforth-opt --convert-forth-to-memref`. Output has `memref<256xi64>` and no `!forth.stack`.
- **Full pipeline to PTX.** Run the three-tool pipe above. PTX on stdout with `.visible .entry main(`; compare with `build/bin/warpforthc $K` output (should match for the same arch).
- **Target options.** Run the `--pass-pipeline` form with `chip=sm_80`. The `gpu.binary` attribute shows `chip = "sm_80"` and `O = 3`; its embedded assembly has `.target sm_80`.
- **Evidence.** `$S/compile.sh $K sm_80` stores `kernel.mlir` and `lowered.mlir` for the same journey.

## Gotchas

- `--warpforth-pipeline` with `compilation-target=llvm` or `bin`/`fatbin` produces non-text or non-PTX objects; `--mlir-to-ptx` expects `isa`.
- Feed `--forth-to-mlir` a file, or `-` for stdin; heredoc the source to keep `\!` intact.
