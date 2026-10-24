# Compile to PTX

A user compiles a `.forth` kernel to PTX assembly with `warpforthc`, choosing the target GPU architecture, and gets a PTX file whose entry point is named after the `\! kernel` header.

## Sub-features

- `compile-stdout` writes PTX to stdout when `-o` is omitted.
- `compile-output` writes PTX to the file named by `-o`.
- `compile-arch` sets `.target` from `--arch` (`sm_70` default; `sm_75 sm_80 sm_86 sm_89 sm_90`).
- `compile-entry` names the `.visible .entry` after `\! kernel <name>` and adds one `.param` per `\! param`.

## How to get to it (user POV)

- `build/bin/warpforthc kernel.forth` (PTX on stdout).
- `build/bin/warpforthc kernel.forth -o kernel.ptx --arch sm_80`.

## Driving it with compile.sh

Preconditions:

- Baseline preconditions hold.

- **Compile for a target.** Run `$S/compile.sh .claude/skills/verify-warpforth/fixtures/smoke/matmul.forth sm_80`. It prints `evidence: <dir> (rc=0)`.
- **Check target and entry.** Run `grep -E '^\.target|\.visible \.entry|_param_' <dir>/kernel.ptx`. Expect `.target sm_80`, `.visible .entry main(`, and three params `main_param_0..2`.
- **Default arch and stdout.** Run `build/bin/warpforthc .claude/skills/verify-warpforth/fixtures/smoke/add.forth | grep '^\.target'`. Expect `.target sm_70`.
- **Kernel name.** Compile a kernel headed `\! kernel saxpy`. Expect `.visible .entry saxpy(`.

## Gotchas

- `.version` in the PTX follows the arch (`7.0` at sm_80). Don't assert a fixed `.version` across arches.
- Successful compilation says nothing about computed values. Pair a semantic change with [GPU execution](./gpu-execution.md).
- The optimizer folds constants aggressively: `3 4 +` compiles to a store of `7`, so a PTX grep for an `add` instruction can fail on a correct build.
