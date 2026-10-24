# Compiler diagnostics

When a kernel is invalid, `warpforthc` exits non-zero and prints a diagnostic pointing at the offending source line and column, with a caret under it; bad flags and missing files get clear one-line errors.

## Sub-features

- `diag-unknown-word` reports `file:line:col: error: unknown word: <WORD>` plus the source line and caret.
- `diag-header` reports header errors (missing `\! kernel`, duplicate params, directives after code).
- `diag-control-flow` reports mismatched `IF/ELSE/THEN`, `BEGIN/UNTIL/REPEAT`, `DO/LOOP`, `LEAVE`, `EXIT` misuse.
- `diag-arch` rejects unsupported `--arch` values and lists the supported ones.
- `diag-input` reports a missing input file.

## How to get to it (user POV)

- `build/bin/warpforthc bad.forth` on any invalid source.
- `build/bin/warpforthc kernel.forth --arch sm_99`.
- `build/bin/warpforthc does-not-exist.forth`.

## Driving it with compile.sh

Preconditions:

- Baseline preconditions hold.
- A scratch file `bad.forth` containing `\! kernel main` then `FOO` on line 2.

- **Unknown word.** Run `$S/compile.sh bad.forth`. Exit 1; `<dir>/warpforthc.stderr` has `bad.forth:2:1: error: unknown word: FOO`, the line `FOO`, a `^`, and `error: failed to parse Forth source`. No `kernel.ptx` is written.
- **Missing kernel header.** Compile a file containing only `1 2 +`. Exit non-zero; stderr contains `\! kernel <name> is required`.
- **Bad arch.** Run `build/bin/warpforthc .claude/skills/verify-warpforth/fixtures/smoke/add.forth --arch sm_99; echo rc=$?`. Stderr names `unsupported GPU architecture 'sm_99'` and lists `sm_70, sm_75, sm_80, sm_86, sm_89, sm_90`; rc non-zero.
- **Missing file.** Run `build/bin/warpforthc nonexist.forth; echo rc=$?`. Stderr: `error: could not open input file 'nonexist.forth'`.

## Gotchas

- The `test/Translation/Forth/*-error.forth` files list the exact expected message for each error class in their `CHECK:` lines. Use them as the oracle; don't invent wording.
- Word lookup is case-insensitive (`dup` = `DUP`), so a lowercase builtin is not an unknown-word case.
- Assert the line:col, not only the message: location tracking is a feature (`test/Pipeline/source-locations.forth`).
