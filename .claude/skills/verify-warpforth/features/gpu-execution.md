# GPU execution

A user runs compiled PTX on an NVIDIA GPU through `warpforth_gpu.client.Device`, uploading array buffers, passing scalar values, choosing a grid and block shape, and reading buffers back as numpy arrays. This is the only journey that proves a kernel computes the right numbers.

## Sub-features

- `gpu-int` i64 arithmetic and memory (`@ !`, `CELLS`) produce expected values.
- `gpu-threads` `GLOBAL-ID` / `TID-X` etc. with `grid`/`block` > 1 write one result per thread.
- `gpu-float` f64 literals, `F@ F!`, float intrinsics (`FSQRT`, `FEXP`, ...) round-trip through f64 params.
- `gpu-scalar-params` `\! param N i64` / `f64` scalars reach the kernel by value.
- `gpu-shared` `\! shared` buffers with `BARRIER` across a block.

## How to get to it (user POV)

- Compile locally with `warpforthc`, then on a GPU host, from the repo root: `Device.spawn(worker_command())`, `device.load(ptx).kernel("main").launch(grid=..., block=..., args=[device.upload(array), 42])`, `buffer.read(np.int64)`. For a remote host, prefix the ssh argv: `Device.spawn([*ssh_argv, shlex.join(worker_command())])`.
- `uv run pytest -v -m gpu` (regression suite, same harness).

## Driving it with gpu_run.py

Preconditions:

- Baseline preconditions hold, including user approval for the Vast.ai spend.
- `$S/doctor.sh --remote` exits 0.

- **Smoke run.** Run `uv run python $S/gpu_run.py .claude/skills/verify-warpforth/fixtures/smoke/spec.json`. It prints `evidence: <dir>` first, logs the instance id and SSH readiness, then one JSON line per job. Exit 0.
- **Int.** `<dir>/add/result.json` has `"output": [7, 0, 0, 0]`, `"pass": true`.
- **Threads.** `<dir>/matmul/result.json` has `"output": [70, 80, 90, 158, 184, 210]` from `grid` `[6,1,1]` (hand-computed `A(2x4)·B(4x3)`).
- **Float.** `<dir>/fsqrt/result.json` output is `sqrt(2)` = `1.4142135623730951`.
- **Cleanup proof.** `<dir>/summary.json` has `"cleanup_verified": true`, and `session.log` ends with `Verified cleanup for Vast.ai label=...`.
- **Your change.** Write a spec with kernels exercising the changed behavior (scalar params, shared memory, etc.) and an `expect` computed independently; run it the same way.

## Gotchas

- Rental can fail with `No suitable GPU offers found` under the $0.50/hr cap; retry later or report it, don't raise the cap without the user.
- `output_param` must index an array param; scalars cannot be read back.
- A CUDA error during a launch (for example an illegal address) is fatal to the worker; `KernelRunner` starts a fresh one for the next job, and that job's `result.json` shows the `DeviceError`.
- `params` values are padded with zeros to the declared array size; `expect` is a prefix match, so trailing zeros need not be listed.
- Higher `arch` in the spec (e.g. `sm_90`) narrows offers to newer GPUs and raises cost and wait.
- Instance boot takes minutes; run in the background and don't kill it mid-run. If you do, run `cleanup.py` on its evidence dir.
