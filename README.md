# WarpForth

An MLIR-based Forth compiler for programming GPU kernels. WarpForth defines a custom MLIR dialect for Forth stack operations and lowers through a pipeline of passes to PTX assembly.

See the [documentation](https://tetsuo-cpp.github.io/warpforth/).

## Dependencies

- LLVM/MLIR
- CMake
- C++17 compiler
- An NVIDIA driver and Python 3 on the GPU host (for GPU execution)
- [uv](https://github.com/astral-sh/uv) (for Python test tooling)

## Building

```bash
# Configure
cmake -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm

# Build
cmake --build build
```

## Quick Start

Write a naive integer matrix multiply kernel (M=2, N=3, K=4, one thread per output element):

```forth
\! kernel main
\! param A i64[8]
\! param B i64[12]
\! param C i64[6]

\ One thread computes C[row, col] where gid = row*N + col.
GLOBAL-ID
DUP 3 /
SWAP 3 MOD
0
4 0 DO
  2 PICK
  I SWAP 4 * +
  CELLS A + @
  I 3 * 3 PICK + CELLS B + @
  * +
LOOP
2 PICK 3 * 2 PICK +
CELLS C + !
```

Compile to PTX:

```bash
./build/bin/warpforthc matmul.forth -o matmul.ptx --arch sm_80
```

Run it on a machine with an NVIDIA GPU, with the repository root on the Python
path (for example `PYTHONPATH=. uv run python run_matmul.py`). A is 2x4
row-major, B is 4x3 row-major, C is 2x3 output:

```python
from pathlib import Path

import numpy as np

from warpforth_gpu.client import Device, worker_command

with Device.spawn(worker_command()) as device:
    kernel = device.load(Path("matmul.ptx").read_text()).kernel("main")
    a = device.upload(np.arange(1, 9, dtype=np.int64))
    b = device.upload(np.arange(1, 13, dtype=np.int64))
    c = device.alloc(6 * 8)
    kernel.launch(grid=(6, 1, 1), block=(1, 1, 1), args=[a, b, c])
    print(c.read(np.int64))  # [ 70  80  90 158 184 210]
```

`worker_command()` runs the device worker from its inlined source, so the
same call works over SSH on a remote GPU host:
`Device.spawn(["ssh", host, shlex.join(worker_command())])`.

## Toolchain

| Tool | Description |
|------|-------------|
| `warpforthc` | Compiles Forth source to PTX |
| `warpforth-translate` | Translates from Forth source to MLIR and MLIR to PTX assembly |
| `warpforth-opt` | Runs individual MLIR passes or entire pipeline |
| `warpforth_gpu` | Python package that loads PTX and runs kernels on an NVIDIA GPU, locally or over SSH |

These tools can be composed for debugging or inspecting intermediate stages:

```bash
./build/bin/warpforth-translate --forth-to-mlir kernel.forth | \
  ./build/bin/warpforth-opt --warpforth-pipeline | \
  ./build/bin/warpforth-translate --mlir-to-ptx
```

The pipeline accepts standard MLIR pipeline options for its NVVM target and
output format. For example, to target Ampere while retaining textual PTX
output:

```bash
./build/bin/warpforth-opt \
  --pass-pipeline='builtin.module(warpforth-pipeline{chip=sm_80 features=+ptx70 opt-level=3 compilation-target=isa})' \
  kernel.mlir
```

Available options are `chip` (default `sm_70`), `features` (default `+ptx60`),
`libdevice` (the configured libdevice path by default), `opt-level` (`0`, `1`,
`2`, or `3`; default `2`), and `compilation-target` (`llvm`, `isa`, `bin`, or
`fatbin`; default `isa`, textual PTX). The pipeline's `chip` option accepts the
same architectures as `warpforthc --arch`.

## Language Reference

WarpForth supports stack operations, integer and float arithmetic, control flow, global and shared memory access, reduced-width memory types, user-defined words with local variables, and GPU-specific operations.

See [docs/language.md](docs/language.md) for the full language reference.

## Architecture

WarpForth compiles Forth through a series of MLIR dialect lowerings, each replacing higher-level abstractions with lower-level ones until the program is expressed entirely in LLVM IR and can be handed to the NVPTX backend.

| Stage | Pass | Description |
|-------|-------------|-------------|
| **Parsing** | `warpforth-translate --forth-to-mlir` | Parses Forth source into the `forth` dialect. The kernel is represented as a series of stack ops on an abstract `!forth.stack` type. |
| **Stack lowering** | `warpforth-opt --convert-forth-to-memref` | The abstract `!forth.stack` type is materialized as a `memref<256xi64>` buffer and `index` pair. Stack ops become explicit loads, stores, and pointer arithmetic. |
| **GPU wrapping** | `warpforth-opt --convert-forth-to-gpu` | Functions are wrapped in a `gpu.module`, the kernel entry point is marked as a `gpu.kernel` and GPU intrinsic words are lowered to `gpu` ops. |
| **NVVM/LLVM lowering** | Standard MLIR passes | GPU→NVVM, math→LLVM intrinsics and NVVM→LLVM. |
| **Code generation** | `warpforth-translate --mlir-to-ptx` | The GPU module is serialized to PTX assembly via LLVM's NVPTX backend. |

## Demo

The `demo/` directory contains a GPT-2 text generation demo that routes scaled dot-product attention through a WarpForth-compiled kernel. See [docs/demo.md](docs/demo.md) for setup instructions.

## Testing

```bash
# Run the LIT test suite
cmake --build build --target check-warpforth

# Run end-to-end GPU tests (requires Vast.ai API key)
VASTAI_API_KEY=xxx uv run pytest -v -m gpu --arch sm_80
```

The GPU tests pass `--arch` to `warpforthc` and only consider Vast.ai GPUs with
a compatible compute capability.
