# WarpForth

An MLIR-based Forth compiler for programming GPU kernels. WarpForth defines a custom MLIR dialect for Forth stack operations and lowers through a pipeline of passes to PTX assembly.

## Dependencies

- LLVM/MLIR
- CMake
- C++17 compiler
- CUDA toolkit (for GPU execution)
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
./build/bin/warpforthc matmul.forth -o matmul.ptx
```

Test on a GPU (A is 2x4 row-major, B is 4x3 row-major, C is 2x3 output):

```bash
./build/bin/warpforth-runner matmul.ptx \
  --param 'i64[]:1,2,3,4,5,6,7,8' \
  --param 'i64[]:1,2,3,4,5,6,7,8,9,10,11,12' \
  --param 'i64[]:0,0,0,0,0,0' \
  --grid 6,1,1 --block 1,1,1 \
  --output-param 2 --output-count 6
```

## Toolchain

| Tool | Description |
|------|-------------|
| `warpforthc` | Compiles Forth source to PTX |
| `warpforth-translate` | Translates from Forth source to MLIR and MLIR to PTX assembly |
| `warpforth-opt` | Runs individual MLIR passes or entire pipeline |
| `warpforth-runner` | Executes PTX kernels on a GPU for testing |

These tools can be composed for debugging or inspecting intermediate stages:

```bash
./build/bin/warpforth-translate --forth-to-mlir kernel.forth | \
  ./build/bin/warpforth-opt --warpforth-pipeline | \
  ./build/bin/warpforth-translate --mlir-to-ptx
```

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

The `demo/` directory contains a GPT-2 text generation demo that routes scaled dot-product attention through a WarpForth-compiled kernel. See [demo/README.md](demo/README.md) for setup instructions.

## Testing

```bash
# Run the LIT test suite
cmake --build build --target check-warpforth

# Run end-to-end GPU tests (requires Vast.ai API key)
VASTAI_API_KEY=xxx uv run pytest -v -m gpu
```
