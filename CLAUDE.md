# CLAUDE.md

WarpForth is an MLIR-based compiler for the Forth programming language targeting GPU kernels. It implements a custom MLIR dialect for Forth stack operations and converts them to executable PTX.

## Build System

```bash
# Build from root directory
cmake --build build

# Format code
cmake --build build --target format

# Run tests (requires: uv sync)
cmake --build build --target check-warpforth
```

Requires MLIR/LLVM with `MLIR_DIR` and `LLVM_DIR` configured in CMake.

## Key Files

**Dialect definition:**
- `include/warpforth/Dialect/Forth/ForthOps.td` - Add new operations here
- `include/warpforth/Dialect/Forth/ForthDialect.td` - Type definitions

**Implementation:**
- `lib/Conversion/ForthToMemRef/ForthToMemRef.cpp` - Stack to MemRef conversion patterns
- `lib/Conversion/ForthToGPU/ForthToGPU.cpp` - GPU conversion logic
- `lib/Translation/ForthToMLIR/ForthToMLIR.cpp` - Forth parser and translator

**Tools:**
- `tools/warpforth-translate/warpforth-translate.cpp` - Translation tool entry point
- `tools/warpforth-opt/warpforth-opt.cpp` - Optimization tool entry point
- `tools/warpforth-runner/warpforth-runner.cpp` - PTX execution tool for GPU kernels

## Tools Usage

```bash
# Forth to MLIR
./build/bin/warpforth-translate --forth-to-mlir test/example.forth

# Run conversion passes
./build/bin/warpforth-opt --convert-forth-to-memref input.mlir
./build/bin/warpforth-opt --convert-forth-to-gpu input.mlir

# Full pipeline to PTX
./build/bin/warpforth-translate --forth-to-mlir test/example.forth | \
  ./build/bin/warpforth-opt --warpforth-pipeline | \
  ./build/bin/warpforth-translate --mlir-to-ptx > kernel.ptx

# Execute PTX on GPU
./warpforth-runner kernel.ptx --param i64[]:1,2,3 --param i64:42 --output-param 0 --output-count 3
```

## Adding New Operations

Define in `include/warpforth/Dialect/Forth/ForthOps.td`:

```tablegen
def Forth_NewOp : Forth_Op<"opname", [Pure]> {
  let summary = "Brief description";
  let description = [{ Stack effect: ( input -- output ) }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}
```

Add corresponding conversion pattern in `lib/Conversion/ForthToMemRef/ForthToMemRef.cpp`.

## GPU Tests

End-to-end GPU execution tests live in `gpu_test/`. They compile Forth kernels locally, rent a GPU on Vast.ai, and verify output.

```bash
# Run GPU tests
VASTAI_API_KEY=xxx uv run pytest -v -m gpu

# Lint and format Python code
uv run ruff check gpu_test/
uv run ruff format gpu_test/
```

## Architecture Notes

- **Stack Type**: `!forth.stack` - untyped stack, programmer ensures type safety
- **Operations**: All take stack as input and produce stack as output (except `forth.stack`)
- **Supported Words**: literals, `DUP DROP SWAP OVER ROT NIP TUCK PICK ROLL`, `+ - * / MOD`, `AND OR XOR NOT LSHIFT RSHIFT`, `= < > <> <= >= 0=`, `@ !`, `CELLS`, `IF ELSE THEN`, `BEGIN UNTIL`, `BEGIN WHILE REPEAT`, `DO LOOP +LOOP I J K`, `LEAVE UNLOOP EXIT`, `TID-X/Y/Z BID-X/Y/Z BDIM-X/Y/Z GDIM-X/Y/Z GLOBAL-ID` (GPU indexing).
- **Kernel Parameters**: Declared in the `\!` header. `\! kernel <name>` is required and must appear first. `\! param <name> i64[<N>]` becomes a `memref<Nxi64>` argument; `\! param <name> i64` becomes an `i64` argument. Using a param name in code emits `forth.param_ref` (arrays push address; scalars push value).
- **Conversion**: `!forth.stack` → `memref<256xi64>` with explicit stack pointer
- **GPU**: Functions wrapped in `gpu.module`, `main` gets `gpu.kernel` attribute, configured with bare pointers for NVVM conversion
- **User-defined Words**: Modeled as `func.func` with signature `(!forth.stack) -> !forth.stack`, called via `func.call`

## Conventions

- Follow LLVM/MLIR naming (CamelCase)
- Document stack effects as `( input -- output )`
- C++17 required
- Use `clang-format` (config in `.clang-format`)

## Agent Instructions

- Use context7 MCP for MLIR API documentation: query with `/websites/mlir_llvm` for MLIR dialects, operations, types, and conversion patterns
