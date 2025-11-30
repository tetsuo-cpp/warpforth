# CLAUDE.md

WarpForth is an MLIR-based compiler for the Forth programming language targeting GPU kernels. It implements a custom MLIR dialect for Forth stack operations and converts them to executable PTX.

## Build System

```bash
# Build from root directory
cmake --build build

# Format code
cmake --build build --target format
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

## Architecture Notes

- **Stack Type**: `!forth.stack` - untyped stack, programmer ensures type safety
- **Operations**: All take stack as input and produce stack as output (except `forth.stack`)
- **Supported Words**: literals, `dup drop swap over rot`, `+ - * / mod`
- **Conversion**: `!forth.stack` → `memref<256xi64>` with explicit stack pointer
- **GPU**: Functions wrapped in `gpu.module`, `main` gets `gpu.kernel` attribute

## Conventions

- Follow LLVM/MLIR naming (CamelCase)
- Document stack effects as `( input -- output )`
- C++17 required
- Use `clang-format` (config in `.clang-format`)