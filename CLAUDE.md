# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WarpForth is a FORTH dialect for GPU kernel programming built on MLIR (Multi-Level Intermediate Representation). It's a stack-based programming language that lowers to MLIR's GPU dialect for cross-platform GPU execution (CUDA, ROCm, Vulkan, etc.).

## Build System

### Prerequisites
- CMake 3.20+
- C++17 compiler
- LLVM/MLIR 17+ (must be installed separately)

### Building the Project

First, set environment variables for MLIR/LLVM paths:
```bash
export LLVM_DIR=/path/to/llvm/install/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/install/lib/cmake/mlir
```

Build with Ninja (preferred):
```bash
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

Or with Make:
```bash
mkdir build && cd build
cmake .. -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Verify the build:
```bash
./bin/warpforth-opt --help
```

### Testing Changes

Use `warpforth-opt` to test dialect operations and transformations:
```bash
# Parse and verify FORTH dialect
./bin/warpforth-opt input.mlir

# Lower FORTH to GPU dialect
./bin/warpforth-opt input.mlir --lower-forth-to-gpu

# Test from stdin
echo 'func.func @test() { ... }' | ./bin/warpforth-opt --lower-forth-to-gpu
```

## Architecture

### Compilation Pipeline

```
FORTH Source Code (future)
       ↓
  FORTH Dialect (warpforth) - Stack-based operations
       ↓
  Arith Dialect (MLIR) - Standard arithmetic operations
       ↓
  GPU Dialect (MLIR) - Platform-agnostic GPU operations
       ↓
  Target Backend (CUDA/ROCm/Vulkan/etc.)
```

### Key Components

**1. Dialect Definition (`include/warpforth/Dialect/Forth/`)**
- `ForthOps.td`: TableGen file defining all dialect operations (forth.constant, forth.add, forth.dup, etc.)
- `ForthDialect.h/.cpp`: Dialect registration and initialization
- `ForthOps.h`: Generated operation declarations (from TableGen)

**2. Dialect Operations (8 operations)**
Stack operations:
- `forth.dup`: Duplicate top value
- `forth.drop`: Remove top value
- `forth.swap`: Swap top two values

Arithmetic operations (binary):
- `forth.add`, `forth.sub`, `forth.mul`, `forth.div`: Standard arithmetic
- `forth.constant`: Push constant value

All operations except `forth.drop` are marked as `Pure` (no side effects).

**3. Lowering Pass (`lib/Conversion/ForthToGPU/`)**
- `ForthToGPU.cpp`: Implements conversion patterns from Forth dialect to Arith dialect
- Pattern-based rewriting: Each Forth op has a corresponding Arith op mapping
  - `forth.constant` → `arith.constant`
  - `forth.add` → `arith.addi` (integer add)
  - `forth.sub` → `arith.subi`
  - `forth.mul` → `arith.muli`
  - `forth.div` → `arith.divsi` (signed integer divide)
- Pass registration name: `--lower-forth-to-gpu`

**4. Optimizer Tool (`tools/warpforth-opt/`)**
- Standard MLIR optimizer driver (similar to mlir-opt)
- Registers Forth dialect and lowering passes
- Used for testing and applying transformations

### MLIR Integration Details

- Uses MLIR's TableGen for operation code generation
- Operations defined in `ForthOps.td` generate C++ code via TableGen
- Dialect namespace: `mlir::forth`
- Lowering uses MLIR's `ConversionTarget` and pattern rewriting framework
- Currently lowers to Arith dialect (not directly to GPU dialect despite pass name)

## Adding New Operations

1. **Define operation in TableGen** (`include/warpforth/Dialect/Forth/ForthOps.td`):
   ```tablegen
   def Forth_MyOp : Forth_Op<"myop", [Pure]> {
     let summary = "Brief description";
     let arguments = (ins AnyType:$input);
     let results = (outs AnyType:$result);
     let assemblyFormat = "$input attr-dict `:` type($result)";
   }
   ```

2. **Implement custom logic** (if needed) in `lib/Dialect/Forth/ForthOps.cpp`

3. **Add lowering pattern** in `lib/Conversion/ForthToGPU/ForthToGPU.cpp`:
   - For arithmetic ops: Add to `populateForthToGPUConversionPatterns()`
   - Use `ForthArithToArithPattern` template for simple binary ops
   - Create custom `OpRewritePattern` for complex transformations

4. **Rebuild**: TableGen will auto-generate operation classes from .td file

## Development Notes

- **No tests currently**: The project has no test directory or test infrastructure yet
- **MLIR syntax only**: No FORTH source parser exists; operations written directly in MLIR syntax
- **Current lowering target**: Operations lower to Arith dialect, not GPU dialect (future work)
- **Stack semantics**: Operations use SSA form (not runtime stack); stack concepts are in operation semantics only

## Roadmap Features (Not Yet Implemented)

- Control flow operations (IF, WHILE, DO-LOOP)
- Memory operations (load, store)
- GPU-specific operations (thread indexing, barriers)
- FORTH source language parser
- JIT execution support
- Optimization passes (stack fusion, dead code elimination)
- Test infrastructure

## File Organization

```
warpforth/
├── include/warpforth/        # Public headers
│   ├── Dialect/Forth/         # FORTH dialect definitions (TableGen + headers)
│   └── Conversion/ForthToGPU/ # Lowering pass headers
├── lib/                       # Implementation files
│   ├── Dialect/Forth/         # Dialect registration and ops implementation
│   └── Conversion/ForthToGPU/ # Lowering pass implementation
└── tools/warpforth-opt/       # Optimizer driver tool
```

## CMake Build Structure

- Root `CMakeLists.txt` finds MLIR/LLVM and sets up build environment
- Uses MLIR's CMake modules: `TableGen`, `AddLLVM`, `AddMLIR`
- TableGen processes `.td` files during build to generate C++ operation code
- Build outputs to `build/bin/` (executables) and `build/lib/` (libraries)
