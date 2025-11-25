# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WarpForth is an MLIR-based compiler infrastructure for the Forth programming language, designed for GPU kernel programming. It implements a custom MLIR dialect that represents Forth language constructs and stack operations.

## Build System

This project uses CMake with LLVM/MLIR infrastructure:

```bash
# Initial build configuration
mkdir -p build && cd build
cmake .. -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm
cmake --build .

# Build from root directory
cmake --build build

# Format all source files
cmake --build build --target format
```

**Note**: This project requires MLIR and LLVM to be built and installed. The CMakeLists.txt expects `MLIR_DIR` and `LLVM_DIR` to be configured.

## Architecture

### MLIR Dialect Structure

The project follows MLIR's standard dialect organization pattern:

- **Dialect Definition** (`ForthDialect.td`): Defines the `forth` dialect namespace and base classes
- **Operations** (`ForthOps.td`): TableGen definitions for all Forth operations
- **Types** (`ForthDialect.td`): Custom type definitions like `Forth_StackType`
- **C++ Implementation** (`ForthDialect.cpp`): Runtime initialization of types and operations

### Key Components

1. **Forth Stack Type**: An untyped stack representation where type correctness is the programmer's responsibility (defined in `ForthDialect.td:43`)

2. **Operation Categories**:
   - Stack initialization: `forth.stack` - initializes an empty stack
   - Literals: `forth.literal` - pushes integer literals onto the stack
   - Stack manipulation ops: `dup`, `drop`, `swap`, `over`, `rot`
   - Arithmetic ops: `add`, `sub`, `mul`, `div`, `mod`
   - All operations take `Forth_StackType` as input and produce `Forth_StackType` as output (except `forth.stack` which creates the initial stack)

3. **Tools**:
   - `warpforth-translate`: MLIR translation tool that converts Forth source to MLIR
   - `warpforth-opt`: MLIR optimization tool for running conversion passes and transformations

4. **Conversion Passes**:
   - `convert-forth-to-memref`: Lowers Forth dialect to MemRef dialect, converting the abstract stack to concrete memory operations
   - `convert-forth-to-gpu`: Converts Forth functions to GPU dialect for GPU execution

### Directory Layout

```
include/warpforth/
  ├── Dialect/Forth/               # Public headers and TableGen files
  │   ├── ForthDialect.td          # Dialect and type definitions
  │   ├── ForthOps.td              # Operation definitions
  │   └── ForthDialect.h           # Main C++ header
  ├── Conversion/                  # Conversion pass headers
  │   ├── Passes.h                 # Pass registration
  │   ├── Passes.td                # TableGen pass definitions
  │   ├── ForthToMemRef/           # Forth to MemRef conversion
  │   │   └── ForthToMemRef.h      # Pass declaration
  │   └── ForthToGPU/              # Forth to GPU conversion
  │       └── ForthToGPU.h         # Pass declaration
  └── Translation/                 # Translation public API
      ├── ForthToMLIR/
      │   └── ForthToMLIR.h        # Forth-to-MLIR translation registration
      └── MLIRToPTX/
          └── MLIRToPTX.h          # MLIR-to-PTX translation registration

lib/
  ├── Dialect/Forth/               # Dialect implementation
  │   └── ForthDialect.cpp         # Dialect initialization
  ├── Conversion/                  # Conversion pass implementations
  │   ├── Passes.cpp               # Pass registration
  │   ├── ForthToMemRef/           # Forth to MemRef conversion
  │   │   └── ForthToMemRef.cpp    # Conversion patterns
  │   └── ForthToGPU/              # Forth to GPU conversion
  │       └── ForthToGPU.cpp       # GPU conversion implementation
  └── Translation/                 # Translation implementations
      ├── ForthToMLIR/
      │   ├── ForthToMLIR.cpp      # Lexer, parser, and translator
      │   └── ForthToMLIR.h        # Private translation headers
      └── MLIRToPTX/
          └── MLIRToPTX.cpp        # PTX extraction from gpu.binary

tools/
  ├── warpforth-translate/         # Translation tool
  │   └── warpforth-translate.cpp  # Main entry point
  └── warpforth-opt/               # Optimization tool
      └── warpforth-opt.cpp        # Pass driver

test/                              # Test files
  ├── example.forth                # Simple arithmetic example
  └── stack-ops.forth              # Stack manipulation example
```

## TableGen Code Generation

MLIR uses TableGen to generate C++ code from `.td` files. The build system automatically generates:
- `ForthOpsDialect.h.inc` / `.cpp.inc` - Dialect boilerplate
- `ForthOpsTypes.h.inc` / `.cpp.inc` - Type definitions
- `ForthOps.h.inc` / `.cpp.inc` - Operation definitions

These generated files are included in `ForthDialect.h` and `ForthDialect.cpp`.

## Adding New Operations

To add a new Forth operation:

1. Define the operation in `include/warpforth/Dialect/Forth/ForthOps.td` following the pattern:
   ```tablegen
   def Forth_NewOp : Forth_Op<"opname", [Pure]> {
     let summary = "Brief description";
     let description = [{ Detailed description with Forth semantics: ( stack-effect ) }];
     let arguments = (ins Forth_StackType:$input_stack);
     let results = (outs Forth_StackType:$output_stack);
     let assemblyFormat = [{
       $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
     }];
   }
   ```

2. Rebuild the project - TableGen will automatically generate the C++ implementation

## Using the Translator

The `warpforth-translate` tool can convert Forth source code to MLIR using the `--forth-to-mlir` flag:

```bash
# Translate Forth source to MLIR
./build/bin/warpforth-translate --forth-to-mlir test/example.forth

# Example Forth input (test/example.forth):
# 5 3 + 2 *

# Generated MLIR output:
# module {
#   func.func private @main() {
#     %0 = forth.stack : !forth.stack
#     %1 = forth.literal %0 5 : !forth.stack -> !forth.stack
#     %2 = forth.literal %1 3 : !forth.stack -> !forth.stack
#     %3 = forth.add %2 : !forth.stack -> !forth.stack
#     %4 = forth.literal %3 2 : !forth.stack -> !forth.stack
#     %5 = forth.mul %4 : !forth.stack -> !forth.stack
#     return
#   }
# }
```

**Supported Forth Words:**
- Numbers: Integer literals (e.g., `42`, `-10`)
- Stack operations: `dup`, `drop`, `swap`, `over`, `rot`
- Arithmetic: `+`, `-`, `*`, `/`, `mod` (or `add`, `sub`, `mul`, `div`)

The translator tokenizes Forth source (whitespace-delimited), generates corresponding MLIR operations, and threads the stack value through each operation.

## Conversion Passes

### Forth to MemRef Conversion

The `convert-forth-to-memref` pass lowers the abstract `!forth.stack` type to concrete `memref<256xi64>` with an explicit stack pointer (`index` type). Use with `warpforth-opt`:

```bash
# Convert Forth dialect to MemRef dialect
./build/bin/warpforth-opt --convert-forth-to-memref input.mlir

# Full pipeline from Forth source
./build/bin/warpforth-translate --forth-to-mlir test/example.forth | \
  ./build/bin/warpforth-opt --convert-forth-to-memref
```

Conversion patterns are in `lib/Conversion/ForthToMemRef/ForthToMemRef.cpp`. When adding new Forth operations, add corresponding conversion patterns there.

### Forth to GPU Conversion

The `convert-forth-to-gpu` pass converts Forth functions to GPU dialect for GPU execution. Use with `warpforth-opt`:

```bash
# Convert Forth functions to GPU dialect
./build/bin/warpforth-opt --convert-forth-to-gpu input.mlir

# Full pipeline from Forth source
./build/bin/warpforth-translate --forth-to-mlir test/example.forth | \
  ./build/bin/warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu
```

The pass wraps `func.func` operations in a `gpu.module` and converts them to `gpu.func`. Functions named "main" receive the `gpu.kernel` attribute. Additionally, the pass annotates `memref.alloca` operations with GPU private address space (address space 5) to ensure thread-local stack allocation in GPU kernels.

### WarpForth Pipeline

The `warpforth-pipeline` is a registered pass pipeline that runs the complete compilation sequence from Forth dialect to PTX assembly suitable for CUDA execution. Use with `warpforth-opt`:

```bash
# Run the complete WarpForth compilation pipeline
./build/bin/warpforth-opt --warpforth-pipeline input.mlir

# Full pipeline from Forth source (equivalent to chaining individual passes)
./build/bin/warpforth-translate --forth-to-mlir test/example.forth | \
  ./build/bin/warpforth-opt --warpforth-pipeline
```

This pipeline internally runs:
1. `convert-forth-to-memref` (as a nested pass on `func.func` operations) - Lowers abstract stack to concrete memref operations
2. `convert-forth-to-gpu` - Wraps functions in GPU modules, annotates with private address space for thread-local stacks
3. Canonicalization pass - Normalizes MemRef operations for GPU
4. `gpu-nvvm-attach-target` - Attaches NVVM target (sm_50) to GPU modules
5. `convert-gpu-ops-to-nvvm-ops` (nested on `gpu.module` ops) - Lowers GPU dialect to NVVM IR
6. `convert-nvvm-to-llvm` - Lowers NVVM intrinsics to LLVM dialect
7. `reconcile-unrealized-casts` - Removes unrealized type conversions
8. `gpu-module-to-binary` - Compiles LLVM IR to PTX assembly and packages it as a `gpu.binary`

The final output is a `gpu.binary` operation containing embedded PTX assembly that can be loaded and executed on NVIDIA GPUs.

The pipeline is registered in `lib/Conversion/Passes.cpp`.

### Extracting PTX Assembly

The `--mlir-to-ptx` translation extracts PTX assembly from `gpu.binary` operations produced by `warpforth-pipeline`:

```bash
# Full pipeline from Forth source to PTX file
./build/bin/warpforth-translate --forth-to-mlir test/example.forth | \
  ./build/bin/warpforth-opt --warpforth-pipeline | \
  ./build/bin/warpforth-translate --mlir-to-ptx > kernel.ptx

# Or use -o flag for output file
... | ./build/bin/warpforth-translate --mlir-to-ptx -o kernel.ptx

# Pipe directly to ptxas for validation or compilation
... | ./build/bin/warpforth-translate --mlir-to-ptx | ptxas -arch=sm_50 -o kernel.cubin -
```

The translation walks the module for `gpu.binary` operations and outputs the embedded PTX assembly to stdout (or file via `-o`).

## Coding Conventions

- Follow LLVM/MLIR naming conventions (CamelCase for types, operations)
- Use LLVM-style file headers with description comments
- All Forth operations should document their stack effects in the description using the notation: `( input -- output )`
- C++17 standard is required
- Use `clang-format` for code formatting (config in `.clang-format`)
