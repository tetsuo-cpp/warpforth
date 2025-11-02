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
   - Stack manipulation ops: `dup`, `drop`, `swap`, `over`, `rot`
   - Arithmetic ops: `add`, `sub`, `mul`, `div`, `mod`
   - All operations take `Forth_StackType` as input and produce `Forth_StackType` as output

3. **Tools**:
   - `warpforth-translate`: MLIR translation tool that registers the Forth dialect alongside standard MLIR dialects

### Directory Layout

```
include/warpforth/Dialect/Forth/  # Public headers and TableGen files
  ├── ForthDialect.td              # Dialect and type definitions
  ├── ForthOps.td                  # Operation definitions
  └── ForthDialect.h               # Main C++ header

lib/Dialect/Forth/                 # Implementation
  └── ForthDialect.cpp             # Dialect initialization

tools/warpforth-translate/         # Translation tool
  └── warpforth-translate.cpp      # Main entry point
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

## Coding Conventions

- Follow LLVM/MLIR naming conventions (CamelCase for types, operations)
- Use LLVM-style file headers with description comments
- All Forth operations should document their stack effects in the description using the notation: `( input -- output )`
- C++17 standard is required
- Use `clang-format` for code formatting (config in `.clang-format`)
