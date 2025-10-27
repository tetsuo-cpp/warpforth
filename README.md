# WarpForth

A FORTH dialect for programming GPU kernels built on top of MLIR (Multi-Level Intermediate Representation).

## Overview

WarpForth is a stack-based programming language dialect inspired by FORTH, specifically designed for GPU kernel development. It leverages MLIR's infrastructure to provide a high-level abstraction for GPU programming that can be lowered to various GPU backends (CUDA, ROCm, Vulkan, etc.) through MLIR's GPU dialect.

## Features

- **Stack-based operations**: Classic FORTH operations (DUP, DROP, SWAP, arithmetic operations)
- **MLIR integration**: Seamless integration with MLIR's dialect ecosystem
- **GPU targeting**: Lower to MLIR's GPU dialect for cross-platform GPU execution
- **Extensible**: Easy to add new operations and transformations

## Architecture

```
FORTH Source Code
       ↓
  FORTH Dialect (warpforth)
       ↓
  GPU Dialect (MLIR)
       ↓
  Target Backend (CUDA/ROCm/Vulkan/etc.)
```

### Dialect Operations

The Forth dialect provides the following operations:

- **forth.constant**: Push constant values onto the stack
- **forth.add**: Add two values (pops two, pushes result)
- **forth.sub**: Subtract two values
- **forth.mul**: Multiply two values
- **forth.div**: Divide two values
- **forth.dup**: Duplicate top stack value
- **forth.drop**: Remove top stack value
- **forth.swap**: Swap top two stack values

## Building

### Prerequisites

- CMake 3.20 or higher
- C++17 compatible compiler
- LLVM/MLIR installation (version 17 or higher recommended)

### Build Instructions

1. **Set MLIR/LLVM path** (if not in standard location):
   ```bash
   export LLVM_DIR=/path/to/llvm/install/lib/cmake/llvm
   export MLIR_DIR=/path/to/llvm/install/lib/cmake/mlir
   ```

2. **Configure and build**:
   ```bash
   mkdir build && cd build
   cmake -G Ninja .. \
     -DMLIR_DIR=$MLIR_DIR \
     -DLLVM_DIR=$LLVM_DIR \
     -DCMAKE_BUILD_TYPE=Release
   cmake --build .
   ```

   Alternatively, using Make:
   ```bash
   mkdir build && cd build
   cmake .. \
     -DMLIR_DIR=$MLIR_DIR \
     -DLLVM_DIR=$LLVM_DIR \
     -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

3. **Verify build**:
   ```bash
   ./bin/warpforth-opt --help
   ```

## Usage

### WarpForth Optimizer Tool

The `warpforth-opt` tool is used to parse, transform, and lower FORTH dialect operations:

```bash
# Parse and verify a FORTH dialect file
./bin/warpforth-opt input.mlir

# Lower FORTH operations to GPU dialect
./bin/warpforth-opt input.mlir --lower-forth-to-gpu

# Apply optimizations and print result
./bin/warpforth-opt input.mlir --lower-forth-to-gpu -o output.mlir
```

### Example MLIR Code

Here's a simple example of FORTH dialect operations in MLIR:

```mlir
module {
  func.func @simple_add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = forth.constant 10 : i32
    %1 = forth.add %arg0, %0 : i32
    %2 = forth.add %1, %arg1 : i32
    return %2 : i32
  }

  func.func @stack_operations(%arg0: i32) -> i32 {
    %0 = forth.dup %arg0 : i32
    %1 = forth.mul %0, %arg0 : i32
    return %1 : i32
  }
}
```

After lowering with `--lower-forth-to-gpu`, FORTH operations are converted to standard arithmetic operations that can be further lowered to GPU-specific code.

## Project Structure

```
warpforth/
├── CMakeLists.txt                    # Root build configuration
├── README.md                         # This file
├── cmake/
│   └── modules/
│       └── FindMLIR.cmake           # MLIR/LLVM detection logic
├── include/
│   └── warpforth/
│       ├── Dialect/
│       │   └── Forth/               # FORTH dialect headers
│       │       ├── ForthDialect.h
│       │       ├── ForthOps.h
│       │       └── ForthOps.td      # TableGen operation definitions
│       └── Conversion/
│           └── ForthToGPU/          # Lowering pass headers
│               └── ForthToGPU.h
├── lib/
│   ├── Dialect/
│   │   └── Forth/                   # FORTH dialect implementation
│   │       ├── ForthDialect.cpp
│   │       └── ForthOps.cpp
│   └── Conversion/
│       └── ForthToGPU/              # Lowering pass implementation
│           └── ForthToGPU.cpp
└── tools/
    └── warpforth-opt/               # Optimizer tool
        └── warpforth-opt.cpp
```

## Development

### Adding New Operations

1. Define the operation in `include/warpforth/Dialect/Forth/ForthOps.td`
2. Implement any custom logic in `lib/Dialect/Forth/ForthOps.cpp`
3. Add lowering patterns in `lib/Conversion/ForthToGPU/ForthToGPU.cpp`

### Testing Transformations

Use `warpforth-opt` to test your transformations:

```bash
echo 'func.func @test() { ... }' | ./bin/warpforth-opt --lower-forth-to-gpu
```

## Roadmap

- [ ] Add control flow operations (IF, WHILE, DO-LOOP)
- [ ] Implement memory operations (load, store)
- [ ] Add GPU-specific operations (thread indexing, barriers)
- [ ] Create FORTH source parser (currently using MLIR syntax)
- [ ] Add JIT execution support
- [ ] Implement optimizations (stack fusion, dead code elimination)
- [ ] Add example GPU kernels (matrix multiplication, reduction, etc.)

## Contributing

Contributions are welcome! This project is in early development, and we're open to:

- Bug reports and fixes
- New operations and transformations
- Documentation improvements
- Example programs and tutorials

## License

This project follows LLVM's licensing. See individual files for details.

## Acknowledgments

Built on top of:
- [LLVM](https://llvm.org/) - Compiler infrastructure
- [MLIR](https://mlir.llvm.org/) - Multi-Level IR framework

Inspired by the FORTH programming language and the need for high-level GPU programming abstractions.
