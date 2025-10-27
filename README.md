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
       ↓ (warpforth-translate)
  FORTH Dialect (warpforth)
       ↓ (warpforth-opt --lower-forth-to-gpu)
  GPU Dialect (MLIR)
       ↓
  Target Backend (CUDA/ROCm/Vulkan/etc.)
```

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

3. **Verify build**:
   ```bash
   ./bin/warpforth-opt --help
   ```

## Usage

### WarpForth Translation Tool

The `warpforth-translate` tool converts FORTH source code to MLIR FORTH dialect:

```bash
# Translate a FORTH source file to MLIR
./bin/warpforth-translate input.forth -o output.mlir

# Translate from stdin
echo '5 10 +' | ./bin/warpforth-translate

# Full pipeline: FORTH source → MLIR → Lowered
./bin/warpforth-translate input.forth | ./bin/warpforth-opt --lower-forth-to-gpu
```

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

### Example FORTH Source Code

Here's a simple FORTH program:

```forth
\ Simple arithmetic
5 10 + 3 *

\ Stack operations
42 DUP *
```

This translates to MLIR FORTH dialect and can be further lowered.
