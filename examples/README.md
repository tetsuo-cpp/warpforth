# WarpForth Example Programs

This directory contains example Forth programs that can be translated to MLIR using the `warpforth-translate` tool.

## Example Files

### simple_arithmetic.forth
Demonstrates basic arithmetic operations: `(5 + 3) * 2 = 16`

### stack_operations.forth
Shows stack manipulation operations: `DUP`, `SWAP`, and `DROP`

### complex_expression.forth
Complex arithmetic expression: `((100 - 20) / 4) + (5 * 3) = 35`

### all_operations.forth
Comprehensive test of all supported Forth operations including negative numbers

## Usage

After building the project (see main README.md), translate a Forth program to MLIR:

```bash
./bin/warpforth-translate examples/simple_arithmetic.forth
```

To save the output to a file:

```bash
./bin/warpforth-translate examples/simple_arithmetic.forth -o output.mlir
```

To process the translated MLIR further (e.g., lower to GPU dialect):

```bash
./bin/warpforth-translate examples/simple_arithmetic.forth | ./bin/warpforth-opt --lower-forth-to-gpu
```

## Supported Forth Operations

- **Arithmetic**: `+`, `-`, `*`, `/`
- **Stack operations**: `DUP`, `SWAP`, `DROP`
- **Constants**: Integer literals (positive and negative)
- **Comments**:
  - Line comments with `\`
  - Parenthetical comments with `( )`

## Forth Syntax

Forth is case-insensitive for words. The following are equivalent:
- `dup`, `DUP`, `Dup`
- `swap`, `SWAP`, `Swap`

Numbers can be positive or negative:
- `42`, `-10`, `0`
