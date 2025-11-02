//===- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// This file contains definitions for conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_CONVERSION_PASSES
#define WARPFORTH_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertForthToMemRef
    : Pass<"convert-forth-to-memref", "mlir::func::FuncOp"> {
  let summary = "Convert Forth dialect to MemRef dialect";
  let description = [{
    This pass converts Forth dialect operations to MemRef dialect operations.

    The abstract `!forth.stack` type is lowered to a concrete representation:
    - Stack buffer: `memref<256xi64>` (fixed-size 256-element array)
    - Stack pointer: `index` type (current top of stack position)

    Forth operations are converted to corresponding memref load/store operations
    with explicit stack pointer arithmetic.
  }];

  let constructor = "mlir::warpforth::createConvertForthToMemRefPass()";

  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect"];
}

#endif // WARPFORTH_CONVERSION_PASSES
