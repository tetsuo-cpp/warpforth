//===- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// This file contains definitions for conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_CONVERSION_PASSES
#define WARPFORTH_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertForthToMemRef
    : Pass<"convert-forth-to-memref", "mlir::ModuleOp"> {
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
                           "mlir::arith::ArithDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::math::MathDialect"];
}

def LowerForthStack : Pass<"lower-forth-stack", "mlir::ModuleOp"> {
  let summary = "Lower the Forth stack to i64 SSA values";
  let description = [{
    Lowers every `!forth.stack` value. Every user word must already be
    inlined, so run `--inline` first. A cell whose depth and slot are known at
    compile time becomes an `i64` SSA value, and each word becomes the `arith`,
    `memref`, or `llvm` ops that compute its result.

    A cell below the known depth, or one that a runtime `PICK` or `ROLL` index
    addresses, lives in a per-thread `memref<256xi64>` with the memref stack's
    layout: index 0 is empty and a push stores at depth + 1.

    A function whose words this pass does not lower yet is left unchanged for
    `convert-forth-to-memref`.
  }];

  let constructor = "mlir::warpforth::createLowerForthStackPass()";

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::forth::ForthDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::math::MathDialect",
                           "mlir::memref::MemRefDialect"];
}

def ConvertForthToGPU : Pass<"convert-forth-to-gpu", "mlir::ModuleOp"> {
  let summary = "Convert Forth functions to GPU dialect";
  let description = [{
    This pass converts Forth functions to the GPU dialect to prepare them for
    GPU execution.

    The pass wraps func.func operations in a gpu.module and converts them to
    gpu.func operations. Functions named "main" receive the gpu.kernel
    attribute, making them launchable from host code.

    Additionally, the pass lowers forth.intrinsic operations to corresponding
    GPU dialect operations (gpu.thread_id, gpu.block_id, gpu.block_dim,
    gpu.grid_dim).
  }];

  let constructor = "mlir::warpforth::createConvertForthToGPUPass()";

  let dependentDialects = ["mlir::gpu::GPUDialect", "mlir::func::FuncDialect"];
}

#endif // WARPFORTH_CONVERSION_PASSES
