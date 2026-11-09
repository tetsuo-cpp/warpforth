//===- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// This file contains definitions for conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_CONVERSION_PASSES
#define WARPFORTH_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LowerForthStack : Pass<"lower-forth-stack", "mlir::ModuleOp"> {
  let summary = "Lower the Forth stack to i64 SSA values";
  let description = [{
    Lowers every `!forth.stack` value. Every user word must already be
    inlined, so run `--inline` first. A cell whose depth and slot are known at
    compile time becomes an `i64` SSA value, and each word becomes the `arith`,
    `memref`, or `llvm` ops that compute its result.

    A cell below the known depth, or one that a runtime `PICK` or `ROLL` index
    addresses, lives in a per-thread `memref<256xi64>`. Index 0 of that buffer
    is empty, and a push stores at depth + 1.
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

    The pass moves the kernel, the func.func marked `forth.kernel`, into a
    gpu.module as a gpu.func with the gpu.kernel attribute, making it
    launchable from host code. Every Forth word must already be inlined, so
    any other func.func is an error.

    Additionally, the pass lowers forth.intrinsic operations to corresponding
    GPU dialect operations (gpu.thread_id, gpu.block_id, gpu.block_dim,
    gpu.grid_dim).
  }];

  let constructor = "mlir::warpforth::createConvertForthToGPUPass()";

  let dependentDialects = ["mlir::gpu::GPUDialect", "mlir::func::FuncDialect"];
}

#endif // WARPFORTH_CONVERSION_PASSES
