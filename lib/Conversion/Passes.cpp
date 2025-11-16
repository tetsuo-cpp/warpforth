//===- Passes.cpp - Conversion Pass Registration ----------------*- C++ -*-===//
//
// This file implements pass registration for conversion passes.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "warpforth/Conversion/ForthToGPU/ForthToGPU.h"
#include "warpforth/Conversion/ForthToMemRef/ForthToMemRef.h"

namespace mlir {
namespace warpforth {

void registerConversionPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return createConvertForthToMemRefPass();
  });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createConvertForthToGPUPass(); });

  // Register WarpForth pipeline
  PassPipelineRegistration<>(
      "warpforth-pipeline", "Complete WarpForth compilation pipeline",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createConvertForthToMemRefPass());
        pm.addPass(createConvertForthToGPUPass());
      });
}

} // namespace warpforth
} // namespace mlir
