//===- Passes.h - Conversion Pass Registration -----------------*- C++ -*-===//
//
// This file declares the registration functions for conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_CONVERSION_PASSES_H
#define WARPFORTH_CONVERSION_PASSES_H

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/Support/CodeGen.h"
#include <memory>
#include <string>

namespace mlir {
class OpPassManager;

namespace warpforth {

/// Target configuration for the WarpForth compilation pipeline.
struct WarpForthPipelineOptions
    : public PassPipelineOptions<WarpForthPipelineOptions> {
  WarpForthPipelineOptions();

  PassOptions::Option<std::string> chip;
  PassOptions::Option<std::string> features;
  PassOptions::Option<std::string> libdevicePath;
  PassOptions::Option<llvm::CodeGenOptLevel> optLevel;
  PassOptions::Option<gpu::CompilationTarget> compilationTarget;
};

/// Register all conversion passes.
void registerConversionPasses();

/// Build the WarpForth compilation pipeline (Forth to PTX).
void buildWarpForthPipeline(OpPassManager &pm);
void buildWarpForthPipeline(OpPassManager &pm,
                            const WarpForthPipelineOptions &options);

} // namespace warpforth
} // namespace mlir

#endif
