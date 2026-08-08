#ifndef WARPFORTH_CONVERSION_PASSES_H
#define WARPFORTH_CONVERSION_PASSES_H

//===- Passes.h - Conversion Pass Registration -----------------*- C++ -*-===//
//
// This file declares the registration functions for conversion passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class OpPassManager;

namespace warpforth {

/// Register all conversion passes.
void registerConversionPasses();

/// Build the WarpForth compilation pipeline (Forth to PTX).
void buildWarpForthPipeline(OpPassManager &pm);

} // namespace warpforth
} // namespace mlir

#endif
