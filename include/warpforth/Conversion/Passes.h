//===- Passes.h - Conversion Pass Registration -----------------*- C++ -*-===//
//
// This file declares the registration functions for conversion passes.
//
//===----------------------------------------------------------------------===//

#pragma once

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
