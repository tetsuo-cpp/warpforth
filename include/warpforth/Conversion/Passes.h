//===- Passes.h - Conversion Pass Registration -----------------*- C++ -*-===//
//
// This file declares the registration functions for conversion passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace warpforth {

/// Creates a pass that converts Forth dialect operations to MemRef dialect.
std::unique_ptr<Pass> createConvertForthToMemRefPass();

/// Register all conversion passes.
void registerConversionPasses();

} // namespace warpforth
} // namespace mlir
