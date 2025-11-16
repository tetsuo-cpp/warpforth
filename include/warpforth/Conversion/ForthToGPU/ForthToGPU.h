//===- ForthToGPU.h - Forth to GPU conversion pass -------------*- C++ -*-===//
//
// This file declares the pass for converting Forth dialect to GPU dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace warpforth {

/// Creates a pass that converts Forth functions to GPU dialect.
/// This pass wraps func.func operations in gpu.module and converts them to
/// gpu.func operations. Functions named "main" receive the gpu.kernel
/// attribute.
std::unique_ptr<Pass> createConvertForthToGPUPass();

} // namespace warpforth
} // namespace mlir
