//===- ForthToMemRef.h - Forth to MemRef conversion ------------*- C++ -*-===//
//
// This file declares the pass for converting Forth dialect operations to
// MemRef dialect operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace warpforth {

/// Creates a pass that converts Forth dialect operations to MemRef dialect.
/// This pass lowers the abstract Forth stack type to a concrete memref<256xi64>
/// representation with an explicit stack pointer (index type) threaded through
/// operations.
std::unique_ptr<Pass> createConvertForthToMemRefPass();

} // namespace warpforth
} // namespace mlir
