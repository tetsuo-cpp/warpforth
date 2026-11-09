//===- ForthToMemRef.h - Forth to MemRef conversion ------------*- C++ -*-===//
//
// This file declares the pass for converting Forth dialect operations to
// MemRef dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_CONVERSION_FORTHTOMEMREF_FORTHTOMEMREF_H
#define WARPFORTH_CONVERSION_FORTHTOMEMREF_FORTHTOMEMREF_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace warpforth {

/// Discardable attribute recording the data-stack depth before an op.
inline constexpr const char kForthKnownDepthAttr[] = "known_depth";

/// Tags Forth ops whose data-stack depth is known at compile time.
std::unique_ptr<Pass> createTagForthStackDepthPass();

/// Creates a pass that converts Forth dialect operations to MemRef dialect.
/// This pass lowers the abstract Forth stack type to a concrete memref<256xi64>
/// representation with an explicit stack pointer (index type) threaded through
/// operations.
std::unique_ptr<Pass> createConvertForthToMemRefPass();

} // namespace warpforth
} // namespace mlir

#endif
