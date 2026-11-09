//===- LowerForthStack.h - Lower the Forth stack ----------------*- C++ -*-===//
//
// This file declares the pass that lowers the Forth data stack.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_CONVERSION_FORTHSTACK_LOWERFORTHSTACK_H
#define WARPFORTH_CONVERSION_FORTHSTACK_LOWERFORTHSTACK_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace warpforth {

/// Creates the pass that lowers every !forth.stack value. Every user word must
/// already be inlined. A cell whose depth and slot are known at compile time
/// becomes an i64 SSA value.
std::unique_ptr<Pass> createLowerForthStackPass();

} // namespace warpforth
} // namespace mlir

#endif
