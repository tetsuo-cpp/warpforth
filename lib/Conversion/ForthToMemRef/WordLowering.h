//===- WordLowering.h - Forth word compute table ---------------*- C++ -*-===//
//
// Fixed-window Forth words. Patterns and the known-stack walk both call
// compute, so each word's arith or memory op is built in one place.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_CONVERSION_FORTHTOMEMREF_WORDLOWERING_H
#define WARPFORTH_CONVERSION_FORTHTOMEMREF_WORDLOWERING_H

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpBuilder;
class Operation;
class Value;

namespace warpforth {

/// inputs[0] is the deeper cell and inputs.back() is the top. compute appends
/// `pushes` cells, deepest first, and does not read or write the memref stack.
struct WordLowering {
  unsigned pops;
  unsigned pushes;
  void (*compute)(Operation *op, ValueRange inputs,
                  SmallVectorImpl<Value> &outputs, OpBuilder &builder);
};

const WordLowering *lookupWord(Operation *op);

} // namespace warpforth
} // namespace mlir

#endif
