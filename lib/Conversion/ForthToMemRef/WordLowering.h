#ifndef WARPFORTH_CONVERSION_FORTHTOMEMREF_WORDLOWERING_H
#define WARPFORTH_CONVERSION_FORTHTOMEMREF_WORDLOWERING_H

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpBuilder;
class Operation;
class Value;

namespace warpforth {

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
