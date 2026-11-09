//===- StackWords.h - Stack effects of Forth words --------------*- C++ -*-===//
//
// This file declares the stack-word table. Each row gives one Forth op its
// stack effect and lowers the op's compute on i64 cells. Where the cells live
// is decided by LowerForthStack.cpp alone.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_LIB_CONVERSION_FORTHSTACK_STACKWORDS_H
#define WARPFORTH_LIB_CONVERSION_FORTHSTACK_STACKWORDS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <functional>

namespace mlir {
namespace warpforth {
namespace stack {

/// How many cells a word pops, then pushes. The table is the only place that
/// declares a stack effect.
struct StackEffect {
  unsigned pops = 0;
  unsigned pushes = 0;
};

/// What a row may do while it lowers one op. Every cell is an i64 value, and a
/// row never learns where a cell lives.
class WordContext {
public:
  virtual ~WordContext();

  /// Positioned immediately before the op being lowered.
  virtual OpBuilder &builder() = 0;

  /// Popped cell `i`, where 0 is the old top. Requires `i < effect.pops`.
  virtual Value popped(unsigned i) = 0;

  /// Pushes one cell. A row pushes exactly `effect.pushes` cells, deepest
  /// first.
  virtual void push(Value cell) = 0;

  /// The cell `n` places below the top of the stack left after the pops,
  /// where 0 is the top. `n` is an i64 cell and need not be a constant.
  virtual Value peek(Value n) = 0;

  /// Moves the cell `n` places below the top of the stack left after the pops
  /// to the top, shifting the cells above it down by one.
  virtual void rollToTop(Value n) = 0;

  /// Replaces every use of the op's non-stack result, such as the i1 of
  /// forth.pop_flag, with `value`.
  virtual void replaceValueResult(Value value) = 0;
};

using LowerFn = std::function<void(Operation *op, WordContext &ctx)>;

/// One row of the stack-word table.
struct StackWord {
  StackEffect effect;
  LowerFn lower;
};

/// The row for `op`, or null when the table has none.
const StackWord *lookupStackWord(Operation *op);

/// Fails, with an error for each one, when a Forth op that takes a
/// !forth.stack has no row. The Forth dialect must be loaded.
LogicalResult verifyStackWordTableCoversDialect(MLIRContext *context);

} // namespace stack
} // namespace warpforth
} // namespace mlir

#endif
