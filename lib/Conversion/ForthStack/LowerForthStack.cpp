//===- LowerForthStack.cpp - Lower the Forth stack --------------*- C++ -*-===//
//
// This file implements the pass that lowers every !forth.stack value. The
// stack-word table says what each word computes. This file decides where each
// cell lives: a cell whose depth and slot are known is an i64 SSA value.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthStack/LowerForthStack.h"
#include "StackWords.h"
#include "mlir/Dialect/Arith/IR/Arith.h" // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h" // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"   // IWYU pragma: keep
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <variant>

namespace mlir {
namespace warpforth {

#define GEN_PASS_DEF_LOWERFORTHSTACK
#include "warpforth/Conversion/Passes.h.inc"

namespace {

using stack::StackEffect;
using stack::StackWord;
using stack::WordContext;

bool isStack(Type type) { return isa<forth::StackType>(type); }

/// Compile-time shape of one !forth.stack value. A static stack holds exactly
/// `cells` cells, all in registers. A residual stack keeps its top `cells`
/// cells in registers above an unknown number of cells in memory.
struct StaticShape {
  unsigned cells;
};
struct ResidualShape {
  unsigned cells;
};
using StackShape = std::variant<StaticShape, ResidualShape>;

unsigned registerCells(StackShape shape) {
  return std::visit([](auto s) { return s.cells; }, shape);
}

/// The shape after an op with `effect`. A static stack that cannot supply the
/// pops reads below its known cells, so the result is residual.
StackShape transfer(StackShape in, StackEffect effect) {
  unsigned cells = registerCells(in);
  if (std::holds_alternative<StaticShape>(in) && effect.pops <= cells)
    return StaticShape{cells - effect.pops + effect.pushes};
  unsigned kept = cells > effect.pops ? cells - effect.pops : 0;
  return ResidualShape{kept + effect.pushes};
}

/// Where the cells of one !forth.stack value live while the pass rewrites it.
/// `cells` are i64 values, deepest first. A residual stack also holds `depth`,
/// the index of its top cell in memory.
struct StaticState {
  SmallVector<Value, 8> cells;
};
struct ResidualState {
  SmallVector<Value, 8> cells;
  Value depth;
};
using StackState = std::variant<StaticState, ResidualState>;

SmallVector<Value, 8> &cellsOf(StackState &state) {
  return std::visit([](auto &s) -> SmallVector<Value, 8> & { return s.cells; },
                    state);
}

/// A word keeps the stack in its signature until the inliner removes it, and
/// this pass gives words no calling convention.
LogicalResult verifyWordsInlined(ModuleOp module) {
  for (auto fn : module.getOps<func::FuncOp>()) {
    FunctionType type = fn.getFunctionType();
    if (llvm::any_of(type.getInputs(), isStack) ||
        llvm::any_of(type.getResults(), isStack))
      return fn.emitError()
             << "Forth word @" << fn.getSymName()
             << " was not inlined; run --inline before --lower-forth-stack";
  }
  return success();
}

/// Whether this pass lowers `fn` yet: it is straight-line code, every stack op
/// has a row, and no pop reaches below the static depth.
bool isLowerable(func::FuncOp fn) {
  if (!fn.getBody().hasOneBlock())
    return false;
  DenseMap<Value, StackShape> shapes;
  WalkResult walk = fn.walk([&](Operation *op) {
    if (isa<forth::StackOp>(op)) {
      shapes.try_emplace(op->getResult(0), StaticShape{0});
      return WalkResult::advance();
    }
    if (llvm::none_of(op->getOperandTypes(), isStack))
      return WalkResult::advance();
    const StackWord *word = stack::lookupStackWord(op);
    if (!word)
      return WalkResult::interrupt();
    StackShape shape = transfer(shapes.at(op->getOperand(0)), word->effect);
    if (!std::holds_alternative<StaticShape>(shape))
      return WalkResult::interrupt();
    shapes.try_emplace(op->getResult(0), shape);
    return WalkResult::advance();
  });
  return !walk.wasInterrupted();
}

/// Rewrites one function. Rows reach the cells only through WordContext.
class StackSimulator final : public WordContext {
public:
  explicit StackSimulator(func::FuncOp fn) : fn(fn), b(fn.getContext()) {}

  /// Lowers every stack word in `fn`, then erases the Forth stack ops.
  void run();

  OpBuilder &builder() override { return b; }
  Value popped(unsigned i) override { return poppedCells[i]; }
  void push(Value cell) override { pushedCells.push_back(cell); }

private:
  void lowerWord(Operation *op, const StackWord &word);

  func::FuncOp fn;
  OpBuilder b;
  /// Keyed by the original !forth.stack value.
  DenseMap<Value, StackState> states;
  SmallVector<Operation *> lowered;

  // Scratch for the op being lowered.
  StackState working;
  SmallVector<Value, 4> poppedCells;
  SmallVector<Value, 4> pushedCells;
};

void StackSimulator::run() {
  for (Operation &op : fn.getBody().front()) {
    if (isa<forth::StackOp>(op)) {
      states[op.getResult(0)] = StaticState{};
      lowered.push_back(&op);
    } else if (const StackWord *word = stack::lookupStackWord(&op)) {
      lowerWord(&op, *word);
    }
  }
  for (Operation *op : lowered)
    op->dropAllReferences();
  for (Operation *op : lowered)
    op->erase();
}

void StackSimulator::lowerWord(Operation *op, const StackWord &word) {
  working = states.at(op->getOperand(0));
  SmallVector<Value, 8> &cells = cellsOf(working);
  poppedCells.clear();
  for (unsigned i = 0; i < word.effect.pops; ++i)
    poppedCells.push_back(cells.pop_back_val());
  pushedCells.clear();
  b.setInsertionPoint(op);
  word.lower(op, *this);
  assert(pushedCells.size() == word.effect.pushes &&
         "a row pushes exactly the cells its effect declares");
  llvm::append_range(cellsOf(working), pushedCells);
  states[op->getResult(0)] = std::move(working);
  lowered.push_back(op);
}

struct LowerForthStackPass
    : public impl::LowerForthStackBase<LowerForthStackPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyWordsInlined(module)))
      return signalPassFailure();
    for (auto fn : module.getOps<func::FuncOp>())
      if (!fn.isExternal() && isLowerable(fn))
        StackSimulator(fn).run();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerForthStackPass() {
  return std::make_unique<LowerForthStackPass>();
}

} // namespace warpforth
} // namespace mlir
