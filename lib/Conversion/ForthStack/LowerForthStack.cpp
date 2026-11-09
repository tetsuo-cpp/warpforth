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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionGraphTraits.h" // IWYU pragma: keep
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <optional>
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
  bool operator==(const StaticShape &other) const {
    return cells == other.cells;
  }
};
struct ResidualShape {
  unsigned cells;
  bool operator==(const ResidualShape &other) const {
    return cells == other.cells;
  }
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

/// The shape where stacks of shapes `a` and `b` meet. Equal shapes are kept.
/// Otherwise the join keeps in registers only the top cells that both sides
/// hold there, and the cells below move to memory.
StackShape join(StackShape a, StackShape b) {
  if (a == b)
    return a;
  return ResidualShape{std::min(registerCells(a), registerCells(b))};
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

/// Whether every op in `body` that takes a stack either has a row or branches.
bool hasRowsFor(Region &body) {
  WalkResult walk = body.walk([](Operation *op) {
    if (llvm::any_of(op->getOperandTypes(), isStack) &&
        !isa<BranchOpInterface>(op) && !stack::lookupStackWord(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !walk.wasInterrupted();
}

/// Joins the shapes that the predecessors seen so far pass to the stack
/// argument `arg`.
std::optional<StackShape>
joinIncoming(BlockArgument arg, const DenseMap<Value, StackShape> &shapes) {
  std::optional<StackShape> joined;
  Block *block = arg.getOwner();
  for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
    auto branch = cast<BranchOpInterface>((*it)->getTerminator());
    Value incoming =
        branch.getSuccessorOperands(it.getSuccessorIndex())[arg.getArgNumber()];
    auto shape = shapes.find(incoming);
    if (shape == shapes.end())
      continue;
    joined = joined ? join(*joined, shape->second) : shape->second;
  }
  return joined;
}

/// The shape of every !forth.stack value in `body`, iterated to a fixpoint in
/// reverse post-order. Requires that every block is reachable and that every
/// op that takes a stack has a row or branches.
///
/// Shapes only move from static to residual or toward fewer register cells, so
/// the iteration ends.
DenseMap<Value, StackShape> inferShapes(Region &body) {
  llvm::ReversePostOrderTraversal<Block *> order(&body.front());
  DenseMap<Value, StackShape> shapes;
  bool changed = true;
  while (changed) {
    changed = false;
    for (Block *block : order) {
      for (BlockArgument arg : block->getArguments()) {
        if (!isStack(arg.getType()))
          continue;
        std::optional<StackShape> joined = joinIncoming(arg, shapes);
        assert(joined && "reverse post-order visits a predecessor first");
        auto [it, inserted] = shapes.try_emplace(arg, *joined);
        if (inserted || !(it->second == *joined)) {
          it->second = *joined;
          changed = true;
        }
      }
      for (Operation &op : *block) {
        if (isa<forth::StackOp>(op)) {
          shapes[op.getResult(0)] = StaticShape{0};
          continue;
        }
        const StackWord *word = stack::lookupStackWord(&op);
        if (!word)
          continue;
        shapes[op.getResult(0)] =
            transfer(shapes.at(op.getOperand(0)), word->effect);
      }
    }
  }
  return shapes;
}

/// Rewrites one function. Rows reach the cells only through WordContext.
class StackSimulator final : public WordContext {
public:
  StackSimulator(func::FuncOp fn, DenseMap<Value, StackShape> shapes)
      : fn(fn), b(fn.getContext()), shapes(std::move(shapes)) {}

  /// Lowers every stack word in `fn`, then erases the Forth stack ops.
  void run();

  OpBuilder &builder() override { return b; }
  Value popped(unsigned i) override { return poppedCells[i]; }
  void push(Value cell) override { pushedCells.push_back(cell); }
  void replaceValueResult(Value value) override {
    current->getResult(1).replaceAllUsesWith(value);
  }

private:
  void expandArguments(Block &block);
  void lowerWord(Operation *op, const StackWord &word);
  void forwardCells(BranchOpInterface branch);

  func::FuncOp fn;
  OpBuilder b;
  DenseMap<Value, StackShape> shapes;
  /// Keyed by the original !forth.stack value.
  DenseMap<Value, StackState> states;
  SmallVector<Operation *> lowered;
  SmallVector<BlockArgument> replacedArguments;

  // Scratch for the op being lowered.
  Operation *current = nullptr;
  StackState working;
  SmallVector<Value, 4> poppedCells;
  SmallVector<Value, 4> pushedCells;
};

void StackSimulator::run() {
  Region &body = fn.getBody();
  for (Block &block : body)
    expandArguments(block);
  for (Block *block : llvm::ReversePostOrderTraversal<Block *>(&body.front())) {
    for (Operation &op : *block) {
      if (isa<forth::StackOp>(op)) {
        states[op.getResult(0)] = StaticState{};
        lowered.push_back(&op);
      } else if (const StackWord *word = stack::lookupStackWord(&op)) {
        lowerWord(&op, *word);
      } else if (auto branch = dyn_cast<BranchOpInterface>(op)) {
        forwardCells(branch);
      }
    }
  }
  for (Operation *op : lowered)
    op->dropAllReferences();
  for (Operation *op : lowered)
    op->erase();
  for (BlockArgument arg : replacedArguments)
    arg.getOwner()->eraseArgument(arg.getArgNumber());
}

/// Gives each stack argument of `block` one i64 argument per cell, placed
/// after it. The stack argument is erased once nothing uses it.
void StackSimulator::expandArguments(Block &block) {
  for (unsigned i = 0; i < block.getNumArguments(); ++i) {
    BlockArgument arg = block.getArgument(i);
    if (!isStack(arg.getType()))
      continue;
    StaticState state;
    for (unsigned cell = 0, e = registerCells(shapes.at(arg)); cell < e; ++cell)
      state.cells.push_back(
          block.insertArgument(i + 1 + cell, b.getI64Type(), arg.getLoc()));
    i += state.cells.size();
    states[arg] = std::move(state);
    replacedArguments.push_back(arg);
  }
}

void StackSimulator::lowerWord(Operation *op, const StackWord &word) {
  current = op;
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

/// Passes each forwarded stack to its successor as the stack's cells.
void StackSimulator::forwardCells(BranchOpInterface branch) {
  for (unsigned i = 0, e = branch->getNumSuccessors(); i < e; ++i) {
    SuccessorOperands operands = branch.getSuccessorOperands(i);
    if (llvm::none_of(operands.getForwardedOperands().getTypes(), isStack))
      continue;
    SmallVector<Value> forwarded;
    for (Value value : operands.getForwardedOperands()) {
      if (!isStack(value.getType())) {
        forwarded.push_back(value);
        continue;
      }
      StackState state = states.at(value);
      llvm::append_range(forwarded, cellsOf(state));
    }
    operands.getMutableForwardedOperands().assign(forwarded);
  }
}

struct LowerForthStackPass
    : public impl::LowerForthStackBase<LowerForthStackPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyWordsInlined(module)))
      return signalPassFailure();
    IRRewriter rewriter(&getContext());
    for (auto fn : module.getOps<func::FuncOp>()) {
      if (fn.isExternal())
        continue;
      (void)eraseUnreachableBlocks(rewriter, fn->getRegions());
      // A function this pass cannot lower yet is left whole for
      // convert-forth-to-memref.
      if (!hasRowsFor(fn.getBody()))
        continue;
      DenseMap<Value, StackShape> shapes = inferShapes(fn.getBody());
      if (llvm::any_of(shapes, [](const auto &entry) {
            return !std::holds_alternative<StaticShape>(entry.second);
          }))
        continue;
      StackSimulator(fn, std::move(shapes)).run();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerForthStackPass() {
  return std::make_unique<LowerForthStackPass>();
}

} // namespace warpforth
} // namespace mlir
