//===- LowerForthStack.cpp - Lower the Forth stack --------------*- C++ -*-===//
//
// This file implements the pass that lowers every !forth.stack value. The
// stack-word table says what each word computes. This file decides where each
// cell lives: a cell whose depth and slot are known is an i64 SSA value, and
// the rest live in a per-thread buffer laid out like the memref stack.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthStack/LowerForthStack.h"
#include "StackWords.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h" // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

/// Cells in the residual buffer. Index 0 is never written by a push, so an
/// empty buffer has depth 0.
constexpr int64_t kResidualCells = 256;

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

/// The block arguments that carry a stack of `shape`: one i64 per register
/// cell, deepest first, then the memory depth of a residual stack.
SmallVector<Type> argumentTypes(StackShape shape, Builder &b) {
  SmallVector<Type> types(registerCells(shape), b.getI64Type());
  if (std::holds_alternative<ResidualShape>(shape))
    types.push_back(b.getIndexType());
  return types;
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

ArrayRef<Value> cellsOf(const StackState &state) {
  return std::visit([](const auto &s) -> ArrayRef<Value> { return s.cells; },
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
      : fn(fn), b(fn.getContext()), loc(fn.getLoc()),
        shapes(std::move(shapes)) {}

  /// Lowers every stack word in `fn`, then erases the Forth stack ops.
  void run();

  OpBuilder &builder() override { return b; }
  Value popped(unsigned i) override { return poppedCells[i]; }
  void push(Value cell) override { pushedCells.push_back(cell); }
  Value peek(Value n) override;
  void rollToTop(Value n) override;
  void replaceValueResult(Value value) override {
    current->getResult(1).replaceAllUsesWith(value);
  }

private:
  void expandArguments(Block &block);
  void lowerWord(Operation *op, const StackWord &word);
  Value popCell();
  void forwardCells(BranchOpInterface branch);
  void appendEdgeOperands(const StackState &state, StackShape target,
                          SmallVectorImpl<Value> &operands);

  Value residualBuffer();
  Value memoryDepth(const StackState &state);
  Value storeCells(ArrayRef<Value> cells, Value depth);
  Value indexConstant(int64_t value);
  static std::optional<unsigned> registerSlot(Value n, size_t cells);

  func::FuncOp fn;
  OpBuilder b;
  /// Location of the op or branch being lowered.
  Location loc;
  DenseMap<Value, StackShape> shapes;
  /// Keyed by the original !forth.stack value.
  DenseMap<Value, StackState> states;
  /// Each block's arguments before stack arguments were expanded.
  DenseMap<Block *, SmallVector<BlockArgument>> originalArguments;
  SmallVector<Operation *> lowered;
  SmallVector<BlockArgument> replacedArguments;
  /// Allocated on first use, so a function whose cells all have static slots
  /// has no buffer.
  Value buffer;

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
    // A dynamic ROLL moves the ops after it into a new block, so follow each
    // op's next op rather than iterating the block.
    Operation *op = &block->front();
    while (op) {
      Operation *next = op->getNextNode();
      if (isa<forth::StackOp>(op)) {
        states[op->getResult(0)] = StaticState{};
        lowered.push_back(op);
      } else if (const StackWord *word = stack::lookupStackWord(op)) {
        lowerWord(op, *word);
      } else if (auto branch = dyn_cast<BranchOpInterface>(op)) {
        forwardCells(branch);
      }
      op = next;
    }
  }
  for (Operation *op : lowered)
    op->dropAllReferences();
  for (Operation *op : lowered)
    op->erase();
  for (BlockArgument arg : replacedArguments)
    arg.getOwner()->eraseArgument(arg.getArgNumber());
}

/// Gives each stack argument of `block` the arguments its shape needs, placed
/// after it. The stack argument is erased once nothing uses it.
void StackSimulator::expandArguments(Block &block) {
  SmallVector<BlockArgument> &original = originalArguments[&block];
  original = llvm::to_vector(block.getArguments());
  for (BlockArgument arg : original) {
    if (!isStack(arg.getType()))
      continue;
    StackShape shape = shapes.at(arg);
    unsigned position = arg.getArgNumber();
    SmallVector<Value, 8> values;
    for (Type type : argumentTypes(shape, b))
      values.push_back(block.insertArgument(++position, type, arg.getLoc()));
    if (std::holds_alternative<StaticShape>(shape)) {
      states[arg] = StaticState{std::move(values)};
    } else {
      Value depth = values.pop_back_val();
      states[arg] = ResidualState{std::move(values), depth};
    }
    replacedArguments.push_back(arg);
  }
}

void StackSimulator::lowerWord(Operation *op, const StackWord &word) {
  current = op;
  loc = op->getLoc();
  b.setInsertionPoint(op);
  working = states.at(op->getOperand(0));
  poppedCells.clear();
  for (unsigned i = 0; i < word.effect.pops; ++i)
    poppedCells.push_back(popCell());
  pushedCells.clear();
  word.lower(op, *this);
  assert(pushedCells.size() == word.effect.pushes &&
         "a row pushes exactly the cells its effect declares");
  llvm::append_range(cellsOf(working), pushedCells);
  states[op->getResult(0)] = std::move(working);
  lowered.push_back(op);
}

/// Pops the top cell of the working stack. Below the register cells the cell
/// is loaded from memory. A static stack has nothing below its cells, so it
/// widens to an empty residual stack and the load reads index 0, as the memref
/// stack did.
Value StackSimulator::popCell() {
  SmallVector<Value, 8> &cells = cellsOf(working);
  if (!cells.empty())
    return cells.pop_back_val();
  if (std::holds_alternative<StaticState>(working))
    working = ResidualState{{}, indexConstant(0)};
  auto &residual = std::get<ResidualState>(working);
  Value cell = b.create<memref::LoadOp>(loc, residualBuffer(), residual.depth);
  residual.depth =
      b.create<arith::SubIOp>(loc, residual.depth, indexConstant(1));
  return cell;
}

/// The register slot that the constant `n` names, counted from the top of
/// `cells` register cells, or none when `n` is not such a constant.
std::optional<unsigned> StackSimulator::registerSlot(Value n, size_t cells) {
  std::optional<int64_t> slot = getConstantIntValue(n);
  if (!slot || *slot < 0 || static_cast<uint64_t>(*slot) >= cells)
    return std::nullopt;
  return static_cast<unsigned>(cells - 1 - static_cast<size_t>(*slot));
}

Value StackSimulator::peek(Value n) {
  SmallVector<Value, 8> &cells = cellsOf(working);
  if (std::optional<unsigned> slot = registerSlot(n, cells.size()))
    return cells[*slot];
  // The register cells are written only as the image this load addresses, and
  // they stay the live stack.
  Value top = storeCells(cells, memoryDepth(working));
  Value offset = b.create<arith::IndexCastOp>(loc, b.getIndexType(), n);
  Value index = b.create<arith::SubIOp>(loc, top, offset);
  return b.create<memref::LoadOp>(loc, residualBuffer(), index);
}

void StackSimulator::rollToTop(Value n) {
  SmallVector<Value, 8> &cells = cellsOf(working);
  if (std::optional<unsigned> slot = registerSlot(n, cells.size())) {
    std::rotate(cells.begin() + *slot, cells.begin() + *slot + 1, cells.end());
    return;
  }
  Value depth = memoryDepth(working);
  Value top = storeCells(cells, depth);
  Value offset = b.create<arith::IndexCastOp>(loc, b.getIndexType(), n);
  Value rolledIndex = b.create<arith::SubIOp>(loc, top, offset);
  Value rolled = b.create<memref::LoadOp>(loc, residualBuffer(), rolledIndex);

  // for (iv = rolledIndex; iv < top; ++iv) buffer[iv] = buffer[iv + 1]
  Block *before = b.getInsertionBlock();
  Block *after = before->splitBlock(b.getInsertionPoint());
  Type indexType = b.getIndexType();
  Block *header = b.createBlock(after, indexType, loc);
  Block *body = b.createBlock(after, indexType, loc);

  b.setInsertionPointToEnd(before);
  b.create<cf::BranchOp>(loc, header, rolledIndex);

  b.setInsertionPointToStart(header);
  Value iv = header->getArgument(0);
  Value shifting =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iv, top);
  b.create<cf::CondBranchOp>(loc, shifting, body, iv, after, ValueRange{});

  b.setInsertionPointToStart(body);
  Value slot = body->getArgument(0);
  Value next = b.create<arith::AddIOp>(loc, slot, indexConstant(1));
  Value shifted = b.create<memref::LoadOp>(loc, residualBuffer(), next);
  b.create<memref::StoreOp>(loc, shifted, residualBuffer(), slot);
  b.create<cf::BranchOp>(loc, header, next);

  b.setInsertionPointToStart(after);
  b.create<memref::StoreOp>(loc, rolled, residualBuffer(), top);
  Value index = depth;
  for (Value &cell : cells) {
    index = b.create<arith::AddIOp>(loc, index, indexConstant(1));
    cell = b.create<memref::LoadOp>(loc, residualBuffer(), index);
  }
}

/// Passes each forwarded stack to its successor in the successor's shape.
void StackSimulator::forwardCells(BranchOpInterface branch) {
  loc = branch->getLoc();
  b.setInsertionPoint(branch);
  for (unsigned i = 0, e = branch->getNumSuccessors(); i < e; ++i) {
    SuccessorOperands operands = branch.getSuccessorOperands(i);
    if (llvm::none_of(operands.getForwardedOperands().getTypes(), isStack))
      continue;
    ArrayRef<BlockArgument> targets =
        originalArguments.find(branch->getSuccessor(i))->second;
    unsigned first = operands.getProducedOperandCount();
    SmallVector<Value> forwarded;
    for (auto [j, value] : llvm::enumerate(operands.getForwardedOperands())) {
      if (!isStack(value.getType())) {
        forwarded.push_back(value);
        continue;
      }
      appendEdgeOperands(states.at(value), shapes.at(targets[first + j]),
                         forwarded);
    }
    operands.getMutableForwardedOperands().assign(forwarded);
  }
}

/// Appends the operands that pass a stack in `state` to an argument of shape
/// `target`. The cells below the target's register cells are stored to memory
/// first. Every edge into a join stores a cell at the same index, so the
/// stores agree whichever edge runs.
void StackSimulator::appendEdgeOperands(const StackState &state,
                                        StackShape target,
                                        SmallVectorImpl<Value> &operands) {
  ArrayRef<Value> cells = cellsOf(state);
  unsigned kept = registerCells(target);
  assert(kept <= cells.size() && "a join keeps cells that every edge holds");
  if (std::holds_alternative<StaticShape>(target)) {
    assert(std::holds_alternative<StaticState>(state) && kept == cells.size() &&
           "a static join has equal static edges");
    llvm::append_range(operands, cells);
    return;
  }
  Value depth = storeCells(cells.drop_back(kept), memoryDepth(state));
  llvm::append_range(operands, cells.take_back(kept));
  operands.push_back(depth);
}

Value StackSimulator::residualBuffer() {
  if (!buffer) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&fn.getBody().front());
    buffer = b.create<memref::AllocaOp>(
        loc, MemRefType::get({kResidualCells}, b.getI64Type()));
  }
  return buffer;
}

/// The index of the top memory cell under `state`'s register cells.
Value StackSimulator::memoryDepth(const StackState &state) {
  if (const auto *residual = std::get_if<ResidualState>(&state))
    return residual->depth;
  return indexConstant(0);
}

/// Pushes `cells` to memory above `depth`, deepest first, and returns the new
/// depth.
Value StackSimulator::storeCells(ArrayRef<Value> cells, Value depth) {
  for (Value cell : cells) {
    depth = b.create<arith::AddIOp>(loc, depth, indexConstant(1));
    b.create<memref::StoreOp>(loc, cell, residualBuffer(), depth);
  }
  return depth;
}

Value StackSimulator::indexConstant(int64_t value) {
  return b.create<arith::ConstantIndexOp>(loc, value);
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
      // A function that uses a word with no row yet is left whole for
      // convert-forth-to-memref.
      if (!hasRowsFor(fn.getBody()))
        continue;
      StackSimulator(fn, inferShapes(fn.getBody())).run();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerForthStackPass() {
  return std::make_unique<LowerForthStackPass>();
}

} // namespace warpforth
} // namespace mlir
