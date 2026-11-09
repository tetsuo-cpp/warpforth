//===- TagForthStackDepth.cpp - Known Forth stack depth --------*- C++ -*-===//
//
// This file tags Forth ops with the data-stack depth before each op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "warpforth/Conversion/ForthToMemRef/ForthToMemRef.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace warpforth {

#define GEN_PASS_DEF_TAGFORTHSTACKDEPTH
#include "warpforth/Conversion/Passes.h.inc"

namespace {

struct StackEffect {
  unsigned pops;
  unsigned pushes;
};

static bool isForthStack(Type type) { return isa<forth::StackType>(type); }

static bool usesForthStack(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), isForthStack) ||
         llvm::any_of(op->getResultTypes(), isForthStack);
}

static unsigned countStackArgs(Block *block) {
  unsigned count = 0;
  for (BlockArgument arg : block->getArguments())
    if (isForthStack(arg.getType()))
      ++count;
  return count;
}

static bool hasStackSignature(func::FuncOp func) {
  return llvm::any_of(func.getArgumentTypes(), isForthStack) ||
         llvm::any_of(func.getResultTypes(), isForthStack);
}

static bool forwardsStack(OperandRange operands) {
  return llvm::any_of(
      operands, [](Value value) { return isForthStack(value.getType()); });
}

static std::optional<StackEffect> stackEffect(Operation *op) {
  if (isa<forth::DupOp>(op))
    return StackEffect{1, 2};
  if (isa<forth::DropOp, forth::PopFlagOp>(op))
    return StackEffect{1, 0};
  if (isa<forth::SwapOp>(op))
    return StackEffect{2, 2};
  if (isa<forth::OverOp, forth::TuckOp>(op))
    return StackEffect{2, 3};
  if (isa<forth::RotOp>(op))
    return StackEffect{3, 3};
  if (isa<forth::NipOp>(op))
    return StackEffect{2, 1};
  if (isa<forth::ConstantOp, forth::ParamRefOp, forth::ThreadIdXOp,
          forth::ThreadIdYOp, forth::ThreadIdZOp, forth::BlockIdXOp,
          forth::BlockIdYOp, forth::BlockIdZOp, forth::BlockDimXOp,
          forth::BlockDimYOp, forth::BlockDimZOp, forth::GridDimXOp,
          forth::GridDimYOp, forth::GridDimZOp, forth::GlobalIdOp,
          forth::LaneIdOp>(op))
    return StackEffect{0, 1};
  if (isa<forth::AddIOp, forth::SubIOp, forth::MulIOp, forth::DivIOp,
          forth::ModOp, forth::AddFOp, forth::SubFOp, forth::MulFOp,
          forth::DivFOp, forth::MaxFOp, forth::MinFOp, forth::AndOp,
          forth::OrOp, forth::XorOp, forth::LshiftOp, forth::RshiftOp,
          forth::EqIOp, forth::LtIOp, forth::GtIOp, forth::NeIOp, forth::LeIOp,
          forth::GeIOp, forth::EqFOp, forth::LtFOp, forth::GtFOp, forth::NeFOp,
          forth::LeFOp, forth::GeFOp, forth::ShuffleOp>(op))
    return StackEffect{2, 1};
  if (isa<forth::ExpFOp, forth::SqrtFOp, forth::LogFOp, forth::AbsFOp,
          forth::NegFOp, forth::NotOp, forth::ZeroEqOp, forth::IToFOp,
          forth::FToIOp, forth::LoadIOp, forth::LoadFOp, forth::SharedLoadIOp,
          forth::SharedLoadFOp, forth::LoadI8Op, forth::SharedLoadI8Op,
          forth::LoadI16Op, forth::SharedLoadI16Op, forth::LoadI32Op,
          forth::SharedLoadI32Op, forth::LoadF16Op, forth::SharedLoadF16Op,
          forth::LoadBF16Op, forth::SharedLoadBF16Op, forth::LoadF32Op,
          forth::SharedLoadF32Op>(op))
    return StackEffect{1, 1};
  if (isa<forth::StoreIOp, forth::StoreFOp, forth::SharedStoreIOp,
          forth::SharedStoreFOp, forth::StoreI8Op, forth::SharedStoreI8Op,
          forth::StoreI16Op, forth::SharedStoreI16Op, forth::StoreI32Op,
          forth::SharedStoreI32Op, forth::StoreF16Op, forth::SharedStoreF16Op,
          forth::StoreBF16Op, forth::SharedStoreBF16Op, forth::StoreF32Op,
          forth::SharedStoreF32Op>(op))
    return StackEffect{2, 0};
  return std::nullopt;
}

struct JoinDepth {
  DenseMap<Block *, unsigned> at;
  Block *entry = nullptr;

  bool note(Block *dest, OperandRange operands, std::optional<unsigned> depth,
            SmallVectorImpl<Block *> &work) {
    if (dest == entry)
      return false;
    unsigned stackOperands = 0;
    for (Value operand : operands)
      if (isForthStack(operand.getType()))
        ++stackOperands;
    unsigned stackArgs = countStackArgs(dest);
    if (stackOperands != stackArgs || stackArgs > 1)
      return false;
    if (stackArgs == 0)
      return true;
    if (!depth.has_value())
      return false;
    auto [it, inserted] = at.try_emplace(dest, *depth);
    if (!inserted)
      return it->second == *depth;
    work.push_back(dest);
    return true;
  }
};

static bool
simulateBlock(Block *block, std::optional<unsigned> depth, JoinDepth &joins,
              SmallVectorImpl<Block *> &work,
              SmallVectorImpl<std::pair<Operation *, unsigned>> &tags,
              bool &sawStack) {
  for (Operation &op : *block) {
    if (isa<forth::BarrierOp, forth::IntrinsicOp>(op))
      continue;
    if (isa<forth::StackOp>(op)) {
      if (depth.has_value())
        return false;
      depth = 0;
      sawStack = true;
      tags.push_back({&op, 0});
      continue;
    }
    if (isa<forth::PickOp, forth::RollOp>(op))
      return false;
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (usesForthStack(call.getOperation()))
        return false;
      continue;
    }
    if (auto branch = dyn_cast<cf::BranchOp>(op)) {
      if (!joins.note(branch.getDest(), branch.getDestOperands(), depth, work))
        return false;
      if (depth.has_value() && forwardsStack(branch.getDestOperands()))
        tags.push_back({&op, *depth});
      continue;
    }
    if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
      if (!joins.note(branch.getTrueDest(), branch.getTrueDestOperands(), depth,
                      work) ||
          !joins.note(branch.getFalseDest(), branch.getFalseDestOperands(),
                      depth, work))
        return false;
      if (depth.has_value() && (forwardsStack(branch.getTrueDestOperands()) ||
                                forwardsStack(branch.getFalseDestOperands())))
        tags.push_back({&op, *depth});
      continue;
    }
    if (!usesForthStack(&op))
      continue;
    std::optional<StackEffect> effect = stackEffect(&op);
    if (!effect || !depth.has_value() || *depth < effect->pops)
      return false;
    tags.push_back({&op, *depth});
    depth = *depth - effect->pops + effect->pushes;
  }
  return true;
}

static bool tagKnownDepths(func::FuncOp func) {
  if (func.isExternal() || hasStackSignature(func))
    return false;

  JoinDepth joins;
  joins.entry = &func.front();
  SmallVector<Block *, 8> work;
  SmallPtrSet<Block *, 8> seen;
  SmallVector<std::pair<Operation *, unsigned>, 16> tags;
  bool sawStack = false;
  auto run = [&](Block *block, std::optional<unsigned> depth) {
    if (!seen.insert(block).second)
      return true;
    return simulateBlock(block, depth, joins, work, tags, sawStack);
  };
  if (!run(joins.entry, std::nullopt))
    return false;
  while (!work.empty()) {
    Block *block = work.pop_back_val();
    if (!run(block, joins.at.lookup(block)))
      return false;
  }
  if (!sawStack || joins.at.contains(joins.entry))
    return false;
  if (!llvm::all_of(func.getBody(),
                    [&](Block &block) { return seen.contains(&block); }))
    return false;

  IntegerType i64 = IntegerType::get(func.getContext(), 64);
  for (auto [op, depth] : tags)
    op->setAttr(kForthKnownDepthAttr, IntegerAttr::get(i64, depth));
  return true;
}

struct TagForthStackDepthPass
    : public impl::TagForthStackDepthBase<TagForthStackDepthPass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation().walk([&](func::FuncOp func) {
      if (func.isExternal())
        return;
      (void)eraseUnreachableBlocks(rewriter, func->getRegions());
      (void)tagKnownDepths(func);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTagForthStackDepthPass() {
  return std::make_unique<TagForthStackDepthPass>();
}

} // namespace warpforth
} // namespace mlir
