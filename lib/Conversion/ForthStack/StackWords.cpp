//===- StackWords.cpp - Stack effects of Forth words ------------*- C++ -*-===//
//
// This file implements the stack-word table: one row per Forth op that takes
// a !forth.stack operand.
//
//===----------------------------------------------------------------------===//

#include "StackWords.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace warpforth {
namespace stack {

WordContext::~WordContext() = default;

namespace {

// Global addresses are numerically equal to their generic addresses, so
// inttoptr into this space is valid and lets NVPTX emit ld.global/st.global
// after address arithmetic.
constexpr unsigned kGlobalAddressSpace =
    static_cast<unsigned>(NVVM::kGlobalMemorySpace);
constexpr unsigned kSharedAddressSpace =
    static_cast<unsigned>(NVVM::kSharedMemorySpace);

enum class MemType { I64, F64, I8, I16, I32, F16, BF16, F32 };

Type memElementType(MLIRContext *ctx, MemType type) {
  switch (type) {
  case MemType::I64:
    return IntegerType::get(ctx, 64);
  case MemType::I32:
    return IntegerType::get(ctx, 32);
  case MemType::I16:
    return IntegerType::get(ctx, 16);
  case MemType::I8:
    return IntegerType::get(ctx, 8);
  case MemType::F64:
    return Float64Type::get(ctx);
  case MemType::F32:
    return Float32Type::get(ctx);
  case MemType::F16:
    return Float16Type::get(ctx);
  case MemType::BF16:
    return BFloat16Type::get(ctx);
  }
  llvm_unreachable("unhandled MemType");
}

Value narrow(OpBuilder &b, Location loc, Value cell, MemType type) {
  Type elementType = memElementType(b.getContext(), type);
  switch (type) {
  case MemType::I64:
    return cell;
  case MemType::I8:
  case MemType::I16:
  case MemType::I32:
    return b.create<arith::TruncIOp>(loc, elementType, cell);
  case MemType::F64:
    return b.create<arith::BitcastOp>(loc, elementType, cell);
  case MemType::F16:
  case MemType::BF16:
  case MemType::F32: {
    Value f64 = b.create<arith::BitcastOp>(loc, b.getF64Type(), cell);
    return b.create<arith::TruncFOp>(loc, elementType, f64);
  }
  }
  llvm_unreachable("unhandled MemType");
}

Value widen(OpBuilder &b, Location loc, Value element, MemType type) {
  switch (type) {
  case MemType::I64:
    return element;
  case MemType::I8:
  case MemType::I16:
  case MemType::I32:
    return b.create<arith::ExtSIOp>(loc, b.getI64Type(), element);
  case MemType::F64:
    return b.create<arith::BitcastOp>(loc, b.getI64Type(), element);
  case MemType::F16:
  case MemType::BF16:
  case MemType::F32: {
    Value f64 = b.create<arith::ExtFOp>(loc, b.getF64Type(), element);
    return b.create<arith::BitcastOp>(loc, b.getI64Type(), f64);
  }
  }
  llvm_unreachable("unhandled MemType");
}

Value pointerTo(OpBuilder &b, Location loc, Value address,
                unsigned addressSpace) {
  auto pointerType = LLVM::LLVMPointerType::get(b.getContext(), addressSpace);
  return b.create<LLVM::IntToPtrOp>(loc, pointerType, address);
}

Value toF64(OpBuilder &b, Location loc, Value cell) {
  return b.create<arith::BitcastOp>(loc, b.getF64Type(), cell);
}

Value toCell(OpBuilder &b, Location loc, Value f64) {
  return b.create<arith::BitcastOp>(loc, b.getI64Type(), f64);
}

/// A Forth flag: true is -1, false is 0.
Value flagCell(OpBuilder &b, Location loc, Value bit) {
  return b.create<arith::ExtSIOp>(loc, b.getI64Type(), bit);
}

Value gpuIndex(OpBuilder &b, Location loc, StringRef intrinsic) {
  return b.create<forth::IntrinsicOp>(loc, b.getIndexType(),
                                      b.getStringAttr(intrinsic));
}

struct StackWordRow {
  TypeID op;
  WordLowering word;
};

template <typename OpT> StackWordRow row(WordLowering word) {
  return {TypeID::get<OpT>(), std::move(word)};
}

template <typename OpT>
StackWordRow row(StackEffect effect, void (*lower)(OpT, WordContext &)) {
  return {TypeID::get<OpT>(),
          {effect, [lower](Operation *op, WordContext &ctx) {
             lower(cast<OpT>(op), ctx);
           }}};
}

/// A word that only reorders cells. It pushes popped(i) for each i in
/// `order`, deepest first.
WordLowering shuffle(unsigned pops, std::initializer_list<unsigned> order) {
  SmallVector<unsigned, 3> pushed(order);
  auto pushes = static_cast<unsigned>(pushed.size());
  return {{pops, pushes}, [pushed](Operation *, WordContext &ctx) {
            for (unsigned i : pushed)
              ctx.push(ctx.popped(i));
          }};
}

/// ( a b -- a op b )
template <typename ArithOp> WordLowering intBinary() {
  return {{2, 1}, [](Operation *op, WordContext &ctx) {
            ctx.push(ctx.builder().create<ArithOp>(op->getLoc(), ctx.popped(1),
                                                   ctx.popped(0)));
          }};
}

/// ( a b -- a op b ) on the f64 bits of each cell.
template <typename ArithOp> WordLowering floatBinary() {
  return {{2, 1}, [](Operation *op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Location loc = op->getLoc();
            Value lhs = toF64(b, loc, ctx.popped(1));
            Value rhs = toF64(b, loc, ctx.popped(0));
            ctx.push(toCell(b, loc, b.create<ArithOp>(loc, lhs, rhs)));
          }};
}

/// ( x -- op x ) on the f64 bits of the cell.
template <typename UnaryOp> WordLowering floatUnary() {
  return {{1, 1}, [](Operation *op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Location loc = op->getLoc();
            Value x = toF64(b, loc, ctx.popped(0));
            ctx.push(toCell(b, loc, b.create<UnaryOp>(loc, x)));
          }};
}

/// ( a b -- flag )
WordLowering intCompare(arith::CmpIPredicate predicate) {
  return {{2, 1}, [predicate](Operation *op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Location loc = op->getLoc();
            Value bit = b.create<arith::CmpIOp>(loc, predicate, ctx.popped(1),
                                                ctx.popped(0));
            ctx.push(flagCell(b, loc, bit));
          }};
}

/// ( a b -- flag ) on the f64 bits of each cell.
WordLowering floatCompare(arith::CmpFPredicate predicate) {
  return {{2, 1}, [predicate](Operation *op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Location loc = op->getLoc();
            Value lhs = toF64(b, loc, ctx.popped(1));
            Value rhs = toF64(b, loc, ctx.popped(0));
            Value bit = b.create<arith::CmpFOp>(loc, predicate, lhs, rhs);
            ctx.push(flagCell(b, loc, bit));
          }};
}

/// ( addr -- x )
WordLowering load(MemType type, unsigned addressSpace) {
  return {{1, 1}, [type, addressSpace](Operation *op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Location loc = op->getLoc();
            Value pointer = pointerTo(b, loc, ctx.popped(0), addressSpace);
            Value element = b.create<LLVM::LoadOp>(
                loc, memElementType(b.getContext(), type), pointer);
            ctx.push(widen(b, loc, element, type));
          }};
}

/// ( x addr -- )
WordLowering store(MemType type, unsigned addressSpace) {
  return {{2, 0}, [type, addressSpace](Operation *op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Location loc = op->getLoc();
            Value pointer = pointerTo(b, loc, ctx.popped(0), addressSpace);
            b.create<LLVM::StoreOp>(loc, narrow(b, loc, ctx.popped(1), type),
                                    pointer);
          }};
}

/// ( -- x ), where `intrinsic` names a GPU index such as "tid-x".
WordLowering gpuIndexWord(StringRef intrinsic) {
  return {{0, 1}, [intrinsic](Operation *op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Value index = gpuIndex(b, op->getLoc(), intrinsic);
            ctx.push(b.create<arith::IndexCastOp>(op->getLoc(), b.getI64Type(),
                                                  index));
          }};
}

ArrayRef<StackWordRow> stackWordRows() {
  using arith::CmpFPredicate;
  using arith::CmpIPredicate;
  static const StackWordRow rows[] = {
      row<forth::ConstantOp>({0, 1},
                             [](forth::ConstantOp op, WordContext &ctx) {
                               OpBuilder &b = ctx.builder();
                               auto value = cast<TypedAttr>(op.getValueAttr());
                               if (isa<FloatAttr>(value)) {
                                 Value f64 = b.create<arith::ConstantOp>(
                                     op.getLoc(), b.getF64Type(), value);
                                 ctx.push(toCell(b, op.getLoc(), f64));
                                 return;
                               }
                               ctx.push(b.create<arith::ConstantOp>(
                                   op.getLoc(), b.getI64Type(), value));
                             }),
      row<forth::ParamRefOp>(
          {0, 1},
          [](forth::ParamRefOp op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Location loc = op.getLoc();
            Value parameter = op.getParameter();
            if (isa<MemRefType>(parameter.getType())) {
              Value pointer = b.create<memref::ExtractAlignedPointerAsIndexOp>(
                  loc, parameter);
              ctx.push(
                  b.create<arith::IndexCastOp>(loc, b.getI64Type(), pointer));
              return;
            }
            if (parameter.getType().isF64()) {
              ctx.push(toCell(b, loc, parameter));
              return;
            }
            ctx.push(parameter);
          }),

      row<forth::DupOp>(shuffle(1, {0, 0})),     // ( a -- a a )
      row<forth::DropOp>(shuffle(1, {})),        // ( a -- )
      row<forth::SwapOp>(shuffle(2, {0, 1})),    // ( a b -- b a )
      row<forth::OverOp>(shuffle(2, {1, 0, 1})), // ( a b -- a b a )
      row<forth::RotOp>(shuffle(3, {1, 0, 2})),  // ( a b c -- b c a )
      row<forth::NipOp>(shuffle(2, {0})),        // ( a b -- b )
      row<forth::TuckOp>(shuffle(2, {0, 1, 0})), // ( a b -- b a b )
      row<forth::PickOp>({1, 1},
                         [](forth::PickOp, WordContext &ctx) {
                           ctx.push(ctx.peek(ctx.popped(0)));
                         }),
      row<forth::RollOp>({1, 0},
                         [](forth::RollOp, WordContext &ctx) {
                           ctx.rollToTop(ctx.popped(0));
                         }),

      row<forth::AddIOp>(intBinary<arith::AddIOp>()),
      row<forth::SubIOp>(intBinary<arith::SubIOp>()),
      row<forth::MulIOp>(intBinary<arith::MulIOp>()),
      row<forth::DivIOp>(intBinary<arith::DivSIOp>()),
      row<forth::ModOp>(intBinary<arith::RemSIOp>()),
      row<forth::AndOp>(intBinary<arith::AndIOp>()),
      row<forth::OrOp>(intBinary<arith::OrIOp>()),
      row<forth::XorOp>(intBinary<arith::XOrIOp>()),
      row<forth::LshiftOp>(intBinary<arith::ShLIOp>()),
      row<forth::RshiftOp>(intBinary<arith::ShRUIOp>()),
      row<forth::NotOp>({1, 1},
                        [](forth::NotOp op, WordContext &ctx) {
                          OpBuilder &b = ctx.builder();
                          Value allOnes = b.create<arith::ConstantIntOp>(
                              op.getLoc(), -1, 64);
                          ctx.push(b.create<arith::XOrIOp>(
                              op.getLoc(), ctx.popped(0), allOnes));
                        }),

      row<forth::AddFOp>(floatBinary<arith::AddFOp>()),
      row<forth::SubFOp>(floatBinary<arith::SubFOp>()),
      row<forth::MulFOp>(floatBinary<arith::MulFOp>()),
      row<forth::DivFOp>(floatBinary<arith::DivFOp>()),
      row<forth::MaxFOp>(floatBinary<arith::MaximumFOp>()),
      row<forth::MinFOp>(floatBinary<arith::MinimumFOp>()),
      row<forth::ExpFOp>(floatUnary<math::ExpOp>()),
      row<forth::SqrtFOp>(floatUnary<math::SqrtOp>()),
      row<forth::LogFOp>(floatUnary<math::LogOp>()),
      row<forth::AbsFOp>(floatUnary<math::AbsFOp>()),
      row<forth::NegFOp>(floatUnary<arith::NegFOp>()),

      row<forth::EqIOp>(intCompare(CmpIPredicate::eq)),
      row<forth::LtIOp>(intCompare(CmpIPredicate::slt)),
      row<forth::GtIOp>(intCompare(CmpIPredicate::sgt)),
      row<forth::NeIOp>(intCompare(CmpIPredicate::ne)),
      row<forth::LeIOp>(intCompare(CmpIPredicate::sle)),
      row<forth::GeIOp>(intCompare(CmpIPredicate::sge)),
      row<forth::EqFOp>(floatCompare(CmpFPredicate::OEQ)),
      row<forth::LtFOp>(floatCompare(CmpFPredicate::OLT)),
      row<forth::GtFOp>(floatCompare(CmpFPredicate::OGT)),
      row<forth::NeFOp>(floatCompare(CmpFPredicate::ONE)),
      row<forth::LeFOp>(floatCompare(CmpFPredicate::OLE)),
      row<forth::GeFOp>(floatCompare(CmpFPredicate::OGE)),
      row<forth::ZeroEqOp>({1, 1},
                           [](forth::ZeroEqOp op, WordContext &ctx) {
                             OpBuilder &b = ctx.builder();
                             Value zero = b.create<arith::ConstantIntOp>(
                                 op.getLoc(), 0, 64);
                             Value bit = b.create<arith::CmpIOp>(
                                 op.getLoc(), CmpIPredicate::eq, ctx.popped(0),
                                 zero);
                             ctx.push(flagCell(b, op.getLoc(), bit));
                           }),

      row<forth::IToFOp>({1, 1},
                         [](forth::IToFOp op, WordContext &ctx) {
                           OpBuilder &b = ctx.builder();
                           Value f64 = b.create<arith::SIToFPOp>(
                               op.getLoc(), b.getF64Type(), ctx.popped(0));
                           ctx.push(toCell(b, op.getLoc(), f64));
                         }),
      row<forth::FToIOp>({1, 1},
                         [](forth::FToIOp op, WordContext &ctx) {
                           OpBuilder &b = ctx.builder();
                           Value f64 = toF64(b, op.getLoc(), ctx.popped(0));
                           ctx.push(b.create<arith::FPToSIOp>(
                               op.getLoc(), b.getI64Type(), f64));
                         }),

      row<forth::LoadIOp>(load(MemType::I64, kGlobalAddressSpace)),
      row<forth::StoreIOp>(store(MemType::I64, kGlobalAddressSpace)),
      row<forth::LoadFOp>(load(MemType::F64, kGlobalAddressSpace)),
      row<forth::StoreFOp>(store(MemType::F64, kGlobalAddressSpace)),
      row<forth::LoadI8Op>(load(MemType::I8, kGlobalAddressSpace)),
      row<forth::StoreI8Op>(store(MemType::I8, kGlobalAddressSpace)),
      row<forth::LoadI16Op>(load(MemType::I16, kGlobalAddressSpace)),
      row<forth::StoreI16Op>(store(MemType::I16, kGlobalAddressSpace)),
      row<forth::LoadI32Op>(load(MemType::I32, kGlobalAddressSpace)),
      row<forth::StoreI32Op>(store(MemType::I32, kGlobalAddressSpace)),
      row<forth::LoadF16Op>(load(MemType::F16, kGlobalAddressSpace)),
      row<forth::StoreF16Op>(store(MemType::F16, kGlobalAddressSpace)),
      row<forth::LoadBF16Op>(load(MemType::BF16, kGlobalAddressSpace)),
      row<forth::StoreBF16Op>(store(MemType::BF16, kGlobalAddressSpace)),
      row<forth::LoadF32Op>(load(MemType::F32, kGlobalAddressSpace)),
      row<forth::StoreF32Op>(store(MemType::F32, kGlobalAddressSpace)),
      row<forth::SharedLoadIOp>(load(MemType::I64, kSharedAddressSpace)),
      row<forth::SharedStoreIOp>(store(MemType::I64, kSharedAddressSpace)),
      row<forth::SharedLoadFOp>(load(MemType::F64, kSharedAddressSpace)),
      row<forth::SharedStoreFOp>(store(MemType::F64, kSharedAddressSpace)),
      row<forth::SharedLoadI8Op>(load(MemType::I8, kSharedAddressSpace)),
      row<forth::SharedStoreI8Op>(store(MemType::I8, kSharedAddressSpace)),
      row<forth::SharedLoadI16Op>(load(MemType::I16, kSharedAddressSpace)),
      row<forth::SharedStoreI16Op>(store(MemType::I16, kSharedAddressSpace)),
      row<forth::SharedLoadI32Op>(load(MemType::I32, kSharedAddressSpace)),
      row<forth::SharedStoreI32Op>(store(MemType::I32, kSharedAddressSpace)),
      row<forth::SharedLoadF16Op>(load(MemType::F16, kSharedAddressSpace)),
      row<forth::SharedStoreF16Op>(store(MemType::F16, kSharedAddressSpace)),
      row<forth::SharedLoadBF16Op>(load(MemType::BF16, kSharedAddressSpace)),
      row<forth::SharedStoreBF16Op>(store(MemType::BF16, kSharedAddressSpace)),
      row<forth::SharedLoadF32Op>(load(MemType::F32, kSharedAddressSpace)),
      row<forth::SharedStoreF32Op>(store(MemType::F32, kSharedAddressSpace)),

      row<forth::ThreadIdXOp>(gpuIndexWord("tid-x")),
      row<forth::ThreadIdYOp>(gpuIndexWord("tid-y")),
      row<forth::ThreadIdZOp>(gpuIndexWord("tid-z")),
      row<forth::BlockIdXOp>(gpuIndexWord("bid-x")),
      row<forth::BlockIdYOp>(gpuIndexWord("bid-y")),
      row<forth::BlockIdZOp>(gpuIndexWord("bid-z")),
      row<forth::BlockDimXOp>(gpuIndexWord("bdim-x")),
      row<forth::BlockDimYOp>(gpuIndexWord("bdim-y")),
      row<forth::BlockDimZOp>(gpuIndexWord("bdim-z")),
      row<forth::GridDimXOp>(gpuIndexWord("gdim-x")),
      row<forth::GridDimYOp>(gpuIndexWord("gdim-y")),
      row<forth::GridDimZOp>(gpuIndexWord("gdim-z")),
      row<forth::GlobalIdOp>({0, 1},
                             [](forth::GlobalIdOp op, WordContext &ctx) {
                               OpBuilder &b = ctx.builder();
                               Location loc = op.getLoc();
                               Value blockId = gpuIndex(b, loc, "bid-x");
                               Value blockDim = gpuIndex(b, loc, "bdim-x");
                               Value threadId = gpuIndex(b, loc, "tid-x");
                               Value product = b.create<arith::MulIOp>(
                                   loc, blockId, blockDim);
                               Value globalId = b.create<arith::AddIOp>(
                                   loc, product, threadId);
                               ctx.push(b.create<arith::IndexCastOp>(
                                   loc, b.getI64Type(), globalId));
                             }),

      row<forth::PushValueOp>({0, 1},
                              [](forth::PushValueOp op, WordContext &ctx) {
                                ctx.push(op.getValue());
                              }),
      row<forth::PopOp>({1, 0},
                        [](forth::PopOp, WordContext &ctx) {
                          ctx.replaceValueResult(ctx.popped(0));
                        }),
      row<forth::PopFlagOp>({1, 0},
                            [](forth::PopFlagOp op, WordContext &ctx) {
                              OpBuilder &b = ctx.builder();
                              Value zero = b.create<arith::ConstantIntOp>(
                                  op.getLoc(), 0, 64);
                              ctx.replaceValueResult(b.create<arith::CmpIOp>(
                                  op.getLoc(), CmpIPredicate::ne, ctx.popped(0),
                                  zero));
                            }),
  };
  return rows;
}

const DenseMap<TypeID, const WordLowering *> &stackWords() {
  static const DenseMap<TypeID, const WordLowering *> words = [] {
    DenseMap<TypeID, const WordLowering *> words;
    for (const StackWordRow &row : stackWordRows()) {
      bool inserted = words.try_emplace(row.op, &row.word).second;
      assert(inserted && "duplicate stack-word row");
      (void)inserted;
    }
    return words;
  }();
  return words;
}

} // namespace

const WordLowering *lookupStackWord(Operation *op) {
  return stackWords().lookup(op->getName().getTypeID());
}

LogicalResult verifyStackWordTableCoversDialect(MLIRContext *context) {
  LogicalResult result = success();
  for (RegisteredOperationName name : context->getRegisteredOperationsByDialect(
           forth::ForthDialect::getDialectNamespace())) {
    // The only Forth ops without a !forth.stack operand.
    if (name.getTypeID() == TypeID::get<forth::StackOp>() ||
        name.getTypeID() == TypeID::get<forth::BarrierOp>() ||
        name.getTypeID() == TypeID::get<forth::IntrinsicOp>())
      continue;
    if (stackWords().contains(name.getTypeID()))
      continue;
    result = emitError(UnknownLoc::get(context))
             << "'" << name.getStringRef()
             << "' takes a !forth.stack but has no row in the stack-word table";
  }
  return result;
}

} // namespace stack
} // namespace warpforth
} // namespace mlir
