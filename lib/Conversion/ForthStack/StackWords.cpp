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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
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

/// Element type of a memory access. Cells are i64; a narrower integer is
/// sign-extended and a float is carried as the bits of an f64.
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

/// The cell as a value of the memory element type.
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

struct StackWordRow {
  TypeID op;
  StackWord word;
};

/// A row built by a family helper, which carries its own effect.
template <typename OpT> StackWordRow row(StackWord word) {
  return {TypeID::get<OpT>(), std::move(word)};
}

/// A row whose lowering reads the typed op.
template <typename OpT>
StackWordRow row(StackEffect effect, void (*lower)(OpT, WordContext &)) {
  return {TypeID::get<OpT>(),
          {effect, [lower](Operation *op, WordContext &ctx) {
             lower(cast<OpT>(op), ctx);
           }}};
}

/// ( a b -- a op b )
template <typename ArithOp> StackWord intBinary() {
  return {{2, 1}, [](Operation *op, WordContext &ctx) {
            ctx.push(ctx.builder().create<ArithOp>(op->getLoc(), ctx.popped(1),
                                                   ctx.popped(0)));
          }};
}

/// ( x addr -- )
StackWord store(MemType type, unsigned addressSpace) {
  return {{2, 0}, [type, addressSpace](Operation *op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Location loc = op->getLoc();
            auto pointerType =
                LLVM::LLVMPointerType::get(b.getContext(), addressSpace);
            Value pointer =
                b.create<LLVM::IntToPtrOp>(loc, pointerType, ctx.popped(0));
            b.create<LLVM::StoreOp>(loc, narrow(b, loc, ctx.popped(1), type),
                                    pointer);
          }};
}

ArrayRef<StackWordRow> stackWordRows() {
  static const StackWordRow rows[] = {
      row<forth::ConstantOp>({0, 1},
                             [](forth::ConstantOp op, WordContext &ctx) {
                               OpBuilder &b = ctx.builder();
                               auto value = cast<TypedAttr>(op.getValueAttr());
                               if (isa<FloatAttr>(value)) {
                                 Value f64 = b.create<arith::ConstantOp>(
                                     op.getLoc(), b.getF64Type(), value);
                                 ctx.push(b.create<arith::BitcastOp>(
                                     op.getLoc(), b.getI64Type(), f64));
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
              ctx.push(
                  b.create<arith::BitcastOp>(loc, b.getI64Type(), parameter));
              return;
            }
            ctx.push(parameter);
          }),
      row<forth::PushValueOp>({0, 1},
                              [](forth::PushValueOp op, WordContext &ctx) {
                                ctx.push(op.getValue());
                              }),
      row<forth::PopOp>({1, 0},
                        [](forth::PopOp, WordContext &ctx) {
                          ctx.replaceValueResult(ctx.popped(0));
                        }),
      row<forth::PopFlagOp>(
          {1, 0},
          [](forth::PopFlagOp op, WordContext &ctx) {
            OpBuilder &b = ctx.builder();
            Value zero = b.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
            ctx.replaceValueResult(b.create<arith::CmpIOp>(
                op.getLoc(), arith::CmpIPredicate::ne, ctx.popped(0), zero));
          }),
      row<forth::AddIOp>(intBinary<arith::AddIOp>()),
      row<forth::SubIOp>(intBinary<arith::SubIOp>()),
      row<forth::MulIOp>(intBinary<arith::MulIOp>()),
      row<forth::DivIOp>(intBinary<arith::DivSIOp>()),
      row<forth::ModOp>(intBinary<arith::RemSIOp>()),
      row<forth::StoreIOp>(store(MemType::I64, kGlobalAddressSpace)),
  };
  return rows;
}

} // namespace

const StackWord *lookupStackWord(Operation *op) {
  static const DenseMap<TypeID, const StackWord *> words = [] {
    DenseMap<TypeID, const StackWord *> words;
    for (const StackWordRow &row : stackWordRows()) {
      bool inserted = words.try_emplace(row.op, &row.word).second;
      assert(inserted && "duplicate stack-word row");
      (void)inserted;
    }
    return words;
  }();
  return words.lookup(op->getName().getTypeID());
}

} // namespace stack
} // namespace warpforth
} // namespace mlir
