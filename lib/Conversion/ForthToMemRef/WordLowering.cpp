//===- WordLowering.cpp - Forth word compute table -------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "WordLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"

namespace mlir {
namespace warpforth {
namespace {

// NVPTX global memory. Global addresses match their generic addresses, so
// inttoptr into this space is what makes NVPTX emit st.global.
constexpr unsigned kGlobalMemorySpace =
    static_cast<unsigned>(NVVM::kGlobalMemorySpace);

template <typename ArithOp>
void computeBinaryInt(Operation *op, ValueRange inputs,
                      SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  outputs.push_back(
      builder.create<ArithOp>(op->getLoc(), inputs[0], inputs[1]));
}

void computeConstant(Operation *op, ValueRange /*inputs*/,
                     SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  auto typedValue = cast<TypedAttr>(cast<forth::ConstantOp>(op).getValueAttr());
  if (isa<FloatAttr>(typedValue)) {
    Value f64Value = builder.create<arith::ConstantOp>(
        loc, builder.getF64Type(), typedValue);
    outputs.push_back(
        builder.create<arith::BitcastOp>(loc, builder.getI64Type(), f64Value));
    return;
  }
  outputs.push_back(
      builder.create<arith::ConstantOp>(loc, builder.getI64Type(), typedValue));
}

void computeStoreI64(Operation *op, ValueRange inputs,
                     SmallVectorImpl<Value> & /*outputs*/, OpBuilder &builder) {
  Location loc = op->getLoc();
  auto ptrType =
      LLVM::LLVMPointerType::get(op->getContext(), kGlobalMemorySpace);
  Value ptr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, inputs[1]);
  builder.create<LLVM::StoreOp>(loc, inputs[0], ptr);
}

const WordLowering kConstant{0, 1, computeConstant};
const WordLowering kAddI{2, 1, computeBinaryInt<arith::AddIOp>};
const WordLowering kSubI{2, 1, computeBinaryInt<arith::SubIOp>};
const WordLowering kMulI{2, 1, computeBinaryInt<arith::MulIOp>};
const WordLowering kDivI{2, 1, computeBinaryInt<arith::DivSIOp>};
const WordLowering kMod{2, 1, computeBinaryInt<arith::RemSIOp>};
const WordLowering kStoreI{2, 0, computeStoreI64};

} // namespace

const WordLowering *lookupWord(Operation *op) {
  if (isa<forth::ConstantOp>(op))
    return &kConstant;
  if (isa<forth::AddIOp>(op))
    return &kAddI;
  if (isa<forth::SubIOp>(op))
    return &kSubI;
  if (isa<forth::MulIOp>(op))
    return &kMulI;
  if (isa<forth::DivIOp>(op))
    return &kDivI;
  if (isa<forth::ModOp>(op))
    return &kMod;
  if (isa<forth::StoreIOp>(op))
    return &kStoreI;
  return nullptr;
}

} // namespace warpforth
} // namespace mlir
