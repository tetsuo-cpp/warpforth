//===- WordLowering.cpp - Forth word compute table -------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "WordLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"

namespace mlir {
namespace warpforth {
namespace {

// NVPTX global memory. Global addresses match their generic addresses, so
// inttoptr into this space is what makes NVPTX emit ld.global/st.global.
constexpr unsigned kGlobalMemorySpace =
    static_cast<unsigned>(NVVM::kGlobalMemorySpace);
constexpr unsigned kWorkgroupMemorySpace =
    static_cast<unsigned>(NVVM::kSharedMemorySpace);

enum class MemType { I64, F64, I8, I16, I32, F16, BF16, F32 };

Type memElemType(MLIRContext *ctx, MemType mt) {
  switch (mt) {
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

template <MemType MT>
inline constexpr bool kIsReducedInt =
    (MT == MemType::I8 || MT == MemType::I16 || MT == MemType::I32);

template <MemType MT>
inline constexpr bool kIsReducedFloat =
    (MT == MemType::F16 || MT == MemType::BF16 || MT == MemType::F32);

Value i64Flag(OpBuilder &builder, Location loc, Value cmp) {
  return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), cmp);
}

void computeDup(Operation * /*op*/, ValueRange inputs,
                SmallVectorImpl<Value> &outputs, OpBuilder & /*builder*/) {
  outputs.push_back(inputs[0]);
  outputs.push_back(inputs[0]);
}

void computeDrop(Operation * /*op*/, ValueRange /*inputs*/,
                 SmallVectorImpl<Value> & /*outputs*/,
                 OpBuilder & /*builder*/) {}

void computePopFlag(Operation *op, ValueRange inputs,
                    SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value zero = builder.create<arith::ConstantOp>(loc, builder.getI64Type(),
                                                 builder.getI64IntegerAttr(0));
  outputs.push_back(builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  inputs[0], zero));
}

void computeSwap(Operation * /*op*/, ValueRange inputs,
                 SmallVectorImpl<Value> &outputs, OpBuilder & /*builder*/) {
  outputs.push_back(inputs[1]);
  outputs.push_back(inputs[0]);
}

void computeOver(Operation * /*op*/, ValueRange inputs,
                 SmallVectorImpl<Value> &outputs, OpBuilder & /*builder*/) {
  outputs.push_back(inputs[0]);
  outputs.push_back(inputs[1]);
  outputs.push_back(inputs[0]);
}

void computeRot(Operation * /*op*/, ValueRange inputs,
                SmallVectorImpl<Value> &outputs, OpBuilder & /*builder*/) {
  outputs.push_back(inputs[1]);
  outputs.push_back(inputs[2]);
  outputs.push_back(inputs[0]);
}

void computeNip(Operation * /*op*/, ValueRange inputs,
                SmallVectorImpl<Value> &outputs, OpBuilder & /*builder*/) {
  outputs.push_back(inputs[1]);
}

void computeTuck(Operation * /*op*/, ValueRange inputs,
                 SmallVectorImpl<Value> &outputs, OpBuilder & /*builder*/) {
  outputs.push_back(inputs[1]);
  outputs.push_back(inputs[0]);
  outputs.push_back(inputs[1]);
}

template <typename ArithOp>
void computeBinaryInt(Operation *op, ValueRange inputs,
                      SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  outputs.push_back(
      builder.create<ArithOp>(op->getLoc(), inputs[0], inputs[1]));
}

template <typename ArithOp>
void computeBinaryFloat(Operation *op, ValueRange inputs,
                        SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  auto f64Type = builder.getF64Type();
  Value lhs = builder.create<arith::BitcastOp>(loc, f64Type, inputs[0]);
  Value rhs = builder.create<arith::BitcastOp>(loc, f64Type, inputs[1]);
  Value result = builder.create<ArithOp>(loc, lhs, rhs);
  outputs.push_back(
      builder.create<arith::BitcastOp>(loc, builder.getI64Type(), result));
}

template <typename MathOp>
void computeUnaryFloat(Operation *op, ValueRange inputs,
                       SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value input =
      builder.create<arith::BitcastOp>(loc, builder.getF64Type(), inputs[0]);
  Value result = builder.create<MathOp>(loc, input);
  outputs.push_back(
      builder.create<arith::BitcastOp>(loc, builder.getI64Type(), result));
}

template <typename CmpOp, auto Predicate, bool IsFloat>
void computeCmp(Operation *op, ValueRange inputs,
                SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value lhs = inputs[0];
  Value rhs = inputs[1];
  if constexpr (IsFloat) {
    auto f64Type = builder.getF64Type();
    lhs = builder.create<arith::BitcastOp>(loc, f64Type, lhs);
    rhs = builder.create<arith::BitcastOp>(loc, f64Type, rhs);
  }
  Value cmp = builder.create<CmpOp>(loc, Predicate, lhs, rhs);
  outputs.push_back(i64Flag(builder, loc, cmp));
}

void computeNot(Operation *op, ValueRange inputs,
                SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value allOnes = builder.create<arith::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(-1));
  outputs.push_back(builder.create<arith::XOrIOp>(loc, inputs[0], allOnes));
}

void computeZeroEq(Operation *op, ValueRange inputs,
                   SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value zero = builder.create<arith::ConstantOp>(loc, builder.getI64Type(),
                                                 builder.getI64IntegerAttr(0));
  Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            inputs[0], zero);
  outputs.push_back(i64Flag(builder, loc, cmp));
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

void computeIToF(Operation *op, ValueRange inputs,
                 SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value f64Value =
      builder.create<arith::SIToFPOp>(loc, builder.getF64Type(), inputs[0]);
  outputs.push_back(
      builder.create<arith::BitcastOp>(loc, builder.getI64Type(), f64Value));
}

void computeFToI(Operation *op, ValueRange inputs,
                 SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value f64Value =
      builder.create<arith::BitcastOp>(loc, builder.getF64Type(), inputs[0]);
  outputs.push_back(
      builder.create<arith::FPToSIOp>(loc, builder.getI64Type(), f64Value));
}

void computeParamRef(Operation *op, ValueRange /*inputs*/,
                     SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value parameter = cast<forth::ParamRefOp>(op).getParameter();
  if (isa<MemRefType>(parameter.getType())) {
    Value ptrIndex =
        builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, parameter);
    outputs.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getI64Type(), ptrIndex));
    return;
  }
  if (parameter.getType().isInteger(64)) {
    outputs.push_back(parameter);
    return;
  }
  if (parameter.getType().isF64()) {
    outputs.push_back(
        builder.create<arith::BitcastOp>(loc, builder.getI64Type(), parameter));
    return;
  }
  llvm_unreachable("unsupported param_ref type");
}

template <MemType MT, unsigned AddressSpace>
void computeLoad(Operation *op, ValueRange inputs,
                 SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  auto ptrType = LLVM::LLVMPointerType::get(op->getContext(), AddressSpace);
  Value ptr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, inputs[0]);
  Type elemType = memElemType(op->getContext(), MT);
  Value loaded = builder.create<LLVM::LoadOp>(loc, elemType, ptr);
  if constexpr (MT == MemType::I64) {
    outputs.push_back(loaded);
  } else if constexpr (MT == MemType::F64) {
    outputs.push_back(
        builder.create<arith::BitcastOp>(loc, builder.getI64Type(), loaded));
  } else if constexpr (kIsReducedInt<MT>) {
    outputs.push_back(
        builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), loaded));
  } else if constexpr (kIsReducedFloat<MT>) {
    Value extended =
        builder.create<arith::ExtFOp>(loc, builder.getF64Type(), loaded);
    outputs.push_back(
        builder.create<arith::BitcastOp>(loc, builder.getI64Type(), extended));
  }
}

template <MemType MT, unsigned AddressSpace>
void computeStore(Operation *op, ValueRange inputs,
                  SmallVectorImpl<Value> & /*outputs*/, OpBuilder &builder) {
  Location loc = op->getLoc();
  auto ptrType = LLVM::LLVMPointerType::get(op->getContext(), AddressSpace);
  Value ptr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, inputs[1]);
  Value value = inputs[0];
  if constexpr (MT == MemType::I64) {
    builder.create<LLVM::StoreOp>(loc, value, ptr);
  } else if constexpr (MT == MemType::F64) {
    Value f64Value =
        builder.create<arith::BitcastOp>(loc, builder.getF64Type(), value);
    builder.create<LLVM::StoreOp>(loc, f64Value, ptr);
  } else if constexpr (kIsReducedInt<MT>) {
    Type elemType = memElemType(op->getContext(), MT);
    Value truncated = builder.create<arith::TruncIOp>(loc, elemType, value);
    builder.create<LLVM::StoreOp>(loc, truncated, ptr);
  } else if constexpr (kIsReducedFloat<MT>) {
    Type elemType = memElemType(op->getContext(), MT);
    Value f64Value =
        builder.create<arith::BitcastOp>(loc, builder.getF64Type(), value);
    Value truncated = builder.create<arith::TruncFOp>(loc, elemType, f64Value);
    builder.create<LLVM::StoreOp>(loc, truncated, ptr);
  }
}

void emitGpuIndex(Operation *op, StringRef name,
                  SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value intrinsic = builder.create<forth::IntrinsicOp>(
      loc, builder.getIndexType(), builder.getStringAttr(name));
  outputs.push_back(
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), intrinsic));
}

void computeTidX(Operation *op, ValueRange /*inputs*/,
                 SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "tid-x", outputs, builder);
}
void computeTidY(Operation *op, ValueRange /*inputs*/,
                 SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "tid-y", outputs, builder);
}
void computeTidZ(Operation *op, ValueRange /*inputs*/,
                 SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "tid-z", outputs, builder);
}
void computeBidX(Operation *op, ValueRange /*inputs*/,
                 SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "bid-x", outputs, builder);
}
void computeBidY(Operation *op, ValueRange /*inputs*/,
                 SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "bid-y", outputs, builder);
}
void computeBidZ(Operation *op, ValueRange /*inputs*/,
                 SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "bid-z", outputs, builder);
}
void computeBdimX(Operation *op, ValueRange /*inputs*/,
                  SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "bdim-x", outputs, builder);
}
void computeBdimY(Operation *op, ValueRange /*inputs*/,
                  SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "bdim-y", outputs, builder);
}
void computeBdimZ(Operation *op, ValueRange /*inputs*/,
                  SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "bdim-z", outputs, builder);
}
void computeGdimX(Operation *op, ValueRange /*inputs*/,
                  SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "gdim-x", outputs, builder);
}
void computeGdimY(Operation *op, ValueRange /*inputs*/,
                  SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "gdim-y", outputs, builder);
}
void computeGdimZ(Operation *op, ValueRange /*inputs*/,
                  SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "gdim-z", outputs, builder);
}
void computeLaneId(Operation *op, ValueRange /*inputs*/,
                   SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  emitGpuIndex(op, "lane-id", outputs, builder);
}

void computeGlobalId(Operation *op, ValueRange /*inputs*/,
                     SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  auto index = [&](StringRef name) {
    return builder.create<forth::IntrinsicOp>(loc, builder.getIndexType(),
                                              builder.getStringAttr(name));
  };
  Value bidX = index("bid-x");
  Value bdimX = index("bdim-x");
  Value tidX = index("tid-x");
  Value product = builder.create<arith::MulIOp>(loc, bidX, bdimX);
  Value globalId = builder.create<arith::AddIOp>(loc, product, tidX);
  outputs.push_back(
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), globalId));
}

gpu::ShuffleMode toGpuShuffleMode(forth::ShuffleMode mode) {
  switch (mode) {
  case forth::ShuffleMode::Down:
    return gpu::ShuffleMode::DOWN;
  case forth::ShuffleMode::Up:
    return gpu::ShuffleMode::UP;
  case forth::ShuffleMode::Xor:
    return gpu::ShuffleMode::XOR;
  case forth::ShuffleMode::Idx:
    return gpu::ShuffleMode::IDX;
  }
  llvm_unreachable("unknown forth::ShuffleMode");
}

void computeShuffle(Operation *op, ValueRange inputs,
                    SmallVectorImpl<Value> &outputs, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value offset32 =
      builder.create<arith::TruncIOp>(loc, builder.getI32Type(), inputs[1]);
  Value laneMask = builder.create<arith::ConstantIntOp>(
      loc, forth::kWarpSize - 1, builder.getI32Type());
  Value offset = builder.create<arith::AndIOp>(loc, offset32, laneMask);
  Value width = builder.create<arith::ConstantIntOp>(loc, forth::kWarpSize,
                                                     builder.getI32Type());
  Value shuffled =
      builder
          .create<gpu::ShuffleOp>(
              loc, inputs[0], offset, width,
              toGpuShuffleMode(cast<forth::ShuffleOp>(op).getMode()))
          .getShuffleResult();
  outputs.push_back(shuffled);
}

const WordLowering kDup{1, 2, computeDup};
const WordLowering kDrop{1, 0, computeDrop};
const WordLowering kPopFlag{1, 0, computePopFlag};
const WordLowering kSwap{2, 2, computeSwap};
const WordLowering kOver{2, 3, computeOver};
const WordLowering kRot{3, 3, computeRot};
const WordLowering kNip{2, 1, computeNip};
const WordLowering kTuck{2, 3, computeTuck};
const WordLowering kConstant{0, 1, computeConstant};
const WordLowering kAddI{2, 1, computeBinaryInt<arith::AddIOp>};
const WordLowering kSubI{2, 1, computeBinaryInt<arith::SubIOp>};
const WordLowering kMulI{2, 1, computeBinaryInt<arith::MulIOp>};
const WordLowering kDivI{2, 1, computeBinaryInt<arith::DivSIOp>};
const WordLowering kMod{2, 1, computeBinaryInt<arith::RemSIOp>};
const WordLowering kAddF{2, 1, computeBinaryFloat<arith::AddFOp>};
const WordLowering kSubF{2, 1, computeBinaryFloat<arith::SubFOp>};
const WordLowering kMulF{2, 1, computeBinaryFloat<arith::MulFOp>};
const WordLowering kDivF{2, 1, computeBinaryFloat<arith::DivFOp>};
const WordLowering kMaxF{2, 1, computeBinaryFloat<arith::MaximumFOp>};
const WordLowering kMinF{2, 1, computeBinaryFloat<arith::MinimumFOp>};
const WordLowering kExpF{1, 1, computeUnaryFloat<math::ExpOp>};
const WordLowering kSqrtF{1, 1, computeUnaryFloat<math::SqrtOp>};
const WordLowering kLogF{1, 1, computeUnaryFloat<math::LogOp>};
const WordLowering kAbsF{1, 1, computeUnaryFloat<math::AbsFOp>};
const WordLowering kNegF{1, 1, computeUnaryFloat<arith::NegFOp>};
const WordLowering kAnd{2, 1, computeBinaryInt<arith::AndIOp>};
const WordLowering kOr{2, 1, computeBinaryInt<arith::OrIOp>};
const WordLowering kXor{2, 1, computeBinaryInt<arith::XOrIOp>};
const WordLowering kLshift{2, 1, computeBinaryInt<arith::ShLIOp>};
const WordLowering kRshift{2, 1, computeBinaryInt<arith::ShRUIOp>};
const WordLowering kNot{1, 1, computeNot};
const WordLowering kEqI{
    2, 1, computeCmp<arith::CmpIOp, arith::CmpIPredicate::eq, false>};
const WordLowering kLtI{
    2, 1, computeCmp<arith::CmpIOp, arith::CmpIPredicate::slt, false>};
const WordLowering kGtI{
    2, 1, computeCmp<arith::CmpIOp, arith::CmpIPredicate::sgt, false>};
const WordLowering kNeI{
    2, 1, computeCmp<arith::CmpIOp, arith::CmpIPredicate::ne, false>};
const WordLowering kLeI{
    2, 1, computeCmp<arith::CmpIOp, arith::CmpIPredicate::sle, false>};
const WordLowering kGeI{
    2, 1, computeCmp<arith::CmpIOp, arith::CmpIPredicate::sge, false>};
const WordLowering kEqF{
    2, 1, computeCmp<arith::CmpFOp, arith::CmpFPredicate::OEQ, true>};
const WordLowering kLtF{
    2, 1, computeCmp<arith::CmpFOp, arith::CmpFPredicate::OLT, true>};
const WordLowering kGtF{
    2, 1, computeCmp<arith::CmpFOp, arith::CmpFPredicate::OGT, true>};
const WordLowering kNeF{
    2, 1, computeCmp<arith::CmpFOp, arith::CmpFPredicate::ONE, true>};
const WordLowering kLeF{
    2, 1, computeCmp<arith::CmpFOp, arith::CmpFPredicate::OLE, true>};
const WordLowering kGeF{
    2, 1, computeCmp<arith::CmpFOp, arith::CmpFPredicate::OGE, true>};
const WordLowering kZeroEq{1, 1, computeZeroEq};
const WordLowering kIToF{1, 1, computeIToF};
const WordLowering kFToI{1, 1, computeFToI};
const WordLowering kParamRef{0, 1, computeParamRef};
const WordLowering kLoadI{1, 1, computeLoad<MemType::I64, kGlobalMemorySpace>};
const WordLowering kStoreI{2, 0,
                           computeStore<MemType::I64, kGlobalMemorySpace>};
const WordLowering kLoadF{1, 1, computeLoad<MemType::F64, kGlobalMemorySpace>};
const WordLowering kStoreF{2, 0,
                           computeStore<MemType::F64, kGlobalMemorySpace>};
const WordLowering kSharedLoadI{
    1, 1, computeLoad<MemType::I64, kWorkgroupMemorySpace>};
const WordLowering kSharedStoreI{
    2, 0, computeStore<MemType::I64, kWorkgroupMemorySpace>};
const WordLowering kSharedLoadF{
    1, 1, computeLoad<MemType::F64, kWorkgroupMemorySpace>};
const WordLowering kSharedStoreF{
    2, 0, computeStore<MemType::F64, kWorkgroupMemorySpace>};
const WordLowering kLoadI8{1, 1, computeLoad<MemType::I8, kGlobalMemorySpace>};
const WordLowering kStoreI8{2, 0,
                            computeStore<MemType::I8, kGlobalMemorySpace>};
const WordLowering kSharedLoadI8{
    1, 1, computeLoad<MemType::I8, kWorkgroupMemorySpace>};
const WordLowering kSharedStoreI8{
    2, 0, computeStore<MemType::I8, kWorkgroupMemorySpace>};
const WordLowering kLoadI16{1, 1,
                            computeLoad<MemType::I16, kGlobalMemorySpace>};
const WordLowering kStoreI16{2, 0,
                             computeStore<MemType::I16, kGlobalMemorySpace>};
const WordLowering kSharedLoadI16{
    1, 1, computeLoad<MemType::I16, kWorkgroupMemorySpace>};
const WordLowering kSharedStoreI16{
    2, 0, computeStore<MemType::I16, kWorkgroupMemorySpace>};
const WordLowering kLoadI32{1, 1,
                            computeLoad<MemType::I32, kGlobalMemorySpace>};
const WordLowering kStoreI32{2, 0,
                             computeStore<MemType::I32, kGlobalMemorySpace>};
const WordLowering kSharedLoadI32{
    1, 1, computeLoad<MemType::I32, kWorkgroupMemorySpace>};
const WordLowering kSharedStoreI32{
    2, 0, computeStore<MemType::I32, kWorkgroupMemorySpace>};
const WordLowering kLoadF16{1, 1,
                            computeLoad<MemType::F16, kGlobalMemorySpace>};
const WordLowering kStoreF16{2, 0,
                             computeStore<MemType::F16, kGlobalMemorySpace>};
const WordLowering kSharedLoadF16{
    1, 1, computeLoad<MemType::F16, kWorkgroupMemorySpace>};
const WordLowering kSharedStoreF16{
    2, 0, computeStore<MemType::F16, kWorkgroupMemorySpace>};
const WordLowering kLoadBF16{1, 1,
                             computeLoad<MemType::BF16, kGlobalMemorySpace>};
const WordLowering kStoreBF16{2, 0,
                              computeStore<MemType::BF16, kGlobalMemorySpace>};
const WordLowering kSharedLoadBF16{
    1, 1, computeLoad<MemType::BF16, kWorkgroupMemorySpace>};
const WordLowering kSharedStoreBF16{
    2, 0, computeStore<MemType::BF16, kWorkgroupMemorySpace>};
const WordLowering kLoadF32{1, 1,
                            computeLoad<MemType::F32, kGlobalMemorySpace>};
const WordLowering kStoreF32{2, 0,
                             computeStore<MemType::F32, kGlobalMemorySpace>};
const WordLowering kSharedLoadF32{
    1, 1, computeLoad<MemType::F32, kWorkgroupMemorySpace>};
const WordLowering kSharedStoreF32{
    2, 0, computeStore<MemType::F32, kWorkgroupMemorySpace>};
const WordLowering kTidX{0, 1, computeTidX};
const WordLowering kTidY{0, 1, computeTidY};
const WordLowering kTidZ{0, 1, computeTidZ};
const WordLowering kBidX{0, 1, computeBidX};
const WordLowering kBidY{0, 1, computeBidY};
const WordLowering kBidZ{0, 1, computeBidZ};
const WordLowering kBdimX{0, 1, computeBdimX};
const WordLowering kBdimY{0, 1, computeBdimY};
const WordLowering kBdimZ{0, 1, computeBdimZ};
const WordLowering kGdimX{0, 1, computeGdimX};
const WordLowering kGdimY{0, 1, computeGdimY};
const WordLowering kGdimZ{0, 1, computeGdimZ};
const WordLowering kGlobalId{0, 1, computeGlobalId};
const WordLowering kShuffle{2, 1, computeShuffle};
const WordLowering kLaneId{0, 1, computeLaneId};

template <typename OpTy> bool isOp(Operation *op) { return isa<OpTy>(op); }

struct ListedWord {
  bool (*is)(Operation *);
  const WordLowering *word;
};

const ListedWord kWords[] = {
    {isOp<forth::DupOp>, &kDup},
    {isOp<forth::DropOp>, &kDrop},
    {isOp<forth::PopFlagOp>, &kPopFlag},
    {isOp<forth::SwapOp>, &kSwap},
    {isOp<forth::OverOp>, &kOver},
    {isOp<forth::RotOp>, &kRot},
    {isOp<forth::NipOp>, &kNip},
    {isOp<forth::TuckOp>, &kTuck},
    {isOp<forth::ConstantOp>, &kConstant},
    {isOp<forth::AddIOp>, &kAddI},
    {isOp<forth::SubIOp>, &kSubI},
    {isOp<forth::MulIOp>, &kMulI},
    {isOp<forth::DivIOp>, &kDivI},
    {isOp<forth::ModOp>, &kMod},
    {isOp<forth::AddFOp>, &kAddF},
    {isOp<forth::SubFOp>, &kSubF},
    {isOp<forth::MulFOp>, &kMulF},
    {isOp<forth::DivFOp>, &kDivF},
    {isOp<forth::MaxFOp>, &kMaxF},
    {isOp<forth::MinFOp>, &kMinF},
    {isOp<forth::ExpFOp>, &kExpF},
    {isOp<forth::SqrtFOp>, &kSqrtF},
    {isOp<forth::LogFOp>, &kLogF},
    {isOp<forth::AbsFOp>, &kAbsF},
    {isOp<forth::NegFOp>, &kNegF},
    {isOp<forth::AndOp>, &kAnd},
    {isOp<forth::OrOp>, &kOr},
    {isOp<forth::XorOp>, &kXor},
    {isOp<forth::LshiftOp>, &kLshift},
    {isOp<forth::RshiftOp>, &kRshift},
    {isOp<forth::NotOp>, &kNot},
    {isOp<forth::EqIOp>, &kEqI},
    {isOp<forth::LtIOp>, &kLtI},
    {isOp<forth::GtIOp>, &kGtI},
    {isOp<forth::NeIOp>, &kNeI},
    {isOp<forth::LeIOp>, &kLeI},
    {isOp<forth::GeIOp>, &kGeI},
    {isOp<forth::EqFOp>, &kEqF},
    {isOp<forth::LtFOp>, &kLtF},
    {isOp<forth::GtFOp>, &kGtF},
    {isOp<forth::NeFOp>, &kNeF},
    {isOp<forth::LeFOp>, &kLeF},
    {isOp<forth::GeFOp>, &kGeF},
    {isOp<forth::ZeroEqOp>, &kZeroEq},
    {isOp<forth::IToFOp>, &kIToF},
    {isOp<forth::FToIOp>, &kFToI},
    {isOp<forth::ParamRefOp>, &kParamRef},
    {isOp<forth::LoadIOp>, &kLoadI},
    {isOp<forth::StoreIOp>, &kStoreI},
    {isOp<forth::LoadFOp>, &kLoadF},
    {isOp<forth::StoreFOp>, &kStoreF},
    {isOp<forth::SharedLoadIOp>, &kSharedLoadI},
    {isOp<forth::SharedStoreIOp>, &kSharedStoreI},
    {isOp<forth::SharedLoadFOp>, &kSharedLoadF},
    {isOp<forth::SharedStoreFOp>, &kSharedStoreF},
    {isOp<forth::LoadI8Op>, &kLoadI8},
    {isOp<forth::StoreI8Op>, &kStoreI8},
    {isOp<forth::SharedLoadI8Op>, &kSharedLoadI8},
    {isOp<forth::SharedStoreI8Op>, &kSharedStoreI8},
    {isOp<forth::LoadI16Op>, &kLoadI16},
    {isOp<forth::StoreI16Op>, &kStoreI16},
    {isOp<forth::SharedLoadI16Op>, &kSharedLoadI16},
    {isOp<forth::SharedStoreI16Op>, &kSharedStoreI16},
    {isOp<forth::LoadI32Op>, &kLoadI32},
    {isOp<forth::StoreI32Op>, &kStoreI32},
    {isOp<forth::SharedLoadI32Op>, &kSharedLoadI32},
    {isOp<forth::SharedStoreI32Op>, &kSharedStoreI32},
    {isOp<forth::LoadF16Op>, &kLoadF16},
    {isOp<forth::StoreF16Op>, &kStoreF16},
    {isOp<forth::SharedLoadF16Op>, &kSharedLoadF16},
    {isOp<forth::SharedStoreF16Op>, &kSharedStoreF16},
    {isOp<forth::LoadBF16Op>, &kLoadBF16},
    {isOp<forth::StoreBF16Op>, &kStoreBF16},
    {isOp<forth::SharedLoadBF16Op>, &kSharedLoadBF16},
    {isOp<forth::SharedStoreBF16Op>, &kSharedStoreBF16},
    {isOp<forth::LoadF32Op>, &kLoadF32},
    {isOp<forth::StoreF32Op>, &kStoreF32},
    {isOp<forth::SharedLoadF32Op>, &kSharedLoadF32},
    {isOp<forth::SharedStoreF32Op>, &kSharedStoreF32},
    {isOp<forth::ThreadIdXOp>, &kTidX},
    {isOp<forth::ThreadIdYOp>, &kTidY},
    {isOp<forth::ThreadIdZOp>, &kTidZ},
    {isOp<forth::BlockIdXOp>, &kBidX},
    {isOp<forth::BlockIdYOp>, &kBidY},
    {isOp<forth::BlockIdZOp>, &kBidZ},
    {isOp<forth::BlockDimXOp>, &kBdimX},
    {isOp<forth::BlockDimYOp>, &kBdimY},
    {isOp<forth::BlockDimZOp>, &kBdimZ},
    {isOp<forth::GridDimXOp>, &kGdimX},
    {isOp<forth::GridDimYOp>, &kGdimY},
    {isOp<forth::GridDimZOp>, &kGdimZ},
    {isOp<forth::GlobalIdOp>, &kGlobalId},
    {isOp<forth::ShuffleOp>, &kShuffle},
    {isOp<forth::LaneIdOp>, &kLaneId},
};

} // namespace

const WordLowering *lookupWord(Operation *op) {
  for (const ListedWord &row : kWords)
    if (row.is(op))
      return row.word;
  return nullptr;
}

} // namespace warpforth
} // namespace mlir
