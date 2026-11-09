//===- ForthToMemRef.cpp - Forth to MemRef conversion ----------*- C++ -*-===//
//
// This file implements the conversion from Forth dialect to MemRef dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthToMemRef/ForthToMemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <type_traits>

namespace mlir {
namespace warpforth {

#define GEN_PASS_DEF_CONVERTFORTHTOMEMREF
#include "warpforth/Conversion/Passes.h.inc"

namespace {

constexpr int64_t kStackSize = 256;

// NVPTX global memory. Global addresses match their generic addresses, so
// inttoptr into this space is what makes NVPTX emit ld.global/st.global.
constexpr unsigned kGlobalMemorySpace =
    static_cast<unsigned>(NVVM::kGlobalMemorySpace);
constexpr unsigned kWorkgroupMemorySpace =
    static_cast<unsigned>(NVVM::kSharedMemorySpace);

enum class MemType { I64, F64, I8, I16, I32, F16, BF16, F32 };

static Type memElemType(MLIRContext *ctx, MemType mt) {
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

static std::optional<unsigned> getKnownDepth(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>(kForthKnownDepthAttr);
  if (!attr)
    return std::nullopt;
  return static_cast<unsigned>(attr.getInt());
}

class ForthToMemRefTypeConverter : public TypeConverter {
public:
  ForthToMemRefTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(
        [&](forth::StackType type,
            SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          auto memrefType = MemRefType::get(
              {kStackSize}, IntegerType::get(type.getContext(), 64));
          auto indexType = IndexType::get(type.getContext());
          results.push_back(memrefType);
          results.push_back(indexType);
          return success();
        });
  }
};

/// Depth 0 is the top cell. MemRef and SSA cursors implement the same slots.
class StackCursor {
public:
  virtual ~StackCursor() = default;
  virtual Value loadAt(Location loc, OpBuilder &builder,
                       unsigned depthFromTop) = 0;
  virtual void storeAt(Location loc, OpBuilder &builder, unsigned depthFromTop,
                       Value value) = 0;
  virtual void push(Location loc, OpBuilder &builder, Value value) = 0;
  virtual void drop(Location loc, OpBuilder &builder, unsigned count) = 0;
  virtual Value stackPointer() const = 0;
};

class MemRefStackCursor : public StackCursor {
public:
  MemRefStackCursor(Value memref, Value stackPointer)
      : memref(memref), sp(stackPointer) {}

  Value loadAt(Location loc, OpBuilder &builder,
               unsigned depthFromTop) override {
    return builder.create<memref::LoadOp>(loc, memref,
                                          indexAt(loc, builder, depthFromTop));
  }

  void storeAt(Location loc, OpBuilder &builder, unsigned depthFromTop,
               Value value) override {
    builder.create<memref::StoreOp>(loc, value, memref,
                                    indexAt(loc, builder, depthFromTop));
  }

  void push(Location loc, OpBuilder &builder, Value value) override {
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    sp = builder.create<arith::AddIOp>(loc, sp, one);
    builder.create<memref::StoreOp>(loc, value, memref, sp);
    cached.clear();
  }

  void drop(Location loc, OpBuilder &builder, unsigned count) override {
    for (unsigned i = 0; i < count; ++i) {
      if (cached.size() > 1 && cached[1])
        sp = cached[1];
      else {
        Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
        sp = builder.create<arith::SubIOp>(loc, sp, one);
      }
      cached.clear();
    }
  }

  Value stackPointer() const override { return sp; }

private:
  Value indexAt(Location loc, OpBuilder &builder, unsigned depthFromTop) {
    if (depthFromTop == 0)
      return sp;
    if (cached.size() <= depthFromTop)
      cached.resize(depthFromTop + 1);
    if (cached[depthFromTop])
      return cached[depthFromTop];
    Value offset = builder.create<arith::ConstantIndexOp>(loc, depthFromTop);
    cached[depthFromTop] = builder.create<arith::SubIOp>(loc, sp, offset);
    return cached[depthFromTop];
  }

  Value memref;
  Value sp;
  SmallVector<Value, 4> cached;
};

class SSAStackCursor : public StackCursor {
public:
  explicit SSAStackCursor(SmallVectorImpl<Value> &cells) : cells(cells) {}

  Value loadAt(Location, OpBuilder &, unsigned depthFromTop) override {
    if (depthFromTop >= cells.size())
      llvm_unreachable("known stack underflow");
    return cells[cells.size() - 1 - depthFromTop];
  }

  void storeAt(Location, OpBuilder &, unsigned depthFromTop,
               Value value) override {
    if (depthFromTop >= cells.size())
      llvm_unreachable("known stack underflow");
    cells[cells.size() - 1 - depthFromTop] = value;
  }

  void push(Location, OpBuilder &, Value value) override {
    cells.push_back(value);
  }

  void drop(Location, OpBuilder &, unsigned count) override {
    if (count > cells.size())
      llvm_unreachable("known stack underflow");
    cells.truncate(cells.size() - count);
  }

  Value stackPointer() const override { return {}; }

private:
  SmallVectorImpl<Value> &cells;
};

class KnownRewrite {
public:
  virtual ~KnownRewrite() = default;
  virtual bool matches(Operation *op) const = 0;
  virtual LogicalResult applyKnown(Operation *op, StackCursor &stack,
                                   OpBuilder &builder) const = 0;
};

template <typename ForthOp>
struct StackEffectConversion : public OpConversionPattern<ForthOp>,
                               public KnownRewrite {
  using OpConversionPattern<ForthOp>::OpConversionPattern;
  using OneToNOpAdaptor =
      typename OpConversionPattern<ForthOp>::OneToNOpAdaptor;

  bool matches(Operation *op) const final { return isa<ForthOp>(op); }

  LogicalResult applyKnown(Operation *op, StackCursor &stack,
                           OpBuilder &builder) const final {
    return emitStack(cast<ForthOp>(op), stack, builder);
  }

  LogicalResult
  matchAndRewrite(ForthOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (getKnownDepth(op.getOperation()))
      return failure();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    MemRefStackCursor stack(memref, inputStack[1]);
    if (failed(emitStack(op, stack, rewriter)))
      return failure();
    rewriter.replaceOpWithMultiple(op, {{memref, stack.stackPointer()}});
    return success();
  }

  virtual LogicalResult emitStack(ForthOp op, StackCursor &stack,
                                  OpBuilder &builder) const = 0;
};

/// Memref-only stack words. Dynamic pick/roll and push_value never carry a
/// known depth.
template <typename ForthOp>
struct MemRefStackConversion : public OpConversionPattern<ForthOp> {
  using OpConversionPattern<ForthOp>::OpConversionPattern;
  using OneToNOpAdaptor =
      typename OpConversionPattern<ForthOp>::OneToNOpAdaptor;

  virtual FailureOr<Value>
  rewriteStack(ForthOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr, ConversionPatternRewriter &rewriter) const = 0;

  LogicalResult
  matchAndRewrite(ForthOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (getKnownDepth(op.getOperation()))
      return failure();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    FailureOr<Value> newSP =
        rewriteStack(op, adaptor, memref, inputStack[1], rewriter);
    if (failed(newSP))
      return failure();
    rewriter.replaceOpWithMultiple(op, {{memref, *newSP}});
    return success();
  }
};

static Value i64Flag(OpBuilder &builder, Location loc, Value cmp) {
  return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), cmp);
}

static Value bitcastF64(OpBuilder &builder, Location loc, Value cell) {
  return builder.create<arith::BitcastOp>(loc, builder.getF64Type(), cell);
}

static Value bitcastI64(OpBuilder &builder, Location loc, Value value) {
  return builder.create<arith::BitcastOp>(loc, builder.getI64Type(), value);
}

struct StackOpConversion : public OpConversionPattern<forth::StackOp> {
  StackOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::StackOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::StackOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (getKnownDepth(op.getOperation()))
      return failure();
    auto loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();
    auto memrefType = MemRefType::get({kStackSize}, i64Type);
    Value stackBuffer = rewriter.create<memref::AllocaOp>(loc, memrefType);
    Value initialSP = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.replaceOpWithMultiple(op, {{stackBuffer, initialSP}});
    return success();
  }
};

struct ConstantOpConversion : public StackEffectConversion<forth::ConstantOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::ConstantOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    auto typedValue = cast<TypedAttr>(op.getValueAttr());
    Value cell;
    if (isa<FloatAttr>(typedValue)) {
      Value f64Value = builder.create<arith::ConstantOp>(
          loc, builder.getF64Type(), typedValue);
      cell = bitcastI64(builder, loc, f64Value);
    } else {
      cell = builder.create<arith::ConstantOp>(loc, builder.getI64Type(),
                                               typedValue);
    }
    stack.push(loc, builder, cell);
    return success();
  }
};

/// ( a -- a a )
struct DupOpConversion : public StackEffectConversion<forth::DupOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::DupOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    stack.push(loc, builder, stack.loadAt(loc, builder, 0));
    return success();
  }
};

/// ( a -- )
struct DropOpConversion : public StackEffectConversion<forth::DropOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::DropOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    stack.drop(op.getLoc(), builder, 1);
    return success();
  }
};

/// ( a b -- b a )
struct SwapOpConversion : public StackEffectConversion<forth::SwapOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::SwapOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value top = stack.loadAt(loc, builder, 0);
    Value second = stack.loadAt(loc, builder, 1);
    stack.storeAt(loc, builder, 0, second);
    stack.storeAt(loc, builder, 1, top);
    return success();
  }
};

/// ( a b -- a b a )
struct OverOpConversion : public StackEffectConversion<forth::OverOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::OverOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    stack.push(loc, builder, stack.loadAt(loc, builder, 1));
    return success();
  }
};

/// ( a b c -- b c a )
struct RotOpConversion : public StackEffectConversion<forth::RotOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::RotOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value c = stack.loadAt(loc, builder, 0);
    Value b = stack.loadAt(loc, builder, 1);
    Value a = stack.loadAt(loc, builder, 2);
    stack.storeAt(loc, builder, 2, b);
    stack.storeAt(loc, builder, 1, c);
    stack.storeAt(loc, builder, 0, a);
    return success();
  }
};

/// ( a b -- b )
struct NipOpConversion : public StackEffectConversion<forth::NipOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::NipOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value top = stack.loadAt(loc, builder, 0);
    stack.storeAt(loc, builder, 1, top);
    stack.drop(loc, builder, 1);
    return success();
  }
};

/// ( a b -- b a b )
struct TuckOpConversion : public StackEffectConversion<forth::TuckOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::TuckOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value b = stack.loadAt(loc, builder, 0);
    Value a = stack.loadAt(loc, builder, 1);
    stack.storeAt(loc, builder, 1, b);
    stack.storeAt(loc, builder, 0, a);
    stack.push(loc, builder, b);
    return success();
  }
};

struct PickOpConversion : public MemRefStackConversion<forth::PickOp> {
  using MemRefStackConversion::MemRefStackConversion;

  FailureOr<Value>
  rewriteStack(forth::PickOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value nI64 = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spAfterPop = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value nIdx =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), nI64);
    Value targetAddr = rewriter.create<arith::SubIOp>(loc, spAfterPop, nIdx);
    Value pickedValue =
        rewriter.create<memref::LoadOp>(loc, memref, targetAddr);
    rewriter.create<memref::StoreOp>(loc, pickedValue, memref, stackPtr);
    return stackPtr;
  }
};

struct RollOpConversion : public MemRefStackConversion<forth::RollOp> {
  using MemRefStackConversion::MemRefStackConversion;

  FailureOr<Value>
  rewriteStack(forth::RollOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto indexType = rewriter.getIndexType();
    Value nI64 = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spAfterPop = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value nIdx = rewriter.create<arith::IndexCastOp>(loc, indexType, nI64);
    Value rolledAddr = rewriter.create<arith::SubIOp>(loc, spAfterPop, nIdx);
    Value rolledValue =
        rewriter.create<memref::LoadOp>(loc, memref, rolledAddr);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, op->getIterator());
    Block *headerBlock = rewriter.createBlock(continueBlock);
    headerBlock->addArgument(indexType, loc);
    Block *bodyBlock = rewriter.createBlock(continueBlock);
    bodyBlock->addArgument(indexType, loc);

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<cf::BranchOp>(loc, headerBlock, ValueRange{rolledAddr});

    rewriter.setInsertionPointToStart(headerBlock);
    Value iv = headerBlock->getArgument(0);
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                iv, spAfterPop);
    rewriter.create<cf::CondBranchOp>(loc, cond, bodyBlock, ValueRange{iv},
                                      continueBlock, ValueRange{});

    rewriter.setInsertionPointToStart(bodyBlock);
    Value biv = bodyBlock->getArgument(0);
    Value next = rewriter.create<arith::AddIOp>(loc, biv, one);
    Value shiftedVal = rewriter.create<memref::LoadOp>(loc, memref, next);
    rewriter.create<memref::StoreOp>(loc, shiftedVal, memref, biv);
    rewriter.create<cf::BranchOp>(loc, headerBlock, ValueRange{next});

    rewriter.setInsertionPoint(op);
    rewriter.create<memref::StoreOp>(loc, rolledValue, memref, spAfterPop);
    return spAfterPop;
  }
};

template <typename ForthOp, typename ArithOp, bool IsFloat>
struct BinaryArithOpConversion : public StackEffectConversion<ForthOp> {
  using StackEffectConversion<ForthOp>::StackEffectConversion;

  LogicalResult emitStack(ForthOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value rhs = stack.loadAt(loc, builder, 0);
    Value lhs = stack.loadAt(loc, builder, 1);
    Value result;
    if constexpr (IsFloat) {
      Value lhsF = bitcastF64(builder, loc, lhs);
      Value rhsF = bitcastF64(builder, loc, rhs);
      Value resF = builder.create<ArithOp>(loc, lhsF, rhsF);
      result = bitcastI64(builder, loc, resF);
    } else {
      result = builder.create<ArithOp>(loc, lhs, rhs);
    }
    stack.storeAt(loc, builder, 1, result);
    stack.drop(loc, builder, 1);
    return success();
  }
};

using AddIOpConversion =
    BinaryArithOpConversion<forth::AddIOp, arith::AddIOp, false>;
using SubIOpConversion =
    BinaryArithOpConversion<forth::SubIOp, arith::SubIOp, false>;
using MulIOpConversion =
    BinaryArithOpConversion<forth::MulIOp, arith::MulIOp, false>;
using DivIOpConversion =
    BinaryArithOpConversion<forth::DivIOp, arith::DivSIOp, false>;
using ModOpConversion =
    BinaryArithOpConversion<forth::ModOp, arith::RemSIOp, false>;
using AndOpConversion =
    BinaryArithOpConversion<forth::AndOp, arith::AndIOp, false>;
using OrOpConversion =
    BinaryArithOpConversion<forth::OrOp, arith::OrIOp, false>;
using XorOpConversion =
    BinaryArithOpConversion<forth::XorOp, arith::XOrIOp, false>;
using LshiftOpConversion =
    BinaryArithOpConversion<forth::LshiftOp, arith::ShLIOp, false>;
using RshiftOpConversion =
    BinaryArithOpConversion<forth::RshiftOp, arith::ShRUIOp, false>;
using AddFOpConversion =
    BinaryArithOpConversion<forth::AddFOp, arith::AddFOp, true>;
using SubFOpConversion =
    BinaryArithOpConversion<forth::SubFOp, arith::SubFOp, true>;
using MulFOpConversion =
    BinaryArithOpConversion<forth::MulFOp, arith::MulFOp, true>;
using DivFOpConversion =
    BinaryArithOpConversion<forth::DivFOp, arith::DivFOp, true>;
using MaxFOpConversion =
    BinaryArithOpConversion<forth::MaxFOp, arith::MaximumFOp, true>;
using MinFOpConversion =
    BinaryArithOpConversion<forth::MinFOp, arith::MinimumFOp, true>;

template <typename ForthOp, typename MathOp>
struct UnaryFloatOpConversion : public StackEffectConversion<ForthOp> {
  using StackEffectConversion<ForthOp>::StackEffectConversion;

  LogicalResult emitStack(ForthOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value input = bitcastF64(builder, loc, stack.loadAt(loc, builder, 0));
    Value result = builder.create<MathOp>(loc, input);
    stack.storeAt(loc, builder, 0, bitcastI64(builder, loc, result));
    return success();
  }
};

using ExpFOpConversion = UnaryFloatOpConversion<forth::ExpFOp, math::ExpOp>;
using SqrtFOpConversion = UnaryFloatOpConversion<forth::SqrtFOp, math::SqrtOp>;
using LogFOpConversion = UnaryFloatOpConversion<forth::LogFOp, math::LogOp>;
using AbsFOpConversion = UnaryFloatOpConversion<forth::AbsFOp, math::AbsFOp>;
using NegFOpConversion = UnaryFloatOpConversion<forth::NegFOp, arith::NegFOp>;

template <typename ForthOp, typename CmpOpTy, auto Predicate, bool IsFloat>
struct BinaryCmpOpConversion : public StackEffectConversion<ForthOp> {
  using StackEffectConversion<ForthOp>::StackEffectConversion;

  LogicalResult emitStack(ForthOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value rhs = stack.loadAt(loc, builder, 0);
    Value lhs = stack.loadAt(loc, builder, 1);
    if constexpr (IsFloat) {
      lhs = bitcastF64(builder, loc, lhs);
      rhs = bitcastF64(builder, loc, rhs);
    }
    Value cmp = builder.create<CmpOpTy>(loc, Predicate, lhs, rhs);
    stack.storeAt(loc, builder, 1, i64Flag(builder, loc, cmp));
    stack.drop(loc, builder, 1);
    return success();
  }
};

using EqIOpConversion = BinaryCmpOpConversion<forth::EqIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::eq, false>;
using LtIOpConversion = BinaryCmpOpConversion<forth::LtIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::slt, false>;
using GtIOpConversion = BinaryCmpOpConversion<forth::GtIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::sgt, false>;
using NeIOpConversion = BinaryCmpOpConversion<forth::NeIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::ne, false>;
using LeIOpConversion = BinaryCmpOpConversion<forth::LeIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::sle, false>;
using GeIOpConversion = BinaryCmpOpConversion<forth::GeIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::sge, false>;
using EqFOpConversion = BinaryCmpOpConversion<forth::EqFOp, arith::CmpFOp,
                                              arith::CmpFPredicate::OEQ, true>;
using LtFOpConversion = BinaryCmpOpConversion<forth::LtFOp, arith::CmpFOp,
                                              arith::CmpFPredicate::OLT, true>;
using GtFOpConversion = BinaryCmpOpConversion<forth::GtFOp, arith::CmpFOp,
                                              arith::CmpFPredicate::OGT, true>;
using NeFOpConversion = BinaryCmpOpConversion<forth::NeFOp, arith::CmpFOp,
                                              arith::CmpFPredicate::ONE, true>;
using LeFOpConversion = BinaryCmpOpConversion<forth::LeFOp, arith::CmpFOp,
                                              arith::CmpFPredicate::OLE, true>;
using GeFOpConversion = BinaryCmpOpConversion<forth::GeFOp, arith::CmpFOp,
                                              arith::CmpFPredicate::OGE, true>;

struct NotOpConversion : public StackEffectConversion<forth::NotOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::NotOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value allOnes = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(-1));
    Value result = builder.create<arith::XOrIOp>(
        loc, stack.loadAt(loc, builder, 0), allOnes);
    stack.storeAt(loc, builder, 0, result);
    return success();
  }
};

struct ZeroEqOpConversion : public StackEffectConversion<forth::ZeroEqOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::ZeroEqOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value zero = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
    Value cmp = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, stack.loadAt(loc, builder, 0), zero);
    stack.storeAt(loc, builder, 0, i64Flag(builder, loc, cmp));
    return success();
  }
};

struct ParamRefOpConversion : public StackEffectConversion<forth::ParamRefOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::ParamRefOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value parameter = op.getParameter();
    if (isa<MemRefType>(parameter.getType())) {
      Value ptrIndex = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
          loc, parameter);
      stack.push(loc, builder,
                 builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                    ptrIndex));
      return success();
    }
    if (parameter.getType().isInteger(64)) {
      stack.push(loc, builder, parameter);
      return success();
    }
    if (parameter.getType().isF64()) {
      stack.push(loc, builder, bitcastI64(builder, loc, parameter));
      return success();
    }
    return failure();
  }
};

template <typename ForthOp, MemType MT, unsigned AddressSpace>
struct MemoryLoadOpConversion : public StackEffectConversion<ForthOp> {
  using StackEffectConversion<ForthOp>::StackEffectConversion;

  LogicalResult emitStack(ForthOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value addr = stack.loadAt(loc, builder, 0);
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext(), AddressSpace);
    Value ptr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, addr);
    Type elemType = memElemType(op.getContext(), MT);
    Value loaded = builder.create<LLVM::LoadOp>(loc, elemType, ptr);
    Value cell;
    if constexpr (MT == MemType::I64)
      cell = loaded;
    else if constexpr (MT == MemType::F64)
      cell = bitcastI64(builder, loc, loaded);
    else if constexpr (kIsReducedInt<MT>)
      cell = builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), loaded);
    else if constexpr (kIsReducedFloat<MT>) {
      Value extended =
          builder.create<arith::ExtFOp>(loc, builder.getF64Type(), loaded);
      cell = bitcastI64(builder, loc, extended);
    } else
      llvm_unreachable("unhandled memory type");
    stack.storeAt(loc, builder, 0, cell);
    return success();
  }
};

template <typename ForthOp, MemType MT, unsigned AddressSpace>
struct MemoryStoreOpConversion : public StackEffectConversion<ForthOp> {
  using StackEffectConversion<ForthOp>::StackEffectConversion;

  LogicalResult emitStack(ForthOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value addr = stack.loadAt(loc, builder, 0);
    Value value = stack.loadAt(loc, builder, 1);
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext(), AddressSpace);
    Value ptr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, addr);
    if constexpr (MT == MemType::I64) {
      builder.create<LLVM::StoreOp>(loc, value, ptr);
    } else if constexpr (MT == MemType::F64) {
      builder.create<LLVM::StoreOp>(loc, bitcastF64(builder, loc, value), ptr);
    } else if constexpr (kIsReducedInt<MT>) {
      Type elemType = memElemType(op.getContext(), MT);
      Value truncated = builder.create<arith::TruncIOp>(loc, elemType, value);
      builder.create<LLVM::StoreOp>(loc, truncated, ptr);
    } else if constexpr (kIsReducedFloat<MT>) {
      Type elemType = memElemType(op.getContext(), MT);
      Value f64Value = bitcastF64(builder, loc, value);
      Value truncated =
          builder.create<arith::TruncFOp>(loc, elemType, f64Value);
      builder.create<LLVM::StoreOp>(loc, truncated, ptr);
    }
    stack.drop(loc, builder, 2);
    return success();
  }
};

using LoadIOpConversion =
    MemoryLoadOpConversion<forth::LoadIOp, MemType::I64, kGlobalMemorySpace>;
using StoreIOpConversion =
    MemoryStoreOpConversion<forth::StoreIOp, MemType::I64, kGlobalMemorySpace>;
using LoadFOpConversion =
    MemoryLoadOpConversion<forth::LoadFOp, MemType::F64, kGlobalMemorySpace>;
using StoreFOpConversion =
    MemoryStoreOpConversion<forth::StoreFOp, MemType::F64, kGlobalMemorySpace>;
using SharedLoadIOpConversion =
    MemoryLoadOpConversion<forth::SharedLoadIOp, MemType::I64,
                           kWorkgroupMemorySpace>;
using SharedStoreIOpConversion =
    MemoryStoreOpConversion<forth::SharedStoreIOp, MemType::I64,
                            kWorkgroupMemorySpace>;
using SharedLoadFOpConversion =
    MemoryLoadOpConversion<forth::SharedLoadFOp, MemType::F64,
                           kWorkgroupMemorySpace>;
using SharedStoreFOpConversion =
    MemoryStoreOpConversion<forth::SharedStoreFOp, MemType::F64,
                            kWorkgroupMemorySpace>;
using LoadI8OpConversion =
    MemoryLoadOpConversion<forth::LoadI8Op, MemType::I8, kGlobalMemorySpace>;
using StoreI8OpConversion =
    MemoryStoreOpConversion<forth::StoreI8Op, MemType::I8, kGlobalMemorySpace>;
using SharedLoadI8OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadI8Op, MemType::I8,
                           kWorkgroupMemorySpace>;
using SharedStoreI8OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreI8Op, MemType::I8,
                            kWorkgroupMemorySpace>;
using LoadI16OpConversion =
    MemoryLoadOpConversion<forth::LoadI16Op, MemType::I16, kGlobalMemorySpace>;
using StoreI16OpConversion =
    MemoryStoreOpConversion<forth::StoreI16Op, MemType::I16,
                            kGlobalMemorySpace>;
using SharedLoadI16OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadI16Op, MemType::I16,
                           kWorkgroupMemorySpace>;
using SharedStoreI16OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreI16Op, MemType::I16,
                            kWorkgroupMemorySpace>;
using LoadI32OpConversion =
    MemoryLoadOpConversion<forth::LoadI32Op, MemType::I32, kGlobalMemorySpace>;
using StoreI32OpConversion =
    MemoryStoreOpConversion<forth::StoreI32Op, MemType::I32,
                            kGlobalMemorySpace>;
using SharedLoadI32OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadI32Op, MemType::I32,
                           kWorkgroupMemorySpace>;
using SharedStoreI32OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreI32Op, MemType::I32,
                            kWorkgroupMemorySpace>;
using LoadF16OpConversion =
    MemoryLoadOpConversion<forth::LoadF16Op, MemType::F16, kGlobalMemorySpace>;
using StoreF16OpConversion =
    MemoryStoreOpConversion<forth::StoreF16Op, MemType::F16,
                            kGlobalMemorySpace>;
using SharedLoadF16OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadF16Op, MemType::F16,
                           kWorkgroupMemorySpace>;
using SharedStoreF16OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreF16Op, MemType::F16,
                            kWorkgroupMemorySpace>;
using LoadBF16OpConversion =
    MemoryLoadOpConversion<forth::LoadBF16Op, MemType::BF16,
                           kGlobalMemorySpace>;
using StoreBF16OpConversion =
    MemoryStoreOpConversion<forth::StoreBF16Op, MemType::BF16,
                            kGlobalMemorySpace>;
using SharedLoadBF16OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadBF16Op, MemType::BF16,
                           kWorkgroupMemorySpace>;
using SharedStoreBF16OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreBF16Op, MemType::BF16,
                            kWorkgroupMemorySpace>;
using LoadF32OpConversion =
    MemoryLoadOpConversion<forth::LoadF32Op, MemType::F32, kGlobalMemorySpace>;
using StoreF32OpConversion =
    MemoryStoreOpConversion<forth::StoreF32Op, MemType::F32,
                            kGlobalMemorySpace>;
using SharedLoadF32OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadF32Op, MemType::F32,
                           kWorkgroupMemorySpace>;
using SharedStoreF32OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreF32Op, MemType::F32,
                            kWorkgroupMemorySpace>;

struct IToFOpConversion : public StackEffectConversion<forth::IToFOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::IToFOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value f64Value = builder.create<arith::SIToFPOp>(
        loc, builder.getF64Type(), stack.loadAt(loc, builder, 0));
    stack.storeAt(loc, builder, 0, bitcastI64(builder, loc, f64Value));
    return success();
  }
};

struct FToIOpConversion : public StackEffectConversion<forth::FToIOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::FToIOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value f64Value = bitcastF64(builder, loc, stack.loadAt(loc, builder, 0));
    Value i64Value =
        builder.create<arith::FPToSIOp>(loc, builder.getI64Type(), f64Value);
    stack.storeAt(loc, builder, 0, i64Value);
    return success();
  }
};

template <typename ForthOp>
struct IntrinsicOpConversion : public StackEffectConversion<ForthOp> {
  using StackEffectConversion<ForthOp>::StackEffectConversion;

  IntrinsicOpConversion(const TypeConverter &typeConverter,
                        MLIRContext *context, StringRef intrinsicName)
      : StackEffectConversion<ForthOp>(typeConverter, context),
        intrinsicName(intrinsicName) {}

  LogicalResult emitStack(ForthOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value intrinsic = builder.create<forth::IntrinsicOp>(
        loc, builder.getIndexType(), builder.getStringAttr(intrinsicName));
    stack.push(loc, builder,
               builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                  intrinsic));
    return success();
  }

  StringRef intrinsicName;
};

struct GlobalIdOpConversion : public StackEffectConversion<forth::GlobalIdOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::GlobalIdOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    auto index = [&](StringRef name) {
      return builder.create<forth::IntrinsicOp>(loc, builder.getIndexType(),
                                                builder.getStringAttr(name));
    };
    Value bidX = index("bid-x");
    Value bdimX = index("bdim-x");
    Value tidX = index("tid-x");
    Value product = builder.create<arith::MulIOp>(loc, bidX, bdimX);
    Value globalId = builder.create<arith::AddIOp>(loc, product, tidX);
    stack.push(loc, builder,
               builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                  globalId));
    return success();
  }
};

static gpu::ShuffleMode toGpuShuffleMode(forth::ShuffleMode mode) {
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

struct ShuffleOpConversion : public StackEffectConversion<forth::ShuffleOp> {
  using StackEffectConversion::StackEffectConversion;

  LogicalResult emitStack(forth::ShuffleOp op, StackCursor &stack,
                          OpBuilder &builder) const override {
    Location loc = op.getLoc();
    Value offsetCell = stack.loadAt(loc, builder, 0);
    Value value = stack.loadAt(loc, builder, 1);
    Value offset32 =
        builder.create<arith::TruncIOp>(loc, builder.getI32Type(), offsetCell);
    Value laneMask = builder.create<arith::ConstantIntOp>(
        loc, forth::kWarpSize - 1, builder.getI32Type());
    Value offset = builder.create<arith::AndIOp>(loc, offset32, laneMask);
    Value width = builder.create<arith::ConstantIntOp>(loc, forth::kWarpSize,
                                                       builder.getI32Type());
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, value, offset, width,
                                                 toGpuShuffleMode(op.getMode()))
                         .getShuffleResult();
    stack.storeAt(loc, builder, 1, shuffled);
    stack.drop(loc, builder, 1);
    return success();
  }
};

static Value popFlag(Operation *op, StackCursor &stack, OpBuilder &builder) {
  Location loc = op->getLoc();
  Value top = stack.loadAt(loc, builder, 0);
  stack.drop(loc, builder, 1);
  Value zero = builder.create<arith::ConstantOp>(loc, builder.getI64Type(),
                                                 builder.getI64IntegerAttr(0));
  return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, top,
                                       zero);
}

struct PopFlagOpConversion : public OpConversionPattern<forth::PopFlagOp>,
                             public KnownRewrite {
  using OpConversionPattern::OpConversionPattern;

  bool matches(Operation *op) const override {
    return isa<forth::PopFlagOp>(op);
  }

  LogicalResult applyKnown(Operation *op, StackCursor &stack,
                           OpBuilder &builder) const override {
    Value flag = popFlag(op, stack, builder);
    cast<forth::PopFlagOp>(op).getFlag().replaceAllUsesWith(flag);
    return success();
  }

  LogicalResult
  matchAndRewrite(forth::PopFlagOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (getKnownDepth(op.getOperation()))
      return failure();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    MemRefStackCursor stack(memref, inputStack[1]);
    Value flag = popFlag(op, stack, rewriter);
    rewriter.replaceOpWithMultiple(op,
                                   {{memref, stack.stackPointer()}, {flag}});
    return success();
  }
};

struct PopOpConversion : public OpConversionPattern<forth::PopOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(forth::PopOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (getKnownDepth(op.getOperation()))
      return failure();
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];
    Value topValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    rewriter.replaceOpWithMultiple(op, {{memref, newSP}, {topValue}});
    return success();
  }
};

struct PushValueOpConversion
    : public MemRefStackConversion<forth::PushValueOp> {
  using MemRefStackConversion::MemRefStackConversion;

  FailureOr<Value>
  rewriteStack(forth::PushValueOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    MemRefStackCursor stack(memref, stackPtr);
    stack.push(op.getLoc(), rewriter, adaptor.getOperands()[1][0]);
    return stack.stackPointer();
  }
};

struct BranchOpConversion : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> newOperands;
    for (ValueRange vals : adaptor.getOperands())
      llvm::append_range(newOperands, vals);
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(), newOperands);
    return success();
  }
};

struct CondBranchOpConversion : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> trueOperands, falseOperands;
    Value condition = adaptor.getOperands()[0][0];
    unsigned trueCount = op.getTrueDestOperands().size();
    for (unsigned i = 1; i <= trueCount; ++i)
      llvm::append_range(trueOperands, adaptor.getOperands()[i]);
    for (unsigned i = 1 + trueCount; i < adaptor.getOperands().size(); ++i)
      llvm::append_range(falseOperands, adaptor.getOperands()[i]);
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, condition, op.getTrueDest(), trueOperands, op.getFalseDest(),
        falseOperands);
    return success();
  }
};

static bool isForthStack(Type type) { return isa<forth::StackType>(type); }

static bool forwardsStack(OperandRange operands) {
  return llvm::any_of(
      operands, [](Value value) { return isForthStack(value.getType()); });
}

static void rewriteDestOperands(MutableOperandRange destOperands, Block *dest,
                                ArrayRef<Value> cells) {
  bool sawStack = false;
  SmallVector<Value, 4> others;
  for (Value operand : destOperands.getAsOperandRange()) {
    if (isForthStack(operand.getType()))
      sawStack = true;
    else
      others.push_back(operand);
  }
  if (!sawStack)
    return;

  unsigned numArgs = dest->getNumArguments();
  unsigned cellArgs = cells.size();
  SmallVector<Value, 8> rewritten;
  unsigned otherIdx = 0;
  unsigned cellIdx = 0;
  for (unsigned i = 0; i < numArgs; ++i) {
    if (isForthStack(dest->getArgument(i).getType()))
      continue;
    if (cellArgs > 0 && i >= numArgs - cellArgs)
      rewritten.push_back(cells[cellIdx++]);
    else
      rewritten.push_back(others[otherIdx++]);
  }
  if (cellIdx != cells.size() || otherIdx != others.size())
    llvm_unreachable("branch operands do not match the destination arguments");
  destOperands.assign(rewritten);
}

static bool forwardsStackTo(Operation *terminator, Block *dest) {
  if (auto branch = dyn_cast<cf::BranchOp>(terminator))
    return branch.getDest() == dest && forwardsStack(branch.getDestOperands());
  if (auto branch = dyn_cast<cf::CondBranchOp>(terminator)) {
    if (branch.getTrueDest() == dest &&
        forwardsStack(branch.getTrueDestOperands()))
      return true;
    if (branch.getFalseDest() == dest &&
        forwardsStack(branch.getFalseDestOperands()))
      return true;
  }
  return false;
}

static std::optional<unsigned> incomingDepth(Block &block) {
  for (Block *pred : block.getPredecessors()) {
    Operation *terminator = pred->getTerminator();
    if (!forwardsStackTo(terminator, &block))
      continue;
    if (std::optional<unsigned> depth = getKnownDepth(terminator))
      return depth;
  }
  return std::nullopt;
}

static bool functionHasKnownDepth(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (getKnownDepth(op))
      found = true;
  });
  return found;
}

static LogicalResult lowerFromDepthAttr(Operation *op,
                                        SmallVectorImpl<Value> &cells,
                                        SSAStackCursor &cursor,
                                        ArrayRef<KnownRewrite *> rewrites,
                                        OpBuilder &builder) {
  std::optional<unsigned> depth = getKnownDepth(op);
  if (!depth)
    return failure();
  if (isa<forth::StackOp>(op)) {
    if (*depth != 0 || !cells.empty())
      return failure();
    cells.clear();
    return success();
  }
  if (cells.size() != *depth)
    return failure();
  if (auto branch = dyn_cast<cf::BranchOp>(op)) {
    rewriteDestOperands(branch.getDestOperandsMutable(), branch.getDest(),
                        cells);
    return success();
  }
  if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
    rewriteDestOperands(branch.getTrueDestOperandsMutable(),
                        branch.getTrueDest(), cells);
    rewriteDestOperands(branch.getFalseDestOperandsMutable(),
                        branch.getFalseDest(), cells);
    return success();
  }
  for (KnownRewrite *rewrite : rewrites) {
    if (!rewrite->matches(op))
      continue;
    builder.setInsertionPoint(op);
    return rewrite->applyKnown(op, cursor, builder);
  }
  return failure();
}

static LogicalResult lowerTaggedOps(func::FuncOp func,
                                    ArrayRef<KnownRewrite *> rewrites) {
  OpBuilder builder(func.getContext());
  Block &entry = func.front();
  DenseMap<Block *, unsigned> incoming;
  for (Block &block : func.getBody()) {
    if (&block == &entry)
      continue;
    if (std::optional<unsigned> depth = incomingDepth(block))
      incoming[&block] = *depth;
  }
  for (auto [block, depth] : incoming) {
    if (depth == 0)
      continue;
    SmallVector<Type, 8> types(depth, builder.getI64Type());
    SmallVector<Location, 8> locs(depth, block->front().getLoc());
    block->addArguments(types, locs);
  }

  for (Block &block : func.getBody()) {
    SmallVector<Value, 8> cells;
    if (auto it = incoming.find(&block); it != incoming.end()) {
      unsigned depth = it->second;
      unsigned numArgs = block.getNumArguments();
      for (unsigned i = numArgs - depth; i < numArgs; ++i)
        cells.push_back(block.getArgument(i));
    }
    SSAStackCursor cursor(cells);

    SmallVector<Operation *, 8> erase;
    for (Operation &op : block) {
      if (!getKnownDepth(&op))
        continue;
      if (failed(lowerFromDepthAttr(&op, cells, cursor, rewrites, builder)))
        return failure();
      if (isa<cf::BranchOp, cf::CondBranchOp>(op))
        op.removeAttr(kForthKnownDepthAttr);
      else
        erase.push_back(&op);
    }
    for (Operation *op : llvm::reverse(erase))
      op->erase();
  }

  for (Block &block : func.getBody()) {
    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      if (!isForthStack(block.getArgument(i).getType()))
        continue;
      block.eraseArgument(i);
      break;
    }
  }
  return success();
}

struct ConvertForthToMemRefPass
    : public impl::ConvertForthToMemRefBase<ConvertForthToMemRefPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto module = getOperation();

    IRRewriter eraser(context);
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp.isExternal())
        return;
      (void)eraseUnreachableBlocks(eraser, funcOp->getRegions());
    });

    ConversionTarget target(*context);
    target.addIllegalDialect<forth::ForthDialect>();
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                           LLVM::LLVMDialect, cf::ControlFlowDialect,
                           math::MathDialect>();
    target.addLegalOp<forth::IntrinsicOp>();
    target.addLegalOp<forth::BarrierOp>();
    target.addLegalOp<gpu::ShuffleOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto isStack = [](Type t) { return isa<forth::StackType>(t); };
      if (llvm::any_of(op.getFunctionType().getInputs(), isStack) ||
          llvm::any_of(op.getFunctionType().getResults(), isStack))
        return false;
      for (Block &block : op.getFunctionBody())
        if (llvm::any_of(block.getArgumentTypes(), isStack))
          return false;
      return true;
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return llvm::none_of(op.getOperandTypes(),
                           [&](Type t) { return isa<forth::StackType>(t); }) &&
             llvm::none_of(op.getResultTypes(),
                           [&](Type t) { return isa<forth::StackType>(t); });
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return llvm::none_of(op.getOperandTypes(),
                           [&](Type t) { return isa<forth::StackType>(t); });
    });
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&](Operation *op) {
          return llvm::none_of(op->getOperandTypes(),
                               [](Type t) { return isa<forth::StackType>(t); });
        });

    ForthToMemRefTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    SmallVector<KnownRewrite *, 64> known;
    auto add = [&](auto pattern) {
      using Pattern = typename std::decay_t<decltype(pattern)>::element_type;
      if constexpr (std::is_base_of<KnownRewrite, Pattern>::value)
        known.push_back(pattern.get());
      patterns.add(std::move(pattern));
    };
    auto make = [&](auto pattern) { add(std::move(pattern)); };

    make(std::make_unique<StackOpConversion>(typeConverter, context));
    make(std::make_unique<ConstantOpConversion>(typeConverter, context));
    make(std::make_unique<DupOpConversion>(typeConverter, context));
    make(std::make_unique<DropOpConversion>(typeConverter, context));
    make(std::make_unique<SwapOpConversion>(typeConverter, context));
    make(std::make_unique<OverOpConversion>(typeConverter, context));
    make(std::make_unique<RotOpConversion>(typeConverter, context));
    make(std::make_unique<NipOpConversion>(typeConverter, context));
    make(std::make_unique<TuckOpConversion>(typeConverter, context));
    make(std::make_unique<PickOpConversion>(typeConverter, context));
    make(std::make_unique<RollOpConversion>(typeConverter, context));
    make(std::make_unique<AddIOpConversion>(typeConverter, context));
    make(std::make_unique<SubIOpConversion>(typeConverter, context));
    make(std::make_unique<MulIOpConversion>(typeConverter, context));
    make(std::make_unique<DivIOpConversion>(typeConverter, context));
    make(std::make_unique<ModOpConversion>(typeConverter, context));
    make(std::make_unique<AddFOpConversion>(typeConverter, context));
    make(std::make_unique<SubFOpConversion>(typeConverter, context));
    make(std::make_unique<MulFOpConversion>(typeConverter, context));
    make(std::make_unique<DivFOpConversion>(typeConverter, context));
    make(std::make_unique<MaxFOpConversion>(typeConverter, context));
    make(std::make_unique<MinFOpConversion>(typeConverter, context));
    make(std::make_unique<ExpFOpConversion>(typeConverter, context));
    make(std::make_unique<SqrtFOpConversion>(typeConverter, context));
    make(std::make_unique<LogFOpConversion>(typeConverter, context));
    make(std::make_unique<AbsFOpConversion>(typeConverter, context));
    make(std::make_unique<NegFOpConversion>(typeConverter, context));
    make(std::make_unique<AndOpConversion>(typeConverter, context));
    make(std::make_unique<OrOpConversion>(typeConverter, context));
    make(std::make_unique<XorOpConversion>(typeConverter, context));
    make(std::make_unique<NotOpConversion>(typeConverter, context));
    make(std::make_unique<LshiftOpConversion>(typeConverter, context));
    make(std::make_unique<RshiftOpConversion>(typeConverter, context));
    make(std::make_unique<EqIOpConversion>(typeConverter, context));
    make(std::make_unique<LtIOpConversion>(typeConverter, context));
    make(std::make_unique<GtIOpConversion>(typeConverter, context));
    make(std::make_unique<NeIOpConversion>(typeConverter, context));
    make(std::make_unique<LeIOpConversion>(typeConverter, context));
    make(std::make_unique<GeIOpConversion>(typeConverter, context));
    make(std::make_unique<EqFOpConversion>(typeConverter, context));
    make(std::make_unique<LtFOpConversion>(typeConverter, context));
    make(std::make_unique<GtFOpConversion>(typeConverter, context));
    make(std::make_unique<NeFOpConversion>(typeConverter, context));
    make(std::make_unique<LeFOpConversion>(typeConverter, context));
    make(std::make_unique<GeFOpConversion>(typeConverter, context));
    make(std::make_unique<ZeroEqOpConversion>(typeConverter, context));
    make(std::make_unique<ParamRefOpConversion>(typeConverter, context));
    make(std::make_unique<LoadIOpConversion>(typeConverter, context));
    make(std::make_unique<StoreIOpConversion>(typeConverter, context));
    make(std::make_unique<LoadFOpConversion>(typeConverter, context));
    make(std::make_unique<StoreFOpConversion>(typeConverter, context));
    make(std::make_unique<SharedLoadIOpConversion>(typeConverter, context));
    make(std::make_unique<SharedStoreIOpConversion>(typeConverter, context));
    make(std::make_unique<SharedLoadFOpConversion>(typeConverter, context));
    make(std::make_unique<SharedStoreFOpConversion>(typeConverter, context));
    make(std::make_unique<LoadI8OpConversion>(typeConverter, context));
    make(std::make_unique<StoreI8OpConversion>(typeConverter, context));
    make(std::make_unique<SharedLoadI8OpConversion>(typeConverter, context));
    make(std::make_unique<SharedStoreI8OpConversion>(typeConverter, context));
    make(std::make_unique<LoadI16OpConversion>(typeConverter, context));
    make(std::make_unique<StoreI16OpConversion>(typeConverter, context));
    make(std::make_unique<SharedLoadI16OpConversion>(typeConverter, context));
    make(std::make_unique<SharedStoreI16OpConversion>(typeConverter, context));
    make(std::make_unique<LoadI32OpConversion>(typeConverter, context));
    make(std::make_unique<StoreI32OpConversion>(typeConverter, context));
    make(std::make_unique<SharedLoadI32OpConversion>(typeConverter, context));
    make(std::make_unique<SharedStoreI32OpConversion>(typeConverter, context));
    make(std::make_unique<LoadF16OpConversion>(typeConverter, context));
    make(std::make_unique<StoreF16OpConversion>(typeConverter, context));
    make(std::make_unique<SharedLoadF16OpConversion>(typeConverter, context));
    make(std::make_unique<SharedStoreF16OpConversion>(typeConverter, context));
    make(std::make_unique<LoadBF16OpConversion>(typeConverter, context));
    make(std::make_unique<StoreBF16OpConversion>(typeConverter, context));
    make(std::make_unique<SharedLoadBF16OpConversion>(typeConverter, context));
    make(std::make_unique<SharedStoreBF16OpConversion>(typeConverter, context));
    make(std::make_unique<LoadF32OpConversion>(typeConverter, context));
    make(std::make_unique<StoreF32OpConversion>(typeConverter, context));
    make(std::make_unique<SharedLoadF32OpConversion>(typeConverter, context));
    make(std::make_unique<SharedStoreF32OpConversion>(typeConverter, context));
    make(std::make_unique<IToFOpConversion>(typeConverter, context));
    make(std::make_unique<FToIOpConversion>(typeConverter, context));
    make(std::make_unique<PopFlagOpConversion>(typeConverter, context));
    make(std::make_unique<PopOpConversion>(typeConverter, context));
    make(std::make_unique<PushValueOpConversion>(typeConverter, context));
    make(std::make_unique<IntrinsicOpConversion<forth::ThreadIdXOp>>(
        typeConverter, context, "tid-x"));
    make(std::make_unique<IntrinsicOpConversion<forth::ThreadIdYOp>>(
        typeConverter, context, "tid-y"));
    make(std::make_unique<IntrinsicOpConversion<forth::ThreadIdZOp>>(
        typeConverter, context, "tid-z"));
    make(std::make_unique<IntrinsicOpConversion<forth::BlockIdXOp>>(
        typeConverter, context, "bid-x"));
    make(std::make_unique<IntrinsicOpConversion<forth::BlockIdYOp>>(
        typeConverter, context, "bid-y"));
    make(std::make_unique<IntrinsicOpConversion<forth::BlockIdZOp>>(
        typeConverter, context, "bid-z"));
    make(std::make_unique<IntrinsicOpConversion<forth::BlockDimXOp>>(
        typeConverter, context, "bdim-x"));
    make(std::make_unique<IntrinsicOpConversion<forth::BlockDimYOp>>(
        typeConverter, context, "bdim-y"));
    make(std::make_unique<IntrinsicOpConversion<forth::BlockDimZOp>>(
        typeConverter, context, "bdim-z"));
    make(std::make_unique<IntrinsicOpConversion<forth::GridDimXOp>>(
        typeConverter, context, "gdim-x"));
    make(std::make_unique<IntrinsicOpConversion<forth::GridDimYOp>>(
        typeConverter, context, "gdim-y"));
    make(std::make_unique<IntrinsicOpConversion<forth::GridDimZOp>>(
        typeConverter, context, "gdim-z"));
    make(std::make_unique<IntrinsicOpConversion<forth::LaneIdOp>>(
        typeConverter, context, "lane-id"));
    make(std::make_unique<GlobalIdOpConversion>(typeConverter, context));
    make(std::make_unique<ShuffleOpConversion>(typeConverter, context));
    make(std::make_unique<BranchOpConversion>(typeConverter, context));
    make(std::make_unique<CondBranchOpConversion>(typeConverter, context));

    WalkResult tagged = module.walk([&](func::FuncOp funcOp) {
      if (funcOp.isExternal() || !functionHasKnownDepth(funcOp))
        return WalkResult::advance();
      if (failed(lowerTaggedOps(funcOp, known)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (tagged.wasInterrupted())
      return signalPassFailure();

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertForthToMemRefPass() {
  return std::make_unique<ConvertForthToMemRefPass>();
}

} // namespace warpforth
} // namespace mlir
