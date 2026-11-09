//===- ForthToMemRef.cpp - Forth to MemRef conversion ----------*- C++ -*-===//
//
// This file implements the conversion from Forth dialect to MemRef dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthToMemRef/ForthToMemRef.h"
#include "WordLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace warpforth {

#define GEN_PASS_DEF_CONVERTFORTHTOMEMREF
#include "warpforth/Conversion/Passes.h.inc"

namespace {

// Stack configuration constants
constexpr int64_t kStackSize = 256;
constexpr const char kKnownDepthAttr[] = "known_depth";

static std::optional<unsigned> getKnownDepth(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>(kKnownDepthAttr);
  if (!attr)
    return std::nullopt;
  return static_cast<unsigned>(attr.getInt());
}
/// Type converter for forth.stack -> memref + index
class ForthToMemRefTypeConverter : public TypeConverter {
public:
  ForthToMemRefTypeConverter() {
    // Pass-through for all non-stack types (must be registered first)
    addConversion([](Type type) { return type; });

    // Stack type: !forth.stack -> memref<256xi64> + index
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

/// Push an i64 value onto the stack. Returns the new stack pointer.
static Value pushValue(Location loc, ConversionPatternRewriter &rewriter,
                       Value memref, Value stackPtr, Value value) {
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);
  rewriter.create<memref::StoreOp>(loc, value, memref, newSP);
  return newSP;
}

/// Base for patterns that rewrite one !forth.stack operand into one output
/// stack. Subclasses emit the stack effect on the unpacked (memref, SP) pair
/// and return the new SP.
template <typename ForthOp>
struct StackEffectConversion : public OpConversionPattern<ForthOp> {
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

/// Conversion pattern for forth.stack operation.
/// Allocates a memref<256xi64> on the stack and initializes SP to 0.
struct StackOpConversion : public OpConversionPattern<forth::StackOp> {
  StackOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::StackOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::StackOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();
    auto memrefType = MemRefType::get({kStackSize}, i64Type);

    // Allocate stack buffer
    Value stackBuffer = rewriter.create<memref::AllocaOp>(loc, memrefType);

    // Initialize stack pointer to 0
    Value initialSP = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    rewriter.replaceOpWithMultiple(op, {{stackBuffer, initialSP}});
    return success();
  }
};

/// Conversion pattern for forth.constant operation.
/// Handles both integer and float constants.
struct ConstantOpConversion : public StackEffectConversion<forth::ConstantOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::ConstantOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, ValueRange(), produced, rewriter);
    return pushValue(loc, rewriter, memref, stackPtr, produced.front());
  }
};

/// Conversion pattern for forth.dup operation.
/// Duplicates the value at the top of the stack: (a -- a a)
struct DupOpConversion : public StackEffectConversion<forth::DupOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::DupOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value topValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    SmallVector<Value, 2> produced;
    lookupWord(op)->compute(op, topValue, produced, rewriter);
    return pushValue(loc, rewriter, memref, stackPtr, produced.back());
  }
};

/// Conversion pattern for forth.drop operation.
/// Removes the top element from the stack: (a -- )
struct DropOpConversion : public StackEffectConversion<forth::DropOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::DropOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, ValueRange(), produced, rewriter);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    return newSP;
  }
};

/// Conversion pattern for forth.swap operation.
/// Swaps the top two elements: (a b -- b a)
struct SwapOpConversion : public StackEffectConversion<forth::SwapOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::SwapOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value topValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value secondValue = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);
    Value cells[] = {secondValue, topValue};
    SmallVector<Value, 2> produced;
    lookupWord(op)->compute(op, cells, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced[1], memref, stackPtr);
    rewriter.create<memref::StoreOp>(loc, produced[0], memref, spMinus1);

    return stackPtr;
  }
};

/// Conversion pattern for forth.over operation.
/// Copies the second element to top: (a b -- a b a)
struct OverOpConversion : public StackEffectConversion<forth::OverOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::OverOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value secondValue = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);
    Value topValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value cells[] = {secondValue, topValue};
    SmallVector<Value, 3> produced;
    lookupWord(op)->compute(op, cells, produced, rewriter);
    return pushValue(loc, rewriter, memref, stackPtr, produced.back());
  }
};

/// Conversion pattern for forth.rot operation.
/// Rotates top three elements: (a b c -- b c a)
struct RotOpConversion : public StackEffectConversion<forth::RotOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::RotOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);

    Value c = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value b = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);
    Value spMinus2 = rewriter.create<arith::SubIOp>(loc, stackPtr, two);
    Value a = rewriter.create<memref::LoadOp>(loc, memref, spMinus2);
    Value cells[] = {a, b, c};
    SmallVector<Value, 3> produced;
    lookupWord(op)->compute(op, cells, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced[0], memref, spMinus2);
    rewriter.create<memref::StoreOp>(loc, produced[1], memref, spMinus1);
    rewriter.create<memref::StoreOp>(loc, produced[2], memref, stackPtr);

    return stackPtr;
  }
};

/// Conversion pattern for forth.nip operation.
/// Removes the second element: (a b -- b)
struct NipOpConversion : public StackEffectConversion<forth::NipOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::NipOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value b = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value a = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);
    Value cells[] = {a, b};
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, cells, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced[0], memref, spMinus1);
    return spMinus1;
  }
};

/// Conversion pattern for forth.tuck operation.
/// Copies top before second: (a b -- b a b)
struct TuckOpConversion : public StackEffectConversion<forth::TuckOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::TuckOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value b = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value a = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);
    Value cells[] = {a, b};
    SmallVector<Value, 3> produced;
    lookupWord(op)->compute(op, cells, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced[0], memref, spMinus1);
    rewriter.create<memref::StoreOp>(loc, produced[1], memref, stackPtr);
    Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);
    rewriter.create<memref::StoreOp>(loc, produced[2], memref, newSP);
    return newSP;
  }
};

/// Conversion pattern for forth.pick operation.
/// Copies nth element to top: ( xn ... x0 n -- xn ... x0 xn )
struct PickOpConversion : public StackEffectConversion<forth::PickOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::PickOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Pop n from stack
    Value nI64 = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spAfterPop = rewriter.create<arith::SubIOp>(loc, stackPtr, one);

    // Cast n to index
    Value nIdx =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), nI64);

    // Compute target address: SP' - n
    Value targetAddr = rewriter.create<arith::SubIOp>(loc, spAfterPop, nIdx);

    // Load the picked value
    Value pickedValue =
        rewriter.create<memref::LoadOp>(loc, memref, targetAddr);

    // Store at SP (where n was), effectively pushing the picked value
    rewriter.create<memref::StoreOp>(loc, pickedValue, memref, stackPtr);

    // Net effect: SP unchanged (popped n, pushed xn)
    return stackPtr;
  }
};

/// Conversion pattern for forth.roll operation.
/// Rotates nth element to top: ( xn ... x0 n -- xn-1 ... x0 xn )
/// Uses a CF-based loop to shift elements down.
struct RollOpConversion : public StackEffectConversion<forth::RollOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::RollOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto indexType = rewriter.getIndexType();

    // Pop n from stack
    Value nI64 = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spAfterPop = rewriter.create<arith::SubIOp>(loc, stackPtr, one);

    // Cast n to index
    Value nIdx = rewriter.create<arith::IndexCastOp>(loc, indexType, nI64);

    // Compute address of the element to roll: SP' - n
    Value rolledAddr = rewriter.create<arith::SubIOp>(loc, spAfterPop, nIdx);

    // Save the value to be rolled to top
    Value rolledValue =
        rewriter.create<memref::LoadOp>(loc, memref, rolledAddr);

    // Split block to create CF-based shift loop.
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, op->getIterator());

    // Create loop header and body blocks (inserted before continueBlock).
    Block *headerBlock = rewriter.createBlock(continueBlock);
    headerBlock->addArgument(indexType, loc); // induction variable
    Block *bodyBlock = rewriter.createBlock(continueBlock);
    bodyBlock->addArgument(indexType, loc); // induction variable

    // currentBlock -> headerBlock(rolledAddr)
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<cf::BranchOp>(loc, headerBlock, ValueRange{rolledAddr});

    // headerBlock: check iv < spAfterPop
    rewriter.setInsertionPointToStart(headerBlock);
    Value iv = headerBlock->getArgument(0);
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                iv, spAfterPop);
    rewriter.create<cf::CondBranchOp>(loc, cond, bodyBlock, ValueRange{iv},
                                      continueBlock, ValueRange{});

    // bodyBlock: shift memref[iv] = memref[iv+1], branch back to header
    rewriter.setInsertionPointToStart(bodyBlock);
    Value biv = bodyBlock->getArgument(0);
    Value next = rewriter.create<arith::AddIOp>(loc, biv, one);
    Value shiftedVal = rewriter.create<memref::LoadOp>(loc, memref, next);
    rewriter.create<memref::StoreOp>(loc, shiftedVal, memref, biv);
    rewriter.create<cf::BranchOp>(loc, headerBlock, ValueRange{next});

    // continueBlock: store rolled value at top, then rest of original block
    rewriter.setInsertionPoint(op);
    rewriter.create<memref::StoreOp>(loc, rolledValue, memref, spAfterPop);

    // Net effect: SP' = SP - 1 (consumed n)
    return spAfterPop;
  }
};

template <typename ForthOp>
struct BinaryArithOpConversion : public StackEffectConversion<ForthOp> {
  using Base = StackEffectConversion<ForthOp>;
  using Base::Base;
  using typename Base::OneToNOpAdaptor;

  FailureOr<Value>
  rewriteStack(ForthOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Load top two values (b at SP, a at SP-1)
    Value b = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value a = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    Value cells[] = {a, b};
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, cells, produced, rewriter);
    Value result = produced.front();

    // Store result at SP-1 (effectively popping both and pushing result)
    rewriter.create<memref::StoreOp>(loc, result, memref, spMinus1);

    // New SP is SP-1 (net effect: two pops, one push)
    return spMinus1;
  }
};

using AddIOpConversion = BinaryArithOpConversion<forth::AddIOp>;
using SubIOpConversion = BinaryArithOpConversion<forth::SubIOp>;
using MulIOpConversion = BinaryArithOpConversion<forth::MulIOp>;
using DivIOpConversion = BinaryArithOpConversion<forth::DivIOp>;
using ModOpConversion = BinaryArithOpConversion<forth::ModOp>;
using AndOpConversion = BinaryArithOpConversion<forth::AndOp>;
using OrOpConversion = BinaryArithOpConversion<forth::OrOp>;
using XorOpConversion = BinaryArithOpConversion<forth::XorOp>;
using LshiftOpConversion = BinaryArithOpConversion<forth::LshiftOp>;
using RshiftOpConversion = BinaryArithOpConversion<forth::RshiftOp>;
using AddFOpConversion = BinaryArithOpConversion<forth::AddFOp>;
using SubFOpConversion = BinaryArithOpConversion<forth::SubFOp>;
using MulFOpConversion = BinaryArithOpConversion<forth::MulFOp>;
using DivFOpConversion = BinaryArithOpConversion<forth::DivFOp>;
using MaxFOpConversion = BinaryArithOpConversion<forth::MaxFOp>;
using MinFOpConversion = BinaryArithOpConversion<forth::MinFOp>;

template <typename ForthOp>
struct UnaryFloatOpConversion : public StackEffectConversion<ForthOp> {
  using Base = StackEffectConversion<ForthOp>;
  using Base::Base;
  using typename Base::OneToNOpAdaptor;

  FailureOr<Value>
  rewriteStack(ForthOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value a = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, a, produced, rewriter);
    Value result = produced.front();
    rewriter.create<memref::StoreOp>(loc, result, memref, stackPtr);

    return stackPtr;
  }
};

using ExpFOpConversion = UnaryFloatOpConversion<forth::ExpFOp>;
using SqrtFOpConversion = UnaryFloatOpConversion<forth::SqrtFOp>;
using LogFOpConversion = UnaryFloatOpConversion<forth::LogFOp>;
using AbsFOpConversion = UnaryFloatOpConversion<forth::AbsFOp>;
using NegFOpConversion = UnaryFloatOpConversion<forth::NegFOp>;

/// Pops two values, compares, pushes -1 (true) or 0 (false): (a b -- flag)
template <typename ForthOp>
struct BinaryCmpOpConversion : public StackEffectConversion<ForthOp> {
  using Base = StackEffectConversion<ForthOp>;
  using Base::Base;
  using typename Base::OneToNOpAdaptor;

  FailureOr<Value>
  rewriteStack(ForthOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Load top two values (b at SP, a at SP-1)
    Value b = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value a = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    Value cells[] = {a, b};
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, cells, produced, rewriter);
    Value result = produced.front();

    // Store result at SP-1
    rewriter.create<memref::StoreOp>(loc, result, memref, spMinus1);

    // New SP is SP-1 (net: two pops, one push)
    return spMinus1;
  }
};

using EqIOpConversion = BinaryCmpOpConversion<forth::EqIOp>;
using LtIOpConversion = BinaryCmpOpConversion<forth::LtIOp>;
using GtIOpConversion = BinaryCmpOpConversion<forth::GtIOp>;
using NeIOpConversion = BinaryCmpOpConversion<forth::NeIOp>;
using LeIOpConversion = BinaryCmpOpConversion<forth::LeIOp>;
using GeIOpConversion = BinaryCmpOpConversion<forth::GeIOp>;
using EqFOpConversion = BinaryCmpOpConversion<forth::EqFOp>;
using LtFOpConversion = BinaryCmpOpConversion<forth::LtFOp>;
using GtFOpConversion = BinaryCmpOpConversion<forth::GtFOp>;
using NeFOpConversion = BinaryCmpOpConversion<forth::NeFOp>;
using LeFOpConversion = BinaryCmpOpConversion<forth::LeFOp>;
using GeFOpConversion = BinaryCmpOpConversion<forth::GeFOp>;

/// Conversion pattern for forth.not operation (bitwise NOT).
/// Unary: pops one value, XORs with -1 (all bits set), pushes result: (a -- ~a)
struct NotOpConversion : public StackEffectConversion<forth::NotOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::NotOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value a = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, a, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced.front(), memref, stackPtr);

    return stackPtr;
  }
};

/// Conversion pattern for forth.zero_eq operation (0=).
/// Unary: pops one value, pushes -1 if zero, 0 otherwise: (a -- flag)
struct ZeroEqOpConversion : public StackEffectConversion<forth::ZeroEqOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::ZeroEqOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value a = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, a, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced.front(), memref, stackPtr);

    return stackPtr;
  }
};

/// Conversion pattern for forth.param_ref operation.
/// Pushes an array parameter's byte address or a scalar parameter's value.
struct ParamRefOpConversion : public StackEffectConversion<forth::ParamRefOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::ParamRefOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value parameter = adaptor.getOperands()[1].front();
    if (!isa<MemRefType>(parameter.getType()) &&
        !parameter.getType().isInteger(64) && !parameter.getType().isF64())
      return rewriter.notifyMatchFailure(
          op, "unsupported param argument type for param_ref");

    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, ValueRange(), produced, rewriter);
    return pushValue(loc, rewriter, memref, stackPtr, produced.front());
  }
};

template <typename ForthOp>
struct MemoryLoadOpConversion : public StackEffectConversion<ForthOp> {
  using Base = StackEffectConversion<ForthOp>;
  using Base::Base;
  using typename Base::OneToNOpAdaptor;

  FailureOr<Value>
  rewriteStack(ForthOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value addrValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, addrValue, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced.front(), memref, stackPtr);
    return stackPtr;
  }
};

using LoadIOpConversion = MemoryLoadOpConversion<forth::LoadIOp>;
using LoadFOpConversion = MemoryLoadOpConversion<forth::LoadFOp>;
using SharedLoadIOpConversion = MemoryLoadOpConversion<forth::SharedLoadIOp>;
using SharedLoadFOpConversion = MemoryLoadOpConversion<forth::SharedLoadFOp>;
using LoadI8OpConversion = MemoryLoadOpConversion<forth::LoadI8Op>;
using SharedLoadI8OpConversion = MemoryLoadOpConversion<forth::SharedLoadI8Op>;
using LoadI16OpConversion = MemoryLoadOpConversion<forth::LoadI16Op>;
using SharedLoadI16OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadI16Op>;
using LoadI32OpConversion = MemoryLoadOpConversion<forth::LoadI32Op>;
using SharedLoadI32OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadI32Op>;
using LoadF16OpConversion = MemoryLoadOpConversion<forth::LoadF16Op>;
using SharedLoadF16OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadF16Op>;
using LoadBF16OpConversion = MemoryLoadOpConversion<forth::LoadBF16Op>;
using SharedLoadBF16OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadBF16Op>;
using LoadF32OpConversion = MemoryLoadOpConversion<forth::LoadF32Op>;
using SharedLoadF32OpConversion =
    MemoryLoadOpConversion<forth::SharedLoadF32Op>;

template <typename ForthOp>
struct MemoryStoreOpConversion : public StackEffectConversion<ForthOp> {
  using Base = StackEffectConversion<ForthOp>;
  using Base::Base;
  using typename Base::OneToNOpAdaptor;

  FailureOr<Value>
  rewriteStack(ForthOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value addrValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value value = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);
    Value cells[] = {value, addrValue};
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, cells, produced, rewriter);
    Value spMinus2 = rewriter.create<arith::SubIOp>(loc, spMinus1, one);
    return spMinus2;
  }
};

using StoreIOpConversion = MemoryStoreOpConversion<forth::StoreIOp>;
using StoreFOpConversion = MemoryStoreOpConversion<forth::StoreFOp>;
using SharedStoreIOpConversion = MemoryStoreOpConversion<forth::SharedStoreIOp>;
using SharedStoreFOpConversion = MemoryStoreOpConversion<forth::SharedStoreFOp>;
using StoreI8OpConversion = MemoryStoreOpConversion<forth::StoreI8Op>;
using SharedStoreI8OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreI8Op>;
using StoreI16OpConversion = MemoryStoreOpConversion<forth::StoreI16Op>;
using SharedStoreI16OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreI16Op>;
using StoreI32OpConversion = MemoryStoreOpConversion<forth::StoreI32Op>;
using SharedStoreI32OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreI32Op>;
using StoreF16OpConversion = MemoryStoreOpConversion<forth::StoreF16Op>;
using SharedStoreF16OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreF16Op>;
using StoreBF16OpConversion = MemoryStoreOpConversion<forth::StoreBF16Op>;
using SharedStoreBF16OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreBF16Op>;
using StoreF32OpConversion = MemoryStoreOpConversion<forth::StoreF32Op>;
using SharedStoreF32OpConversion =
    MemoryStoreOpConversion<forth::SharedStoreF32Op>;

/// Conversion pattern for forth.itof (S>F).
/// Pops i64, converts to f64 via sitofp, bitcasts back to i64, pushes.
struct IToFOpConversion : public StackEffectConversion<forth::IToFOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::IToFOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value i64Value = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, i64Value, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced.front(), memref, stackPtr);
    return stackPtr;
  }
};

/// Conversion pattern for forth.ftoi (F>S).
/// Pops i64 (f64 bits), bitcasts to f64, converts to i64 via fptosi, pushes.
struct FToIOpConversion : public StackEffectConversion<forth::FToIOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::FToIOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value i64Bits = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, i64Bits, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced.front(), memref, stackPtr);
    return stackPtr;
  }
};

template <typename ForthOp>
struct IntrinsicOpConversion : public StackEffectConversion<ForthOp> {
  using Base = StackEffectConversion<ForthOp>;
  using Base::Base;
  using typename Base::OneToNOpAdaptor;

  FailureOr<Value>
  rewriteStack(ForthOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, ValueRange(), produced, rewriter);
    return pushValue(loc, rewriter, memref, stackPtr, produced.front());
  }
};

/// Conversion pattern for forth.global_id operation.
/// Computes global_id = bid-x * bdim-x + tid-x using intrinsics.
struct GlobalIdOpConversion : public StackEffectConversion<forth::GlobalIdOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::GlobalIdOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, ValueRange(), produced, rewriter);
    return pushValue(loc, rewriter, memref, stackPtr, produced.front());
  }
};

struct ShuffleOpConversion : public StackEffectConversion<forth::ShuffleOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::ShuffleOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value b = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value x = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);
    Value cells[] = {x, b};
    SmallVector<Value, 1> produced;
    lookupWord(op)->compute(op, cells, produced, rewriter);
    rewriter.create<memref::StoreOp>(loc, produced.front(), memref, spMinus1);
    return spMinus1;
  }
};

/// Conversion pattern for forth.pop_flag operation.
/// Pops top of stack, compares != 0, returns (memref, newSP, i1 flag).
struct PopFlagOpConversion : public OpConversionPattern<forth::PopFlagOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(forth::PopFlagOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    Value topValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::SubIOp>(loc, stackPtr, one);

    SmallVector<Value, 2> produced;
    lookupWord(op)->compute(op, topValue, produced, rewriter);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}, {produced[0]}});
    return success();
  }
};

/// Conversion pattern for forth.pop operation.
/// Pops top of stack, returns (memref, newSP, i64 value).
struct PopOpConversion : public OpConversionPattern<forth::PopOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(forth::PopOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load top value
    Value topValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Decrement SP
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::SubIOp>(loc, stackPtr, one);

    // Result 0: output_stack -> {memref, newSP}
    // Result 1: value -> i64 (passes through unchanged)
    rewriter.replaceOpWithMultiple(op, {{memref, newSP}, {topValue}});
    return success();
  }
};

/// Conversion pattern for forth.push_value operation.
/// Pushes a dynamic i64 value onto the stack.
struct PushValueOpConversion
    : public StackEffectConversion<forth::PushValueOp> {
  using StackEffectConversion::StackEffectConversion;

  FailureOr<Value>
  rewriteStack(forth::PushValueOp op, OneToNOpAdaptor adaptor, Value memref,
               Value stackPtr,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // The value operand (i64) passes through as-is (not a converted type).
    Value value = adaptor.getOperands()[1][0];

    Value newSP = pushValue(loc, rewriter, memref, stackPtr, value);

    return newSP;
  }
};

/// Conversion pattern for cf::BranchOp with 1:N type conversion.
/// The built-in populateBranchOpInterfaceTypeConversionPattern uses the old
/// ArrayRef<Value> signature and crashes on 1:N conversions.
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

/// Conversion pattern for cf::CondBranchOp with 1:N type conversion.
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

static bool isForthStack(Type type);
static unsigned countStackArgs(Block *block);

struct KnownStack {
  llvm::SmallVector<Value, 8> cells;

  unsigned depth() const { return cells.size(); }
  void push(Value cell) { cells.push_back(cell); }
  void pushCells(ArrayRef<Value> pushed) {
    cells.append(pushed.begin(), pushed.end());
  }
  ArrayRef<Value> topCells(unsigned n) const {
    return ArrayRef<Value>(cells).take_back(n);
  }
  void dropTop(unsigned n) { cells.truncate(cells.size() - n); }
  void clear() { cells.clear(); }
};

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
    const WordLowering *word = lookupWord(&op);
    if (!word || !depth.has_value() || *depth < word->pops)
      return false;
    tags.push_back({&op, *depth});
    depth = *depth - word->pops + word->pushes;
  }
  return true;
}

static bool tagKnownDepths(func::FuncOp func) {
  if (hasStackSignature(func))
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
    op->setAttr(kKnownDepthAttr, IntegerAttr::get(i64, depth));
  return true;
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

static void applyWord(Operation &op, const WordLowering *word,
                      KnownStack &stack, OpBuilder &builder) {
  SmallVector<Value, 4> inputs(stack.topCells(word->pops));
  SmallVector<Value, 4> outputs;
  builder.setInsertionPoint(&op);
  word->compute(&op, inputs, outputs, builder);
  unsigned expected = word->pushes;
  for (Value result : op.getResults())
    if (!isForthStack(result.getType()))
      ++expected;
  if (outputs.size() != expected)
    llvm_unreachable("word compute returned the wrong number of values");
  stack.dropTop(word->pops);
  stack.pushCells(ArrayRef<Value>(outputs).take_front(word->pushes));
  unsigned extra = word->pushes;
  for (Value result : op.getResults()) {
    if (isForthStack(result.getType()))
      continue;
    result.replaceAllUsesWith(outputs[extra]);
    ++extra;
  }
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

static LogicalResult lowerFromDepthAttr(Operation *op, KnownStack &stack,
                                        OpBuilder &builder) {
  std::optional<unsigned> depth = getKnownDepth(op);
  if (!depth)
    return failure();
  if (isa<forth::StackOp>(op)) {
    if (*depth != 0 || stack.depth() != 0)
      return failure();
    stack.clear();
    return success();
  }
  if (stack.depth() != *depth)
    return failure();
  if (auto branch = dyn_cast<cf::BranchOp>(op)) {
    rewriteDestOperands(branch.getDestOperandsMutable(), branch.getDest(),
                        stack.cells);
    return success();
  }
  if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
    rewriteDestOperands(branch.getTrueDestOperandsMutable(),
                        branch.getTrueDest(), stack.cells);
    rewriteDestOperands(branch.getFalseDestOperandsMutable(),
                        branch.getFalseDest(), stack.cells);
    return success();
  }
  const WordLowering *word = lookupWord(op);
  if (!word || *depth < word->pops)
    return failure();
  applyWord(*op, word, stack, builder);
  return success();
}

static LogicalResult lowerTaggedOps(func::FuncOp func) {
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
    KnownStack stack;
    if (auto it = incoming.find(&block); it != incoming.end()) {
      unsigned depth = it->second;
      unsigned numArgs = block.getNumArguments();
      for (unsigned i = numArgs - depth; i < numArgs; ++i)
        stack.push(block.getArgument(i));
    }

    SmallVector<Operation *, 8> erase;
    for (Operation &op : block) {
      if (!getKnownDepth(&op))
        continue;
      if (failed(lowerFromDepthAttr(&op, stack, builder)))
        return failure();
      if (isa<cf::BranchOp, cf::CondBranchOp>(op))
        op.removeAttr(kKnownDepthAttr);
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

/// Conversion pass implementation.
struct ConvertForthToMemRefPass
    : public impl::ConvertForthToMemRefBase<ConvertForthToMemRefPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto module = getOperation();

    // Dialect conversion does not rewrite unreachable blocks. Remove parser
    // continuations before converting !forth.stack values 1:N.
    IRRewriter rewriter(context);
    module.walk([&](func::FuncOp funcOp) {
      (void)eraseUnreachableBlocks(rewriter, funcOp->getRegions());
    });

    WalkResult tagged = module.walk([&](func::FuncOp funcOp) {
      if (tagKnownDepths(funcOp) && failed(lowerTaggedOps(funcOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (tagged.wasInterrupted())
      return signalPassFailure();

    ConversionTarget target(*context);

    // Mark Forth dialect as illegal (to be converted)
    target.addIllegalDialect<forth::ForthDialect>();

    // Mark MemRef, Arith, LLVM, and CF dialects as legal
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                           LLVM::LLVMDialect, cf::ControlFlowDialect,
                           math::MathDialect>();

    target.addLegalOp<forth::IntrinsicOp>();
    target.addLegalOp<forth::BarrierOp>();
    target.addLegalOp<gpu::ShuffleOp>();

    // Use dynamic legality for func operations to ensure they're properly
    // converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto isStack = [](Type t) { return isa<forth::StackType>(t); };
      if (llvm::any_of(op.getFunctionType().getInputs(), isStack) ||
          llvm::any_of(op.getFunctionType().getResults(), isStack))
        return false;
      // Also check non-entry block args (CF control flow)
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

    // CF ops are legal but need dynamic legality for type-converted block args
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&](Operation *op) {
          return llvm::none_of(op->getOperandTypes(),
                               [](Type t) { return isa<forth::StackType>(t); });
        });

    ForthToMemRefTypeConverter typeConverter;
    RewritePatternSet patterns(context);

    // Add Forth operation conversion patterns
    patterns.add<
        StackOpConversion, ConstantOpConversion, DupOpConversion,
        DropOpConversion, SwapOpConversion, OverOpConversion, RotOpConversion,
        NipOpConversion, TuckOpConversion, PickOpConversion, RollOpConversion,
        // Integer arithmetic
        AddIOpConversion, SubIOpConversion, MulIOpConversion, DivIOpConversion,
        ModOpConversion,
        // Float arithmetic
        AddFOpConversion, SubFOpConversion, MulFOpConversion, DivFOpConversion,
        // Float math intrinsics
        ExpFOpConversion, SqrtFOpConversion, LogFOpConversion, AbsFOpConversion,
        NegFOpConversion, MaxFOpConversion, MinFOpConversion,
        // Bitwise
        AndOpConversion, OrOpConversion, XorOpConversion, NotOpConversion,
        LshiftOpConversion, RshiftOpConversion,
        // Integer comparisons
        EqIOpConversion, LtIOpConversion, GtIOpConversion, NeIOpConversion,
        LeIOpConversion, GeIOpConversion,
        // Float comparisons
        EqFOpConversion, LtFOpConversion, GtFOpConversion, NeFOpConversion,
        LeFOpConversion, GeFOpConversion,
        // Other
        ZeroEqOpConversion, ParamRefOpConversion,
        // Memory ops (int + float, global + shared)
        LoadIOpConversion, StoreIOpConversion, LoadFOpConversion,
        StoreFOpConversion, SharedLoadIOpConversion, SharedStoreIOpConversion,
        SharedLoadFOpConversion, SharedStoreFOpConversion,
        // Reduced-width memory ops
        LoadI8OpConversion, StoreI8OpConversion, SharedLoadI8OpConversion,
        SharedStoreI8OpConversion, LoadI16OpConversion, StoreI16OpConversion,
        SharedLoadI16OpConversion, SharedStoreI16OpConversion,
        LoadI32OpConversion, StoreI32OpConversion, SharedLoadI32OpConversion,
        SharedStoreI32OpConversion, LoadF16OpConversion, StoreF16OpConversion,
        SharedLoadF16OpConversion, SharedStoreF16OpConversion,
        LoadBF16OpConversion, StoreBF16OpConversion, SharedLoadBF16OpConversion,
        SharedStoreBF16OpConversion, LoadF32OpConversion, StoreF32OpConversion,
        SharedLoadF32OpConversion, SharedStoreF32OpConversion,
        // Type conversions
        IToFOpConversion, FToIOpConversion,
        // Control flow
        PopFlagOpConversion, PopOpConversion, PushValueOpConversion>(
        typeConverter, context);

    // Add GPU indexing op conversion patterns
    patterns.add<IntrinsicOpConversion<forth::ThreadIdXOp>,
                 IntrinsicOpConversion<forth::ThreadIdYOp>,
                 IntrinsicOpConversion<forth::ThreadIdZOp>,
                 IntrinsicOpConversion<forth::BlockIdXOp>,
                 IntrinsicOpConversion<forth::BlockIdYOp>,
                 IntrinsicOpConversion<forth::BlockIdZOp>,
                 IntrinsicOpConversion<forth::BlockDimXOp>,
                 IntrinsicOpConversion<forth::BlockDimYOp>,
                 IntrinsicOpConversion<forth::BlockDimZOp>,
                 IntrinsicOpConversion<forth::GridDimXOp>,
                 IntrinsicOpConversion<forth::GridDimYOp>,
                 IntrinsicOpConversion<forth::GridDimZOp>,
                 IntrinsicOpConversion<forth::LaneIdOp>>(typeConverter,
                                                         context);

    // GlobalIdOp has custom pattern
    patterns.add<GlobalIdOpConversion>(typeConverter, context);

    patterns.add<ShuffleOpConversion>(typeConverter, context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    // Custom branch patterns are required for 1:N type conversion.
    patterns.add<BranchOpConversion, CondBranchOpConversion>(typeConverter,
                                                             context);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertForthToMemRefPass() {
  return std::make_unique<::mlir::warpforth::ConvertForthToMemRefPass>();
}

} // namespace warpforth
} // namespace mlir
