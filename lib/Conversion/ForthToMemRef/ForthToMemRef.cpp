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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"

namespace mlir {
namespace warpforth {

#define GEN_PASS_DEF_CONVERTFORTHTOMEMREF
#include "warpforth/Conversion/Passes.h.inc"

namespace {

// Stack configuration constants
constexpr int64_t kStackSize = 256;
constexpr unsigned kWorkgroupAddressSpace =
    static_cast<unsigned>(gpu::AddressSpace::Workgroup);

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
struct ConstantOpConversion : public OpConversionPattern<forth::ConstantOp> {
  ConstantOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::ConstantOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::ConstantOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    Value valueToPush;
    auto typedValue = cast<TypedAttr>(op.getValueAttr());
    if (isa<FloatAttr>(typedValue)) {
      // Float: create f64 constant, bitcast to i64
      Value f64Value = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF64Type(), typedValue);
      valueToPush = rewriter.create<arith::BitcastOp>(
          loc, rewriter.getI64Type(), f64Value);
    } else {
      // Integer: create i64 constant directly
      valueToPush = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64Type(), typedValue);
    }
    Value newSP = pushValue(loc, rewriter, memref, stackPtr, valueToPush);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Conversion pattern for forth.dup operation.
/// Duplicates the value at the top of the stack: (a -- a a)
struct DupOpConversion : public OpConversionPattern<forth::DupOp> {
  DupOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::DupOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::DupOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load value at current SP and push duplicate
    Value topValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value newSP = pushValue(loc, rewriter, memref, stackPtr, topValue);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Conversion pattern for forth.drop operation.
/// Removes the top element from the stack: (a -- )
struct DropOpConversion : public OpConversionPattern<forth::DropOp> {
  DropOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::DropOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::DropOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Decrement SP
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::SubIOp>(loc, stackPtr, one);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Conversion pattern for forth.swap operation.
/// Swaps the top two elements: (a b -- b a)
struct SwapOpConversion : public OpConversionPattern<forth::SwapOp> {
  SwapOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::SwapOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::SwapOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load top two values
    Value topValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value secondValue = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    // Store swapped
    rewriter.create<memref::StoreOp>(loc, secondValue, memref, stackPtr);
    rewriter.create<memref::StoreOp>(loc, topValue, memref, spMinus1);

    rewriter.replaceOpWithMultiple(op, {{memref, stackPtr}});
    return success();
  }
};

/// Conversion pattern for forth.over operation.
/// Copies the second element to top: (a b -- a b a)
struct OverOpConversion : public OpConversionPattern<forth::OverOp> {
  OverOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::OverOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::OverOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load second element (SP - 1) and push copy
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value secondValue = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);
    Value newSP = pushValue(loc, rewriter, memref, stackPtr, secondValue);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Conversion pattern for forth.rot operation.
/// Rotates top three elements: (a b c -- b c a)
struct RotOpConversion : public OpConversionPattern<forth::RotOp> {
  RotOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::RotOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::RotOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);

    // Load top three values
    Value c = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value b = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    Value spMinus2 = rewriter.create<arith::SubIOp>(loc, stackPtr, two);
    Value a = rewriter.create<memref::LoadOp>(loc, memref, spMinus2);

    // Store rotated: (a b c -- b c a)
    rewriter.create<memref::StoreOp>(loc, b, memref, spMinus2);
    rewriter.create<memref::StoreOp>(loc, c, memref, spMinus1);
    rewriter.create<memref::StoreOp>(loc, a, memref, stackPtr);

    rewriter.replaceOpWithMultiple(op, {{memref, stackPtr}});
    return success();
  }
};

/// Conversion pattern for forth.nip operation.
/// Removes the second element: (a b -- b)
struct NipOpConversion : public OpConversionPattern<forth::NipOp> {
  NipOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::NipOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::NipOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load top value (b at SP)
    Value b = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Store b at SP-1 (overwriting a)
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    rewriter.create<memref::StoreOp>(loc, b, memref, spMinus1);

    // Net effect: SP-1 (removed one element)
    rewriter.replaceOpWithMultiple(op, {{memref, spMinus1}});
    return success();
  }
};

/// Conversion pattern for forth.tuck operation.
/// Copies top before second: (a b -- b a b)
struct TuckOpConversion : public OpConversionPattern<forth::TuckOp> {
  TuckOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::TuckOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::TuckOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Load top two values
    Value b = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value a = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    // Store: b at SP-1, a at SP, b at SP+1
    rewriter.create<memref::StoreOp>(loc, b, memref, spMinus1);
    rewriter.create<memref::StoreOp>(loc, a, memref, stackPtr);
    Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);
    rewriter.create<memref::StoreOp>(loc, b, memref, newSP);

    // Net effect: SP+1 (added one element)
    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Conversion pattern for forth.pick operation.
/// Copies nth element to top: ( xn ... x0 n -- xn ... x0 xn )
struct PickOpConversion : public OpConversionPattern<forth::PickOp> {
  PickOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::PickOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::PickOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

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
    rewriter.replaceOpWithMultiple(op, {{memref, stackPtr}});
    return success();
  }
};

/// Conversion pattern for forth.roll operation.
/// Rotates nth element to top: ( xn ... x0 n -- xn-1 ... x0 xn )
/// Uses a CF-based loop to shift elements down.
struct RollOpConversion : public OpConversionPattern<forth::RollOp> {
  RollOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::RollOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::RollOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

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
    rewriter.replaceOpWithMultiple(op, {{memref, spAfterPop}});
    return success();
  }
};

/// Base template for binary arithmetic operations.
/// Pops two values, applies operation, pushes result: (a b -- result)
/// When IsFloat=true, bitcasts i64->f64 before the op and f64->i64 after.
template <typename ForthOp, typename ArithOp, bool IsFloat = false>
struct BinaryArithOpConversion : public OpConversionPattern<ForthOp> {
  BinaryArithOpConversion(const TypeConverter &typeConverter,
                          MLIRContext *context)
      : OpConversionPattern<ForthOp>(typeConverter, context) {}
  using OneToNOpAdaptor =
      typename OpConversionPattern<ForthOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(ForthOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Load top two values (b at SP, a at SP-1)
    Value b = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value a = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    Value result;
    if constexpr (IsFloat) {
      // Bitcast i64 -> f64
      auto f64Type = rewriter.getF64Type();
      Value aF = rewriter.create<arith::BitcastOp>(loc, f64Type, a);
      Value bF = rewriter.create<arith::BitcastOp>(loc, f64Type, b);
      Value resF = rewriter.create<ArithOp>(loc, aF, bF);
      // Bitcast f64 -> i64
      result =
          rewriter.create<arith::BitcastOp>(loc, rewriter.getI64Type(), resF);
    } else {
      result = rewriter.create<ArithOp>(loc, a, b);
    }

    // Store result at SP-1 (effectively popping both and pushing result)
    rewriter.create<memref::StoreOp>(loc, result, memref, spMinus1);

    // New SP is SP-1 (net effect: two pops, one push)
    rewriter.replaceOpWithMultiple(op, {{memref, spMinus1}});
    return success();
  }
};

// Integer arithmetic
using AddIOpConversion = BinaryArithOpConversion<forth::AddIOp, arith::AddIOp>;
using SubIOpConversion = BinaryArithOpConversion<forth::SubIOp, arith::SubIOp>;
using MulIOpConversion = BinaryArithOpConversion<forth::MulIOp, arith::MulIOp>;
using DivIOpConversion = BinaryArithOpConversion<forth::DivIOp, arith::DivSIOp>;
using ModOpConversion = BinaryArithOpConversion<forth::ModOp, arith::RemSIOp>;
using AndOpConversion = BinaryArithOpConversion<forth::AndOp, arith::AndIOp>;
using OrOpConversion = BinaryArithOpConversion<forth::OrOp, arith::OrIOp>;
using XorOpConversion = BinaryArithOpConversion<forth::XorOp, arith::XOrIOp>;
using LshiftOpConversion =
    BinaryArithOpConversion<forth::LshiftOp, arith::ShLIOp>;
using RshiftOpConversion =
    BinaryArithOpConversion<forth::RshiftOp, arith::ShRUIOp>;

// Float arithmetic
using AddFOpConversion =
    BinaryArithOpConversion<forth::AddFOp, arith::AddFOp, true>;
using SubFOpConversion =
    BinaryArithOpConversion<forth::SubFOp, arith::SubFOp, true>;
using MulFOpConversion =
    BinaryArithOpConversion<forth::MulFOp, arith::MulFOp, true>;
using DivFOpConversion =
    BinaryArithOpConversion<forth::DivFOp, arith::DivFOp, true>;

/// Base template for binary comparison operations.
/// Pops two values, compares, pushes -1 (true) or 0 (false): (a b -- flag)
/// When IsFloat=true, bitcasts i64->f64 before comparing.
template <typename ForthOp, typename CmpOp, auto Predicate,
          bool IsFloat = false>
struct BinaryCmpOpConversion : public OpConversionPattern<ForthOp> {
  BinaryCmpOpConversion(const TypeConverter &typeConverter,
                        MLIRContext *context)
      : OpConversionPattern<ForthOp>(typeConverter, context) {}
  using OneToNOpAdaptor =
      typename OpConversionPattern<ForthOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(ForthOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Load top two values (b at SP, a at SP-1)
    Value b = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value a = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    Value cmp;
    if constexpr (IsFloat) {
      auto f64Type = rewriter.getF64Type();
      Value aF = rewriter.create<arith::BitcastOp>(loc, f64Type, a);
      Value bF = rewriter.create<arith::BitcastOp>(loc, f64Type, b);
      cmp = rewriter.create<CmpOp>(loc, Predicate, aF, bF);
    } else {
      cmp = rewriter.create<CmpOp>(loc, Predicate, a, b);
    }

    // Extend i1 to i64: true = -1 (all bits set), false = 0
    Value result =
        rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), cmp);

    // Store result at SP-1
    rewriter.create<memref::StoreOp>(loc, result, memref, spMinus1);

    // New SP is SP-1 (net: two pops, one push)
    rewriter.replaceOpWithMultiple(op, {{memref, spMinus1}});
    return success();
  }
};

// Integer comparisons
using EqIOpConversion = BinaryCmpOpConversion<forth::EqIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::eq>;
using LtIOpConversion = BinaryCmpOpConversion<forth::LtIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::slt>;
using GtIOpConversion = BinaryCmpOpConversion<forth::GtIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::sgt>;
using NeIOpConversion = BinaryCmpOpConversion<forth::NeIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::ne>;
using LeIOpConversion = BinaryCmpOpConversion<forth::LeIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::sle>;
using GeIOpConversion = BinaryCmpOpConversion<forth::GeIOp, arith::CmpIOp,
                                              arith::CmpIPredicate::sge>;

// Float comparisons (ordered predicates)
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

/// Conversion pattern for forth.not operation (bitwise NOT).
/// Unary: pops one value, XORs with -1 (all bits set), pushes result: (a -- ~a)
struct NotOpConversion : public OpConversionPattern<forth::NotOp> {
  NotOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::NotOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::NotOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load top value
    Value a = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // XOR with -1 (all bits set) to flip all bits
    Value allOnes = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(-1));
    Value result = rewriter.create<arith::XOrIOp>(loc, a, allOnes);

    // Store result at same position (SP unchanged)
    rewriter.create<memref::StoreOp>(loc, result, memref, stackPtr);

    rewriter.replaceOpWithMultiple(op, {{memref, stackPtr}});
    return success();
  }
};

/// Conversion pattern for forth.zero_eq operation (0=).
/// Unary: pops one value, pushes -1 if zero, 0 otherwise: (a -- flag)
struct ZeroEqOpConversion : public OpConversionPattern<forth::ZeroEqOp> {
  ZeroEqOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::ZeroEqOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::ZeroEqOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load top value
    Value a = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Compare with zero
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    Value cmp =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, a, zero);

    // Extend i1 to i64: true = -1, false = 0
    Value result =
        rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), cmp);

    // Store result at same position (SP unchanged)
    rewriter.create<memref::StoreOp>(loc, result, memref, stackPtr);

    rewriter.replaceOpWithMultiple(op, {{memref, stackPtr}});
    return success();
  }
};

/// Conversion pattern for forth.param_ref operation.
/// Pushes the byte address of a named kernel parameter onto the stack.
struct ParamRefOpConversion : public OpConversionPattern<forth::ParamRefOp> {
  ParamRefOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::ParamRefOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::ParamRefOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Find the function argument with matching forth.param_name
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp)
      return rewriter.notifyMatchFailure(op, "not inside a func.func");

    StringRef paramName = op.getParamName();
    Value memrefArg;
    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
      auto nameAttr =
          funcOp.getArgAttrOfType<StringAttr>(i, "forth.param_name");
      if (nameAttr && nameAttr.getValue() == paramName) {
        memrefArg = funcOp.getArgument(i);
        break;
      }
    }
    if (!memrefArg)
      return rewriter.notifyMatchFailure(
          op, "no function argument with param_name: " + paramName);

    Value valueToPush;
    if (auto memrefType = dyn_cast<MemRefType>(memrefArg.getType())) {
      // Extract pointer as index, then cast to i64
      Value ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
          loc, memrefArg);
      valueToPush = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), ptrIndex);
    } else if (memrefArg.getType().isInteger(64)) {
      // Scalar i64 param: push value directly.
      valueToPush = memrefArg;
    } else if (memrefArg.getType().isF64()) {
      // Scalar f64 param: bitcast to i64 for stack storage.
      valueToPush = rewriter.create<arith::BitcastOp>(
          loc, rewriter.getI64Type(), memrefArg);
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported param argument type for param_ref");
    }

    // Push onto stack
    Value newSP = pushValue(loc, rewriter, memref, stackPtr, valueToPush);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Generalized memory load template.
/// Pops address from stack, loads value via pointer, pushes value.
/// When IsFloat=true, loads f64 from memory and bitcasts to i64 for stack.
/// AddressSpace selects global (0) or workgroup memory.
template <typename ForthOp, bool IsFloat = false, unsigned AddressSpace = 0>
struct MemoryLoadOpConversion : public OpConversionPattern<ForthOp> {
  MemoryLoadOpConversion(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<ForthOp>(typeConverter, context) {}
  using OneToNOpAdaptor =
      typename OpConversionPattern<ForthOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(ForthOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    auto ptrType =
        LLVM::LLVMPointerType::get(rewriter.getContext(), AddressSpace);

    // Load address from stack
    Value addrValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Load value from memory via pointer
    Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, addrValue);

    Value valueToPush;
    if constexpr (IsFloat) {
      // Load f64 from memory, then bitcast to i64 for stack storage
      Value loadedF64 =
          rewriter.create<LLVM::LoadOp>(loc, rewriter.getF64Type(), ptr);
      valueToPush = rewriter.create<arith::BitcastOp>(
          loc, rewriter.getI64Type(), loadedF64);
    } else {
      valueToPush =
          rewriter.create<LLVM::LoadOp>(loc, rewriter.getI64Type(), ptr);
    }

    // Store loaded value back at same position (replaces address)
    rewriter.create<memref::StoreOp>(loc, valueToPush, memref, stackPtr);

    rewriter.replaceOpWithMultiple(op, {{memref, stackPtr}});
    return success();
  }
};

// Memory load instantiations
using LoadIOpConversion = MemoryLoadOpConversion<forth::LoadIOp>;
using LoadFOpConversion = MemoryLoadOpConversion<forth::LoadFOp, true>;
using SharedLoadIOpConversion =
    MemoryLoadOpConversion<forth::SharedLoadIOp, false, kWorkgroupAddressSpace>;
using SharedLoadFOpConversion =
    MemoryLoadOpConversion<forth::SharedLoadFOp, true, kWorkgroupAddressSpace>;

/// Generalized memory store template.
/// Pops address and value from stack, stores value to memory.
/// When IsFloat=true, bitcasts i64->f64 before storing.
template <typename ForthOp, bool IsFloat = false, unsigned AddressSpace = 0>
struct MemoryStoreOpConversion : public OpConversionPattern<ForthOp> {
  MemoryStoreOpConversion(const TypeConverter &typeConverter,
                          MLIRContext *context)
      : OpConversionPattern<ForthOp>(typeConverter, context) {}
  using OneToNOpAdaptor =
      typename OpConversionPattern<ForthOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(ForthOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    auto ptrType =
        LLVM::LLVMPointerType::get(rewriter.getContext(), AddressSpace);

    // Pop address from stack
    Value addrValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Pop value from stack
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value value = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    // Store value to memory via pointer
    Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, addrValue);

    if constexpr (IsFloat) {
      // Bitcast i64 -> f64 before storing
      Value f64Value =
          rewriter.create<arith::BitcastOp>(loc, rewriter.getF64Type(), value);
      rewriter.create<LLVM::StoreOp>(loc, f64Value, ptr);
    } else {
      rewriter.create<LLVM::StoreOp>(loc, value, ptr);
    }

    // New stack pointer is SP-2 (popped both address and value)
    Value spMinus2 = rewriter.create<arith::SubIOp>(loc, spMinus1, one);
    rewriter.replaceOpWithMultiple(op, {{memref, spMinus2}});
    return success();
  }
};

// Memory store instantiations
using StoreIOpConversion = MemoryStoreOpConversion<forth::StoreIOp>;
using StoreFOpConversion = MemoryStoreOpConversion<forth::StoreFOp, true>;
using SharedStoreIOpConversion =
    MemoryStoreOpConversion<forth::SharedStoreIOp, false,
                            kWorkgroupAddressSpace>;
using SharedStoreFOpConversion =
    MemoryStoreOpConversion<forth::SharedStoreFOp, true,
                            kWorkgroupAddressSpace>;

/// Conversion pattern for forth.itof (S>F).
/// Pops i64, converts to f64 via sitofp, bitcasts back to i64, pushes.
struct IToFOpConversion : public OpConversionPattern<forth::IToFOp> {
  IToFOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::IToFOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::IToFOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load i64 value from top of stack
    Value i64Value = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Convert i64 -> f64 via SIToFPOp
    Value f64Value =
        rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), i64Value);

    // Bitcast f64 -> i64 for stack storage
    Value result =
        rewriter.create<arith::BitcastOp>(loc, rewriter.getI64Type(), f64Value);

    // Store result (SP unchanged — unary op)
    rewriter.create<memref::StoreOp>(loc, result, memref, stackPtr);

    rewriter.replaceOpWithMultiple(op, {{memref, stackPtr}});
    return success();
  }
};

/// Conversion pattern for forth.ftoi (F>S).
/// Pops i64 (f64 bits), bitcasts to f64, converts to i64 via fptosi, pushes.
struct FToIOpConversion : public OpConversionPattern<forth::FToIOp> {
  FToIOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::FToIOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::FToIOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load i64 (f64 bit pattern) from top of stack
    Value i64Bits = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Bitcast i64 -> f64
    Value f64Value =
        rewriter.create<arith::BitcastOp>(loc, rewriter.getF64Type(), i64Bits);

    // Convert f64 -> i64 via FPToSIOp
    Value result =
        rewriter.create<arith::FPToSIOp>(loc, rewriter.getI64Type(), f64Value);

    // Store result (SP unchanged — unary op)
    rewriter.create<memref::StoreOp>(loc, result, memref, stackPtr);

    rewriter.replaceOpWithMultiple(op, {{memref, stackPtr}});
    return success();
  }
};

/// Template for converting GPU indexing ops to intrinsic ops.
/// Creates an intrinsic op with the specified name and pushes the value onto
/// the stack.
template <typename ForthOp>
struct IntrinsicOpConversion : public OpConversionPattern<ForthOp> {
  IntrinsicOpConversion(const TypeConverter &typeConverter,
                        MLIRContext *context, StringRef intrinsicName)
      : OpConversionPattern<ForthOp>(typeConverter, context),
        intrinsicName(intrinsicName) {}

  using OneToNOpAdaptor =
      typename OpConversionPattern<ForthOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(ForthOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Create intrinsic op and push onto stack
    Value intrinsicValue = rewriter.create<forth::IntrinsicOp>(
        loc, rewriter.getIndexType(), rewriter.getStringAttr(intrinsicName));
    Value intrinsicI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), intrinsicValue);
    Value newSP = pushValue(loc, rewriter, memref, stackPtr, intrinsicI64);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }

  std::string intrinsicName;
};

/// Conversion pattern for forth.global_id operation.
/// Computes global_id = bid-x * bdim-x + tid-x using intrinsics.
struct GlobalIdOpConversion : public OpConversionPattern<forth::GlobalIdOp> {
  GlobalIdOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::GlobalIdOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::GlobalIdOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Create intrinsic ops for each component
    Value bidX = rewriter.create<forth::IntrinsicOp>(
        loc, rewriter.getIndexType(), rewriter.getStringAttr("bid-x"));
    Value bdimX = rewriter.create<forth::IntrinsicOp>(
        loc, rewriter.getIndexType(), rewriter.getStringAttr("bdim-x"));
    Value tidX = rewriter.create<forth::IntrinsicOp>(
        loc, rewriter.getIndexType(), rewriter.getStringAttr("tid-x"));

    // Compute: bid-x * bdim-x + tid-x
    Value product = rewriter.create<arith::MulIOp>(loc, bidX, bdimX);
    Value globalId = rewriter.create<arith::AddIOp>(loc, product, tidX);
    Value globalIdI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), globalId);
    Value newSP = pushValue(loc, rewriter, memref, stackPtr, globalIdI64);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Conversion pattern for forth.pop_flag operation.
/// Pops top of stack, compares != 0, returns (memref, newSP, i1 flag).
struct PopFlagOpConversion : public OpConversionPattern<forth::PopFlagOp> {
  PopFlagOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::PopFlagOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::PopFlagOp op, OneToNOpAdaptor adaptor,
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

    // Compare != 0
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value flag = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                topValue, zero);

    // Result 0: output_stack -> {memref, newSP}
    // Result 1: flag -> i1 (passes through unchanged)
    rewriter.replaceOpWithMultiple(op, {{memref, newSP}, {flag}});
    return success();
  }
};

/// Conversion pattern for forth.pop operation.
/// Pops top of stack, returns (memref, newSP, i64 value).
struct PopOpConversion : public OpConversionPattern<forth::PopOp> {
  PopOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::PopOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

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
struct PushValueOpConversion : public OpConversionPattern<forth::PushValueOp> {
  PushValueOpConversion(const TypeConverter &typeConverter,
                        MLIRContext *context)
      : OpConversionPattern<forth::PushValueOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::PushValueOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // The value operand (i64) passes through as-is (not a converted type).
    Value value = adaptor.getOperands()[1][0];

    Value newSP = pushValue(loc, rewriter, memref, stackPtr, value);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Custom FuncOp conversion that calls convertRegionTypes to convert ALL
/// block args (including non-entry blocks used by CF branch ops).
/// The built-in pattern only converts the entry block.
struct FuncOpConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = funcOp.getFunctionType();

    TypeConverter::SignatureConversion result(type.getNumInputs());
    SmallVector<Type, 1> newResults;
    if (failed(getTypeConverter()->convertSignatureArgs(type.getInputs(),
                                                        result)) ||
        failed(getTypeConverter()->convertTypes(type.getResults(), newResults)))
      return failure();

    if (!funcOp.getFunctionBody().empty()) {
      if (failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                             *getTypeConverter(), &result)))
        return failure();
    }

    auto newType = FunctionType::get(rewriter.getContext(),
                                     result.getConvertedTypes(), newResults);
    rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newType); });
    return success();
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

/// Conversion pass implementation.
struct ConvertForthToMemRefPass
    : public impl::ConvertForthToMemRefBase<ConvertForthToMemRefPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto module = getOperation();

    ConversionTarget target(*context);

    // Mark Forth dialect as illegal (to be converted)
    target.addIllegalDialect<forth::ForthDialect>();

    // Mark MemRef, Arith, LLVM, and CF dialects as legal
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                           LLVM::LLVMDialect, cf::ControlFlowDialect>();

    // Mark IntrinsicOp and BarrierOp as legal (to be lowered later)
    target.addLegalOp<forth::IntrinsicOp>();
    target.addLegalOp<forth::BarrierOp>();

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
        // Type conversions
        IToFOpConversion, FToIOpConversion,
        // Control flow
        PopFlagOpConversion, PopOpConversion, PushValueOpConversion>(
        typeConverter, context);

    // Add GPU indexing op conversion patterns
    patterns.add<IntrinsicOpConversion<forth::ThreadIdXOp>>(typeConverter,
                                                            context, "tid-x");
    patterns.add<IntrinsicOpConversion<forth::ThreadIdYOp>>(typeConverter,
                                                            context, "tid-y");
    patterns.add<IntrinsicOpConversion<forth::ThreadIdZOp>>(typeConverter,
                                                            context, "tid-z");
    patterns.add<IntrinsicOpConversion<forth::BlockIdXOp>>(typeConverter,
                                                           context, "bid-x");
    patterns.add<IntrinsicOpConversion<forth::BlockIdYOp>>(typeConverter,
                                                           context, "bid-y");
    patterns.add<IntrinsicOpConversion<forth::BlockIdZOp>>(typeConverter,
                                                           context, "bid-z");
    patterns.add<IntrinsicOpConversion<forth::BlockDimXOp>>(typeConverter,
                                                            context, "bdim-x");
    patterns.add<IntrinsicOpConversion<forth::BlockDimYOp>>(typeConverter,
                                                            context, "bdim-y");
    patterns.add<IntrinsicOpConversion<forth::BlockDimZOp>>(typeConverter,
                                                            context, "bdim-z");
    patterns.add<IntrinsicOpConversion<forth::GridDimXOp>>(typeConverter,
                                                           context, "gdim-x");
    patterns.add<IntrinsicOpConversion<forth::GridDimYOp>>(typeConverter,
                                                           context, "gdim-y");
    patterns.add<IntrinsicOpConversion<forth::GridDimZOp>>(typeConverter,
                                                           context, "gdim-z");

    // GlobalIdOp has custom pattern
    patterns.add<GlobalIdOpConversion>(typeConverter, context);

    // Custom FuncOp + branch patterns for 1:N type conversion
    patterns.add<FuncOpConversion, BranchOpConversion, CondBranchOpConversion>(
        typeConverter, context);
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
