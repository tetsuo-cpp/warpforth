//===- ForthToMemRef.cpp - Forth to MemRef conversion ----------*- C++ -*-===//
//
// This file implements the conversion from Forth dialect to MemRef dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthToMemRef/ForthToMemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

/// Type converter for forth.stack -> memref + index
class ForthToMemRefTypeConverter : public TypeConverter {
public:
  ForthToMemRefTypeConverter() {
    addConversion(
        [&](Type type,
            SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          if (auto stackType = dyn_cast<forth::StackType>(type)) {
            auto memrefType = MemRefType::get(
                {kStackSize}, IntegerType::get(type.getContext(), 64));
            auto indexType = IndexType::get(type.getContext());
            results.push_back(memrefType);
            results.push_back(indexType);
            return success();
          }
          return std::nullopt;
        });
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

/// Conversion pattern for forth.literal operation.
/// Increments SP and stores the literal value at the new SP position.
struct LiteralOpConversion : public OpConversionPattern<forth::LiteralOp> {
  LiteralOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::LiteralOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::LiteralOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Increment stack pointer
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);

    // Store literal value at new SP position
    Value literalValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), op.getValueAttr());
    rewriter.create<memref::StoreOp>(loc, literalValue, memref, newSP);

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

    // Load value at current SP
    Value topValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Increment SP
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);

    // Store duplicate at new SP
    rewriter.create<memref::StoreOp>(loc, topValue, memref, newSP);

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

    // Load second element (SP - 1)
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value secondValue = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    // Increment SP
    Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);

    // Store second value at new SP
    rewriter.create<memref::StoreOp>(loc, secondValue, memref, newSP);

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

/// Base template for binary arithmetic operations.
/// Pops two values, applies operation, pushes result: (a b -- result)
template <typename ForthOp, typename ArithOp>
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

    // Perform arithmetic operation
    Value result = rewriter.create<ArithOp>(loc, a, b);

    // Store result at SP-1 (effectively popping both and pushing result)
    rewriter.create<memref::StoreOp>(loc, result, memref, spMinus1);

    // New SP is SP-1 (net effect: two pops, one push)
    rewriter.replaceOpWithMultiple(op, {{memref, spMinus1}});
    return success();
  }
};

// Instantiate arithmetic operation conversions
using AddOpConversion = BinaryArithOpConversion<forth::AddOp, arith::AddIOp>;
using SubOpConversion = BinaryArithOpConversion<forth::SubOp, arith::SubIOp>;
using MulOpConversion = BinaryArithOpConversion<forth::MulOp, arith::MulIOp>;
using DivOpConversion = BinaryArithOpConversion<forth::DivOp, arith::DivSIOp>;
using ModOpConversion = BinaryArithOpConversion<forth::ModOp, arith::RemSIOp>;
using AndOpConversion = BinaryArithOpConversion<forth::AndOp, arith::AndIOp>;
using OrOpConversion = BinaryArithOpConversion<forth::OrOp, arith::OrIOp>;
using XorOpConversion = BinaryArithOpConversion<forth::XorOp, arith::XOrIOp>;
using LshiftOpConversion =
    BinaryArithOpConversion<forth::LshiftOp, arith::ShLIOp>;
using RshiftOpConversion =
    BinaryArithOpConversion<forth::RshiftOp, arith::ShRUIOp>;

/// Base template for binary comparison operations.
/// Pops two values, compares, pushes -1 (true) or 0 (false): (a b -- flag)
template <typename ForthOp, arith::CmpIPredicate predicate>
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

    // Compare
    Value cmp = rewriter.create<arith::CmpIOp>(loc, predicate, a, b);

    // Extend i1 to i64: true → -1 (all bits set), false → 0
    Value result =
        rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), cmp);

    // Store result at SP-1
    rewriter.create<memref::StoreOp>(loc, result, memref, spMinus1);

    // New SP is SP-1 (net: two pops, one push)
    rewriter.replaceOpWithMultiple(op, {{memref, spMinus1}});
    return success();
  }
};

// Instantiate comparison operation conversions
using EqOpConversion =
    BinaryCmpOpConversion<forth::EqOp, arith::CmpIPredicate::eq>;
using LtOpConversion =
    BinaryCmpOpConversion<forth::LtOp, arith::CmpIPredicate::slt>;
using GtOpConversion =
    BinaryCmpOpConversion<forth::GtOp, arith::CmpIPredicate::sgt>;

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

    // Extend i1 to i64: true → -1, false → 0
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

    // Extract pointer as index, then cast to i64
    Value ptrIndex =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memrefArg);
    Value ptrI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), ptrIndex);

    // Push onto stack
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);
    rewriter.create<memref::StoreOp>(loc, ptrI64, memref, newSP);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Conversion pattern for forth.load operation (@).
/// Pops address from stack, loads value via pointer, pushes value: ( addr --
/// value )
struct LoadOpConversion : public OpConversionPattern<forth::LoadOp> {
  LoadOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::LoadOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::LoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Load address from stack
    Value addrValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Load value from memory via pointer
    Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, addrValue);
    Value loadedValue = rewriter.create<LLVM::LoadOp>(loc, i64Type, ptr);

    // Store loaded value back at same position (replaces address)
    rewriter.create<memref::StoreOp>(loc, loadedValue, memref, stackPtr);

    rewriter.replaceOpWithMultiple(op, {{memref, stackPtr}});
    return success();
  }
};

/// Conversion pattern for forth.store operation (!).
/// Pops address and value from stack, stores value to memory: ( x addr -- )
struct StoreOpConversion : public OpConversionPattern<forth::StoreOp> {
  StoreOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::StoreOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::StoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Pop address from stack
    Value addrValue = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Pop value from stack
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value value = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    // Store value to memory via pointer
    Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, addrValue);
    rewriter.create<LLVM::StoreOp>(loc, value, ptr);

    // New stack pointer is SP-2 (popped both address and value)
    Value spMinus2 = rewriter.create<arith::SubIOp>(loc, spMinus1, one);
    rewriter.replaceOpWithMultiple(op, {{memref, spMinus2}});
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

    // Increment stack pointer
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);

    // Create intrinsic op
    Value intrinsicValue = rewriter.create<forth::IntrinsicOp>(
        loc, rewriter.getIndexType(), rewriter.getStringAttr(intrinsicName));
    Value intrinsicI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), intrinsicValue);

    // Store at new SP
    rewriter.create<memref::StoreOp>(loc, intrinsicI64, memref, newSP);

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

    // Increment SP and store result
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);
    rewriter.create<memref::StoreOp>(loc, globalIdI64, memref, newSP);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
    return success();
  }
};

/// Conversion pattern for forth.yield operation.
/// Context-aware: inside scf.while's `before` region (from BeginUntilOp),
/// emits flag-pop + scf.condition; otherwise emits scf.yield with SP.
struct YieldOpConversion : public OpConversionPattern<forth::YieldOp> {
  YieldOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::YieldOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::YieldOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange adaptedResult = adaptor.getOperands()[0];
    Value memref = adaptedResult[0];
    Value sp = adaptedResult[1]; // index

    // Check if we're inside scf.while's `before` region.
    auto *parentOp = op->getParentOp();
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      if (op->getParentRegion() == &whileOp.getBefore()) {
        // Pop flag from stack top.
        Value flag = rewriter.create<memref::LoadOp>(loc, memref, sp);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value spAfterPop = rewriter.create<arith::SubIOp>(loc, sp, one);

        // UNTIL exits on non-zero; scf.while continues on true.
        // So keep going when flag == 0.
        Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
        Value keepGoing = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, flag, zero);

        rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, keepGoing,
                                                      ValueRange{spAfterPop});
        return success();
      }
    }

    // Default: emit scf.yield with just the SP.
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, ValueRange{sp});
    return success();
  }
};

/// Conversion pattern for forth.if operation.
/// Loads the flag from the stack top, creates scf.if with the condition,
/// and inlines the region content after converting block args.
struct IfOpConversion : public OpConversionPattern<forth::IfOp> {
  IfOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::IfOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::IfOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Load flag from stack top.
    Value flag = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);

    // Condition: flag != 0.
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                flag, zero);

    // Create scf.if with index result (SP).
    auto indexType = rewriter.getIndexType();
    auto scfIf = rewriter.create<scf::IfOp>(loc, TypeRange{indexType}, cond,
                                            /*addElseBlock=*/true);

    // Convert block signatures and inline regions into scf.if.
    // convertRegionTypes converts !forth.stack block arg → {memref, index}
    // and inserts tracked materializations (unrealized_conversion_cast).
    // We inline into scf.if and mergeBlocks to substitute the converted
    // block args with parent-scope values. The original materialization
    // cast stays intact (tracked by the framework). When the framework
    // later converts the inlined inner ops, their adaptors unwrap the
    // cast to get {memref, index}.
    auto convertRegion = [&](Region &srcRegion,
                             Region &dstRegion) -> LogicalResult {
      if (failed(rewriter.convertRegionTypes(&srcRegion, *getTypeConverter())))
        return failure();

      rewriter.eraseBlock(&dstRegion.front());
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());

      Block &blockWithArgs = dstRegion.front();
      Block *newBlock = rewriter.createBlock(&dstRegion);
      rewriter.mergeBlocks(&blockWithArgs, newBlock, {memref, stackPtr});
      return success();
    };

    if (failed(convertRegion(op.getThenRegion(), scfIf.getThenRegion())))
      return failure();
    if (failed(convertRegion(op.getElseRegion(), scfIf.getElseRegion())))
      return failure();

    // Replace forth.if with {memref, scf.if result SP}.
    rewriter.replaceOpWithMultiple(op, {{memref, scfIf.getResult(0)}});
    return success();
  }
};

/// Conversion pattern for forth.begin_until operation.
/// Creates scf.while with the body as the `before` region (condition test),
/// and an identity `after` region.
struct BeginUntilOpConversion
    : public OpConversionPattern<forth::BeginUntilOp> {
  BeginUntilOpConversion(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<forth::BeginUntilOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::BeginUntilOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    auto indexType = rewriter.getIndexType();

    // Create scf.while with index result, stackPtr as iter arg.
    auto whileOp = rewriter.create<scf::WhileOp>(loc, TypeRange{indexType},
                                                 ValueRange{stackPtr});

    // --- Before region (body): convert + inline ---
    Region &bodyRegion = op.getBodyRegion();
    if (failed(rewriter.convertRegionTypes(&bodyRegion, *getTypeConverter())))
      return failure();

    // scf.while's before region starts empty (no auto-created blocks).
    // Inline the body region into before.
    rewriter.inlineRegionBefore(bodyRegion, whileOp.getBefore(),
                                whileOp.getBefore().end());

    // Merge the block args: replace converted {memref, index} with
    // {memref, beforeSP}.
    Block &beforeBlock = whileOp.getBefore().front();
    Block *newBeforeBlock = rewriter.createBlock(&whileOp.getBefore());
    newBeforeBlock->addArgument(indexType, loc);
    Value beforeSP = newBeforeBlock->getArgument(0);
    rewriter.mergeBlocks(&beforeBlock, newBeforeBlock, {memref, beforeSP});

    // --- After region (identity): just yield the SP ---
    Block *afterBlock = rewriter.createBlock(&whileOp.getAfter());
    afterBlock->addArgument(indexType, loc);
    Value afterSP = afterBlock->getArgument(0);
    rewriter.setInsertionPointToStart(afterBlock);
    rewriter.create<scf::YieldOp>(loc, ValueRange{afterSP});

    // Replace forth.begin_until with {memref, whileOp result}.
    rewriter.replaceOpWithMultiple(op, {{memref, whileOp.getResult(0)}});
    return success();
  }
};

/// Conversion pattern for forth.do_loop operation.
/// Pops start and limit from the stack, creates scf.for from start to limit.
struct DoLoopOpConversion : public OpConversionPattern<forth::DoLoopOp> {
  DoLoopOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<forth::DoLoopOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::DoLoopOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    auto indexType = rewriter.getIndexType();
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Pop start (TOS): load memref[SP], SP -= 1
    Value startI64 = rewriter.create<memref::LoadOp>(loc, memref, stackPtr);
    Value spAfterStart = rewriter.create<arith::SubIOp>(loc, stackPtr, one);

    // Pop limit (new TOS): load memref[SP-1], SP -= 1
    Value limitI64 = rewriter.create<memref::LoadOp>(loc, memref, spAfterStart);
    Value spAfterPops = rewriter.create<arith::SubIOp>(loc, spAfterStart, one);

    // Cast i64 → index for scf.for bounds
    Value startIdx =
        rewriter.create<arith::IndexCastOp>(loc, indexType, startI64);
    Value limitIdx =
        rewriter.create<arith::IndexCastOp>(loc, indexType, limitI64);
    Value stepIdx = one; // step = 1

    // Create scf.for %iv = start to limit step 1 iter_args(%sp = spAfterPops)
    auto forOp = rewriter.create<scf::ForOp>(loc, startIdx, limitIdx, stepIdx,
                                             ValueRange{spAfterPops});

    // Convert body region types and inline into scf.for
    Region &bodyRegion = op.getBodyRegion();
    if (failed(rewriter.convertRegionTypes(&bodyRegion, *getTypeConverter())))
      return failure();

    // Erase the auto-created body block of scf.for
    rewriter.eraseBlock(forOp.getBody());

    // Inline the converted body region into scf.for
    rewriter.inlineRegionBefore(bodyRegion, forOp.getRegion(),
                                forOp.getRegion().end());

    // Merge block args: the converted block has {memref, index} args.
    // Replace with {memref, iter_arg SP}.
    Block &bodyBlock = forOp.getRegion().front();
    Block *newBlock = rewriter.createBlock(&forOp.getRegion());
    // scf.for block has: induction var, then iter args
    newBlock->addArgument(indexType, loc); // induction variable
    newBlock->addArgument(indexType, loc); // SP iter arg
    Value spIterArg = newBlock->getArgument(1);
    rewriter.mergeBlocks(&bodyBlock, newBlock, {memref, spIterArg});

    // Replace forth.do_loop with {memref, forOp result SP}
    rewriter.replaceOpWithMultiple(op, {{memref, forOp.getResult(0)}});
    return success();
  }
};

/// Conversion pattern for forth.loop_index operation (I word).
/// Finds the enclosing scf.for and pushes its induction variable onto the
/// stack.
struct LoopIndexOpConversion : public OpConversionPattern<forth::LoopIndexOp> {
  LoopIndexOpConversion(const TypeConverter &typeConverter,
                        MLIRContext *context)
      : OpConversionPattern<forth::LoopIndexOp>(typeConverter, context) {}
  using OneToNOpAdaptor = OpConversionPattern::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(forth::LoopIndexOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ValueRange inputStack = adaptor.getOperands()[0];
    Value memref = inputStack[0];
    Value stackPtr = inputStack[1];

    // Find enclosing scf.for
    auto forOp = op->getParentOfType<scf::ForOp>();
    if (!forOp)
      return rewriter.notifyMatchFailure(op, "not inside an scf.for");

    // Get induction variable and cast index → i64
    Value iv = forOp.getInductionVar();
    Value ivI64 =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), iv);

    // Push onto stack: SP += 1, store at new SP
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::AddIOp>(loc, stackPtr, one);
    rewriter.create<memref::StoreOp>(loc, ivI64, memref, newSP);

    rewriter.replaceOpWithMultiple(op, {{memref, newSP}});
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

    // Mark MemRef, Arith, LLVM, and SCF dialects as legal
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                           LLVM::LLVMDialect, scf::SCFDialect>();

    // Mark IntrinsicOp as legal (to be lowered later)
    target.addLegalOp<forth::IntrinsicOp>();

    // Use dynamic legality for func operations to ensure they're properly
    // converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      // Function is legal if its signature doesn't contain forth.stack types
      return llvm::none_of(op.getFunctionType().getInputs(),
                           [&](Type t) { return isa<forth::StackType>(t); }) &&
             llvm::none_of(op.getFunctionType().getResults(),
                           [&](Type t) { return isa<forth::StackType>(t); });
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

    ForthToMemRefTypeConverter typeConverter;
    RewritePatternSet patterns(context);

    // Add Forth operation conversion patterns
    patterns
        .add<StackOpConversion, LiteralOpConversion, DupOpConversion,
             DropOpConversion, SwapOpConversion, OverOpConversion,
             RotOpConversion, AddOpConversion, SubOpConversion, MulOpConversion,
             DivOpConversion, ModOpConversion, AndOpConversion, OrOpConversion,
             XorOpConversion, NotOpConversion, LshiftOpConversion,
             RshiftOpConversion, EqOpConversion, LtOpConversion, GtOpConversion,
             ZeroEqOpConversion, ParamRefOpConversion, LoadOpConversion,
             StoreOpConversion, IfOpConversion, BeginUntilOpConversion,
             DoLoopOpConversion, LoopIndexOpConversion, YieldOpConversion>(
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

    // Add built-in function conversion patterns
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
  return std::make_unique<::mlir::warpforth::ConvertForthToMemRefPass>();
}

} // namespace warpforth
} // namespace mlir
