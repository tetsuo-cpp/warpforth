//===- ForthToMemRef.cpp - Forth to MemRef conversion ----------*- C++ -*-===//
//
// This file implements the conversion from Forth dialect to MemRef dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthToMemRef/ForthToMemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
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

/// Conversion pattern for forth.load operation (@).
/// Pops address from stack, loads from buffer, pushes value: ( addr -- value )
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

    // Get the buffer parameter from the function
    auto funcOp = dyn_cast<func::FuncOp>(op->getParentOp());
    if (!funcOp || funcOp.getNumArguments() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected function with buffer parameter");
    }
    Value bufferArg = funcOp.getArgument(0);

    // Pop address from stack
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value newSP = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value addrValue = rewriter.create<memref::LoadOp>(loc, memref, newSP);

    // Convert address from i64 to index for buffer access
    Value addrIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), addrValue);

    // Load value from buffer at address
    Value loadedValue =
        rewriter.create<memref::LoadOp>(loc, bufferArg, addrIndex);

    // Push loaded value onto stack
    Value finalSP = rewriter.create<arith::AddIOp>(loc, newSP, one);
    rewriter.create<memref::StoreOp>(loc, loadedValue, memref, finalSP);

    rewriter.replaceOpWithMultiple(op, {{memref, finalSP}});
    return success();
  }
};

/// Conversion pattern for forth.store operation (!).
/// Pops address and value from stack, stores value to buffer: ( addr value -- )
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

    // Get the buffer parameter from the function
    auto funcOp = dyn_cast<func::FuncOp>(op->getParentOp());
    if (!funcOp || funcOp.getNumArguments() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected function with buffer parameter");
    }
    Value bufferArg = funcOp.getArgument(0);

    // Pop value from stack
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value spMinus1 = rewriter.create<arith::SubIOp>(loc, stackPtr, one);
    Value value = rewriter.create<memref::LoadOp>(loc, memref, spMinus1);

    // Pop address from stack
    Value spMinus2 = rewriter.create<arith::SubIOp>(loc, spMinus1, one);
    Value addrValue = rewriter.create<memref::LoadOp>(loc, memref, spMinus2);

    // Convert address from i64 to index for buffer access
    Value addrIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), addrValue);

    // Store value to buffer at address
    rewriter.create<memref::StoreOp>(loc, value, bufferArg, addrIndex);

    // New stack pointer is SP-2 (popped both address and value)
    rewriter.replaceOpWithMultiple(op, {{memref, spMinus2}});
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

    // Mark MemRef and Arith dialects as legal
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect>();

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
      return llvm::none_of(op.getResultTypes(),
                           [&](Type t) { return isa<forth::StackType>(t); });
    });

    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return llvm::none_of(op.getOperandTypes(),
                           [&](Type t) { return isa<forth::StackType>(t); });
    });

    ForthToMemRefTypeConverter typeConverter;
    RewritePatternSet patterns(context);

    // Add Forth operation conversion patterns
    patterns.add<StackOpConversion, LiteralOpConversion, DupOpConversion,
                 DropOpConversion, SwapOpConversion, OverOpConversion,
                 RotOpConversion, AddOpConversion, SubOpConversion,
                 MulOpConversion, DivOpConversion, ModOpConversion,
                 LoadOpConversion, StoreOpConversion>(typeConverter, context);

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
