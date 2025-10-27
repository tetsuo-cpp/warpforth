//===- ForthToGPU.cpp - Forth to GPU dialect conversion --------*- C++ -*-===//
//
// This file implements the lowering of Forth operations to GPU dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthToGPU/ForthToGPU.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "warpforth/Dialect/Forth/ForthOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::forth;

namespace {

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert Forth arithmetic operations to standard arithmetic operations.
/// These can then be lowered to GPU-specific operations in a later pass.
template <typename ForthOp, typename ArithOp>
struct ForthArithToArithPattern : public OpRewritePattern<ForthOp> {
  using OpRewritePattern<ForthOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForthOp op,
                                 PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ArithOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

/// Convert forth.constant to arith.constant
struct ForthConstantToArithConstant : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern<ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op,
                                 PatternRewriter &rewriter) const override {
    // Cast AnyAttr to TypedAttr for arith.constant
    auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(op.getValue());
    if (!typedAttr) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, typedAttr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Forth to GPU Pass
//===----------------------------------------------------------------------===//

struct LowerForthToGPUPass
    : public PassWrapper<LowerForthToGPUPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerForthToGPUPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, func::FuncDialect>();
  }

  StringRef getArgument() const final { return "lower-forth-to-gpu"; }

  StringRef getDescription() const final {
    return "Lower Forth dialect operations to GPU dialect operations";
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, gpu::GPUDialect, func::FuncDialect>();
    target.addIllegalDialect<ForthDialect>();

    RewritePatternSet patterns(&getContext());
    populateForthToGPUConversionPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                       std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::forth::populateForthToGPUConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ForthConstantToArithConstant>(patterns.getContext());
  patterns.add<ForthArithToArithPattern<AddOp, arith::AddIOp>>(
      patterns.getContext());
  patterns.add<ForthArithToArithPattern<SubOp, arith::SubIOp>>(
      patterns.getContext());
  patterns.add<ForthArithToArithPattern<MulOp, arith::MulIOp>>(
      patterns.getContext());
  patterns.add<ForthArithToArithPattern<DivOp, arith::DivSIOp>>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::forth::createLowerForthToGPUPass() {
  return std::make_unique<LowerForthToGPUPass>();
}
