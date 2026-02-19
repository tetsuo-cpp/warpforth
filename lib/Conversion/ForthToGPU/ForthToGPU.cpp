//===- ForthToGPU.cpp - Forth to GPU conversion ----------------*- C++ -*-===//
//
// This file implements the conversion from Forth dialect to GPU dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthToGPU/ForthToGPU.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"

namespace mlir {
namespace warpforth {

#define GEN_PASS_DEF_CONVERTFORTHTOGPU
#include "warpforth/Conversion/Passes.h.inc"

namespace {

/// Maps intrinsic names to GPU dimension operations.
static gpu::Dimension getDimensionFromIntrinsic(StringRef intrinsic) {
  if (intrinsic.ends_with("-x"))
    return gpu::Dimension::x;
  if (intrinsic.ends_with("-y"))
    return gpu::Dimension::y;
  if (intrinsic.ends_with("-z"))
    return gpu::Dimension::z;
  return gpu::Dimension::x; // Default
}

/// Conversion pattern for forth.intrinsic to GPU dialect operations.
/// Maps intrinsic names like "tid-x", "bid-x", etc. to gpu.thread_id,
/// gpu.block_id, gpu.block_dim, gpu.grid_dim operations.
struct IntrinsicOpConversion : public OpConversionPattern<forth::IntrinsicOp> {
  using OpConversionPattern<forth::IntrinsicOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(forth::IntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    StringRef intrinsic = op.getIntrinsic();
    gpu::Dimension dim = getDimensionFromIntrinsic(intrinsic);

    Value result;
    if (intrinsic.starts_with("tid-")) {
      // thread_id: tid-x, tid-y, tid-z
      result =
          rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), dim);
    } else if (intrinsic.starts_with("bid-")) {
      // block_id: bid-x, bid-y, bid-z
      result =
          rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(), dim);
    } else if (intrinsic.starts_with("bdim-")) {
      // block_dim: bdim-x, bdim-y, bdim-z
      result =
          rewriter.create<gpu::BlockDimOp>(loc, rewriter.getIndexType(), dim);
    } else if (intrinsic.starts_with("gdim-")) {
      // grid_dim: gdim-x, gdim-y, gdim-z
      result =
          rewriter.create<gpu::GridDimOp>(loc, rewriter.getIndexType(), dim);
    } else {
      return rewriter.notifyMatchFailure(op, "unknown intrinsic: " + intrinsic);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Applies conversion patterns to a function.
static void applyConversionPatterns(Operation *op, MLIRContext *context) {
  ConversionTarget target(*context);

  // Mark forth.intrinsic as illegal - it must be converted
  target.addIllegalOp<forth::IntrinsicOp>();

  // GPU, Arith, Memref and LLVM dialect ops are legal
  target.addLegalDialect<gpu::GPUDialect, arith::ArithDialect,
                         memref::MemRefDialect, LLVM::LLVMDialect>();

  RewritePatternSet patterns(context);
  patterns.add<IntrinsicOpConversion>(context);

  if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
    return;
  }
}

/// Pass implementation that wraps func.func operations in a single gpu.module
/// and converts them to gpu.func operations.
struct ConvertForthToGPUPass
    : public impl::ConvertForthToGPUBase<ConvertForthToGPUPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());

    SmallVector<func::FuncOp> funcsToConvert;
    module.walk([&](func::FuncOp funcOp) { funcsToConvert.push_back(funcOp); });

    if (funcsToConvert.empty())
      return;

    rewriter.setInsertionPointToStart(&module.getBodyRegion().front());
    auto gpuModule =
        rewriter.create<gpu::GPUModuleOp>(module.getLoc(), "warpforth_module");

    for (auto funcOp : funcsToConvert) {
      convertFuncToGPU(funcOp, gpuModule, rewriter);
    }
  }

private:
  gpu::GPUFuncOp createGPUFunc(func::FuncOp funcOp, gpu::GPUModuleOp gpuModule,
                               IRRewriter &rewriter) {
    rewriter.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
    auto gpuFunc = rewriter.create<gpu::GPUFuncOp>(
        funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType());

    // Copy forth.param_name arg attributes
    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
      if (auto nameAttr =
              funcOp.getArgAttrOfType<StringAttr>(i, "forth.param_name")) {
        gpuFunc.setArgAttr(i, "forth.param_name", nameAttr);
      }
    }

    Block &srcBlock = funcOp.getBody().front(),
          &dstBlock = gpuFunc.getBody().front();

    IRMapping mapping;
    for (auto [srcArg, dstArg] :
         llvm::zip(srcBlock.getArguments(), dstBlock.getArguments())) {
      mapping.map(srcArg, dstArg);
    }

    // Clone all blocks after the entry block.
    for (auto it = std::next(funcOp.getBody().begin());
         it != funcOp.getBody().end(); ++it) {
      Block *newBlock =
          rewriter.createBlock(&gpuFunc.getBody(), gpuFunc.getBody().end());
      mapping.map(&*it, newBlock);
      for (auto arg : it->getArguments()) {
        Value newArg = newBlock->addArgument(arg.getType(), arg.getLoc());
        mapping.map(arg, newArg);
      }
    }

    // Clone ops from each source block into the corresponding destination
    // block, with two transformations:
    // - func.return -> gpu.return
    // - shared memref.alloca -> gpu.func workgroup attribution
    auto *ctx = funcOp.getContext();
    for (auto [srcBlock, dstBlock] :
         llvm::zip(funcOp.getBody(), gpuFunc.getBody())) {
      rewriter.setInsertionPointToEnd(&dstBlock);
      for (Operation &op : srcBlock.getOperations()) {
        if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
          SmallVector<Value> remappedOperands;
          for (Value operand : returnOp.getOperands())
            remappedOperands.push_back(mapping.lookup(operand));
          rewriter.create<gpu::ReturnOp>(returnOp.getLoc(), remappedOperands);
        } else if (auto allocaOp = dyn_cast<memref::AllocaOp>(&op);
                   allocaOp && allocaOp->hasAttr("forth.shared_name")) {
          auto origType = cast<MemRefType>(allocaOp.getType());
          auto addressSpace =
              gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
          auto sharedType =
              MemRefType::get(origType.getShape(), origType.getElementType(),
                              MemRefLayoutAttrInterface{}, addressSpace);
          BlockArgument attr =
              gpuFunc.addWorkgroupAttribution(sharedType, allocaOp.getLoc());
          mapping.map(allocaOp.getResult(), attr);
        } else {
          rewriter.clone(op, mapping);
        }
      }
    }

    // Apply conversion patterns to convert forth.intrinsic ops
    applyConversionPatterns(gpuFunc, funcOp.getContext());

    return gpuFunc;
  }

  void convertFuncToGPU(func::FuncOp funcOp, gpu::GPUModuleOp gpuModule,
                        IRRewriter &rewriter) {
    bool isKernel = funcOp->hasAttr("forth.kernel");

    if (isKernel) {
      auto gpuFunc = createGPUFunc(funcOp, gpuModule, rewriter);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       rewriter.getUnitAttr());
      rewriter.eraseOp(funcOp);
    } else {
      funcOp->moveBefore(&gpuModule.getBodyRegion().front(),
                         gpuModule.getBodyRegion().front().end());
      applyConversionPatterns(funcOp, funcOp.getContext());
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertForthToGPUPass() {
  return std::make_unique<ConvertForthToGPUPass>();
}

} // namespace warpforth
} // namespace mlir
