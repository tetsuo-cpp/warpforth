//===- ForthToGPU.cpp - Forth to GPU conversion ----------------*- C++ -*-===//
//
// This file implements the conversion from Forth dialect to GPU dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthToGPU/ForthToGPU.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace warpforth {

#define GEN_PASS_DEF_CONVERTFORTHTOGPU
#include "warpforth/Conversion/Passes.h.inc"

namespace {

/// Pattern to annotate memref.alloca with private address space for
/// thread-local stacks
struct AllocaAddressSpacePattern : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp allocaOp,
                                PatternRewriter &rewriter) const override {
    auto memRefType = cast<MemRefType>(allocaOp.getType());

    if (memRefType.getMemorySpace())
      return failure();

    auto privateAddrSpace = gpu::AddressSpaceAttr::get(
        allocaOp.getContext(), gpu::AddressSpace::Private);
    auto newMemRefType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        memRefType.getLayout(), privateAddrSpace);

    rewriter.modifyOpInPlace(
        allocaOp, [&] { allocaOp.getResult().setType(newMemRefType); });
    return success();
  }
};

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

    Block &srcBlock = funcOp.getBody().front(),
          &dstBlock = gpuFunc.getBody().front();

    IRMapping mapping;
    for (auto [srcArg, dstArg] :
         llvm::zip(srcBlock.getArguments(), dstBlock.getArguments())) {
      mapping.map(srcArg, dstArg);
    }

    rewriter.setInsertionPointToStart(&dstBlock);
    for (Operation &op : srcBlock.getOperations()) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
        rewriter.create<gpu::ReturnOp>(returnOp.getLoc(),
                                       returnOp.getOperands());
      } else {
        rewriter.clone(op, mapping);
      }
    }

    return gpuFunc;
  }

  void convertFuncToGPU(func::FuncOp funcOp, gpu::GPUModuleOp gpuModule,
                        IRRewriter &rewriter) {
    bool isKernel = funcOp.getName() == "main";
    Operation *targetFunc;

    if (isKernel) {
      targetFunc = createGPUFunc(funcOp, gpuModule, rewriter);
      targetFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                          rewriter.getUnitAttr());
      rewriter.eraseOp(funcOp);
    } else {
      funcOp->moveBefore(&gpuModule.getBodyRegion().front(),
                         gpuModule.getBodyRegion().front().end());
      targetFunc = funcOp;
    }

    RewritePatternSet patterns(targetFunc->getContext());
    patterns.add<AllocaAddressSpacePattern>(targetFunc->getContext());
    (void)applyPatternsGreedily(targetFunc, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<Pass> createConvertForthToGPUPass() {
  return std::make_unique<ConvertForthToGPUPass>();
}

} // namespace warpforth
} // namespace mlir
