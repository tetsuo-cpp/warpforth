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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace warpforth {

#define GEN_PASS_DEF_CONVERTFORTHTOGPU
#include "warpforth/Conversion/Passes.h.inc"

namespace {

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
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertForthToGPUPass() {
  return std::make_unique<ConvertForthToGPUPass>();
}

} // namespace warpforth
} // namespace mlir
