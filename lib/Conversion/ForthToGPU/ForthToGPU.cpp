//===- ForthToGPU.cpp - Forth to GPU conversion ----------------*- C++ -*-===//
//
// This file implements the conversion from Forth dialect to GPU dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthToGPU/ForthToGPU.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
    OpBuilder builder(module.getContext());

    SmallVector<func::FuncOp> funcsToConvert;
    module.walk([&](func::FuncOp funcOp) { funcsToConvert.push_back(funcOp); });

    if (funcsToConvert.empty())
      return;

    builder.setInsertionPointToStart(&module.getBodyRegion().front());
    auto gpuModule =
        builder.create<gpu::GPUModuleOp>(module.getLoc(), "warpforth_module");

    for (auto funcOp : funcsToConvert) {
      convertFuncToGPU(funcOp, gpuModule, builder);
    }
  }

private:
  void convertFuncToGPU(func::FuncOp funcOp, gpu::GPUModuleOp gpuModule,
                        OpBuilder &builder) {
    // Create a gpu.func inside the WarpForth module
    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
    auto gpuFunc = builder.create<gpu::GPUFuncOp>(
        funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType());

    // Add kernel attribute if this is the main function
    bool isKernel = funcOp.getName() == "main";
    if (isKernel) {
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       builder.getUnitAttr());
    }

    Block &srcBlock = funcOp.getBody().front(),
          &dstBlock = gpuFunc.getBody().front();

    IRMapping mapping;
    for (auto [srcArg, dstArg] :
         llvm::zip(srcBlock.getArguments(), dstBlock.getArguments())) {
      mapping.map(srcArg, dstArg);
    }

    // Copy operations from func.func to gpu.func
    builder.setInsertionPointToStart(&dstBlock);
    for (Operation &op : srcBlock.getOperations()) {
      // Replace func.return with gpu.return
      if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
        builder.create<gpu::ReturnOp>(returnOp.getLoc(),
                                      returnOp.getOperands());
      } else {
        builder.clone(op, mapping);
      }
    }

    funcOp.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertForthToGPUPass() {
  return std::make_unique<ConvertForthToGPUPass>();
}

} // namespace warpforth
} // namespace mlir
