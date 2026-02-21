//===- Passes.cpp - Conversion Pass Registration ----------------*- C++ -*-===//
//
// This file implements pass registration for conversion passes.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/Passes.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "warpforth/Conversion/ForthToGPU/ForthToGPU.h"
#include "warpforth/Conversion/ForthToMemRef/ForthToMemRef.h"

namespace mlir {
namespace warpforth {

void buildWarpForthPipeline(OpPassManager &pm) {
  // Stage 1: Lower Forth to MemRef (CF ops pass through as-is)
  pm.addPass(createConvertForthToMemRefPass());

  // Stage 2: Convert to GPU dialect (includes private address space annotation)
  pm.addPass(createConvertForthToGPUPass());

  // Stage 3: Normalize MemRefs for GPU
  pm.addPass(createCanonicalizerPass());

  // Stage 4: Attach NVVM target to GPU modules (sm_70 = Volta architecture)
  pm.addPass(createGpuNVVMAttachTarget());

  // Stage 5: Lower GPU to NVVM with bare pointers
  ConvertGpuOpsToNVVMOpsOptions gpuToNVVMOptions;
  gpuToNVVMOptions.useBarePtrCallConv = true;
  pm.addNestedPass<gpu::GPUModuleOp>(
      createConvertGpuOpsToNVVMOps(gpuToNVVMOptions));

  // Stage 6: Lower math ops to LLVM intrinsics inside GPU module
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToLLVMPass());

  // Stage 7: Lower NVVM to LLVM
  pm.addPass(createConvertNVVMToLLVMPass());

  // Stage 8: Reconcile type conversions
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Stage 9: Compile GPU module to PTX binary
  GpuModuleToBinaryPassOptions binaryOptions;
  binaryOptions.compilationTarget = "isa"; // Output PTX assembly
  pm.addPass(createGpuModuleToBinaryPass(binaryOptions));
}

void registerConversionPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return createConvertForthToMemRefPass();
  });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createConvertForthToGPUPass(); });

  // Register WarpForth pipeline
  PassPipelineRegistration<>("warpforth-pipeline",
                             "WarpForth compilation pipeline (Forth to PTX)",
                             buildWarpForthPipeline);
}

} // namespace warpforth
} // namespace mlir
