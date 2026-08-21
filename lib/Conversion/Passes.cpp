//===- Passes.cpp - Conversion Pass Registration ----------------*- C++ -*-===//
//
// This file implements pass registration for conversion passes.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/Passes.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h" // IWYU pragma: keep
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"   // IWYU pragma: keep
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"   // IWYU pragma: keep
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "warpforth/Conversion/ForthToGPU/ForthToGPU.h"
#include "warpforth/Conversion/ForthToMemRef/ForthToMemRef.h"

namespace mlir {
namespace warpforth {

WarpForthPipelineOptions::WarpForthPipelineOptions()
    : chip(*this, "chip", llvm::cl::desc("NVVM target chip"),
           llvm::cl::init("sm_70")),
      features(*this, "features", llvm::cl::desc("NVVM target features"),
               llvm::cl::init("+ptx60")),
      libdevicePath(*this, "libdevice",
                    llvm::cl::desc("Path to the CUDA libdevice bitcode"),
                    llvm::cl::init(WARPFORTH_LIBDEVICE_PATH)),
      optLevel(
          *this, "opt-level",
          llvm::cl::desc("NVVM target optimization level (0, 1, 2, or 3)"),
          llvm::cl::init(llvm::CodeGenOptLevel::Default),
          llvm::cl::values(
              clEnumValN(llvm::CodeGenOptLevel::None, "0", "No optimization"),
              clEnumValN(llvm::CodeGenOptLevel::Less, "1", "Less optimization"),
              clEnumValN(llvm::CodeGenOptLevel::Default, "2",
                         "Default optimization"),
              clEnumValN(llvm::CodeGenOptLevel::Aggressive, "3",
                         "Aggressive optimization"))),
      compilationTarget(
          *this, "compilation-target",
          llvm::cl::desc("GPU output format (llvm, isa, bin, or fatbin)"),
          llvm::cl::init(gpu::CompilationTarget::Assembly),
          llvm::cl::values(clEnumValN(gpu::CompilationTarget::Offload, "llvm",
                                      "LLVM bitcode"),
                           clEnumValN(gpu::CompilationTarget::Assembly, "isa",
                                      "Target assembly"),
                           clEnumValN(gpu::CompilationTarget::Binary, "bin",
                                      "Target binary"),
                           clEnumValN(gpu::CompilationTarget::Fatbin, "fatbin",
                                      "Fat binary"))) {}

void buildWarpForthPipeline(OpPassManager &pm,
                            const WarpForthPipelineOptions &options) {
  // Stage 1: Lower Forth to MemRef (CF ops pass through as-is)
  pm.addPass(createConvertForthToMemRefPass());

  // Stage 2: Convert to GPU dialect (includes private address space annotation)
  pm.addPass(createConvertForthToGPUPass());

  // Stage 3: Normalize MemRefs for GPU
  pm.addPass(createCanonicalizerPass());

  // Stage 4: Attach the configured NVVM target to GPU modules
  GpuNVVMAttachTargetOptions nvvmOptions;
  nvvmOptions.chip = options.chip.getValue();
  nvvmOptions.features = options.features.getValue();
  nvvmOptions.optLevel = static_cast<unsigned>(options.optLevel.getValue());
  if (!options.libdevicePath.getValue().empty())
    nvvmOptions.linkLibs.push_back(options.libdevicePath.getValue());
  pm.addPass(createGpuNVVMAttachTarget(nvvmOptions));

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
  binaryOptions.compilationTarget =
      gpu::stringifyCompilationTarget(options.compilationTarget.getValue());
  pm.addPass(createGpuModuleToBinaryPass(binaryOptions));
}

void buildWarpForthPipeline(OpPassManager &pm) {
  buildWarpForthPipeline(pm, WarpForthPipelineOptions());
}

void registerConversionPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return createConvertForthToMemRefPass();
  });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createConvertForthToGPUPass(); });

  // Register WarpForth pipeline
  PassPipelineRegistration<WarpForthPipelineOptions>(
      "warpforth-pipeline", "WarpForth compilation pipeline (Forth to PTX)",
      [](OpPassManager &pm, const WarpForthPipelineOptions &options) {
        buildWarpForthPipeline(pm, options);
      });
}

} // namespace warpforth
} // namespace mlir
