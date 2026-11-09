add_mlir_library(MLIRConversionPasses
  Passes.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRForthStackConversion
  MLIRForthToGPUConversion
  MLIRGPUDialect
  MLIRGPUToNVVMTransforms
  MLIRGPUTransforms
  MLIRMathToLLVM
  MLIRNVVMToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
)

target_compile_definitions(obj.MLIRConversionPasses PRIVATE
  WARPFORTH_LIBDEVICE_PATH="${WARPFORTH_LIBDEVICE_PATH}"
)

add_subdirectory(ForthStack)
add_subdirectory(ForthToGPU)
