add_mlir_library(MLIRConversionPasses
  Passes.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRForthToMemRefConversion
  MLIRForthToGPUConversion
  MLIRGPUDialect
  MLIRGPUToNVVMTransforms
  MLIRGPUTransforms
  MLIRMathToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
)

target_compile_definitions(obj.MLIRConversionPasses PRIVATE
  WARPFORTH_LIBDEVICE_PATH="${WARPFORTH_LIBDEVICE_PATH}"
)

add_subdirectory(ForthToMemRef)
add_subdirectory(ForthToGPU)
