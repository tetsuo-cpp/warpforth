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

add_subdirectory(ForthToMemRef)
add_subdirectory(ForthToGPU)
