add_mlir_conversion_library(MLIRForthToGPUConversion
  ForthToGPU.cpp

  DEPENDS
  MLIRConversionPassIncGen
  MLIRForth

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRGPUDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRForth
)
