add_mlir_conversion_library(MLIRForthToGPUConversion
  ForthToGPU.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRGPUDialect
  MLIRFuncDialect
  MLIRMemRefDialect
)
