add_mlir_conversion_library(MLIRForthToGPU
  ForthToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/warpforth/Conversion/ForthToGPU

  LINK_LIBS PUBLIC
  MLIRForth
  MLIRArithDialect
  MLIRGPUDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  )
