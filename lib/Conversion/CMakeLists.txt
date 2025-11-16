add_mlir_library(MLIRConversionPasses
  Passes.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRForthToMemRefConversion
  MLIRForthToGPUConversion
)

add_subdirectory(ForthToMemRef)
add_subdirectory(ForthToGPU)
