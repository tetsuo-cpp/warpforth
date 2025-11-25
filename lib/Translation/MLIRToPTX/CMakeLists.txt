add_mlir_library(MLIRToPTXTranslation
  MLIRToPTX.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRGPUDialect
  MLIRNVVMDialect
)
