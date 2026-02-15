add_mlir_conversion_library(MLIRForthToMemRefConversion
  ForthToMemRef.cpp

  DEPENDS
  MLIRConversionPassIncGen
  MLIRForth

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRForth
)
