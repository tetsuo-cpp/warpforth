add_mlir_conversion_library(MLIRForthStackConversion
  LowerForthStack.cpp
  StackWords.cpp

  DEPENDS
  MLIRConversionPassIncGen
  MLIRForth

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRTransformUtils
  MLIRForth
)
