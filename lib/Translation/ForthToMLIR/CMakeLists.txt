add_mlir_library(MLIRForthTranslation
  ForthToMLIR.cpp

  DEPENDS
  MLIRForth

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTranslateLib
  MLIRForth
  MLIRFuncDialect
)
