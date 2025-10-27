add_mlir_library(MLIRForthTranslation
  ForthImporter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/warpforth/Translation

  LINK_LIBS PUBLIC
  MLIRForth
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  )
