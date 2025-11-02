add_mlir_dialect_library(MLIRForth
  ForthDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/warpforth/Dialect/Forth

  DEPENDS
  MLIRForthOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
