get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(warpforth-translate
  warpforth-translate.cpp
)

llvm_update_compile_flags(warpforth-translate)

target_link_libraries(warpforth-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRForth
  MLIRForthTranslation
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRSupport
  MLIRTranslateLib
)

mlir_check_all_link_libraries(warpforth-translate)
